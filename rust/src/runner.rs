//! Run solver arms over problems and record per-problem results and client events.

use crate::client::{Endpoint, Sampling, Stats};
use crate::orchestrator::{majority_vote, solve_reasoning, Ctx};
use crate::prompts::{AimeItem, SkillPack};
use crate::timeline::{breakdown, Breakdown, Timeline};
use crate::verifier::{VerifierConfig, VerifierPool};
use anyhow::Result;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

pub struct BuildOpts {
    pub endpoints: Vec<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub stream: bool,
    pub sampling: Sampling,
    pub max_tokens: Option<u32>,
    pub request_timeout_s: u32,
    pub deterministic_seed: Option<u32>,
    pub capture_output: bool,
    pub cortheon_src: PathBuf,
    pub python: String,
    pub worker_script: PathBuf,
    pub exec_workers: usize,
    pub light_workers: usize,
    pub sandbox_timeout_s: f64,
    pub agree: usize,
    pub speculative: bool,
}

pub async fn build_ctx(o: &BuildOpts) -> Result<Arc<Ctx>> {
    build_ctx_for_arms(o, &[], 4).await
}

pub async fn build_ctx_for_arms(o: &BuildOpts, arms: &[String], n: usize) -> Result<Arc<Ctx>> {
    anyhow::ensure!(!o.endpoints.is_empty(), "at least one endpoint is required");
    let early_vote = arms.iter().any(|arm| arm == "maj_early");
    if early_vote {
        anyhow::ensure!(
            (1..=16).contains(&n),
            "maj_early needs --n between 1 and 16"
        );
        anyhow::ensure!(o.stream, "maj_early is incompatible with --no-stream");
        anyhow::ensure!(
            o.deterministic_seed.is_none(),
            "maj_early uses distinct sample seeds and is incompatible with --deterministic-seed"
        );
    }
    let max_parallel = if early_vote {
        n.div_ceil(o.endpoints.len()).max(8)
    } else {
        8
    };
    let mut eps: Vec<Arc<Endpoint>> = vec![];
    for e in &o.endpoints {
        let mut endpoint = Endpoint::connect(
            e,
            o.model.as_deref(),
            o.api_key.as_deref(),
            o.stream,
            f64::from(o.request_timeout_s),
            max_parallel,
        )
        .await?;
        endpoint.sampling = o.sampling.clone();
        endpoint.max_tokens = o.max_tokens;
        endpoint.deterministic_seed = o.deterministic_seed;
        endpoint.capture_output = o.capture_output;
        eps.push(Arc::new(endpoint));
    }
    crate::client::check_model_ids(&eps)?;
    let vc = |n: usize| VerifierConfig {
        python: o.python.clone(),
        worker_script: o.worker_script.clone(),
        cortheon_src: Some(o.cortheon_src.clone()),
        workers: n,
    };
    let (exec_pool, light_pool) = if early_vote && arms.iter().all(|arm| arm == "maj_early") {
        (
            VerifierPool::lazy(vc(o.exec_workers)),
            VerifierPool::lazy(vc(o.light_workers)),
        )
    } else {
        tokio::try_join!(
            VerifierPool::start(vc(o.exec_workers)),
            VerifierPool::start(vc(o.light_workers))
        )?
    };
    Ok(Arc::new(Ctx {
        endpoints: eps,
        skills: SkillPack::load(&o.cortheon_src)?,
        exec_pool,
        light_pool,
        sandbox_timeout_s: o.sandbox_timeout_s,
        agree: o.agree,
        speculative: o.speculative,
        serial_attempts: o.deterministic_seed.is_some(),
        stats: Stats::default(),
    }))
}

pub async fn run_one(
    ctx: &Arc<Ctx>,
    arm: &str,
    item: &AimeItem,
    n: usize,
    deep: bool,
) -> Result<(Value, Vec<crate::timeline::Event>, Breakdown)> {
    run_one_with_timeline(ctx, arm, item, n, deep, Timeline::new()).await
}

pub async fn run_one_with_timeline(
    ctx: &Arc<Ctx>,
    arm: &str,
    item: &AimeItem,
    n: usize,
    deep: bool,
    tl: Timeline,
) -> Result<(Value, Vec<crate::timeline::Event>, Breakdown)> {
    ctx.stats.reset();
    tl.emit("problem_start", "", 0, "");
    let samples = sample_count(arm, n);
    let (predicted, detail, extra) = match arm {
        "vanilla_b1" | "pal_b1" => {
            let (prediction, detail) = crate::ablation::single(ctx, &tl, arm, item).await?;
            (prediction, detail, json!({}))
        }
        "cortheon_w1" | "cortheon_full" => {
            let tr = solve_reasoning(
                ctx.clone(),
                &tl,
                &item.prompt,
                n,
                deep,
                arm == "cortheon_full",
            )
            .await?;
            let w1: Vec<String> = tr
                .wave1
                .iter()
                .map(|o| {
                    format!(
                        "{}:{}",
                        o.name,
                        if o.norm_answer.is_empty() {
                            "ERR"
                        } else {
                            &o.norm_answer
                        }
                    )
                })
                .collect();
            let det = format!(
                "{} | wave2={} | w1=[{}]",
                tr.consensus_type,
                tr.wave2_triggered,
                w1.join(", ")
            );
            (
                ctx.light_pool.normalize(&tr.final_answer).await?,
                det,
                json!({"early_aborted": tr.early_aborted, "spec_hits": tr.spec_hits,
                    "wave2_triggered": tr.wave2_triggered, "wave2_reason": tr.wave2_reason,
                    "verified": tr.verified, "final_explanation": tr.final_explanation,
                    "recovered_from_disagreement": tr.recovered_from_disagreement}),
            )
        }
        "maj" | "maj4_prose" | "maj8_prose" => {
            let (p, meta) = majority_vote(ctx.clone(), &tl, &item.prompt, samples).await?;
            let detail = format!(
                "maj{samples}_prose ({}/{samples} votes)",
                meta["winner_count"]
            );
            (p, detail, json!({"vote": meta}))
        }
        "maj_early" => {
            let vote =
                crate::early_vote::run(&ctx.endpoints, &ctx.stats, &tl, &item.prompt, n).await?;
            let predicted = vote.winner.map(|a| a.to_string()).unwrap_or_default();
            let detail = format!(
                "maj_early {}/{} terminals, {}",
                vote.received.len(),
                n,
                vote.error.as_deref().unwrap_or(vote.status())
            );
            (
                predicted,
                detail,
                json!({"status": vote.status(), "vote": vote}),
            )
        }
        other => {
            anyhow::bail!("unknown arm {other:?}")
        }
    };
    let expected = (!item.expected.is_empty()).then_some(&item.expected);
    let (correct, grade) = if extra["status"] == "invalid_stream" {
        (None, "invalid trial".into())
    } else if let Some(expected) = expected {
        if arm == "maj_early" {
            let right = expected
                .parse::<u16>()
                .is_ok_and(|v| v.to_string() == predicted);
            (Some(right), format!("expected {expected}"))
        } else {
            let (right, grade) = ctx.light_pool.grade_aime(&predicted, expected).await?;
            (Some(right), grade)
        }
    } else {
        (None, "unscored".into())
    };
    tl.emit("problem_end", "", 0, "");
    let events = tl.events();
    let b = breakdown(&events);
    let usage: Vec<Value> = events
        .iter()
        .filter(|e| matches!(e.ev.as_str(), "req_end" | "req_cancel"))
        .map(|e| serde_json::from_str::<Value>(&e.note).unwrap_or_default()["server_usage"].clone())
        .collect();
    // A cancelled request often has no final usage block. Only report a per-problem
    // total when every request supplied the field, including any cancelled streams.
    let sum_usage = |path: &str| -> Option<u64> {
        if usage.is_empty() || usage.len() as i64 != b.requests {
            return None;
        }
        usage.iter().map(|u| u.pointer(path)?.as_u64()).sum()
    };
    let pt = sum_usage("/prompt_tokens");
    let ct = sum_usage("/completion_tokens");
    let detail = format!("{detail} | {grade}");
    let detail = if arm != "maj_early" && detail.contains('/') {
        ctx.light_pool.scrub_detail(&detail).await?
    } else {
        detail
    };
    let row = json!({
        "impl": "rust", "arm": arm, "id": item.id, "samples": samples, "expected": expected, "predicted": predicted,
        "correct": correct, "wall_s": b.wall_s, "llm_calls": b.requests, "prompt_tokens": pt,
        "completion_tokens": ct, "reasoning_tokens": sum_usage("/completion_tokens_details/reasoning_tokens"),
        "cached_tokens": sum_usage("/prompt_tokens_details/cached_tokens"),
        "total_tokens": pt.zip(ct).map(|(p, c)| p + c), "detail": detail, "breakdown": b, "extra": extra,
    });
    Ok((row, events, b))
}

fn sample_count(arm: &str, n: usize) -> usize {
    match arm {
        "vanilla_b1" | "pal_b1" => 1,
        "maj4_prose" => 4,
        "maj8_prose" => 8,
        "cortheon_w1" | "cortheon_full" => n.min(4),
        _ => n,
    }
}

pub fn summarize(rows: &[Value]) -> Value {
    let mut arms: Vec<String> = rows
        .iter()
        .filter_map(|r| r["arm"].as_str().map(String::from))
        .collect();
    arms.dedup();
    arms.sort();
    arms.dedup();
    let mut out = serde_json::Map::new();
    for a in arms {
        let rs: Vec<&Value> = rows.iter().filter(|r| r["arm"] == a.as_str()).collect();
        let n = rs.len() as f64;
        let s = |k: &str| {
            rs.iter()
                .map(|r| r["breakdown"][k].as_f64().unwrap_or(0.0))
                .sum::<f64>()
        };
        let wall = s("wall_s");
        let pct = |x: f64| if wall > 0.0 { 100.0 * x / wall } else { 0.0 };
        let k = rs
            .iter()
            .filter(|r| r["correct"].as_bool() == Some(true))
            .count();
        let scored = rs.iter().filter(|r| r["correct"].is_boolean()).count();
        let invalid = rs
            .iter()
            .filter(|r| r["extra"]["status"] == "invalid_stream")
            .count();
        out.insert(
            a.clone(),
            json!({
                "n": rs.len(), "scored": scored, "correct": k, "invalid": invalid,
                "avg_wall_s": wall / n,
                "avg_total_tokens": rs.iter().map(|r| r["total_tokens"].as_f64()).sum::<Option<f64>>().map(|total| total / n),
                "avg_llm_calls": rs.iter().map(|r| r["llm_calls"].as_f64().unwrap_or(0.0)).sum::<f64>() / n,
                "pct": {
                    "gen_prefill": pct(s("gen_prefill_s")), "gen_decode": pct(s("gen_decode_s")), "verify": pct(s("verify_s")),
                    "orch_busy": pct(s("orch_busy_s")), "idle_wait": pct(s("idle_wait_s")),
                },
                "total_s": {
                    "wall": wall, "gen_prefill": s("gen_prefill_s"), "gen_decode": s("gen_decode_s"), "verify": s("verify_s"),
                    "orch_busy": s("orch_busy_s"), "idle_wait": s("idle_wait_s"), "verify_total": s("verify_total_s"),
                    "orchestrator_cpu": s("orchestrator_cpu_s"),
                },
                "cancelled_stream_chunks": rs.iter().map(|r| r["breakdown"]["cancelled_stream_chunks"].as_i64().unwrap_or(0)).sum::<i64>(),
            }),
        );
    }
    Value::Object(out)
}

pub fn write_events(
    w: &mut impl Write,
    arm: &str,
    problem: &str,
    events: &[crate::timeline::Event],
) -> Result<()> {
    for e in events {
        let mut v = serde_json::to_value(e)?;
        let o = v.as_object_mut().unwrap();
        o.insert("impl".into(), json!("rust"));
        o.insert("arm".into(), json!(arm));
        o.insert("problem".into(), json!(problem));
        writeln!(w, "{v}")?;
    }
    Ok(())
}
