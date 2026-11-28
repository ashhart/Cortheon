//! The orchestration hot path: Rust port of reason_forge.solve_reasoning
//! (Wave 1 streaming early-abort, conditional Wave 2 falsifiers, conditional Wave 3,
//! weighted adjudication with order-swapped pairwise judge, fallback extraction)
//! plus the compute-matched majority-vote baseline (maj@N prose voting).

use crate::client::{trunc, CallSpec, CompletionResult, Endpoint, Stats};
use crate::consensus::{branch_weight, cluster, equiv, format_ok, Outcome};
use crate::prompts::{self, Role, SkillPack, SHARED_SYS};
use crate::timeline::Timeline;
use crate::verifier::VerifierPool;
use anyhow::{Context, Result};
use futures::future::{BoxFuture, Fuse, FusedFuture, FutureExt, Shared};
use futures::StreamExt;
use regex::Regex;
use serde_json::{json, Value};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task::JoinSet;

pub struct Ctx {
    pub endpoints: Vec<Arc<Endpoint>>,
    pub skills: SkillPack,
    /// runs model scripts (sandboxed subprocesses)
    pub exec_pool: VerifierPool,
    /// cheap ops: code extraction, prose answers, SymPy equivalence
    pub light_pool: VerifierPool,
    pub sandbox_timeout_s: f64,
    pub agree: usize,
    pub speculative: bool,
    pub serial_attempts: bool,
    pub stats: Stats,
}

#[derive(Clone, Debug, Default)]
pub struct Trace {
    pub wave1: Vec<Outcome>,
    pub wave2: Vec<Outcome>,
    pub wave2_triggered: bool,
    pub wave2_reason: String,
    pub graveyard: Vec<String>,
    pub consensus_type: String,
    pub final_answer: String,
    pub final_explanation: String,
    pub recovered_from_disagreement: bool,
    pub early_aborted: bool,
    pub verified: bool,
    pub spec_hits: usize,
}

impl Ctx {
    fn ep(&self, i: usize) -> Arc<Endpoint> {
        self.endpoints[i % self.endpoints.len()].clone()
    }
}

type ExecFut = Shared<BoxFuture<'static, Result<Arc<Value>, String>>>;

/// The stream polls this same execution that can later supply the final result.
/// Dropping its last owner cancels the worker call and closes the timing span.
fn script_execution(
    pool: VerifierPool,
    tl: Timeline,
    slot: String,
    code: String,
    timeout: f64,
) -> ExecFut {
    async move {
        let _guard = VerifyGuard::new(tl, slot, "exec");
        pool.exec(&code, timeout)
            .await
            .map(Arc::new)
            .map_err(|e| format!("{e:#}"))
    }
    .boxed()
    .shared()
}

struct VerifyGuard {
    tl: Timeline,
    slot: String,
    phase: &'static str,
}
impl VerifyGuard {
    fn new(tl: Timeline, slot: String, phase: &'static str) -> Self {
        tl.emit("verify_start", &slot, 0, phase);
        VerifyGuard { tl, slot, phase }
    }
}
impl Drop for VerifyGuard {
    fn drop(&mut self) {
        self.tl.emit("verify_end", &self.slot, 0, self.phase);
    }
}

pub(crate) struct Candidate {
    code: String,
    execution: ExecFut,
}

fn closed_fences(s: &str) -> usize {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?is)```(?:python|py)?\s*\n.*?```").unwrap())
        .find_iter(s)
        .count()
}

fn apply_exec(out: &mut Outcome, v: &Value) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Execution {
        ok: bool,
        elapsed_s: f64,
        stdout: String,
        raw_answer: String,
        norm_answer: String,
        verified_constraints: bool,
        error: String,
    }
    let result: Execution =
        serde::Deserialize::deserialize(v).context("malformed verifier exec reply")?;
    out.exec_elapsed_s = result.elapsed_s;
    out.exec_ok = Some(result.ok);
    out.exec_stdout = result.stdout;
    out.raw_answer = result.raw_answer;
    out.norm_answer = result.norm_answer;
    out.verified_constraints = result.verified_constraints;
    out.error = result.error;
    Ok(())
}

/// Port of _execute_pal_outcome, with an optional cache of speculative executions.
pub(crate) async fn execute_pal(
    ctx: &Ctx,
    tl: &Timeline,
    out: &mut Outcome,
    slot: &str,
    timeout: f64,
    candidate: Option<Candidate>,
) -> Result<bool> {
    let code = {
        let _guard = VerifyGuard::new(tl.clone(), slot.to_owned(), "extract");
        ctx.light_pool.extract(&out.raw_content).await?
    };
    // A stale candidate must release its worker before the final script queues.
    let candidate = candidate.filter(|c| c.code == code);
    out.code = code.clone();
    if code.is_empty() {
        out.error = "No valid Python code block extracted".into();
        return Ok(false);
    }
    let hit = candidate.is_some();
    let fut = match candidate {
        Some(c) => c.execution,
        None => script_execution(
            ctx.exec_pool.clone(),
            tl.clone(),
            slot.to_string(),
            code,
            timeout,
        ),
    };
    let v = fut.await.map_err(anyhow::Error::msg)?;
    apply_exec(out, &v)?;
    Ok(hit)
}

pub(crate) fn mk_outcome(
    name: &str,
    wave: i32,
    kind: &str,
    res: &CompletionResult,
    ep: &str,
) -> Outcome {
    Outcome {
        name: name.into(),
        wave,
        kind: kind.into(),
        raw_content: res.content.clone(),
        reasoning: res.reasoning.clone(),
        latency_s: res.latency_s,
        completion_tokens: res.completion_tokens,
        error: res.raw_error.clone().unwrap_or_default(),
        endpoint: ep.into(),
        ..Default::default()
    }
}

async fn run_w1_slot(
    ctx: Arc<Ctx>,
    tl: Timeline,
    idx: usize,
    role: Role,
    spec: CallSpec,
) -> Result<(usize, Outcome, bool, f64)> {
    let start = Instant::now();
    let ep = ctx.ep(idx);
    let slot = format!("w1_{}", role.name);
    let mut candidate = None;
    let res = if ctx.speculative && role.kind == "pal" {
        let (tx, rx) = mpsc::unbounded_channel();
        let completion = ep.complete(&spec, &slot, &tl, &ctx.stats, Some(tx));
        tokio::pin!(completion);
        // Both futures belong to the slot. Once completion wins, its listener
        // cannot publish a late candidate or start a job behind the final script.
        tokio::select! {
            result = &mut completion => result,
            () = speculate(&ctx, &tl, &slot, rx, &mut candidate) => completion.await,
        }
    } else {
        ep.complete(&spec, &slot, &tl, &ctx.stats, None).await
    };
    let mut out = mk_outcome(&role.name, 1, &role.kind, &res, &ep.base_url);
    let mut hit = false;
    if res.ok() {
        if role.kind == "pal" {
            hit = execute_pal(&ctx, &tl, &mut out, &slot, ctx.sandbox_timeout_s, candidate).await?;
        } else {
            let (raw, norm) = ctx.light_pool.prose(&res.content).await?;
            out.raw_answer = raw;
            out.norm_answer = norm;
        }
    }
    let elapsed = start.elapsed().as_secs_f64().max(out.latency_s);
    Ok((idx, out, hit, elapsed))
}

fn round_seconds(s: f64) -> f64 {
    (s * 100.0).round_ties_even() / 100.0
}

fn slot_done(tl: &Timeline, idx: usize, out: &Outcome, problem: &str, elapsed: f64) {
    let valid = !out.norm_answer.is_empty() && format_ok(problem, &out.norm_answer);
    let ok = valid && out.error.is_empty() && (out.kind != "pal" || out.exec_ok == Some(true));
    let summary = if ok {
        if out.kind == "pal" {
            "PAL verified"
        } else {
            "analytical"
        }
        .to_owned()
    } else if !out.error.is_empty() {
        trunc(out.error.lines().next().unwrap_or(""), 80)
    } else if !out.norm_answer.is_empty() && !valid {
        "format gate rejected".to_owned()
    } else if out.kind == "pal" {
        "PAL failed".to_owned()
    } else {
        "no answer extracted".to_owned()
    };
    tl.progress(json!({
        "engine": "reason", "event": "slot_done", "wave": out.wave, "slot": idx + 1,
        "role": out.name, "ok": ok, "elapsed_s": round_seconds(elapsed),
        "tokens": out.completion_tokens, "answer": out.norm_answer, "summary": summary,
    }));
}

async fn speculate(
    ctx: &Ctx,
    tl: &Timeline,
    slot: &str,
    rx: mpsc::UnboundedReceiver<String>,
    candidate: &mut Option<Candidate>,
) {
    let scripts = futures::stream::unfold((rx, 0), |(mut rx, mut seen)| async move {
        while let Some(mut snapshot) = rx.recv().await {
            while let Ok(latest) = rx.try_recv() {
                snapshot = latest;
            }
            let count = closed_fences(&snapshot);
            if count > seen {
                seen = count;
                // A speculative extraction is advisory. The final extraction
                // independently checks the complete response.
                let extraction = {
                    let _guard = VerifyGuard::new(tl.clone(), format!("spec:{slot}"), "extract");
                    ctx.light_pool.extract(&snapshot).await
                };
                if let Ok(code) = extraction {
                    return Some((code, (rx, seen)));
                }
            }
        }
        None
    });
    tokio::pin!(scripts);
    let mut running = Fuse::terminated();
    loop {
        tokio::select! {
            update = scripts.next() => {
                let Some(code) = update else { break };
                if candidate.as_ref().is_some_and(|c| c.code == code) {
                    continue;
                }
                *candidate = None;
                running = Fuse::terminated();
                if !code.is_empty() {
                    let execution = script_execution(
                        ctx.exec_pool.clone(),
                        tl.clone(),
                        format!("spec:{slot}"),
                        code.clone(),
                        ctx.sandbox_timeout_s,
                    );
                    running = execution.clone().fuse();
                    *candidate = Some(Candidate { code, execution });
                }
            }
            _ = &mut running, if !running.is_terminated() => {}
        }
    }
}

fn spec_for(
    sys: &str,
    user: String,
    tag: String,
    role_temp: f64,
    seed: i64,
    tb: i64,
    extra: i64,
) -> CallSpec {
    CallSpec {
        messages: vec![("system".into(), sys.into()), ("user".into(), user)],
        tag,
        max_tokens: tb + extra,
        temperature: role_temp,
        seed: Some(seed),
        enable_thinking: true,
        thinking_budget: Some(tb),
        reasoning_effort: None,
        loop_guard: true,
        ..Default::default()
    }
}

async fn batch(
    ctx: &Arc<Ctx>,
    tl: &Timeline,
    specs: Vec<CallSpec>,
) -> Vec<(CompletionResult, String)> {
    let futs = specs.into_iter().enumerate().map(|(i, s)| {
        let ep = ctx.ep(i);
        let (tl, ctx) = (tl.clone(), ctx.clone());
        async move {
            let r = ep.complete(&s, &s.tag, &tl, &ctx.stats, None).await;
            (r, ep.base_url.clone())
        }
    });
    if ctx.serial_attempts {
        let mut results = Vec::new();
        for request in futs {
            results.push(request.await);
        }
        results
    } else {
        futures::future::join_all(futs).await
    }
}

/// Port of reason_forge.solve_reasoning (memory disabled, as in the AIME ablation).
pub async fn solve_reasoning(
    ctx: Arc<Ctx>,
    tl: &Timeline,
    problem: &str,
    n: usize,
    deep: bool,
    enable_wave2: bool,
) -> Result<Trace> {
    let start = Instant::now();
    let mut trace = Trace::default();
    let shared_sys = SHARED_SYS.to_string();
    let prefix = prompts::shared_user_prefix(&ctx.skills, problem);
    let roles = prompts::wave1_roles(&ctx.skills, n, deep)?;

    tl.progress(
        json!({"engine":"reason", "event":"start", "mode":"reason", "streams":roles.len()}),
    );
    let mut set = JoinSet::new();
    let mut ordered: Vec<Option<Outcome>> = vec![None; roles.len()];
    for (i, role) in roles.iter().enumerate() {
        let spec = spec_for(
            &shared_sys,
            format!("{prefix}\n\nPROTOCOL ({}):\n{}", role.name, role.card),
            format!("w1_{}", role.name),
            role.temp,
            role.seed,
            role.think,
            2000,
        );
        let attempt = run_w1_slot(ctx.clone(), tl.clone(), i, role.clone(), spec);
        if ctx.serial_attempts {
            // Capture every role for the matched deterministic comparison.
            let (_, out, hit, elapsed) = attempt.await?;
            slot_done(tl, i, &out, problem, elapsed);
            trace.spec_hits += usize::from(hit);
            ordered[i] = Some(out);
        } else {
            set.spawn(attempt);
        }
    }
    let mut pal_ok: Vec<Outcome> = vec![];
    let collect: Result<()> = async {
    while let Some(j) = set.join_next().await {
        // Match asyncio.wait(FIRST_COMPLETED): consider every already-finished
        // slot, in role order, before judging the batch or cancelling a straggler.
        let mut completed = vec![j??];
        while let Some(ready) = set.try_join_next() {
            completed.push(ready??);
        }
        completed.sort_by_key(|(idx, ..)| *idx);
        for (idx, out, hit, elapsed) in completed {
            slot_done(tl, idx, &out, problem, elapsed);
            trace.spec_hits += usize::from(hit);
            if out.kind == "pal"
                && !out.norm_answer.is_empty()
                && format_ok(problem, &out.norm_answer)
            {
                pal_ok.push(out.clone());
            }
            ordered[idx] = Some(out);
        }
        if !set.is_empty() && !pal_ok.is_empty() {
            let refs: Vec<&Outcome> = pal_ok.iter().collect();
            let cl = cluster(&ctx.light_pool, &refs).await?;
            if let Some((top, members)) = cl.first() {
                if members.len() >= ctx.agree {
                    let mut ana_contradicts = false;
                    for o in ordered.iter().flatten() {
                        if o.kind == "analytical"
                            && !o.norm_answer.is_empty()
                            && !equiv(&ctx.light_pool, &o.norm_answer, top).await?
                        {
                            ana_contradicts = true;
                        }
                    }
                    if !ana_contradicts || members.len() > ctx.agree {
                        tl.emit("consensus", "", 0, top);
                        trace.early_aborted = true;
                        tl.progress(json!({
                            "engine":"reason", "event":"early_abort", "cancelled":set.len(),
                            "elapsed_s":round_seconds(start.elapsed().as_secs_f64()), "consensus_answer":top,
                            "reason":format!("{}/{} PAL consensus", members.len(), roles.len()),
                        }));
                        break;
                    }
                }
            }
        }
    }
    Ok(())
    }.await;
    // Close all in-flight request/verification spans before returning a result or error.
    set.shutdown().await;
    collect?;
    for (i, role) in roles.iter().enumerate() {
        trace.wave1.push(ordered[i].take().unwrap_or(Outcome {
            name: role.name.clone(),
            wave: 1,
            kind: role.kind.clone(),
            error: "cancelled_early_consensus".into(),
            ..Default::default()
        }));
    }

    let valid_pal: Vec<&Outcome> = trace
        .wave1
        .iter()
        .filter(|o| {
            o.kind == "pal" && !o.norm_answer.is_empty() && format_ok(problem, &o.norm_answer)
        })
        .collect();
    let all_valid: Vec<&Outcome> = trace
        .wave1
        .iter()
        .filter(|o| !o.norm_answer.is_empty() && format_ok(problem, &o.norm_answer))
        .collect();
    let pal_clusters = cluster(&ctx.light_pool, &valid_pal).await?;
    let all_clusters = cluster(&ctx.light_pool, &all_valid).await?;
    let mut need_wave2 = false;
    if valid_pal.is_empty() {
        need_wave2 = true;
        trace.wave2_reason = "All Wave-1 Python verification scripts failed or timed out".into();
    } else if pal_clusters.len() > 1 {
        need_wave2 = true;
        let s: Vec<String> = pal_clusters
            .iter()
            .map(|(a, m)| format!("{a} ({})", m.len()))
            .collect();
        trace.wave2_reason = format!("PAL script disagreement: {}", s.join(", "));
    } else if valid_pal.len() == 1 {
        if all_clusters.len() != 1 || all_valid.len() < 2 {
            need_wave2 = true;
            trace.wave2_reason =
                "Only 1 PAL script succeeded without independent corroboration".into();
        }
    } else if pal_clusters.len() == 1 && all_clusters.len() > 1 && valid_pal.len() < 3 {
        need_wave2 = true;
        trace.wave2_reason = format!(
            "PAL ({}) disagreed with analytical stream; running falsifier",
            pal_clusters[0].0
        );
    }
    for o in &trace.wave1 {
        if !o.error.is_empty() {
            trace
                .graveyard
                .push(format!("[{}] FAILED: {}", o.name, trunc(&o.error, 240)));
        }
    }

    if need_wave2 && enable_wave2 {
        trace.wave2_triggered = true;
        let mut conflict_lines = vec![];
        for (rep, members) in &all_clusters {
            let names: Vec<&str> = members
                .iter()
                .map(|&i| all_valid[i].name.as_str())
                .collect();
            let snip = members
                .iter()
                .map(|&i| all_valid[i])
                .find(|m| !m.code.is_empty())
                .map(|m| trunc(&m.code, 700))
                .unwrap_or_default();
            conflict_lines.push(format!(
                "Candidate Answer = {rep} (from: {})\n{}",
                names.join(", "),
                if snip.is_empty() {
                    String::new()
                } else {
                    format!("Excerpt of method:\n```python\n{snip}\n```")
                }
            ));
        }
        let gy = if trace.graveyard.is_empty() {
            "(none)".to_string()
        } else {
            trace.graveyard.join("\n")
        };
        let dossier = format!(
            "PROBLEM:\n{}\n\nWAVE-1 STATUS: {}\n\nCOMPETING CANDIDATES:\n{}\n\nGRAVEYARD (DO NOT REPEAT THESE BUGS):\n{gy}",
            problem.trim(),
            trace.wave2_reason,
            if conflict_lines.is_empty() { "(none succeeded)".to_string() } else { conflict_lines.join("\n---\n") }
        );
        let tb = if deep { 1100 } else { 650 };
        let w2 = [
            ("falsifier_bruteforce", 0.25, 505),
            ("falsifier_reread", 0.45, 606),
            ("falsifier_z3_sympy", 0.40, 707),
        ];
        let mut specs = vec![];
        for (name, temp, seed) in w2 {
            let card = ctx.skills.card(name)?;
            specs.push(spec_for(
                &shared_sys,
                format!("{dossier}\n\nPROTOCOL ({name}):\n{card}"),
                format!("w2_{name}"),
                temp,
                seed,
                tb,
                2200,
            ));
        }
        tl.progress(json!({"engine":"reason", "event":"wave2_start", "reason":trace.wave2_reason, "streams":specs.len()}));
        let wave_start = Instant::now();
        let results = batch(&ctx, tl, specs).await;
        let mut jobs = vec![];
        for ((name, _, _), (res, ep)) in w2.iter().zip(results.iter()) {
            let o = mk_outcome(name, 2, "pal", res, ep);
            if res.ok() {
                jobs.push(trace.wave2.len());
            }
            trace.wave2.push(o);
        }
        let t2 = ctx.sandbox_timeout_s + 4.0;
        exec_many(&ctx, tl, &mut trace.wave2, &jobs, t2).await?;
        let wave_elapsed = wave_start.elapsed().as_secs_f64();
        for (idx, o) in trace.wave2.iter().enumerate() {
            if !o.error.is_empty() {
                trace
                    .graveyard
                    .push(format!("[{}] FAILED: {}", o.name, trunc(&o.error, 240)));
            }
            let elapsed = o.latency_s + o.exec_elapsed_s;
            slot_done(
                tl,
                idx,
                o,
                problem,
                if elapsed > 0.0 { elapsed } else { wave_elapsed },
            );
        }
        let any_valid = trace.wave1.iter().chain(trace.wave2.iter()).any(|o| {
            o.kind == "pal"
                && o.exec_ok == Some(true)
                && !o.norm_answer.is_empty()
                && format_ok(problem, &o.norm_answer)
        });
        if !any_valid {
            wave3(&ctx, tl, &mut trace, problem, &shared_sys).await?;
        }
    }

    adjudicate(&ctx, tl, &mut trace, problem, need_wave2, enable_wave2).await?;
    tl.progress(json!({
        "engine":"reason", "event":"done", "elapsed_s":round_seconds(start.elapsed().as_secs_f64()),
        "verified":trace.verified, "summary":format!("{} -> {}", trace.consensus_type, trace.final_answer),
    }));
    Ok(trace)
}

async fn exec_many(
    ctx: &Arc<Ctx>,
    tl: &Timeline,
    outs: &mut [Outcome],
    jobs: &[usize],
    timeout: f64,
) -> Result<()> {
    let futs = jobs.iter().map(|&i| {
        let mut o = outs[i].clone();
        let (ctx, tl) = (ctx.clone(), tl.clone());
        async move {
            let slot = format!("w{}_{}", o.wave, o.name);
            execute_pal(&ctx, &tl, &mut o, &slot, timeout, None).await?;
            Ok::<_, anyhow::Error>((i, o))
        }
    });
    for (i, o) in futures::future::try_join_all(futs).await? {
        outs[i] = o;
    }
    Ok(())
}

async fn wave3(
    ctx: &Arc<Ctx>,
    tl: &Timeline,
    trace: &mut Trace,
    problem: &str,
    shared_sys: &str,
) -> Result<()> {
    // Python scans reversed(wave2 + wave1), preferring the last Wave-1 script.
    let closest = trace
        .wave1
        .iter()
        .rev()
        .chain(trace.wave2.iter().rev())
        .find(|o| o.kind == "pal" && !o.code.is_empty())
        .cloned();
    let start = trace.graveyard.len().saturating_sub(6);
    let dossier = format!(
        "PROBLEM:\n{}\n\nCRITICAL: Wave-1 and Wave-2 Python scripts failed or produced format-invalid output.\nGRAVEYARD OF PRIOR ATTEMPTS:\n{}{}",
        problem.trim(),
        trace.graveyard[start..].join("\n"),
        closest.map(|c| format!("\n\nCLOSEST SCRIPT (`{}`):\n```python\n{}\n```", c.name, trunc(&c.code, 2200))).unwrap_or_default()
    );
    let roles: [(&str, &str, &str, f64, i64, i64); 3] = [
        ("w3_pal_z3_exact", "pal", "Write a SHORT (<55 lines), rock-solid `z3` (`z3.Optimize()` with `z3.Real`/`z3.Int`) OR `sympy` script directly modeling the equations/constraints (or clean coordinate/numerical verification snapped to exact answer). Print `ANSWER: <exact_value>` on the last line. Output ONLY one ```python``` block.", 0.20, 808, 1000),
        ("w3_analytical_deep_a", "analytical", "Solve this problem completely from first principles with rigorous step-by-step mathematical proof (synthetic geometry, algebra, number theory, or combinatorics — do NOT write a broken script). Verify every step, symmetry factor, and arithmetic calculation twice. End with `ANSWER: <exact_value>`.", 0.25, 909, 1800),
        ("w3_analytical_deep_b", "analytical", "Solve this problem independently via an ALTERNATIVE mathematical approach (e.g., coordinate/trigonometric/algebraic if the obvious route is synthetic, or vice versa). Double-check for off-by-2 symmetry or counting traps. End with `ANSWER: <exact_value>`.", 0.55, 1010, 1800),
    ];
    let specs: Vec<CallSpec> = roles
        .iter()
        .map(|(name, kind, card, temp, seed, tb)| {
            let sys = if *kind == "pal" {
                shared_sys
            } else {
                "You are an IMO/Putnam gold-medalist mathematician. Be rigorous and exact."
            };
            let mut s = spec_for(
                sys,
                format!("{dossier}\n\nPROTOCOL ({name}):\n{card}"),
                name.to_string(),
                *temp,
                *seed,
                *tb,
                2200,
            );
            if *kind == "analytical" {
                s.max_tokens = 8192;
            }
            s
        })
        .collect();
    let wave_start = Instant::now();
    let first = trace.wave2.len();
    let results = batch(ctx, tl, specs).await;
    let mut jobs = vec![];
    for ((name, kind, ..), (res, ep)) in roles.iter().zip(results.iter()) {
        let mut o = mk_outcome(name, 3, kind, res, ep);
        if res.ok() {
            if *kind == "pal" {
                jobs.push(trace.wave2.len());
            } else {
                let (raw, norm) = ctx.light_pool.prose(&res.content).await?;
                o.raw_answer = raw;
                o.norm_answer = norm;
            }
        }
        trace.wave2.push(o);
    }
    let t3 = ctx.sandbox_timeout_s + 6.0;
    exec_many(ctx, tl, &mut trace.wave2, &jobs, t3).await?;
    let wall = wave_start.elapsed().as_secs_f64();
    for (idx, o) in trace.wave2[first..].iter().enumerate() {
        let elapsed = o.latency_s + o.exec_elapsed_s;
        slot_done(
            tl,
            idx,
            o,
            problem,
            if elapsed > 0.0 { elapsed } else { wall },
        );
    }
    Ok(())
}

async fn pairwise_vote(ctx: &Arc<Ctx>, tl: &Timeline, p_ab: String, p_ba: String) -> String {
    let sys = "You are a strict, impartial verifier. Compare Candidate A and Candidate B strictly against the problem specification and edge cases. Reply with only A or B.";
    let mk = |p: String, tag: &str, idx: usize| CallSpec {
        messages: vec![("system".into(), sys.into()), ("user".into(), p)],
        tag: tag.into(),
        max_tokens: if ctx.ep(idx).caps.structures && ctx.ep(idx).caps.is_tensorfold {
            16
        } else {
            64
        },
        temperature: 0.0,
        enable_thinking: false,
        guided_choice: Some(vec!["A".into(), "B".into()]),
        ..Default::default()
    };
    let r = batch(
        ctx,
        tl,
        vec![
            mk(p_ab, "reason_judge_ab", 0),
            mk(p_ba, "reason_judge_ba", 1),
        ],
    )
    .await;
    let (ab, ba) = (&r[0].0, &r[1].0);
    if !(ab.ok() && ba.ok()) {
        return "TIE".into();
    }
    let vab = ab.content.trim();
    let vba = match ba.content.trim() {
        "A" => "B",
        "B" => "A",
        _ => "TIE",
    };
    if vab == vba && (vab == "A" || vab == "B") {
        vab.to_string()
    } else {
        "TIE".into()
    }
}

async fn adjudicate(
    ctx: &Arc<Ctx>,
    tl: &Timeline,
    trace: &mut Trace,
    problem: &str,
    need_wave2: bool,
    enable_wave2: bool,
) -> Result<()> {
    let combined: Vec<Outcome> = trace
        .wave1
        .iter()
        .chain(trace.wave2.iter())
        .cloned()
        .collect();
    let mut valid: Vec<&Outcome> = combined
        .iter()
        .filter(|o| !o.norm_answer.is_empty() && format_ok(problem, &o.norm_answer))
        .collect();
    if valid.is_empty() {
        valid = combined
            .iter()
            .filter(|o| !o.norm_answer.is_empty())
            .collect();
    }
    let clusters = cluster(&ctx.light_pool, &valid).await?;
    if clusters.is_empty() {
        // Python len() counts Unicode characters and max() retains the first tie.
        let best = combined
            .iter()
            .rev()
            .max_by_key(|o| o.raw_content.chars().count());
        let work = best
            .map(|b| trunc(&b.raw_content, 3500))
            .unwrap_or_default();
        let prompt = format!("PROBLEM:\n{}\n\nWORK:\n{work}\n\nExtract ONLY the exact final answer cleanly. End with `ANSWER: <value>`.", problem.trim());
        let schema: Value = json!({
            "type": "object",
            "properties": {
                "final_answer": {"type": "string"},
                "answer_type": {"type": "string", "enum": ["integer", "fraction", "float", "boolean", "string", "expression", "list"]},
                "confidence_notes": {"type": "string"}
            },
            "required": ["final_answer", "answer_type", "confidence_notes"],
            "additionalProperties": false
        });
        let spec = CallSpec {
            messages: vec![("user".into(), prompt)],
            tag: "reason_fallback_extract".into(),
            max_tokens: 256,
            temperature: 0.0,
            enable_thinking: false,
            json_schema: Some(schema),
            schema_name: "answer_extract".into(),
            ..Default::default()
        };
        let r = ctx.endpoints[0]
            .complete(&spec, "reason_fallback_extract", tl, &ctx.stats, None)
            .await;
        if r.ok() {
            trace.final_answer = ctx.light_pool.fallback_answer(&r.content).await?;
            trace.consensus_type = "fallback_extract".into();
        } else {
            trace.final_answer.clear();
            trace.consensus_type = "fallback_extract_failed".into();
        }
        trace.final_explanation = best.map(|out| out.raw_content.clone()).unwrap_or_default();
        return Ok(());
    }
    let mut scored: Vec<(f64, usize, String, Vec<usize>)> = clusters
        .iter()
        .map(|(rep, m)| {
            let mut w: f64 = m.iter().map(|&i| branch_weight(valid[i])).sum();
            let pal_count = m
                .iter()
                .filter(|&&i| valid[i].kind == "pal" && valid[i].exec_ok == Some(true))
                .count();
            let w2 = m
                .iter()
                .filter(|&&i| valid[i].wave == 2 && valid[i].exec_ok == Some(true))
                .count();
            if w2 >= 2 {
                w += 3.5;
            }
            (w, pal_count, rep.clone(), m.clone())
        })
        .collect();
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
    });
    let (_, winner_pals, mut winner, mut members) = scored[0].clone();
    let margin = if winner_pals <= 1 { 2.25 } else { 1.5 };
    if scored.len() >= 2 && scored[0].0 - scored[1].0 <= margin {
        let (ans_a, mem_a) = (scored[0].2.clone(), scored[0].3.clone());
        let (ans_b, mem_b) = (scored[1].2.clone(), scored[1].3.clone());
        let code_of = |m: &Vec<usize>| {
            trunc(
                &m.iter()
                    .map(|&i| valid[i])
                    .find(|o| !o.code.is_empty())
                    .map(|o| o.code.clone())
                    .unwrap_or_else(|| valid[m[0]].raw_content.clone()),
                2500,
            )
        };
        let (ca, cb) = (code_of(&mem_a), code_of(&mem_b));
        let q = "Which candidate faithfully models EVERY word and constraint of the problem with no bug? Reply A or B.";
        let p_ab = format!("PROBLEM:\n{}\n\nCANDIDATE A (Answer: {ans_a}):\n{ca}\n\nCANDIDATE B (Answer: {ans_b}):\n{cb}\n\n{q}", problem.trim());
        let p_ba = format!("PROBLEM:\n{}\n\nCANDIDATE A (Answer: {ans_b}):\n{cb}\n\nCANDIDATE B (Answer: {ans_a}):\n{ca}\n\n{q}", problem.trim());
        let mut vote = pairwise_vote(ctx, tl, p_ab.clone(), p_ba).await;
        let mut audit_failed = false;
        if vote == "TIE" {
            let spec = CallSpec {
                messages: vec![
                    ("system".into(), "Compare Candidate A and Candidate B line by line against every single clue/constraint in the problem. End with WINNER: A or WINNER: B.".into()),
                    ("user".into(), p_ab),
                ],
                tag: "reason_judge_audit".into(),
                max_tokens: 1400,
                temperature: 0.0,
                enable_thinking: true,
                thinking_budget: Some(700),
                loop_guard: true,
                ..Default::default()
            };
            let a = ctx.endpoints[0]
                .complete(&spec, "reason_judge_audit", tl, &ctx.stats, None)
                .await;
            if a.ok() {
                let txt = a.content.to_uppercase();
                if txt.contains("WINNER: B") {
                    vote = "B".into();
                } else if txt.contains("WINNER: A") {
                    vote = "A".into();
                }
            } else {
                audit_failed = true;
            }
        }
        if vote == "B" {
            winner = ans_b;
            members = mem_b;
            trace.consensus_type = "pairwise_tiebreak_B".into();
        } else if audit_failed {
            trace.consensus_type = "pairwise_audit_failed_default_A".into();
        } else {
            trace.consensus_type = "pairwise_tiebreak_A".into();
        }
    } else if trace.wave2_triggered {
        trace.consensus_type = format!("wave2_verified ({winner_pals} PAL)");
    } else if need_wave2 && !enable_wave2 {
        trace.consensus_type = format!("wave1_only ({winner_pals} PAL)");
    } else {
        trace.consensus_type = format!("wave1_consensus ({winner_pals} PAL)");
    }
    trace.verified = members
        .iter()
        .any(|&i| valid[i].kind == "pal" && valid[i].exec_ok == Some(true));
    if trace.wave2_triggered {
        let first_pal = trace
            .wave1
            .iter()
            .find(|o| o.kind == "pal" && !o.norm_answer.is_empty());
        trace.recovered_from_disagreement = match first_pal {
            Some(pal) => !equiv(&ctx.light_pool, &pal.norm_answer, &winner).await?,
            None => true,
        };
    }
    let best = members
        .iter()
        .rev()
        .map(|&i| valid[i])
        .max_by_key(|m| (m.wave, m.verified_constraints))
        .unwrap();
    trace.final_explanation = format!(
        "Verified Answer: **{winner}**\n\nVerification: {} ({}/{} agreeing branches)\n",
        trace.consensus_type,
        members.len(),
        valid.len()
    );
    let preview = trunc(best.exec_stdout.trim(), 800);
    if !preview.is_empty() {
        trace.final_explanation.push_str(&format!(
            "\nVerification Output (`{}`):\n```\n{preview}\n```",
            best.name
        ));
    }
    trace.final_answer = winner;
    Ok(())
}

/// Run upstream's published temperature schedules at widths 4 and 8. Other
/// widths keep the generic maj arm's repeating four-temperature schedule.
/// Verifier errors propagate to the caller so an incomplete vote can be excluded.
pub async fn majority_vote(
    ctx: Arc<Ctx>,
    tl: &Timeline,
    prompt: &str,
    n: usize,
) -> Result<(String, Value)> {
    let temps = if n == 8 {
        &[0.20, 0.30, 0.40, 0.50, 0.35, 0.45, 0.55, 0.60][..]
    } else {
        &[0.25, 0.55, 0.45, 0.35][..]
    };
    let cfgs: Vec<(i64, f64)> = (0..n)
        .map(|i| (101 * (i as i64 + 1), temps[i % temps.len()]))
        .collect();
    let specs: Vec<CallSpec> = cfgs
        .iter()
        .enumerate()
        .map(|(i, (seed, temp))| CallSpec {
            messages: vec![("user".into(), prompt.to_string())],
            tag: format!("maj_s{i}"),
            max_tokens: 4096,
            temperature: *temp,
            top_p: 0.95,
            top_k: 20,
            seed: Some(*seed),
            enable_thinking: true,
            thinking_budget: Some(1000),
            loop_guard: true,
            ..Default::default()
        })
        .collect();
    let results = batch(&ctx, tl, specs).await;
    let mut branches = Vec::with_capacity(n);
    let mut candidates = vec![];
    let mut valid_count = 0;
    for (i, ((_, temp), (res, _))) in cfgs.iter().zip(results.iter()).enumerate() {
        let text = if res.ok() { res.content.as_str() } else { "" };
        let (norm, integer) = ctx.light_pool.aime_candidate(text).await?;
        let valid = !integer.is_empty();
        branches.push(json!({"slot": i, "temp": temp, "norm": norm, "valid_aime_int": integer}));
        if !norm.is_empty() {
            valid_count += usize::from(valid);
            candidates.push((i, *temp, if valid { integer } else { norm }, valid));
        }
    }
    // Rank by most votes, then lowest temperature, then earliest slot.
    struct Cluster {
        answer: String,
        count: usize,
        min_temp: f64,
        first_slot: usize,
    }
    let mut clusters: Vec<Cluster> = vec![];
    for (slot, temp, answer, valid) in candidates {
        if valid_count != 0 && !valid {
            continue;
        }
        let mut found = None;
        for c in &mut clusters {
            if c.answer == answer || (!valid && equiv(&ctx.light_pool, &c.answer, &answer).await?) {
                found = Some(c);
                break;
            }
        }
        if let Some(c) = found {
            c.count += 1;
            c.min_temp = c.min_temp.min(temp);
        } else {
            clusters.push(Cluster {
                answer,
                count: 1,
                min_temp: temp,
                first_slot: slot,
            });
        }
    }
    clusters.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then(a.min_temp.total_cmp(&b.min_temp))
            .then(a.first_slot.cmp(&b.first_slot))
    });
    let first = clusters.first();
    let winner_count = first.map_or(0, |c| c.count);
    let counts: serde_json::Map<_, _> = clusters
        .iter()
        .map(|c| (c.answer.clone(), json!(c.count)))
        .collect();
    Ok((
        first.map_or_else(String::new, |c| c.answer.clone()),
        json!({
            "valid_count": valid_count, "winner_count": winner_count,
            "tie_broken_by_temp": clusters.get(1).is_some_and(|c| c.count == winner_count),
            "vote_counts": counts, "per_branch": branches, "used_fallback": valid_count == 0,
        }),
    ))
}

pub(crate) async fn extract_prose_candidate(light: &VerifierPool, text: &str) -> Result<String> {
    let v = light
        .call(
            json!({"op": "aime_prose", "content": text}),
            std::time::Duration::from_secs(30),
        )
        .await?;
    v["norm"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| anyhow::anyhow!("AIME answer extraction failed: {v}"))
}

pub fn aime_int(norm: &str) -> String {
    if norm.chars().all(|c| c.is_ascii_digit()) && !norm.is_empty() {
        if let Ok(v) = norm.parse::<u64>() {
            if v <= 999 {
                return v.to_string();
            }
        }
    }
    String::new()
}

/// Grade with the upstream Python routine. A grader failure returns an error.
pub async fn grade_aime(
    light: &VerifierPool,
    predicted: &str,
    expected: &str,
) -> Result<(bool, String)> {
    light.grade_aime(predicted, expected).await
}
