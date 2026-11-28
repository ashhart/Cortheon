//! The orchestration hot path: Rust port of reason_forge.solve_reasoning
//! (Wave 1 streaming early-abort, conditional Wave 2 falsifiers, conditional Wave 3,
//! weighted adjudication with order-swapped pairwise judge, fallback extraction)
//! plus the compute-matched majority-vote baseline (maj@N prose voting).

use crate::client::{trunc, CallSpec, CompletionResult, Endpoint, Stats};
use crate::consensus::{branch_weight, cluster, equiv, format_ok, Outcome};
use crate::prompts::{self, Role, SkillPack, SHARED_SYS};
use crate::timeline::Timeline;
use crate::verifier::VerifierPool;
use anyhow::Result;
use futures::future::{BoxFuture, FutureExt, Shared};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
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
    pub early_aborted: bool,
    pub verified: bool,
    pub spec_hits: usize,
}

impl Ctx {
    fn ep(&self, i: usize) -> Arc<Endpoint> {
        self.endpoints[i % self.endpoints.len()].clone()
    }
}

type ExecFut = Shared<BoxFuture<'static, Arc<Value>>>;

/// Start a verifier job immediately (spawned). The returned shared future yields the
/// result; when every clone is dropped (slot cancelled, speculation unused) the job is
/// aborted, which makes the worker kill the sandboxed script.
fn start_exec(
    pool: VerifierPool,
    tl: Timeline,
    slot: String,
    code: String,
    timeout: f64,
) -> ExecFut {
    let guard = VerifyGuard::new(tl, slot);
    let h = tokio::spawn(async move {
        let _g = guard;
        Arc::new(
            pool.exec(&code, timeout)
                .await
                .unwrap_or_else(|e| json!({"ok": false, "error": e.to_string()})),
        )
    });
    let h = ExecHandle(h);
    async move { h.await }.boxed().shared()
}

struct VerifyGuard {
    tl: Timeline,
    slot: String,
}
impl VerifyGuard {
    fn new(tl: Timeline, slot: String) -> Self {
        tl.emit("verify_start", &slot, 0, "exec");
        VerifyGuard { tl, slot }
    }
}
impl Drop for VerifyGuard {
    fn drop(&mut self) {
        self.tl.emit("verify_end", &self.slot, 0, "exec");
    }
}

struct ExecHandle(tokio::task::JoinHandle<Arc<Value>>);
impl std::future::Future for ExecHandle {
    type Output = Arc<Value>;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Arc<Value>> {
        std::pin::Pin::new(&mut self.0).poll(cx).map(|r| {
            r.unwrap_or_else(|e| {
                Arc::new(json!({"ok": false, "error": format!("verifier task: {e}")}))
            })
        })
    }
}
impl Drop for ExecHandle {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn closed_fences(s: &str) -> usize {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?is)```(?:python|py)?\s*\n.*?```").unwrap())
        .find_iter(s)
        .count()
}

fn apply_exec(out: &mut Outcome, v: &Value) {
    out.exec_ok = v.get("ok").and_then(|x| x.as_bool()).or(Some(false));
    if v.get("returncode").map(|r| r.is_null()).unwrap_or(true) && v.get("stdout").is_none() {
        // worker-level failure before execution
        out.exec_ok = None;
    }
    out.exec_stdout = v["stdout"].as_str().unwrap_or("").to_string();
    out.raw_answer = v["raw_answer"].as_str().unwrap_or("").to_string();
    out.norm_answer = v["norm_answer"].as_str().unwrap_or("").to_string();
    out.verified_constraints = v["verified_constraints"].as_bool().unwrap_or(false);
    out.error = v["error"].as_str().unwrap_or("").to_string();
}

/// Port of _execute_pal_outcome, with an optional cache of speculative executions.
async fn execute_pal(
    ctx: &Ctx,
    tl: &Timeline,
    out: &mut Outcome,
    slot: &str,
    timeout: f64,
    spec: Option<&Arc<Mutex<HashMap<String, ExecFut>>>>,
) -> bool {
    tl.emit("verify_start", slot, 0, "extract");
    let code = ctx
        .light_pool
        .extract(&out.raw_content)
        .await
        .unwrap_or_default();
    out.code = code.clone();
    if code.is_empty() {
        out.error = "No valid Python code block extracted".into();
        tl.emit("verify_end", slot, 0, "extract");
        return false;
    }
    let cached = spec.and_then(|m| m.lock().unwrap().get(&code).cloned());
    let hit = cached.is_some();
    let fut = match cached {
        Some(f) => f,
        None => start_exec(
            ctx.exec_pool.clone(),
            tl.clone(),
            slot.to_string(),
            code,
            timeout,
        ),
    };
    let v = fut.await;
    apply_exec(out, &v);
    tl.emit(
        "verify_end",
        slot,
        0,
        if hit { "spec_hit" } else { "extract" },
    );
    hit
}

fn mk_outcome(name: &str, wave: i32, kind: &str, res: &CompletionResult, ep: &str) -> Outcome {
    Outcome {
        name: name.into(),
        wave,
        kind: kind.into(),
        raw_content: res.content.clone(),
        reasoning: res.reasoning.clone(),
        latency_s: res.latency_s,
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
) -> (usize, Outcome, bool) {
    let ep = ctx.ep(idx);
    let slot = format!("w1_{}", role.name);
    let spec_map: Arc<Mutex<HashMap<String, ExecFut>>> = Arc::new(Mutex::new(HashMap::new()));
    let (tx, listener) = if ctx.speculative && role.kind == "pal" {
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let (c2, tl2, m2, slot2) = (ctx.clone(), tl.clone(), spec_map.clone(), slot.clone());
        let timeout = ctx.sandbox_timeout_s;
        let h = tokio::spawn(async move {
            let mut seen = 0usize;
            while let Some(mut snap) = rx.recv().await {
                while let Ok(s) = rx.try_recv() {
                    snap = s;
                }
                let n = closed_fences(&snap);
                if n <= seen {
                    continue;
                }
                seen = n;
                // Safe speculation: run the script that is complete so far. It is only
                // used if the final extracted code is byte-identical.
                if let Ok(code) = c2.light_pool.extract(&snap).await {
                    if !code.is_empty() && !m2.lock().unwrap().contains_key(&code) {
                        let f = start_exec(
                            c2.exec_pool.clone(),
                            tl2.clone(),
                            format!("spec:{slot2}"),
                            code.clone(),
                            timeout,
                        );
                        m2.lock().unwrap().insert(code, f);
                    }
                }
            }
        });
        (Some(tx), Some(AbortOnDrop(h)))
    } else {
        (None, None)
    };
    let res = ep.complete(&spec, &slot, &tl, &ctx.stats, tx).await;
    drop(listener);
    let mut out = mk_outcome(&role.name, 1, &role.kind, &res, &ep.base_url);
    let mut hit = false;
    if res.ok() {
        if role.kind == "pal" {
            hit = execute_pal(
                &ctx,
                &tl,
                &mut out,
                &slot,
                ctx.sandbox_timeout_s,
                Some(&spec_map),
            )
            .await;
        } else if let Ok((raw, norm)) = ctx.light_pool.prose(&res.content).await {
            out.raw_answer = raw;
            out.norm_answer = norm;
        }
    }
    (idx, out, hit)
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
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
    let mut trace = Trace::default();
    let shared_sys = SHARED_SYS.to_string();
    let prefix = prompts::shared_user_prefix(&ctx.skills, problem);
    let roles = prompts::wave1_roles(&ctx.skills, n, deep)?;

    let mut set: JoinSet<(usize, Outcome, bool)> = JoinSet::new();
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
            let (_, out, hit) = attempt.await;
            trace.spec_hits += usize::from(hit);
            ordered[i] = Some(out);
        } else {
            set.spawn(attempt);
        }
    }
    let mut pal_ok: Vec<Outcome> = vec![];
    while let Some(j) = set.join_next().await {
        let Ok((idx, out, hit)) = j else { continue };
        if hit {
            trace.spec_hits += 1;
        }
        ordered[idx] = Some(out.clone());
        if out.kind == "pal" && !out.norm_answer.is_empty() && format_ok(problem, &out.norm_answer)
        {
            pal_ok.push(out);
            let refs: Vec<&Outcome> = pal_ok.iter().collect();
            let cl = cluster(&ctx.light_pool, &refs).await;
            if let Some((top, members)) = cl.first() {
                if members.len() >= ctx.agree {
                    let mut ana_contradicts = false;
                    for o in ordered.iter().flatten() {
                        if o.kind == "analytical"
                            && !o.norm_answer.is_empty()
                            && !equiv(&ctx.light_pool, &o.norm_answer, top).await
                        {
                            ana_contradicts = true;
                        }
                    }
                    if !ana_contradicts || members.len() > ctx.agree {
                        tl.emit("consensus", "", 0, top);
                        trace.early_aborted = true;
                        set.abort_all();
                        while set.join_next().await.is_some() {}
                        break;
                    }
                }
            }
        }
    }
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
    let pal_clusters = cluster(&ctx.light_pool, &valid_pal).await;
    let all_clusters = cluster(&ctx.light_pool, &all_valid).await;
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
        exec_many(&ctx, tl, &mut trace.wave2, &jobs, t2).await;
        for o in &trace.wave2 {
            if !o.error.is_empty() {
                trace
                    .graveyard
                    .push(format!("[{}] FAILED: {}", o.name, trunc(&o.error, 240)));
            }
        }
        let any_valid = trace.wave1.iter().chain(trace.wave2.iter()).any(|o| {
            o.kind == "pal"
                && o.exec_ok == Some(true)
                && !o.norm_answer.is_empty()
                && format_ok(problem, &o.norm_answer)
        });
        if !any_valid {
            wave3(&ctx, tl, &mut trace, problem, &shared_sys).await;
        }
    }

    adjudicate(&ctx, tl, &mut trace, problem, need_wave2, enable_wave2).await;
    Ok(trace)
}

async fn exec_many(
    ctx: &Arc<Ctx>,
    tl: &Timeline,
    outs: &mut [Outcome],
    jobs: &[usize],
    timeout: f64,
) {
    let futs = jobs.iter().map(|&i| {
        let mut o = outs[i].clone();
        let (ctx, tl) = (ctx.clone(), tl.clone());
        async move {
            let slot = format!("w{}_{}", o.wave, o.name);
            execute_pal(&ctx, &tl, &mut o, &slot, timeout, None).await;
            (i, o)
        }
    });
    for (i, o) in futures::future::join_all(futs).await {
        outs[i] = o;
    }
}

async fn wave3(ctx: &Arc<Ctx>, tl: &Timeline, trace: &mut Trace, problem: &str, shared_sys: &str) {
    let closest = trace
        .wave2
        .iter()
        .rev()
        .chain(trace.wave1.iter().rev())
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
    let results = batch(ctx, tl, specs).await;
    let mut jobs = vec![];
    for ((name, kind, ..), (res, ep)) in roles.iter().zip(results.iter()) {
        let mut o = mk_outcome(name, 3, kind, res, ep);
        if res.ok() {
            if *kind == "pal" {
                jobs.push(trace.wave2.len());
            } else if let Ok((raw, norm)) = ctx.light_pool.prose(&res.content).await {
                o.raw_answer = raw;
                o.norm_answer = norm;
            }
        }
        trace.wave2.push(o);
    }
    let t3 = ctx.sandbox_timeout_s + 6.0;
    exec_many(ctx, tl, &mut trace.wave2, &jobs, t3).await;
}

async fn pairwise_vote(ctx: &Arc<Ctx>, tl: &Timeline, p_ab: String, p_ba: String) -> String {
    let use_grammar = ctx.endpoints[0].caps.structures && ctx.endpoints[0].caps.is_tensorfold;
    let sys = "You are a strict, impartial verifier. Compare Candidate A and Candidate B strictly against the problem specification and edge cases. Reply with only A or B.";
    let mk = |p: String, tag: &str| CallSpec {
        messages: vec![("system".into(), sys.into()), ("user".into(), p)],
        tag: tag.into(),
        max_tokens: if use_grammar { 16 } else { 64 },
        temperature: 0.0,
        enable_thinking: false,
        guided_choice: Some(vec!["A".into(), "B".into()]),
        ..Default::default()
    };
    let r = batch(
        ctx,
        tl,
        vec![mk(p_ab, "reason_judge_ab"), mk(p_ba, "reason_judge_ba")],
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
) {
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
    let clusters = cluster(&ctx.light_pool, &valid).await;
    if clusters.is_empty() {
        let best = combined.iter().max_by_key(|o| o.raw_content.len());
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
        let mut raw = json_payload(&r.content)
            .and_then(|o| {
                o.get("final_answer").map(|v| match v {
                    Value::String(s) => s.trim().to_string(),
                    other => other.to_string(),
                })
            })
            .unwrap_or_default();
        if raw.is_empty() {
            raw = ctx
                .light_pool
                .prose(&r.content)
                .await
                .map(|x| x.0)
                .unwrap_or_default();
        }
        trace.final_answer = ctx.light_pool.normalize(&raw).await.unwrap_or_default();
        trace.consensus_type = "fallback_extract".into();
        return;
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
            let txt = a.content.to_uppercase();
            if txt.contains("WINNER: B") {
                vote = "B".into();
            } else if txt.contains("WINNER: A") {
                vote = "A".into();
            }
        }
        if vote == "B" {
            winner = ans_b;
            members = mem_b;
            trace.consensus_type = "pairwise_tiebreak_B".into();
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
    trace.final_answer = winner;
}

fn json_payload(text: &str) -> Option<serde_json::Map<String, Value>> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(t) {
        return Some(o);
    }
    let (s, e) = (t.find('{')?, t.rfind('}')?);
    if e > s {
        if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(&t[s..=e]) {
            return Some(o);
        }
    }
    None
}

/// maj@N prose voting baseline (port of run_arm_maj4_prose + select_maj4_answer).
/// For N=4 the seeds/temperatures are exactly MAJ4_CONFIGS.
pub async fn majority_vote(
    ctx: Arc<Ctx>,
    tl: &Timeline,
    prompt: &str,
    n: usize,
) -> (String, Value) {
    let base = [(101, 0.25), (202, 0.55), (303, 0.45), (404, 0.35)];
    let cfgs: Vec<(i64, f64)> = (0..n)
        .map(|i| {
            if i < 4 {
                base[i]
            } else {
                (101 * (i as i64 + 1), base[i % 4].1)
            }
        })
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
    let mut valid: Vec<(usize, f64, String)> = vec![];
    let mut fallback: Vec<(usize, f64, String)> = vec![];
    for (i, ((_, temp), (res, _))) in cfgs.iter().zip(results.iter()).enumerate() {
        let text = if res.ok() { res.content.as_str() } else { "" };
        let norm = extract_prose_candidate(&ctx.light_pool, text).await;
        let ai = aime_int(&norm);
        if !ai.is_empty() {
            valid.push((i, *temp, ai));
        } else if !norm.is_empty() {
            fallback.push((i, *temp, norm));
        }
    }
    let rank = |cl: &mut Vec<(String, Vec<(usize, f64)>)>| {
        cl.sort_by(|a, b| {
            let ka = (
                std::cmp::Reverse(a.1.len()),
                a.1.iter().map(|x| x.1).fold(f64::INFINITY, f64::min),
                a.1.iter().map(|x| x.0).min().unwrap(),
            );
            let kb = (
                std::cmp::Reverse(b.1.len()),
                b.1.iter().map(|x| x.1).fold(f64::INFINITY, f64::min),
                b.1.iter().map(|x| x.0).min().unwrap(),
            );
            ka.0.cmp(&kb.0)
                .then(ka.1.partial_cmp(&kb.1).unwrap())
                .then(ka.2.cmp(&kb.2))
        });
    };
    if !valid.is_empty() {
        let mut cl: Vec<(String, Vec<(usize, f64)>)> = vec![];
        for (i, t, a) in valid {
            match cl.iter_mut().find(|c| c.0 == a) {
                Some(c) => c.1.push((i, t)),
                None => cl.push((a, vec![(i, t)])),
            }
        }
        rank(&mut cl);
        let votes: Value = cl
            .iter()
            .map(|c| (c.0.clone(), json!(c.1.len())))
            .collect::<serde_json::Map<_, _>>()
            .into();
        return (
            cl[0].0.clone(),
            json!({"winner_count": cl[0].1.len(), "vote_counts": votes, "used_fallback": false}),
        );
    }
    if !fallback.is_empty() {
        let mut cl: Vec<(String, Vec<(usize, f64)>)> = vec![];
        for (i, t, a) in fallback {
            let mut placed = false;
            for c in cl.iter_mut() {
                if equiv(&ctx.light_pool, &c.0, &a).await {
                    c.1.push((i, t));
                    placed = true;
                    break;
                }
            }
            if !placed {
                cl.push((a, vec![(i, t)]));
            }
        }
        rank(&mut cl);
        return (
            cl[0].0.clone(),
            json!({"winner_count": cl[0].1.len(), "used_fallback": true}),
        );
    }
    (
        String::new(),
        json!({"winner_count": 0, "used_fallback": true}),
    )
}

async fn extract_prose_candidate(light: &VerifierPool, text: &str) -> String {
    let (raw, norm) = light.prose(text).await.unwrap_or_default();
    if !raw.is_empty() || text.is_empty() {
        return norm;
    }
    let last = text
        .trim()
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .last()
        .unwrap_or("");
    light.normalize(last).await.unwrap_or_default()
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

/// Port of grade_aime.
pub async fn grade_aime(light: &VerifierPool, predicted: &str, expected: &str) -> (bool, String) {
    let got = light.normalize(predicted).await.unwrap_or_default();
    let exp = expected
        .trim()
        .parse::<i64>()
        .map(|v| v.to_string())
        .unwrap_or_else(|_| expected.to_string());
    if got.is_empty() {
        return (false, format!("empty (expected {exp})"));
    }
    if got.chars().all(|c| c.is_ascii_digit()) {
        let ok = got
            .parse::<u64>()
            .map(|v| v <= 999 && v.to_string() == exp)
            .unwrap_or(false);
        return (
            ok,
            if ok {
                format!("OK ({got})")
            } else {
                format!("WRONG: got={got:?} expected={exp:?}")
            },
        );
    }
    if equiv(light, &got, &exp).await {
        return (true, format!("OK ({got})"));
    }
    (false, format!("WRONG: got={got:?} expected={exp:?}"))
}
