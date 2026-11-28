//! Integration tests against the in-process mock OpenAI server.
//! Uses the parent Cortheon checkout and its Python dependencies. Override the
//! checkout with CORTHEON_SRC and the Python interpreter with CORTHEON_PYTHON.

use cortheon_rs::mock::{self, Rule, Scenario};
use cortheon_rs::orchestrator::{majority_vote, solve_reasoning};
use cortheon_rs::prompts::AimeItem;
use cortheon_rs::runner::{build_ctx, build_ctx_for_arms, run_one, summarize, BuildOpts};
use cortheon_rs::timeline::{breakdown, Timeline};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

const PROBLEM: &str = "Find the sum of all integer bases b>9 for which 17_b divides 97_b.\n\nNote: Every AIME answer is an integer from 000 to 999.\nEnd your final response with `ANSWER: <integer>`.";

fn pal(ans: &str, tail: &str) -> String {
    format!("Plan.\n```python\nfrom fractions import Fraction\nassert 1 + 1 == 2\nprint('ANSWER: {ans}')\n```\n{tail}")
}

fn rule(m: &str, content: String, ttft_ms: f64, tps: f64) -> Rule {
    Rule {
        r#match: m.into(),
        seed: None,
        content,
        ttft_ms,
        tok_per_s: tps,
        chunk_chars: 4,
        reasoning_tokens: 0,
    }
}

async fn ctx_for(
    urls: Vec<String>,
    agree: usize,
    speculative: bool,
    timeout: f64,
) -> std::sync::Arc<cortheon_rs::orchestrator::Ctx> {
    let opts = opts_for(urls, agree, speculative, timeout);
    build_ctx(&opts)
        .await
        .expect("start mock-server client and Cortheon verifier workers")
}

fn opts_for(urls: Vec<String>, agree: usize, speculative: bool, timeout: f64) -> BuildOpts {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src = std::env::var_os("CORTHEON_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            manifest
                .parent()
                .expect("crate has a parent checkout")
                .to_owned()
        });
    assert!(
        src.join("flash_forge/reason_forge.py").is_file(),
        "missing Cortheon checkout at {}",
        src.display()
    );
    BuildOpts {
        endpoints: urls,
        model: None,
        api_key: None,
        stream: true,
        sampling: Default::default(),
        max_tokens: None,
        request_timeout_s: 360,
        deterministic_seed: None,
        capture_output: false,
        cortheon_src: src,
        python: std::env::var("CORTHEON_PYTHON").unwrap_or_else(|_| "python3".into()),
        worker_script: manifest.join("verifier/worker.py"),
        exec_workers: 4,
        light_workers: 2,
        sandbox_timeout_s: timeout,
        agree,
        speculative,
    }
}

async fn stats(url: &str) -> Value {
    let base = url.trim_end_matches("/v1");
    reqwest::get(format!("{base}/mock/stats"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap()
}

fn aime_oracle(opts: &BuildOpts, case: Value) -> Value {
    let output = std::process::Command::new(&opts.python)
        .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/aime_oracle.py"))
        .arg(case.to_string())
        .env("PYTHONPATH", &opts.cortheon_src)
        .output()
        .expect("run upstream Python oracle");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("upstream oracle JSON")
}

#[tokio::test]
async fn published_majorities_match_upstream_schedules_votes_and_grades() {
    for (arm, samples) in [
        (
            "maj8_prose",
            vec![
                "ANSWER: 008",
                "ANSWER: 7",
                "ANSWER: 7",
                "ANSWER: 8",
                "no explicit answer\n701",
                "ANSWER: 701",
                "ANSWER: 8",
                "ANSWER: 7",
            ],
        ),
        (
            "maj4_prose",
            vec![
                "ANSWER: 1/2",
                "ANSWER: 0.5",
                "ANSWER: sqrt(2)",
                "ANSWER: sqrt(2)",
            ],
        ),
        ("maj4_prose", vec!["", "", "", ""]),
    ] {
        let rules = samples
            .iter()
            .enumerate()
            .map(|(i, text)| Rule {
                seed: Some(101 * (i as i64 + 1)),
                ..rule("", text.to_string(), 0.0, 5000.0)
            })
            .collect();
        let (url, _server) = mock::spawn(
            Scenario {
                model: "m".into(),
                rules,
            },
            0,
        )
        .await
        .unwrap();
        let opts = opts_for(vec![url.clone()], 2, false, 10.0);
        let reference = aime_oracle(
            &opts,
            serde_json::json!({"samples": samples, "expected": "008"}),
        );
        let item = AimeItem {
            id: "sample".into(),
            prompt: "Choose an integer.".into(),
            expected: "008".into(),
        };
        let ctx = build_ctx_for_arms(&opts, &[arm.into()], 4).await.unwrap();
        let (row, _, _) = run_one(&ctx, arm, &item, 4, false).await.unwrap();
        assert_eq!(row["samples"], samples.len());
        assert_eq!(row["predicted"], reference["predicted"]);
        assert_eq!(row["extra"]["vote"], reference["vote"]);
        assert_eq!(row["correct"], reference["grade"][0]);
        assert!(row["detail"]
            .as_str()
            .unwrap()
            .ends_with(reference["grade"][1].as_str().unwrap()));
        let observed = stats(&url).await;
        let mut sent: Vec<_> = observed["bodies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|body| {
                (
                    body["seed"].as_i64().unwrap(),
                    body["temperature"].as_f64().unwrap(),
                )
            })
            .collect();
        sent.sort_by_key(|v| v.0);
        assert_eq!(serde_json::to_value(sent).unwrap(), reference["config"]);
    }
}

#[tokio::test]
async fn consensus_matches_upstream_on_mixed_math_and_verified_branches() {
    use cortheon_rs::consensus::{branch_weight, cluster, equiv, format_ok, Outcome};
    use serde_json::json;
    let (url, _server) = mock::spawn(
        Scenario {
            model: "m".into(),
            rules: vec![],
        },
        0,
    )
    .await
    .unwrap();
    let opts = opts_for(vec![url], 2, false, 10.0);
    let outcomes = [
        ("analytical", "0.5", false, false),
        ("pal", "1/2", true, true),
        ("pal", "7", true, false),
        ("pal", "007", false, true),
        ("analytical", "sqrt(2)", false, false),
        ("pal", "", true, true),
    ];
    let formats = [
        ("Give an exact integer", "2/3"),
        ("ANSWER: <int>", "-42"),
        ("integer from 000 to 999", "０７"),
        ("free form", "sqrt(2)"),
        ("free form", ""),
    ];
    let pairs = [
        ("42", "0042"),
        ("1/2", "2/4"),
        ("1/0", "2"),
        ("Alice", "alice"),
        ("0.3333333333", "1/3"),
        ("sqrt(2)", "2**(1/2)"),
        ("１２", "12"),
        (
            "340282366920938463463374607431768211456/2",
            "170141183460469231731687303715884105728",
        ),
    ];
    let reference = aime_oracle(
        &opts,
        json!({"outcomes": outcomes, "format": formats, "equiv": pairs}),
    );
    let ctx = build_ctx(&opts).await.unwrap();
    let values: Vec<_> = outcomes
        .iter()
        .map(|&(kind, answer, ok, checked)| Outcome {
            kind: kind.into(),
            norm_answer: answer.into(),
            exec_ok: Some(ok),
            verified_constraints: checked,
            ..Default::default()
        })
        .collect();
    let refs = values.iter().collect::<Vec<_>>();
    assert_eq!(
        json!(cluster(&ctx.light_pool, &refs).await),
        reference["clusters"]
    );
    assert_eq!(
        json!(values.iter().map(branch_weight).collect::<Vec<_>>()),
        reference["weights"]
    );
    assert_eq!(
        json!(formats.map(|(question, norm)| format_ok(question, norm))),
        reference["format"]
    );
    let mut equivalent = vec![];
    for (a, b) in pairs {
        equivalent.push(equiv(&ctx.light_pool, a, b).await);
    }
    assert_eq!(json!(equivalent), reference["equiv"]);
}

#[tokio::test]
async fn different_selected_models_are_rejected_before_any_dispatch() {
    use cortheon_rs::client::{Endpoint, Stats};
    use std::sync::Arc;
    let mut urls = vec![];
    let mut eps = vec![];
    let mut servers = vec![];
    for model in ["left", "right"] {
        let (url, server) = mock::spawn(
            Scenario {
                model: model.into(),
                rules: vec![],
            },
            0,
        )
        .await
        .unwrap();
        eps.push(Arc::new(
            Endpoint::connect(&url, None, None, true, 10.0, 4)
                .await
                .unwrap(),
        ));
        urls.push(url);
        servers.push(server);
    }
    let mut opts = opts_for(urls.clone(), 2, false, 10.0);
    assert!(build_ctx_for_arms(&opts, &["maj_early".into()], 4)
        .await
        .err()
        .unwrap()
        .to_string()
        .contains("differs"));
    let error =
        cortheon_rs::early_vote::run(&eps, &Stats::default(), &Timeline::new(), "sample", 4)
            .await
            .unwrap_err();
    assert!(error.to_string().contains("differs"));
    opts.model = Some("left".into());
    assert!(build_ctx(&opts).await.is_err());
    for url in urls {
        assert_eq!(stats(&url).await["started"], 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_vote_keeps_partial_chunks_out_of_token_totals() {
    let scenario = Scenario {
        model: "m".into(),
        rules: vec![
            Rule {
                seed: Some(404),
                ..rule("", "Slow output\nANSWER: 8".into(), 1.0, 1.0)
            },
            rule("", "ANSWER: 7".into(), 100.0, 5000.0),
        ],
    };
    let (url, _server) = mock::spawn(scenario, 0).await.unwrap();
    let opts = opts_for(vec![url.clone()], 2, false, 10.0);
    let ctx = build_ctx_for_arms(&opts, &["maj_early".into()], 4)
        .await
        .unwrap();
    let problem = AimeItem {
        id: "seven".into(),
        prompt: "What is 3 + 4?".into(),
        expected: "7".into(),
    };
    let (row, events, b) = run_one(&ctx, "maj_early", &problem, 4, false)
        .await
        .unwrap();

    assert_eq!(row["correct"], true);
    assert_eq!(stats(&url).await["started"], 4);
    assert_eq!(row["llm_calls"], 4);
    assert_eq!(b.cancelled_requests, 1);
    assert_eq!(b.reported_completion_tokens, 9); // Each complete mock response has three pieces.
    for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        assert_eq!(
            row.get(field),
            Some(&Value::Null),
            "missing usage for one request: {field}"
        );
    }
    let canceled = events.iter().find(|e| e.ev == "req_cancel").unwrap();
    let note: Value = serde_json::from_str(&canceled.note).unwrap();
    assert_eq!(note["usage_source"], "unavailable");
    assert_eq!(canceled.tokens, 0);
    assert_eq!(
        note["observed_chunks"].as_i64(),
        Some(b.cancelled_stream_chunks)
    );

    let summary = summarize(&[row]);
    assert_eq!(summary["maj_early"]["avg_total_tokens"], Value::Null);
    assert_eq!(
        summary["maj_early"]["cancelled_stream_chunks"],
        b.cancelled_stream_chunks
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sampling_and_cap_reach_both_endpoints_and_fallback() {
    let scenario = Scenario {
        model: "m".into(),
        rules: vec![
            rule(
                "Extract ONLY",
                r#"{"final_answer":"7","answer_type":"integer","confidence_notes":""}"#.into(),
                1.0,
                5000.0,
            ),
            rule("", String::new(), 1.0, 5000.0),
        ],
    };
    let (first, _h1) = mock::spawn(scenario.clone(), 0).await.unwrap();
    let (second, _h2) = mock::spawn(scenario, 0).await.unwrap();
    let mut opts = opts_for(vec![first.clone(), second.clone()], 2, false, 10.0);
    opts.sampling = r#"{"temperature":1.0,"top_p":1.0,"top_k":50,"min_p":0.0,"chat_template_kwargs":{"reasoning_effort":0.9}}"#.parse().unwrap();
    opts.max_tokens = Some(6144);
    opts.request_timeout_s = 14400;
    let ctx = build_ctx(&opts).await.unwrap();
    let tr = solve_reasoning(ctx.clone(), &Timeline::new(), PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert_eq!(tr.consensus_type, "fallback_extract");
    assert_eq!(tr.final_answer, "7");
    majority_vote(ctx, &Timeline::new(), PROBLEM, 4).await;
    for (url, expected_seeds) in [
        (
            first,
            vec![None, Some(101), Some(101), Some(303), Some(303)],
        ),
        (second, vec![Some(202), Some(202), Some(404), Some(404)]),
    ] {
        let observed = stats(&url).await;
        let mut seeds = vec![];
        for body in observed["bodies"].as_array().unwrap() {
            seeds.push(body["seed"].as_i64());
            assert_eq!(body["temperature"], 1.0);
            assert_eq!(body["top_p"], 1.0);
            assert_eq!(body["top_k"], 50);
            assert_eq!(body["min_p"], 0.0);
            assert_eq!(body["chat_template_kwargs"]["reasoning_effort"], 0.9);
            assert_eq!(body["max_tokens"], 6144);
            if body["seed"].is_null() {
                assert_eq!(
                    body["response_format"]["json_schema"]["name"],
                    "answer_extract"
                );
            }
        }
        seeds.sort();
        assert_eq!(seeds, expected_seeds);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deterministic_capture_preserves_order_seed_and_all_four_attempts() {
    let scenario = Scenario {
        model: "m".into(),
        rules: vec![
            rule(
                "Extract ONLY",
                r#"{"final_answer":"7","answer_type":"integer","confidence_notes":""}"#.into(),
                1.0,
                5000.0,
            ),
            rule("", String::new(), 1.0, 5000.0),
        ],
    };
    let (url, _h) = mock::spawn(scenario, 0).await.unwrap();
    let mut opts = opts_for(vec![url], 2, false, 10.0);
    opts.deterministic_seed = Some(42);
    opts.capture_output = true;
    opts.stream = false;
    let ctx = build_ctx(&opts).await.unwrap();
    let tl = Timeline::new();
    let trace = solve_reasoning(ctx.clone(), &tl, PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert_eq!(trace.final_answer, "7");
    majority_vote(ctx, &tl, PROBLEM, 4).await;
    let events = tl.events();
    let outputs: Vec<_> = events.iter().filter(|e| e.ev == "req_output").collect();
    assert_eq!(
        outputs.iter().map(|e| e.slot.as_str()).collect::<Vec<_>>(),
        [
            "w1_pal_exact",
            "w1_pal_alt_z3",
            "w1_pal_skeptic",
            "w1_analytical",
            "reason_fallback_extract",
            "maj_s0",
            "maj_s1",
            "maj_s2",
            "maj_s3"
        ]
    );
    for (i, e) in outputs.iter().enumerate() {
        let capture: Value = serde_json::from_str(&e.note).unwrap();
        assert_eq!(capture["submission_index"], i);
        assert_eq!(capture["request_body"]["temperature"], 0.0);
        assert_eq!(capture["request_body"]["seed"], 42);
        assert_eq!(capture["request_body"]["stream"], false);
        assert_eq!(capture["request_body"]["logprobs"], true);
        assert_eq!(capture["request_body"]["top_logprobs"], 1);
        assert_eq!(capture["request_body"]["verbose"], true);
        assert_eq!(capture["request_body"]["return_tokens"], true);
        assert_eq!(capture["finish_reason"], "stop");
        assert!(capture["generated_token_ids"].is_null()); // mock doesn't return per-token ids
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn early_abort_cancels_stragglers() {
    let sc = Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (pal_exact)", pal("70", ""), 10.0, 400.0),
            rule("PROTOCOL (pal_skeptic)", pal("70", ""), 10.0, 300.0),
            rule("PROTOCOL (pal_alt_z3)", pal("70", ""), 10.0, 5.0), // ~25 s if not cancelled
            rule(
                "PROTOCOL (analytical)",
                format!("{}ANSWER: 70", "x\n".repeat(200)),
                10.0,
                5.0,
            ),
        ],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url.clone()], 2, false, 10.0).await;
    let tl = Timeline::new();
    tl.emit("problem_start", "", 0, "");
    let t0 = Instant::now();
    let tr = solve_reasoning(ctx.clone(), &tl, PROBLEM, 4, false, false)
        .await
        .unwrap();
    tl.emit("problem_end", "", 0, "");
    let wall = t0.elapsed().as_secs_f64();
    assert_eq!(tr.final_answer, "70");
    assert!(tr.early_aborted, "expected early abort");
    assert!(
        wall < 5.0,
        "stragglers were not cancelled (wall {wall:.2}s)"
    );
    let names: Vec<_> = tr
        .wave1
        .iter()
        .filter(|o| o.error == "cancelled_early_consensus")
        .map(|o| o.name.clone())
        .collect();
    assert_eq!(names, vec!["pal_alt_z3", "analytical"]);
    let b = breakdown(&tl.events());
    assert_eq!(b.cancelled_requests, 2);
    assert!(
        b.cancel_latency_s.unwrap() < 0.05,
        "cancel latency {:?}",
        b.cancel_latency_s
    );
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    let s = stats(&url).await;
    assert_eq!(s["aborted"], 2, "server must observe 2 client aborts: {s}");
    assert_eq!(s["completed"], 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disagreement_waits_for_third_vote() {
    let sc = Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (pal_exact)", pal("70", ""), 5.0, 2000.0),
            rule("PROTOCOL (pal_skeptic)", pal("71", ""), 5.0, 1000.0),
            rule("PROTOCOL (pal_alt_z3)", pal("70", ""), 5.0, 300.0),
            rule(
                "PROTOCOL (analytical)",
                format!("{}ANSWER: 70", "x\n".repeat(400)),
                5.0,
                5.0,
            ),
        ],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url.clone()], 2, false, 10.0).await;
    let tl = Timeline::new();
    let tr = solve_reasoning(ctx, &tl, PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert!(tr.early_aborted);
    assert_eq!(tr.final_answer, "70");
    let got: Vec<_> = tr.wave1.iter().map(|o| o.norm_answer.clone()).collect();
    assert_eq!(got, vec!["70", "70", "71", ""]);
    // 2 PAL clusters at the end of wave 1 -> wave-1-only verdict with disagreement noted
    assert!(
        tr.wave2_reason.contains("PAL script disagreement"),
        "{}",
        tr.wave2_reason
    );
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert_eq!(stats(&url).await["aborted"], 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn analytical_contradiction_blocks_two_vote_abort() {
    // analytical finishes first with 71; two PALs agree on 70 -> needs a 3rd PAL (Cortheon rule)
    let sc = Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (analytical)", "ANSWER: 71".into(), 1.0, 2000.0),
            rule("PROTOCOL (pal_exact)", pal("70", ""), 50.0, 1000.0),
            rule("PROTOCOL (pal_skeptic)", pal("70", ""), 50.0, 800.0),
            rule("PROTOCOL (pal_alt_z3)", pal("70", ""), 50.0, 150.0),
        ],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url.clone()], 2, false, 10.0).await;
    let tl = Timeline::new();
    let tr = solve_reasoning(ctx, &tl, PROBLEM, 4, false, false)
        .await
        .unwrap();
    // The upstream solver records early_abort only while a slot is pending.
    // The third PAL resolves the contradiction and also finishes the last slot.
    assert!(!tr.early_aborted);
    assert!(!tl.events().iter().any(|ev| ev.ev == "req_cancel"));
    assert!(
        tr.wave1
            .iter()
            .all(|o| o.error != "cancelled_early_consensus"),
        "nothing left to cancel"
    );
    assert_eq!(tr.final_answer, "70");
    assert_eq!(stats(&url).await["aborted"], 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn majority_vote_and_temperature_tiebreak() {
    // seeds 101/202/303/404 -> temps .25/.55/.45/.35
    let mk = |seed: i64, ans: &str| Rule {
        seed: Some(seed),
        ..rule("", format!("prose\nANSWER: {ans}"), 1.0, 5000.0)
    };
    let sc = Scenario {
        model: "m".into(),
        rules: vec![mk(101, "5"), mk(202, "7"), mk(303, "7"), mk(404, "9")],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url.clone()], 2, false, 10.0).await;
    let (win, meta) = majority_vote(ctx.clone(), &Timeline::new(), PROBLEM, 4).await;
    assert_eq!(win, "7", "{meta}");
    // tie 2-2: {5 (t=.25), 9 (t=.35)} vs {7,7 (t=.55,.45)} -> lowest temperature cluster wins
    let sc2 = Scenario {
        model: "m".into(),
        rules: vec![mk(101, "5"), mk(202, "7"), mk(303, "7"), mk(404, "5")],
    };
    let (url2, _h2) = mock::spawn(sc2, 0).await.unwrap();
    let ctx2 = ctx_for(vec![url2], 2, false, 10.0).await;
    let (win2, _) = majority_vote(ctx2, &Timeline::new(), PROBLEM, 4).await;
    assert_eq!(win2, "5");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn speculative_verification_reuses_result() {
    let tail = "Explanation of the method. ".repeat(80);
    let sc = Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (pal_exact)", pal("70", &tail), 5.0, 400.0),
            rule("PROTOCOL (pal_skeptic)", pal("70", &tail), 5.0, 350.0),
            rule("PROTOCOL (pal_alt_z3)", pal("70", ""), 5.0, 5.0),
            rule("PROTOCOL (analytical)", "x\n".repeat(300), 5.0, 5.0),
        ],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url], 2, true, 10.0).await;
    let tl = Timeline::new();
    let tr = solve_reasoning(ctx, &tl, PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert_eq!(tr.final_answer, "70");
    assert_eq!(
        tr.spec_hits, 2,
        "both finished PAL slots should reuse their speculative run"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_endpoint_fanout() {
    let mk = || Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (pal_exact)", pal("70", ""), 5.0, 2000.0),
            rule("PROTOCOL (pal_skeptic)", pal("70", ""), 5.0, 2000.0),
            rule("PROTOCOL (pal_alt_z3)", pal("70", ""), 5.0, 2000.0),
            rule("PROTOCOL (analytical)", "ANSWER: 70".into(), 5.0, 2000.0),
        ],
    };
    let (u1, _h1) = mock::spawn(mk(), 0).await.unwrap();
    let (u2, _h2) = mock::spawn(mk(), 0).await.unwrap();
    let ctx = ctx_for(vec![u1.clone(), u2.clone()], 2, false, 10.0).await;
    let tr = solve_reasoning(ctx, &Timeline::new(), PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert_eq!(tr.final_answer, "70");
    let eps: Vec<_> = tr
        .wave1
        .iter()
        .filter(|o| !o.endpoint.is_empty())
        .map(|o| o.endpoint.clone())
        .collect();
    assert!(eps.contains(&u1) && eps.contains(&u2), "{eps:?}");
    assert_eq!(stats(&u1).await["started"], 2);
    assert_eq!(stats(&u2).await["started"], 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn verifier_timeout_does_not_hang() {
    let looping = "```python\nwhile True:\n    pass\n```\n".to_string();
    let sc = Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (pal_exact)", looping.clone(), 1.0, 5000.0),
            rule("PROTOCOL (pal_skeptic)", pal("70", ""), 1.0, 5000.0),
            rule("PROTOCOL (pal_alt_z3)", pal("70", ""), 1.0, 1000.0),
            rule("PROTOCOL (analytical)", "ANSWER: 70".into(), 1.0, 5000.0),
        ],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url], 2, false, 1.0).await;
    let t0 = Instant::now();
    let tr = solve_reasoning(ctx, &Timeline::new(), PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert_eq!(tr.final_answer, "70");
    assert!(t0.elapsed().as_secs_f64() < 5.0);
    let ex = &tr.wave1[0];
    // either it timed out (TIMEOUT) or was abandoned when the other two PALs agreed first
    assert!(
        ex.error.contains("TIMEOUT") || ex.error == "cancelled_early_consensus",
        "{}",
        ex.error
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exec_pool_enforces_sandbox_timeout() {
    let (url, _h) = mock::spawn(
        Scenario {
            model: "m".into(),
            rules: vec![],
        },
        0,
    )
    .await
    .unwrap();
    let ctx = ctx_for(vec![url], 2, false, 1.0).await;
    let t0 = Instant::now();
    let v = ctx
        .exec_pool
        .exec("while True:\n    pass\n", 1.0)
        .await
        .unwrap();
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(v["timed_out"], true, "{v}");
    assert!(v["error"].as_str().unwrap().starts_with("TIMEOUT"), "{v}");
    assert!(dt < 4.0, "took {dt:.2}s");
    // pool is still usable afterwards
    let ok = ctx.exec_pool.exec("print('ANSWER: 5')", 5.0).await.unwrap();
    assert_eq!(ok["norm_answer"], "5");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn straggler_verifier_is_killed_after_consensus() {
    // alt_z3 finishes first with a script that loops forever (10 s sandbox timeout);
    // the two agreeing PALs win -> the looping check must be killed, freeing its worker.
    let looping = "```python\nwhile True:\n    pass\n```\n".to_string();
    let sc = Scenario {
        model: "m".into(),
        rules: vec![
            rule("PROTOCOL (pal_alt_z3)", looping, 1.0, 5000.0),
            rule("PROTOCOL (pal_exact)", pal("70", ""), 150.0, 2000.0),
            rule("PROTOCOL (pal_skeptic)", pal("70", ""), 150.0, 2000.0),
            rule("PROTOCOL (analytical)", "x\n".repeat(400), 1.0, 5.0),
        ],
    };
    let (url, _h) = mock::spawn(sc, 0).await.unwrap();
    let ctx = ctx_for(vec![url], 2, false, 10.0).await;
    let tl = Timeline::new();
    let tr = solve_reasoning(ctx.clone(), &tl, PROBLEM, 4, false, false)
        .await
        .unwrap();
    assert_eq!(tr.final_answer, "70");
    assert!(tr.early_aborted);
    // All 4 exec workers must be free again almost immediately (without the kill, the
    // looping check would hold its worker for the full 10 s sandbox timeout).
    let t0 = Instant::now();
    while ctx.exec_pool.available() < 4 && t0.elapsed().as_secs_f64() < 3.0 {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    let dt = t0.elapsed().as_secs_f64();
    assert_eq!(
        ctx.exec_pool.available(),
        4,
        "a worker was still busy with the abandoned check after {dt:.2}s"
    );
    let ok = ctx.exec_pool.exec("print('ANSWER: 1')", 5.0).await.unwrap();
    assert_eq!(ok["norm_answer"], "1");
    let alt_end = tl
        .events()
        .iter()
        .filter(|e| e.ev == "verify_end" && e.slot == "w1_pal_alt_z3")
        .count();
    assert_eq!(
        alt_end, 1,
        "abandoned check should still close its verify interval"
    );
}
