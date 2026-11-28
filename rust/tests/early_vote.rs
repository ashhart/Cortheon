//! Exercise the public arm against local SSE streams, including trailers that arrive
//! only after a decision. These tests do not need a model, Python, or a dataset.
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cortheon_rs::prompts::AimeItem;
use cortheon_rs::runner::{build_ctx_for_arms, run_one, BuildOpts};
use futures::stream::{self, StreamExt};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{watch, Barrier};

#[derive(Clone)]
struct Fixture {
    all_dispatched: Arc<Barrier>,
    bodies: Arc<Mutex<Vec<Value>>>,
    trailers: watch::Receiver<bool>,
    n: usize,
}

const USAGE: &str =
    "data: {\"choices\":[],\"usage\":{\"completion_tokens\":2,\"prompt_tokens\":8}}\n\n";
const DONE: &str = "data: [DONE]\n\n";

async fn reply(State(state): State<Fixture>, Json(body): Json<Value>) -> Response {
    let idx = body["seed"].as_u64().unwrap() as usize / 101 - 1;
    let case = body["messages"][0]["content"]
        .as_str()
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_owned();
    state.bodies.lock().unwrap().push(body);
    state.all_dispatched.wait().await;
    let reason = match case.as_str() {
        "length" => "length",
        "error finish" => "error",
        _ => "stop",
    };
    let content = match case.as_str() {
        "tie" if idx % 2 == 0 => "ANSWER: 9",
        "empty" => "",
        _ => "work\n**ANSWER:** 007.",
    };
    let choice = json!({"choices":[{"delta":{"content":content, "reasoning_content":"ANSWER: 888"},"finish_reason":reason}]});
    let terminal = format!("data: {choice}\n\n");
    match case.as_str() {
        "http error" => return (StatusCode::SERVICE_UNAVAILABLE, "overloaded").into_response(),
        "bad JSON" => Body::from(format!("data: {{oops\n\n{terminal}{USAGE}{DONE}")),
        "server error" => Body::from("data: {\"error\":\"overloaded\"}\n\n"),
        "missing finish" => Body::from(format!("{USAGE}{DONE}")),
        "missing done" => Body::from(format!("{terminal}{USAGE}")),
        "missing usage" => Body::from(format!("{terminal}{DONE}")),
        "double finish" => Body::from(format!("{terminal}{terminal}{USAGE}{DONE}")),
        "late delta" => {
            let delta = json!({"choices":[{"delta":{"content":"ANSWER: 999"}}]});
            Body::from(format!("{terminal}data: {delta}\n\n{USAGE}{DONE}"))
        }
        "hold" if idx == state.n - 1 => {
            Body::from_stream(stream::pending::<Result<String, Infallible>>())
        }
        "hold" => {
            let head = stream::iter([Ok::<_, Infallible>(terminal)]);
            let tail = stream::once(async move {
                let mut gate = state.trailers;
                gate.wait_for(|open| *open).await.unwrap();
                Ok(format!("{USAGE}{DONE}"))
            });
            Body::from_stream(head.chain(tail))
        }
        _ => Body::from(format!("{terminal}{USAGE}{DONE}")),
    }
    .into_response()
}

async fn server(state: Fixture) -> String {
    let app = Router::new()
        .route(
            "/v1/models",
            get(|| async { Json(json!({"data":[{"id":"mock"}]})) }),
        )
        .route("/v1/chat/completions", post(reply))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{addr}/v1")
}

fn opts(endpoints: Vec<String>) -> BuildOpts {
    BuildOpts {
        endpoints,
        model: None,
        api_key: None,
        stream: true,
        sampling: Default::default(),
        max_tokens: None,
        request_timeout_s: 5,
        deterministic_seed: None,
        capture_output: false,
        cortheon_src: ".".into(),
        python: "unused-python".into(),
        worker_script: "unused-worker".into(),
        exec_workers: 1,
        light_workers: 1,
        sandbox_timeout_s: 1.0,
        agree: 2,
        speculative: false,
    }
}

#[tokio::test]
async fn early_decision_cancels_before_usage_and_stripes_all_slots() {
    use cortheon_rs::{early_vote, timeline::Timeline};
    for (n, hosts) in [(4, 1), (8, 2), (16, 1), (16, 2)] {
        let (gate, trailers) = watch::channel(false);
        let barrier = Arc::new(Barrier::new(n));
        let mut logs = Vec::new();
        let mut urls = Vec::new();
        for _ in 0..hosts {
            let bodies = Arc::new(Mutex::new(Vec::new()));
            logs.push(bodies.clone());
            urls.push(
                server(Fixture {
                    all_dispatched: barrier.clone(),
                    bodies,
                    trailers: trailers.clone(),
                    n,
                })
                .await,
            );
        }
        let ctx = build_ctx_for_arms(&opts(urls), &["maj_early".into()], n)
            .await
            .unwrap();
        let timeline = Timeline::new();
        let vote = early_vote::run(&ctx.endpoints, &ctx.stats, &timeline, "hold", n);
        let release = async {
            // The last stream never votes, and no stream has sent usage yet.
            while !timeline.events().iter().any(|e| e.ev == "req_cancel") {
                tokio::task::yield_now().await;
            }
            gate.send(true).unwrap();
        };
        let (vote, ()) = tokio::time::timeout(Duration::from_secs(8), async {
            tokio::join!(vote, release)
        })
        .await
        .unwrap();
        let vote = vote.unwrap();
        assert_eq!(vote.status(), "winner");
        assert_eq!(vote.winner, Some(7));
        assert_eq!(vote.received.len(), n / 2 + 1);
        assert_eq!(
            timeline
                .events()
                .iter()
                .filter(|e| e.ev == "req_cancel")
                .count(),
            vote.unresolved_at_decision
        );
        for (host, log) in logs.iter().enumerate() {
            let bodies = log.lock().unwrap();
            let mut seeds: Vec<_> = bodies
                .iter()
                .map(|body| body["seed"].as_u64().unwrap())
                .collect();
            seeds.sort_unstable();
            assert_eq!(
                seeds,
                (host..n)
                    .step_by(hosts)
                    .map(|i| 101 * (i as u64 + 1))
                    .collect::<Vec<_>>()
            );
            for body in bodies.iter() {
                assert_eq!(body["temperature"], 1.0);
                assert_eq!(body["top_p"], 1.0);
                assert_eq!(body["top_k"], 50);
                assert_eq!(body["min_p"], 0.0);
                assert_eq!(body["max_tokens"], 4096);
                assert_eq!(body["chat_template_kwargs"]["reasoning_effort"], "low");
                assert_eq!(body["stream_options"]["include_usage"], true);
                assert!(body["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .contains("ANSWER: <integer>"));
            }
        }
    }
}

#[tokio::test]
async fn row_distinguishes_ties_abstentions_and_stream_failures() {
    let n = 4;
    let (_gate, trailers) = watch::channel(true);
    let url = server(Fixture {
        all_dispatched: Arc::new(Barrier::new(n)),
        bodies: Default::default(),
        trailers,
        n,
    })
    .await;
    let ctx = build_ctx_for_arms(&opts(vec![url]), &["maj_early".into()], n)
        .await
        .unwrap();
    for (case, status) in [
        ("tie", "tie"),
        ("length", "no_valid_answers"),
        ("empty", "no_valid_answers"),
        ("bad JSON", "invalid_stream"),
        ("http error", "invalid_stream"),
        ("server error", "invalid_stream"),
        ("missing done", "invalid_stream"),
        ("missing usage", "invalid_stream"),
        ("missing finish", "invalid_stream"),
        ("double finish", "invalid_stream"),
        ("late delta", "invalid_stream"),
        ("error finish", "invalid_stream"),
        ("unscored", "winner"),
        ("valid", "winner"),
    ] {
        let expected = if case == "unscored" { "" } else { "007" };
        let item = AimeItem {
            id: case.into(),
            prompt: case.into(),
            expected: expected.into(),
        };
        let (row, _, _) = run_one(&ctx, "maj_early", &item, n, false).await.unwrap();
        assert_eq!(row["extra"]["status"], status, "{row}");
        if status == "invalid_stream" || case == "unscored" {
            assert!(row["correct"].is_null(), "{row}");
        } else {
            assert_eq!(row["correct"], status == "winner", "{row}");
        }
        if status == "invalid_stream" {
            assert!(row["extra"]["vote"]["error"].is_string(), "{row}");
            assert_eq!(row["predicted"], "");
        }
    }
}
