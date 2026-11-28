//! Exercise the HTTP behavior shared with flash_forge.client.TensorFoldClient.
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use cortheon_rs::client::{CallSpec, Endpoint, Stats};
use cortheon_rs::timeline::Timeline;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

type Replies = Vec<(StatusCode, Value)>;
type Requests = Arc<Mutex<Vec<Value>>>;

#[derive(Clone)]
struct Server {
    replies: Arc<Mutex<VecDeque<(StatusCode, Value)>>>,
    requests: Requests,
}

async fn reply(State(server): State<Server>, Json(body): Json<Value>) -> Response {
    let streaming = body["stream"] == true;
    server.requests.lock().unwrap().push(body);
    let (status, data) = server
        .replies
        .lock()
        .unwrap()
        .pop_front()
        .expect("unexpected request");
    if status != StatusCode::OK || !streaming {
        return (status, Json(data)).into_response();
    }
    let choice = &data["choices"][0];
    let delta =
        json!({"choices":[{"delta":choice["message"],"finish_reason":choice["finish_reason"]}]});
    let usage = json!({"choices":[], "usage":data["usage"]});
    format!("data: {delta}\n\ndata: {usage}\n\ndata: [DONE]\n\n").into_response()
}

async fn server(models: Value, replies: Replies) -> (String, Requests) {
    let requests = Requests::default();
    let state = Server {
        replies: Arc::new(Mutex::new(replies.into())),
        requests: requests.clone(),
    };
    let app = Router::new()
        .route(
            "/v1/models",
            get(move || async move { Json(json!({"data": models})) }),
        )
        .route("/v1/chat/completions", post(reply))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}/v1"), requests)
}

fn completion(content: &str, reasoning: &str, finish: &str, tokens: i64) -> (StatusCode, Value) {
    (
        StatusCode::OK,
        json!({
            "choices":[{"message":{"content":content, "reasoning_content":reasoning},"finish_reason":finish}],
            "usage":{"prompt_tokens":4,"completion_tokens":tokens,
                     "completion_tokens_details":{"reasoning_tokens":if reasoning.is_empty() { 0 } else { tokens }}}
        }),
    )
}

#[tokio::test]
async fn probes_context_and_thinking_controls_for_the_selected_model() {
    let (url, _) = server(
        json!([
            {"id":"other"},
            {"id":"repo/Inkling-2-mini","context_length":0,"meta":{"n_ctx":4096}},
            {"id":"native","capabilities":{"reasoning_optional":false,"thinking_budget":true},
             "reasoning":{"supported_efforts":["minimal","medium"]}}
        ]),
        vec![],
    )
    .await;
    let ep = Endpoint::connect(&url, Some("repo/Inkling-2-mini"), None, false, 3., 1)
        .await
        .unwrap();
    assert!(Endpoint::connect(&url, Some("absent"), None, false, 3., 1)
        .await
        .is_err());
    let mut spec = CallSpec {
        max_tokens: 400,
        thinking_budget: Some(1810),
        ..Default::default()
    };
    let payload = ep.build_payload(&spec);
    assert_eq!(ep.caps.context_length, 4096);
    assert_eq!(payload["max_tokens"], 2048);
    assert_eq!(payload["reasoning_effort"], "medium");
    spec.max_tokens = 1;
    assert_eq!(ep.build_payload(&spec)["max_tokens"], 1);
    let native = Endpoint::connect(&url, Some("native"), None, false, 3., 1)
        .await
        .unwrap();
    spec.enable_thinking = false;
    spec.stop = Some(vec!["DONE".into()]);
    let fast = native.build_payload(&spec);
    assert_eq!(fast["thinking_budget"], 0);
    assert_eq!(fast["reasoning_effort"], "minimal");
    assert_eq!(fast["stop"], json!(["DONE"]));
}

#[tokio::test]
async fn optional_fields_fall_back_in_order_and_rejections_persist() {
    let (url, requests) = server(
        json!([{"id":"generic"}]),
        vec![
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                json!("response_format unsupported"),
            ),
            (
                StatusCode::BAD_REQUEST,
                json!("chat_template_kwargs unsupported"),
            ),
            completion("7", "", "stop", 1),
            completion("8", "", "stop", 1),
        ],
    )
    .await;
    let ep = Endpoint::connect(&url, None, None, false, 3., 1)
        .await
        .unwrap();
    let spec = CallSpec {
        enable_thinking: false,
        json_schema: Some(json!({"type":"object"})),
        ..Default::default()
    };
    let tl = Timeline::new();
    let stats = Stats::default();
    assert_eq!(
        ep.complete(&spec, "one", &tl, &stats, None).await.content,
        "7"
    );
    assert_eq!(
        ep.complete(&spec, "two", &tl, &stats, None).await.content,
        "8"
    );
    let bodies = requests.lock().unwrap();
    assert!(bodies[0]["response_format"].is_object());
    assert_eq!(bodies[0]["chat_template_kwargs"]["enable_thinking"], false);
    assert!(bodies[1]["response_format"].is_null());
    assert!(bodies[1]["chat_template_kwargs"].is_object());
    assert!(bodies[2]["chat_template_kwargs"].is_null());
    assert!(bodies[3]["response_format"].is_object());
    assert!(bodies[3]["chat_template_kwargs"].is_null());
    assert_eq!(stats.snapshot()["calls"], 2);
}

#[tokio::test]
async fn recovers_reasoning_only_length_in_json_and_streaming_modes() {
    for streaming in [false, true] {
        let (url, requests) = server(
            json!([{"id":"unknown"}]),
            vec![
                completion("", "notes", "length", 40),
                completion("7", "", "stop", 2),
                completion("", "scratch", "length", 55),
                completion("8", "", "stop", 1),
            ],
        )
        .await;
        let ep = Endpoint::connect(&url, None, None, streaming, 3., 1)
            .await
            .unwrap();
        let spec = CallSpec {
            messages: vec![("user".into(), "question".into())],
            seed: Some(101),
            thinking_budget: Some(300),
            max_tokens: 100,
            ..Default::default()
        };
        let stats = Stats::default();
        let tl = Timeline::new();
        let first = ep.complete(&spec, "sample", &tl, &stats, None).await;
        assert_eq!(
            (
                first.content.as_str(),
                first.reasoning.as_str(),
                first.completion_tokens
            ),
            ("7", "notes", 42)
        );
        let second = ep.complete(&spec, "sample", &tl, &stats, None).await;
        assert_eq!(
            (second.content.as_str(), second.reasoning_tokens),
            ("8", 55)
        );
        let bodies = requests.lock().unwrap();
        assert_eq!(bodies[0]["max_tokens"], 100);
        assert_eq!(bodies[1]["max_tokens"], 6144);
        assert_eq!(bodies[2]["max_tokens"], 6144);
        assert_eq!(bodies[2]["reasoning_effort"], "low");
        assert_eq!(bodies[3]["max_tokens"], 2048);
        assert_eq!(bodies[3]["temperature"], 0.3);
        assert_eq!(bodies[3]["seed"], 101);
        assert_eq!(bodies[3]["chat_template_kwargs"]["enable_thinking"], false);
        assert!(bodies[3]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .ends_with("Scratchpad excerpt:\nscratch"));
        assert_eq!(stats.snapshot()["calls"], 4);
        assert_eq!(stats.snapshot()["completion_tokens"], 98);
        assert_eq!(tl.events().iter().filter(|e| e.ev == "req_end").count(), 4);
    }
}

#[tokio::test]
async fn fast_route_retries_once_and_an_explicit_budget_keeps_one_attempt() {
    let (url, requests) = server(
        json!([{
            "id":"native","capabilities":{"thinking_budget":true,"reasoning_optional":false}
        }]),
        vec![
            completion("", "notes", "length", 3),
            completion("7", "", "stop", 2),
            completion("", "notes", "length", 3),
        ],
    )
    .await;
    let mut ep = Endpoint::connect(&url, None, None, true, 3., 1)
        .await
        .unwrap();
    let spec = CallSpec {
        enable_thinking: false,
        max_tokens: 64,
        ..Default::default()
    };
    let stats = Stats::default();
    let tl = Timeline::new();
    let recovered = ep.complete(&spec, "fast", &tl, &stats, None).await;
    assert_eq!(recovered.content, "7");
    ep.max_tokens = Some(12);
    let fixed = ep.complete(&spec, "fixed", &tl, &stats, None).await;
    assert_eq!(fixed.finish_reason, "length");
    let bodies = requests.lock().unwrap();
    assert_eq!(
        bodies
            .iter()
            .map(|v| v["max_tokens"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [64, 1024, 12]
    );
}

#[tokio::test]
async fn fallback_surfaces_the_last_error_and_never_silently_drops_explicit_controls() {
    let (url, requests) = server(
        json!([{"id":"generic"}]),
        vec![
            (StatusCode::BAD_REQUEST, json!("original")),
            (StatusCode::SERVICE_UNAVAILABLE, json!("server is busy")),
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                json!("required setting rejected"),
            ),
        ],
    )
    .await;
    let mut ep = Endpoint::connect(&url, None, None, false, 3., 1)
        .await
        .unwrap();
    let spec = CallSpec {
        enable_thinking: false,
        ..Default::default()
    };
    let tl = Timeline::new();
    let stats = Stats::default();
    let failure = ep.complete(&spec, "default", &tl, &stats, None).await;
    assert!(failure
        .raw_error
        .unwrap()
        .contains("HTTP 503: \"server is busy\""));
    ep.sampling = r#"{"chat_template_kwargs":{"reasoning_effort":"low"}}"#
        .parse()
        .unwrap();
    let failure = ep.complete(&spec, "fixed", &tl, &stats, None).await;
    assert!(failure.raw_error.unwrap().contains("HTTP 422"));
    assert_eq!(requests.lock().unwrap().len(), 3);
}

#[tokio::test]
async fn buffered_requests_and_results_match_the_upstream_python_client() {
    let models = json!([{"id":"ds41-flash","meta":{"n_ctx":32768}}]);
    let replies = vec![
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            json!("response_format unsupported"),
        ),
        (
            StatusCode::BAD_REQUEST,
            json!("reasoning effort unsupported"),
        ),
        completion("", "some work", "length", 12),
        completion("ANSWER: 7", "", "stop", 3),
    ];
    let (rust_url, rust_requests) = server(models.clone(), replies.clone()).await;
    let (python_url, python_requests) = server(models, replies).await;
    let ep = Endpoint::connect(&rust_url, None, None, false, 4., 1)
        .await
        .unwrap();
    let spec = CallSpec {
        messages: vec![("user".into(), "question".into())],
        seed: Some(101),
        thinking_budget: Some(900),
        max_tokens: 2500,
        json_schema: Some(json!({"type":"object"})),
        ..Default::default()
    };
    let stats = Stats::default();
    let rust = ep
        .complete(&spec, "a", &Timeline::new(), &stats, None)
        .await;
    assert!(rust.ok());

    let script = r#"
import asyncio, json, sys
from flash_forge.client import CallSpec, TensorFoldClient
async def main():
    async with TensorFoldClient(base_url=sys.argv[1]) as client:
        result = await client.complete(CallSpec(
            messages=[{"role":"user","content":"question"}], seed=101,
            thinking_budget=900, max_tokens=2500, json_schema={"type":"object"}))
        assert result.ok, result.raw_error
        print(json.dumps([result.content, result.reasoning, result.finish_reason,
                          result.completion_tokens, result.prompt_tokens, result.reasoning_tokens,
                          client.stats_dict()]))
asyncio.run(main())
"#;
    let checkout = std::env::var_os("CORTHEON_SRC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_owned()
        });
    let python = std::env::var_os("CORTHEON_PYTHON").unwrap_or_else(|| "python3".into());
    let output = tokio::process::Command::new(python)
        .current_dir(checkout)
        .args(["-c", script, &python_url])
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let reference: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        *rust_requests.lock().unwrap(),
        *python_requests.lock().unwrap()
    );
    assert_eq!(
        reference,
        json!([
            rust.content,
            rust.reasoning,
            rust.finish_reason,
            rust.completion_tokens,
            rust.prompt_tokens,
            rust.reasoning_tokens,
            stats.snapshot()
        ])
    );
}

fn dual_control_model() -> Value {
    json!([{"id":"native","capabilities":{"thinking_switch":true,"reasoning_optional":false},
            "reasoning":{"supported_efforts":["none"]}}])
}

#[tokio::test]
async fn named_rejections_disable_only_the_unsupported_thinking_control() {
    for (bad, good) in [
        ("reasoning_effort", "chat_template_kwargs"),
        ("chat_template_kwargs", "reasoning_effort"),
    ] {
        let (url, requests) = server(
            dual_control_model(),
            vec![
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    json!({"detail":[{"type":"extra_forbidden",
                "loc":["body",bad],"msg":"Extra inputs are not permitted","input":good}]}),
                ),
                completion("7", "", "stop", 1),
                completion("8", "", "stop", 1),
            ],
        )
        .await;
        let ep = Endpoint::connect(&url, None, None, false, 3., 1)
            .await
            .unwrap();
        let spec = CallSpec {
            enable_thinking: false,
            ..Default::default()
        };
        for expected in ["7", "8"] {
            let result = ep
                .complete(&spec, "sample", &Timeline::new(), &Stats::default(), None)
                .await;
            assert_eq!(result.content, expected);
        }
        let bodies = requests.lock().unwrap();
        assert!(!bodies[0][bad].is_null() && !bodies[0][good].is_null());
        for body in &bodies[1..] {
            assert!(body[bad].is_null());
            assert_eq!(body[good], bodies[0][good]);
        }
    }
}

#[tokio::test]
async fn opaque_errors_probe_independently_without_poisoning_later_requests() {
    let (url, requests) = server(
        dual_control_model(),
        vec![
            (StatusCode::BAD_REQUEST, json!("bad request")),
            (StatusCode::BAD_REQUEST, json!("bad request")),
            completion("7", "", "stop", 1),
            completion("8", "", "stop", 1),
        ],
    )
    .await;
    let ep = Endpoint::connect(&url, None, None, false, 3., 1)
        .await
        .unwrap();
    let spec = CallSpec {
        enable_thinking: false,
        ..Default::default()
    };
    for expected in ["7", "8"] {
        assert_eq!(
            ep.complete(&spec, "sample", &Timeline::new(), &Stats::default(), None)
                .await
                .content,
            expected
        );
    }
    let bodies = requests.lock().unwrap();
    assert!(bodies[1]["chat_template_kwargs"].is_null());
    assert_eq!(bodies[1]["reasoning_effort"], "none");
    assert_eq!(bodies[2]["chat_template_kwargs"]["enable_thinking"], false);
    assert!(bodies[2]["reasoning_effort"].is_null());
    assert_eq!(bodies[3], bodies[0]);
}

#[tokio::test]
async fn preserves_explicit_control_while_retrying_the_independent_automatic_control() {
    let (url, requests) = server(
        dual_control_model(),
        vec![
            (
                StatusCode::BAD_REQUEST,
                json!({"error":{"code":"unsupported_parameter","param":"chat_template_kwargs"}}),
            ),
            completion("7", "", "stop", 1),
        ],
    )
    .await;
    let mut ep = Endpoint::connect(&url, None, None, false, 3., 1)
        .await
        .unwrap();
    ep.sampling = r#"{"reasoning_effort":"low"}"#.parse().unwrap();
    let spec = CallSpec {
        enable_thinking: false,
        ..Default::default()
    };
    assert_eq!(
        ep.complete(&spec, "sample", &Timeline::new(), &Stats::default(), None)
            .await
            .content,
        "7"
    );
    let bodies = requests.lock().unwrap();
    assert_eq!(bodies[1]["reasoning_effort"], "low");
    assert!(bodies[1]["chat_template_kwargs"].is_null());
}

#[tokio::test]
async fn generic_server_can_capture_output_without_llama_cpp_extensions() {
    let (url, requests) = server(json!([{"id":"generic"}]), vec![
        (StatusCode::BAD_REQUEST, json!({"error":{"message":"Unrecognized request argument supplied: return_tokens"}})),
        completion("hello", "work", "stop", 2), completion("again", "", "stop", 1),
    ]).await;
    let mut ep = Endpoint::connect(&url, None, None, false, 3., 1)
        .await
        .unwrap();
    ep.capture_output = true;
    let tl = Timeline::new();
    let stats = Stats::default();
    for expected in ["hello", "again"] {
        assert_eq!(
            ep.complete(&CallSpec::default(), "capture", &tl, &stats, None)
                .await
                .content,
            expected
        );
    }
    let bodies = requests.lock().unwrap();
    assert_eq!(bodies[0]["return_tokens"], true);
    assert_eq!(bodies[0]["verbose"], true);
    for request in &bodies[1..] {
        assert!(request["return_tokens"].is_null() && request["verbose"].is_null());
        assert_eq!(request["logprobs"], true);
        assert_eq!(request["top_logprobs"], 1);
    }
    let capture: Value = serde_json::from_str(
        &tl.events()
            .into_iter()
            .find(|e| e.ev == "req_output")
            .unwrap()
            .note,
    )
    .unwrap();
    assert_eq!(capture["prior_request_bodies"], json!([bodies[0]]));
    assert_eq!(capture["request_body"], bodies[1]);
    assert_eq!(capture["content"], "hello");
    assert_eq!(capture["reasoning"], "work");
    assert!(capture["generated_token_ids"].is_null() && capture["token_id_source"].is_null());
    assert_eq!(capture["server_usage"]["completion_tokens"], 2);
}
