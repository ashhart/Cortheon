//! Mock OpenAI-compatible server: streams canned answers at controlled speeds and
//! records which requests completed and which were aborted by the client.
//! Used by the integration tests and by the Python-vs-Rust overhead benchmark.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Rule {
    /// substring that must appear in the last message (empty = always)
    #[serde(default)]
    pub r#match: String,
    #[serde(default)]
    pub seed: Option<i64>,
    pub content: String,
    #[serde(default = "d_ttft")]
    pub ttft_ms: f64,
    #[serde(default = "d_tps")]
    pub tok_per_s: f64,
    #[serde(default = "d_chunk")]
    pub chunk_chars: usize,
    #[serde(default)]
    pub reasoning_tokens: usize,
}
fn d_ttft() -> f64 {
    20.0
}
fn d_tps() -> f64 {
    200.0
}
fn d_chunk() -> usize {
    4
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct Scenario {
    #[serde(default = "d_model")]
    pub model: String,
    pub rules: Vec<Rule>,
}
fn d_model() -> String {
    "mock-model".into()
}

#[derive(Clone, Debug, Serialize, Default)]
pub struct MockEvent {
    pub t: f64,
    pub kind: String,
    pub seed: Option<i64>,
    pub rule: usize,
    pub tokens: usize,
}

#[derive(Default)]
struct Log {
    events: Vec<MockEvent>,
    /// request bodies minus messages/stream fields, for payload-parity checks
    bodies: Vec<Value>,
}

#[derive(Clone)]
pub struct MockState {
    scenario: Arc<Scenario>,
    log: Arc<Mutex<Log>>,
    t0: Instant,
}

impl MockState {
    fn rec(&self, kind: &str, seed: Option<i64>, rule: usize, tokens: usize) {
        self.log.lock().unwrap().events.push(MockEvent {
            t: self.t0.elapsed().as_secs_f64(),
            kind: kind.into(),
            seed,
            rule,
            tokens,
        });
    }
}

struct Guard {
    st: MockState,
    seed: Option<i64>,
    rule: usize,
    sent: usize,
    done: bool,
}
impl Drop for Guard {
    fn drop(&mut self) {
        let kind = if self.done { "completed" } else { "aborted" };
        self.st.rec(kind, self.seed, self.rule, self.sent);
    }
}

fn pick(sc: &Scenario, body: &Value) -> (usize, Rule) {
    let last = body["messages"]
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m["content"].as_str())
        .unwrap_or("");
    let seed = body["seed"].as_i64();
    for (i, r) in sc.rules.iter().enumerate() {
        if (r.r#match.is_empty() || last.contains(&r.r#match))
            && (r.seed.is_none() || r.seed == seed)
        {
            return (i, r.clone());
        }
    }
    (
        usize::MAX,
        Rule {
            r#match: String::new(),
            seed: None,
            content: "ANSWER: 0".into(),
            ttft_ms: 1.0,
            tok_per_s: 1000.0,
            chunk_chars: 4,
            reasoning_tokens: 0,
        },
    )
}

fn chunks_of(s: &str, n: usize) -> Vec<String> {
    let cs: Vec<char> = s.chars().collect();
    cs.chunks(n.max(1)).map(|c| c.iter().collect()).collect()
}

async fn models(State(st): State<MockState>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": st.scenario.model, "object": "model", "owned_by": "mock"}]}),
    )
}

async fn stats(State(st): State<MockState>) -> Json<Value> {
    let log = st.log.lock().unwrap();
    let completed = log.events.iter().filter(|e| e.kind == "completed").count();
    let aborted = log.events.iter().filter(|e| e.kind == "aborted").count();
    let started = log.events.iter().filter(|e| e.kind == "started").count();
    let tokens: usize = log
        .events
        .iter()
        .filter(|e| e.kind != "started")
        .map(|e| e.tokens)
        .sum();
    let aborted_tokens: usize = log
        .events
        .iter()
        .filter(|e| e.kind == "aborted")
        .map(|e| e.tokens)
        .sum();
    Json(
        json!({"started": started, "completed": completed, "aborted": aborted, "tokens_sent": tokens, "aborted_tokens": aborted_tokens, "events": log.events, "bodies": log.bodies}),
    )
}

async fn reset(State(st): State<MockState>) -> Json<Value> {
    let mut l = st.log.lock().unwrap();
    l.events.clear();
    l.bodies.clear();
    Json(json!({"ok": true}))
}

async fn chat(State(st): State<MockState>, Json(body): Json<Value>) -> Response {
    let (ri, rule) = pick(&st.scenario, &body);
    let seed = body["seed"].as_i64();
    let stream = body["stream"].as_bool().unwrap_or(false);
    let prompt_chars: usize = body["messages"]
        .as_array()
        .map(|m| {
            m.iter()
                .map(|x| x["content"].as_str().unwrap_or("").len())
                .sum()
        })
        .unwrap_or(0);
    let prompt_tokens = prompt_chars / 4;
    st.rec("started", seed, ri, 0);
    {
        let mut b = body.clone();
        if let Some(o) = b.as_object_mut() {
            let n_msgs = o
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|m| m.len())
                .unwrap_or(0);
            let chars: usize = o
                .get("messages")
                .and_then(|m| m.as_array())
                .map(|m| {
                    m.iter()
                        .map(|x| x["content"].as_str().unwrap_or("").len())
                        .sum()
                })
                .unwrap_or(0);
            o.remove("messages");
            o.remove("stream");
            o.remove("stream_options");
            o.insert("n_messages".into(), json!(n_msgs));
            o.insert("message_chars".into(), json!(chars));
        }
        st.log.lock().unwrap().bodies.push(b);
    }
    let mut pieces: Vec<(bool, String)> = (0..rule.reasoning_tokens)
        .map(|_| (true, "think ".to_string()))
        .collect();
    pieces.extend(
        chunks_of(&rule.content, rule.chunk_chars)
            .into_iter()
            .map(|c| (false, c)),
    );
    let n = pieces.len();
    let start = tokio::time::Instant::now();
    let ttft = Duration::from_secs_f64(rule.ttft_ms / 1000.0);
    let per = Duration::from_secs_f64(1.0 / rule.tok_per_s.max(1e-3));
    let model = st.scenario.model.clone();
    let guard = Guard {
        st: st.clone(),
        seed,
        rule: ri,
        sent: 0,
        done: false,
    };
    if !stream {
        let mut g = guard;
        for k in 0..n {
            tokio::time::sleep_until(start + ttft + per * k as u32).await;
            g.sent = k + 1;
        }
        g.done = true;
        let content: String = pieces
            .iter()
            .filter(|p| !p.0)
            .map(|p| p.1.as_str())
            .collect();
        let reasoning: String = pieces
            .iter()
            .filter(|p| p.0)
            .map(|p| p.1.as_str())
            .collect();
        return Json(json!({
            "id": "mock", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n,
                       "completion_tokens_details": {"reasoning_tokens": rule.reasoning_tokens}}
        }))
        .into_response();
    }
    let state = (0usize, guard, pieces, false);
    let s = futures::stream::unfold(state, move |(k, mut g, pieces, finished)| {
        let model = model.clone();
        async move {
            if finished {
                return None;
            }
            if k < pieces.len() {
                tokio::time::sleep_until(start + ttft + per * k as u32).await;
                let (is_r, txt) = &pieces[k];
                let delta = if *is_r {
                    json!({"reasoning_content": txt})
                } else {
                    json!({"content": txt})
                };
                let ev = json!({"id": "mock", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": Value::Null}]});
                g.sent = k + 1;
                return Some((
                    Ok::<_, std::io::Error>(format!("data: {ev}\n\n")),
                    (k + 1, g, pieces, false),
                ));
            }
            let n = pieces.len();
            let fin = json!({"id": "mock", "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]});
            let usage = json!({"id": "mock", "object": "chat.completion.chunk", "model": model, "choices": [],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n,
                          "completion_tokens_details": {"reasoning_tokens": pieces.iter().filter(|p| p.0).count()}}});
            g.done = true;
            Some((
                Ok(format!("data: {fin}\n\ndata: {usage}\n\ndata: [DONE]\n\n")),
                (k + 1, g, pieces, true),
            ))
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(s))
        .unwrap()
}

pub fn router(scenario: Scenario) -> Router {
    let st = MockState {
        scenario: Arc::new(scenario),
        log: Arc::new(Mutex::new(Log::default())),
        t0: Instant::now(),
    };
    Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/mock/stats", get(stats))
        .route("/mock/reset", post(reset))
        .with_state(st)
}

/// Bind on 127.0.0.1:port (0 = ephemeral) and serve in the background. Returns the base URL.
pub async fn spawn(
    scenario: Scenario,
    port: u16,
) -> anyhow::Result<(String, tokio::task::JoinHandle<()>)> {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    let addr = listener.local_addr()?;
    let app = router(scenario);
    let h = tokio::spawn(async move {
        // TCP_NODELAY like uvicorn/vLLM and llama.cpp, so small SSE chunks are not held back by Nagle
        axum::serve(listener, app).tcp_nodelay(true).await.ok();
    });
    Ok((format!("http://{addr}/v1"), h))
}
