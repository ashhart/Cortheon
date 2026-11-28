//! Streaming OpenAI-compatible client (port of flash_forge/client.py payload rules).
//!
//! Differences from Cortheon's TensorFoldClient (documented in README):
//!  * requests are sent with `stream: true` (+ `stream_options.include_usage`) so the
//!    orchestrator sees time-to-first-token and can drop a straggler mid-decode;
//!  * a request future that is dropped closes its HTTP connection immediately, which
//!    is how vLLM / llama.cpp learn to abort the generation;
//!  * one client can use several endpoints serving replicas of the same model.

use crate::timeline::Timeline;
use anyhow::{anyhow, Result};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, Semaphore};

#[derive(Clone, Debug, Default)]
pub struct Caps {
    pub model_id: String,
    pub context_length: i64,
    pub reasoning_optional: bool,
    pub structures: bool,
    pub thinking_budget: bool,
    pub thinking_switch: bool,
    pub loop_guard: bool,
    pub supported_efforts: Vec<String>,
    pub is_tensorfold: bool,
}

#[derive(Clone, Debug)]
pub struct CallSpec {
    pub messages: Vec<(String, String)>,
    pub tag: String,
    pub max_tokens: i64,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub min_p: Option<f64>,
    /// llama.cpp exposes template reasoning effort independently of model metadata.
    pub template_reasoning_effort: Option<String>,
    pub seed: Option<i64>,
    pub enable_thinking: bool,
    pub thinking_budget: Option<i64>,
    pub reasoning_effort: Option<String>,
    pub json_schema: Option<Value>,
    pub schema_name: String,
    pub guided_choice: Option<Vec<String>>,
    pub stop: Option<Vec<String>>,
    pub grammar_budget: bool,
    pub loop_guard: bool,
}

impl Default for CallSpec {
    fn default() -> Self {
        CallSpec {
            messages: vec![],
            tag: String::new(),
            max_tokens: 4096,
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: None,
            template_reasoning_effort: None,
            seed: None,
            enable_thinking: true,
            thinking_budget: None,
            reasoning_effort: None,
            json_schema: None,
            schema_name: "response".into(),
            guided_choice: None,
            stop: None,
            grammar_budget: true,
            loop_guard: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompletionResult {
    pub content: String,
    pub reasoning: String,
    pub finish_reason: String,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub reasoning_tokens: i64,
    pub cached_tokens: i64,
    pub latency_s: f64,
    pub ttft_s: Option<f64>,
    pub raw_error: Option<String>,
    pub tag: String,
}

impl CompletionResult {
    pub fn ok(&self) -> bool {
        self.raw_error.is_none()
    }
    fn err(tag: &str, msg: String, dt: f64) -> Self {
        CompletionResult {
            finish_reason: "error".into(),
            latency_s: dt,
            raw_error: Some(msg),
            tag: tag.into(),
            ..Default::default()
        }
    }
}

pub(crate) fn check_model_ids(endpoints: &[Arc<Endpoint>]) -> Result<()> {
    if let Some((first, rest)) = endpoints.split_first() {
        for ep in rest {
            anyhow::ensure!(
                ep.caps.model_id == first.caps.model_id,
                "selected model {:?} on {} differs from {:?} on {}",
                ep.caps.model_id,
                ep.base_url,
                first.caps.model_id,
                first.base_url
            );
        }
    }
    Ok(())
}

pub struct Endpoint {
    pub base_url: String,
    pub caps: Caps,
    http: reqwest::Client,
    sem: Arc<Semaphore>,
    api_key: Option<String>,
    pub stream: bool,
    pub sampling: Sampling,
    pub max_tokens: Option<u32>,
    pub deterministic_seed: Option<u32>,
    pub capture_output: bool,
    observed_reasoning: AtomicBool,
    reasoning_token_floor: AtomicI64,
    template_kwargs_rejected: AtomicBool,
    reasoning_effort_rejected: AtomicBool,
}

/// Only sampling fields can be overridden. Routing, prompts, seeds and response
/// formats stay under the orchestrator's control.
#[derive(Clone, Debug, Default)]
pub struct Sampling(Map<String, Value>);

impl FromStr for Sampling {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        let fields: Map<String, Value> = serde_json::from_str(raw)?;
        for (key, value) in &fields {
            let valid = match key.as_str() {
                "temperature" => value.as_f64().is_some_and(|n| n >= 0.0),
                "top_p" | "min_p" => value.as_f64().is_some_and(|n| (0.0..=1.0).contains(&n)),
                "top_k" => value.as_u64().is_some(),
                "reasoning_effort" => value.is_string(),
                "chat_template_kwargs" => value.as_object().is_some_and(|kwargs| {
                    kwargs
                        .iter()
                        .all(|(k, v)| k == "reasoning_effort" && (v.is_string() || v.is_number()))
                }),
                _ => false,
            };
            anyhow::ensure!(valid, "invalid sampling option {key}={value}");
        }
        Ok(Self(fields))
    }
}

#[derive(Default)]
pub struct Stats {
    pub calls: AtomicI64,
    pub prompt_tokens: AtomicI64,
    pub completion_tokens: AtomicI64,
    pub reasoning_tokens: AtomicI64,
    pub cached_tokens: AtomicI64,
}

impl Stats {
    pub fn reset(&self) {
        for a in [
            &self.calls,
            &self.prompt_tokens,
            &self.completion_tokens,
            &self.reasoning_tokens,
            &self.cached_tokens,
        ] {
            a.store(0, Ordering::SeqCst);
        }
    }
    pub fn snapshot(&self) -> Value {
        json!({
            "calls": self.calls.load(Ordering::SeqCst),
            "prompt_tokens": self.prompt_tokens.load(Ordering::SeqCst),
            "completion_tokens": self.completion_tokens.load(Ordering::SeqCst),
            "reasoning_tokens": self.reasoning_tokens.load(Ordering::SeqCst),
            "cached_tokens": self.cached_tokens.load(Ordering::SeqCst),
        })
    }
}

#[derive(Clone, Copy)]
enum Listener<'a> {
    Content(Option<&'a mpsc::UnboundedSender<String>>),
    Ballot(&'a (dyn Fn(&str, &str) + Sync)),
}

impl Endpoint {
    pub async fn connect(
        base_url: &str,
        model: Option<&str>,
        api_key: Option<&str>,
        stream: bool,
        timeout_s: f64,
        max_parallel: usize,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs_f64(timeout_s))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(16)
            .tcp_nodelay(true)
            .build()?;
        let base_url = base_url.trim_end_matches('/').to_string();
        let mut req = http.get(format!("{base_url}/models"));
        if let Some(k) = api_key {
            req = req.bearer_auth(k);
        }
        let v: Value = req.send().await?.error_for_status()?.json().await?;
        let data = v
            .get("data")
            .and_then(|d| d.as_array())
            .cloned()
            .unwrap_or_default();
        if data.is_empty() {
            return Err(anyhow!("No models reported by {base_url}/models"));
        }
        let entry = match model {
            Some(m) => data
                .iter()
                .find(|e| e["id"].as_str() == Some(m))
                .ok_or_else(|| anyhow!("model {m:?} is absent from {base_url}/models"))?,
            None => &data[0],
        };
        let caps_raw = entry.get("capabilities").cloned().unwrap_or(json!({}));
        let caps_nonempty = caps_raw.as_object().map(|o| !o.is_empty()).unwrap_or(false);
        let is_tf =
            caps_nonempty || entry.get("owned_by").and_then(|x| x.as_str()) == Some("tensorfold");
        let b = |k: &str| caps_raw.get(k).and_then(|x| x.as_bool()).unwrap_or(false);
        let context_length = [
            &entry["context_length"],
            &entry["max_model_len"],
            &entry["meta"]["n_ctx"],
        ]
        .into_iter()
        .filter_map(|v| v.as_i64().or_else(|| v.as_str()?.parse().ok()))
        .find(|&n| n > 0)
        .unwrap_or(32768);
        let caps = Caps {
            model_id: entry["id"]
                .as_str()
                .filter(|id| !id.is_empty())
                .ok_or_else(|| anyhow!("model entry from {base_url}/models is missing its id"))?
                .to_string(),
            context_length,
            reasoning_optional: caps_raw["reasoning_optional"].as_bool().unwrap_or(true),
            structures: if is_tf { b("structures") } else { true },
            thinking_budget: b("thinking_budget"),
            thinking_switch: b("thinking_switch"),
            loop_guard: b("loop_guard"),
            supported_efforts: entry
                .get("reasoning")
                .and_then(|r| r.get("supported_efforts"))
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            is_tensorfold: is_tf,
        };
        Ok(Endpoint {
            base_url,
            caps,
            http,
            sem: Arc::new(Semaphore::new(max_parallel)),
            api_key: api_key.map(String::from),
            stream,
            sampling: Sampling::default(),
            max_tokens: None,
            deterministic_seed: None,
            capture_output: false,
            observed_reasoning: AtomicBool::new(false),
            reasoning_token_floor: AtomicI64::new(0),
            template_kwargs_rejected: AtomicBool::new(false),
            reasoning_effort_rejected: AtomicBool::new(false),
        })
    }

    /// Port of TensorFoldClient._build_payload.
    pub fn build_payload(&self, spec: &CallSpec) -> Value {
        let caps = &self.caps;
        let msgs: Vec<Value> = spec
            .messages
            .iter()
            .map(|(r, c)| json!({"role": r, "content": c}))
            .collect();
        let mut p = json!({
            "model": caps.model_id,
            "messages": msgs,
            "max_tokens": spec.max_tokens,
            "temperature": spec.temperature,
        });
        let o = p.as_object_mut().unwrap();
        if spec.temperature > 0.0 {
            o.insert("top_p".into(), json!(spec.top_p));
            o.insert("top_k".into(), json!(spec.top_k));
            if let Some(min_p) = spec.min_p {
                o.insert("min_p".into(), json!(min_p));
            }
        }
        if let Some(s) = spec.seed {
            o.insert("seed".into(), json!(s));
        }
        if let Some(stop) = spec.stop.as_ref().filter(|s| !s.is_empty()) {
            o.insert("stop".into(), json!(stop));
        }
        let eff = |e: &str| caps.supported_efforts.iter().any(|x| x == e);
        static REASONER: LazyLock<regex::Regex> = LazyLock::new(|| {
            regex::Regex::new(r"(?i)(?:^|[^a-z0-9])(?:deepseek|ds\d|r1|qwq|qwen3|glm[-_]?5|nemotron|inkling|o[134](?:[-_]|$)|thinking|reasoner)").unwrap()
        });
        let generic_reasoner = !caps.is_tensorfold
            && (self.observed_reasoning.load(Ordering::Relaxed)
                || REASONER.is_match(&caps.model_id));
        if caps.thinking_switch {
            o.insert(
                "chat_template_kwargs".into(),
                json!({"enable_thinking": spec.enable_thinking}),
            );
        } else if !spec.enable_thinking {
            if eff("none") {
                o.insert("reasoning_effort".into(), json!("none"));
            }
            if caps.is_tensorfold || !self.template_kwargs_rejected.load(Ordering::Relaxed) {
                o.insert(
                    "chat_template_kwargs".into(),
                    json!({"enable_thinking": false}),
                );
            }
        }
        if !spec.enable_thinking && !caps.reasoning_optional {
            if caps.thinking_budget {
                o.insert("thinking_budget".into(), json!(0));
            }
            if let Some(effort) = ["none", "minimal"].into_iter().find(|e| eff(e)) {
                o.insert("reasoning_effort".into(), json!(effort));
            }
        }
        if spec.enable_thinking {
            if let (Some(tb), true) = (spec.thinking_budget, caps.thinking_budget) {
                o.insert("thinking_budget".into(), json!(tb));
            }
            if let Some(re) = spec.reasoning_effort.as_deref().filter(|r| eff(r)) {
                o.insert("reasoning_effort".into(), json!(re));
            } else if let (Some(tb), false) =
                (spec.thinking_budget, caps.supported_efforts.is_empty())
            {
                let pick = if tb <= 550 && eff("minimal") {
                    Some("minimal")
                } else if tb <= 1200 && eff("low") {
                    Some("low")
                } else if eff("medium") {
                    Some("medium")
                } else if eff("high") {
                    Some("high")
                } else {
                    None
                };
                if let Some(x) = pick {
                    o.insert("reasoning_effort".into(), json!(x));
                }
            } else if generic_reasoner
                && !caps.thinking_budget
                && spec.template_reasoning_effort.is_none()
                && !self.reasoning_effort_rejected.load(Ordering::Relaxed)
            {
                let effort = spec
                    .reasoning_effort
                    .as_deref()
                    .filter(|e| matches!(*e, "low" | "medium" | "high"))
                    .or_else(|| {
                        spec.thinking_budget
                            .map(|tb| if tb <= 1200 { "low" } else { "medium" })
                    });
                if let Some(effort) = effort {
                    o.insert("reasoning_effort".into(), json!(effort));
                }
            }
            if spec.template_reasoning_effort.is_none()
                && spec.max_tokens > 1
                && !caps.thinking_budget
                && (!caps.supported_efforts.is_empty() || generic_reasoner)
            {
                let safe = (caps.context_length / 2).clamp(1024, 16384);
                let floor = self.reasoning_token_floor.load(Ordering::Relaxed).max(6144);
                let mt = spec.max_tokens.max(floor.min(safe));
                o.insert("max_tokens".into(), json!(mt));
            }
        }
        if spec.loop_guard && caps.loop_guard {
            o.insert("loop_guard".into(), json!(true));
        }
        if caps.structures {
            if let (Some(gc), true) = (&spec.guided_choice, caps.is_tensorfold) {
                o.insert("guided_choice".into(), json!(gc));
            } else if let Some(js) = &spec.json_schema {
                o.insert(
                    "response_format".into(),
                    json!({"type": "json_schema", "json_schema": {"name": spec.schema_name, "schema": js}}),
                );
                if spec.grammar_budget && caps.is_tensorfold {
                    o.insert("grammar_budget".into(), json!(true));
                }
            }
        }
        if caps.is_tensorfold {
            o.insert("cortheon".into(), json!(false));
        }
        if let Some(effort) = &spec.template_reasoning_effort {
            o.entry("chat_template_kwargs").or_insert_with(|| json!({}))["reasoning_effort"] =
                json!(effort);
        }
        if let Some(cap) = self.max_tokens {
            o.insert("max_tokens".into(), json!(cap));
        }
        for (key, value) in &self.sampling.0 {
            if key == "chat_template_kwargs" {
                let kwargs = o.entry(key.clone()).or_insert_with(|| json!({}));
                kwargs
                    .as_object_mut()
                    .unwrap()
                    .extend(value.as_object().unwrap().clone());
            } else {
                o.insert(key.clone(), value.clone());
            }
        }
        if let Some(seed) = self.deterministic_seed {
            o.insert("temperature".into(), json!(0.0));
            o.insert("seed".into(), json!(seed));
        }
        if self.capture_output {
            o.insert("logprobs".into(), json!(true));
            o.insert("top_logprobs".into(), json!(1));
            o.insert("stream".into(), json!(self.stream));
            if !self.stream {
                o.insert("verbose".into(), json!(true));
                o.insert("return_tokens".into(), json!(true));
            }
        }
        p
    }

    fn post(&self, payload: &Value) -> reqwest::RequestBuilder {
        let mut r = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .json(payload);
        if let Some(k) = &self.api_key {
            r = r.bearer_auth(k);
        }
        r
    }

    async fn post_with_fallback(
        &self,
        mut payload: Value,
        guard: &mut CancelGuard,
        preserve_thinking: bool,
    ) -> Result<reqwest::Response> {
        let mut rejected_controls = (false, false);
        loop {
            let response = self.post(&payload).send().await?;
            guard.read_headers(&response);
            let code = response.status().as_u16();
            if code == 200 {
                if rejected_controls.0 {
                    self.template_kwargs_rejected.store(true, Ordering::Relaxed);
                }
                if rejected_controls.1 {
                    self.reasoning_effort_rejected
                        .store(true, Ordering::Relaxed);
                }
                return Ok(response);
            }
            let fields = payload.as_object_mut().unwrap();
            let grammar = ["response_format", "guided_choice", "grammar_budget"];
            if matches!(code, 400 | 422) && grammar.iter().any(|k| fields.contains_key(*k)) {
                for key in grammar {
                    fields.remove(key);
                }
            } else if matches!(code, 400 | 422)
                && !preserve_thinking
                && (fields.contains_key("chat_template_kwargs")
                    || fields.contains_key("reasoning_effort"))
            {
                rejected_controls = (
                    fields.remove("chat_template_kwargs").is_some(),
                    fields.remove("reasoning_effort").is_some(),
                );
            } else {
                let message = response.text().await.unwrap_or_default();
                return Err(anyhow!("HTTP {code}: {}", trunc(&message, 300)));
            }
            if let Some(capture) = &mut guard.capture {
                capture.prior_request_bodies.push(std::mem::replace(
                    &mut capture.request_body,
                    payload.clone(),
                ));
            }
        }
    }

    fn recovery_payload(
        &self,
        spec: &CallSpec,
        sent: &Value,
        first: &CompletionResult,
    ) -> Option<Value> {
        // Fixed benchmark profiles specify the exact request and number of samples.
        if self.max_tokens.is_some()
            || self.deterministic_seed.is_some()
            || spec.max_tokens <= 1
            || !first.ok()
            || first.finish_reason != "length"
            || !first.content.trim().is_empty()
            || (first.reasoning_tokens == 0 && first.reasoning.is_empty())
            || (self.caps.thinking_budget && (spec.enable_thinking || self.caps.reasoning_optional))
        {
            return None;
        }
        let sent_max = sent["max_tokens"].as_i64()?;
        let cap = sent_max
            .max((self.caps.context_length / 2).max(1024))
            .min(16384);
        let next_max = if !spec.enable_thinking {
            sent_max.saturating_mul(4).max(1024).min(cap)
        } else if sent_max < 6144 && cap > sent_max {
            sent_max.saturating_mul(2).max(6144).min(cap)
        } else if !self.template_kwargs_rejected.load(Ordering::Relaxed) {
            let mut commit = spec.clone();
            let mut message = String::from("Your reasoning scratchpad reached the token limit before emitting the final answer. Do not continue brainstorming. Output ONLY the final requested answer/code directly now.");
            let tail = first
                .reasoning
                .char_indices()
                .rev()
                .nth(2399)
                .map_or(0, |(i, _)| i);
            let notes = first.reasoning[tail..].trim();
            if !notes.is_empty() {
                message.push_str("\n\nScratchpad excerpt:\n");
                message.push_str(notes);
            }
            match commit.messages.last_mut() {
                Some((role, content)) if role == "user" => {
                    content.push_str("\n\n");
                    content.push_str(&message);
                }
                _ => commit.messages.push(("user".into(), message)),
            }
            commit.max_tokens = spec.max_tokens.clamp(2048, 4096);
            commit.temperature = spec.temperature.min(0.3);
            commit.enable_thinking = false;
            commit.thinking_budget = None;
            commit.reasoning_effort = None;
            return Some(self.build_payload(&commit));
        } else {
            sent_max.saturating_mul(2).min(cap)
        };
        if next_max <= sent_max {
            return None;
        }
        if spec.enable_thinking {
            self.reasoning_token_floor
                .fetch_max(next_max, Ordering::Relaxed);
        }
        let mut retry = sent.clone();
        retry["max_tokens"] = json!(next_max);
        Some(retry)
    }

    /// Run one completion. Dropping the returned future cancels the HTTP request
    /// (connection closed) and records a `req_cancel` event with the tokens seen so far.
    /// `content_tx`, when given, receives the accumulated *content* every time a
    /// chunk containing a code fence arrives (used for speculative verification).
    pub async fn complete(
        &self,
        spec: &CallSpec,
        slot: &str,
        tl: &Timeline,
        stats: &Stats,
        content_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> CompletionResult {
        self.read_completion(
            spec,
            slot,
            tl,
            stats,
            Listener::Content(content_tx.as_ref()),
        )
        .await
    }

    /// Report the terminal choice immediately, then drain and validate its usage and [DONE].
    /// This lets voting cancel undecided streams without cutting off the winners' trailers.
    pub(crate) async fn complete_ballot(
        &self,
        spec: &CallSpec,
        tl: &Timeline,
        stats: &Stats,
        on_finish: impl Fn(&str, &str) + Sync,
    ) -> CompletionResult {
        self.read_completion(spec, &spec.tag, tl, stats, Listener::Ballot(&on_finish))
            .await
    }

    async fn read_completion(
        &self,
        spec: &CallSpec,
        slot: &str,
        tl: &Timeline,
        stats: &Stats,
        listener: Listener<'_>,
    ) -> CompletionResult {
        let payload = self.build_payload(spec);
        let _permit = self.sem.acquire().await.expect("semaphore");
        let started = Instant::now();
        let mut first = self
            .read_attempt(spec, slot, tl, stats, listener, payload.clone())
            .await;
        if first.reasoning_tokens > 0 || !first.reasoning.is_empty() {
            self.observed_reasoning.store(true, Ordering::Relaxed);
        }
        // A strict ballot is closed at its first terminal choice.
        if matches!(listener, Listener::Content(_)) {
            if let Some(retry) = self.recovery_payload(spec, &payload, &first) {
                let second = self
                    .read_attempt(spec, slot, tl, stats, listener, retry)
                    .await;
                first.latency_s = started.elapsed().as_secs_f64();
                if second.ok() {
                    first.prompt_tokens += second.prompt_tokens;
                    first.completion_tokens += second.completion_tokens;
                    first.cached_tokens += second.cached_tokens;
                    first.reasoning_tokens += second.reasoning_tokens;
                    if !second.content.trim().is_empty() {
                        first.content = second.content;
                        first.finish_reason = second.finish_reason;
                    }
                    if first.reasoning.is_empty() {
                        first.reasoning = second.reasoning;
                    }
                }
            }
        }
        first
    }

    async fn read_attempt(
        &self,
        spec: &CallSpec,
        slot: &str,
        tl: &Timeline,
        stats: &Stats,
        listener: Listener<'_>,
        mut payload: Value,
    ) -> CompletionResult {
        let on_finish = match listener {
            Listener::Ballot(notify) => Some(notify),
            Listener::Content(_) => None,
        };
        let mut guard = CancelGuard::new(tl.clone(), slot.to_string(), self.stream);
        tl.emit("req_start", slot, 0, &self.base_url);
        if self.max_tokens.is_some()
            || !self.sampling.0.is_empty()
            || self.deterministic_seed.is_some()
        {
            let mut settings = Map::new();
            for key in [
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "seed",
                "reasoning_effort",
                "chat_template_kwargs",
            ] {
                if let Some(value) = payload.get(key) {
                    settings.insert(key.to_owned(), value.clone());
                }
            }
            tl.emit(
                "req_settings",
                slot,
                0,
                &Value::Object(settings).to_string(),
            );
        }
        let t0 = Instant::now();
        if self.stream {
            let o = payload.as_object_mut().unwrap();
            o.insert("stream".into(), json!(true));
            o.insert("stream_options".into(), json!({"include_usage": true}));
        }
        if self.capture_output {
            guard.capture = Some(OutputCapture::new(tl.next_capture_index(), payload.clone()));
        }
        let preserve_thinking = on_finish.is_some()
            || self.sampling.0.contains_key("chat_template_kwargs")
            || self.sampling.0.contains_key("reasoning_effort");
        let resp = match self
            .post_with_fallback(payload, &mut guard, preserve_thinking)
            .await
        {
            Ok(response) => response,
            Err(error) => {
                return guard.finish(
                    stats,
                    CompletionResult::err(&spec.tag, error.to_string(), t0.elapsed().as_secs_f64()),
                )
            }
        };
        let mut out = CompletionResult {
            tag: spec.tag.clone(),
            finish_reason: "stop".into(),
            ..Default::default()
        };
        if !self.stream {
            let v: Value = match resp.json().await {
                Ok(v) => v,
                Err(e) => {
                    return guard.finish(
                        stats,
                        CompletionResult::err(&spec.tag, e.to_string(), t0.elapsed().as_secs_f64()),
                    )
                }
            };
            guard.read_response(&v);
            let Some(ch) = v["choices"].get(0).filter(|choice| choice.is_object()) else {
                let err = guard
                    .stream_error
                    .clone()
                    .unwrap_or_else(|| "completion response has no choice".into());
                return guard.finish(
                    stats,
                    CompletionResult::err(&spec.tag, err, t0.elapsed().as_secs_f64()),
                );
            };
            if let Some(err) = guard.stream_error.clone() {
                return guard.finish(
                    stats,
                    CompletionResult::err(&spec.tag, err, t0.elapsed().as_secs_f64()),
                );
            }
            guard.finish_reason = ch["finish_reason"].as_str().map(str::to_owned);
            out.content = ch["message"]["content"].as_str().unwrap_or("").to_string();
            out.reasoning = ch["message"]["reasoning_content"]
                .as_str()
                .or_else(|| ch["message"]["reasoning"].as_str())
                .unwrap_or("")
                .to_string();
            out.finish_reason = ch["finish_reason"].as_str().unwrap_or("stop").to_string();
            if out.finish_reason == "error" {
                out.raw_error = Some("backend returned an error finish reason".into());
            }
            read_usage(&v, &mut out);
            out.latency_s = t0.elapsed().as_secs_f64();
            return guard.finish(stats, out);
        }
        let mut body = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::with_capacity(8192);
        let mut saw_choice = false;
        'outer: while let Some(item) = body.next().await {
            let bytes = match item {
                Ok(b) => b,
                Err(e) => {
                    out.raw_error = Some(e.to_string());
                    out.finish_reason = "error".into();
                    break;
                }
            };
            buf.extend_from_slice(&bytes);
            while let Some(pos) = buf.iter().position(|&c| c == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data == "[DONE]" {
                    guard.done_seen = true;
                    break 'outer;
                }
                let v = match serde_json::from_str::<Value>(data) {
                    Ok(v) => v,
                    Err(e) => {
                        out.raw_error = Some(format!("invalid SSE JSON: {e}"));
                        break 'outer;
                    }
                };
                guard.read_response(&v);
                if let Some(error) = &guard.stream_error {
                    out.raw_error = Some(error.clone());
                    break 'outer;
                }
                if on_finish.is_some() {
                    if !v["choices"]
                        .as_array()
                        .is_some_and(|choices| choices.len() <= 1)
                    {
                        out.raw_error = Some("expected a stream with one sampled choice".into());
                        break 'outer;
                    }
                }
                if v.get("usage").map(|u| u.is_object()).unwrap_or(false) {
                    read_usage(&v, &mut out);
                }
                if let Some(ch) = v.get("choices").and_then(|c| c.get(0)) {
                    saw_choice = true;
                    if on_finish.is_some() && guard.finish_reason.is_some() {
                        out.raw_error = Some("received a choice after the terminal choice".into());
                        break 'outer;
                    }
                    let d = &ch["delta"];
                    let c = d.get("content").and_then(|x| x.as_str()).unwrap_or("");
                    let r = d
                        .get("reasoning_content")
                        .and_then(|x| x.as_str())
                        .or_else(|| d.get("reasoning").and_then(|x| x.as_str()))
                        .unwrap_or("");
                    if !c.is_empty() || !r.is_empty() {
                        guard.chunks += 1;
                        guard.last_delta_s = Some(t0.elapsed().as_secs_f64());
                        if out.ttft_s.is_none() {
                            out.ttft_s = guard.last_delta_s;
                            guard.first_delta_s = out.ttft_s;
                            tl.emit("first_token", slot, 0, "");
                        }
                        out.reasoning.push_str(r);
                        if !c.is_empty() {
                            out.content.push_str(c);
                            if c.contains('`') {
                                if let Listener::Content(Some(tx)) = listener {
                                    let _ = tx.send(out.content.clone());
                                }
                            }
                        }
                    }
                    if let Some(fr) = ch.get("finish_reason").and_then(|x| x.as_str()) {
                        if fr == "error" {
                            out.raw_error = Some("backend returned an error finish reason".into());
                            break 'outer;
                        }
                        if let Some(notify) = on_finish {
                            notify(&out.content, fr);
                        }
                        out.finish_reason = fr.to_string();
                        guard.finish_reason = Some(fr.to_owned());
                    }
                }
            }
        }
        if out.raw_error.is_none() {
            if !saw_choice {
                out.raw_error = Some("stream contains no completion choice".into());
            } else if guard.finish_reason.is_none() && !guard.done_seen {
                out.raw_error = Some("stream closed without a terminal response".into());
            }
        }
        if on_finish.is_some() && out.raw_error.is_none() {
            if guard.finish_reason.is_none() || !guard.done_seen {
                out.raw_error =
                    Some("stream ended without both a terminal choice and [DONE]".into());
            } else if !guard
                .server_usage
                .as_ref()
                .is_some_and(|u| u["completion_tokens"].is_u64() && u["prompt_tokens"].is_u64())
            {
                out.raw_error = Some("stream ended without server token usage".into());
            }
        }
        out.latency_s = t0.elapsed().as_secs_f64();
        guard.finish(stats, out)
    }
}

fn read_usage(v: &Value, out: &mut CompletionResult) {
    let u = &v["usage"];
    out.prompt_tokens = u["prompt_tokens"].as_i64().unwrap_or(0);
    out.completion_tokens = u["completion_tokens"].as_i64().unwrap_or(0);
    out.cached_tokens = u["prompt_tokens_details"]["cached_tokens"]
        .as_i64()
        .unwrap_or(0);
    out.reasoning_tokens = u["completion_tokens_details"]["reasoning_tokens"]
        .as_i64()
        .unwrap_or(0);
}

pub fn trunc(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

struct OutputCapture {
    submission_index: usize,
    request_body: Value,
    prior_request_bodies: Vec<Value>,
    content: String,
    reasoning: String,
    tokens: Vec<(Option<i64>, Option<Vec<u8>>)>,
    native_ids: Option<Vec<i64>>,
    server_rendered_text: Option<String>,
}

impl OutputCapture {
    fn new(submission_index: usize, request_body: Value) -> Self {
        Self {
            submission_index,
            request_body,
            prior_request_bodies: vec![],
            content: String::new(),
            reasoning: String::new(),
            tokens: vec![],
            native_ids: None,
            server_rendered_text: None,
        }
    }

    fn observe(&mut self, response: &Value) {
        let choice = &response["choices"][0];
        if let Some(native) = response.get("__verbose") {
            self.native_ids = native["tokens"].as_array().and_then(|ids| {
                ids.iter()
                    .map(|v| v.as_i64().filter(|&id| id >= 0))
                    .collect()
            });
            self.server_rendered_text = native["content"].as_str().map(str::to_owned);
        }
        let message = choice.get("delta").unwrap_or(&choice["message"]);
        if let Some(content) = message["content"].as_str() {
            self.content.push_str(content);
        }
        if let Some(reasoning) = message["reasoning_content"]
            .as_str()
            .or_else(|| message["reasoning"].as_str())
        {
            self.reasoning.push_str(reasoning);
        }
        if let Some(items) = choice["logprobs"]["content"].as_array() {
            for item in items {
                // Read the sampled token. top_logprobs contains alternatives.
                let bytes = item["bytes"].as_array().and_then(|arr| {
                    arr.iter()
                        .map(|v| v.as_u64().and_then(|n| u8::try_from(n).ok()))
                        .collect()
                });
                self.tokens
                    .push((item["id"].as_i64().filter(|&id| id >= 0), bytes));
            }
        }
    }

    fn note(&self, guard: &CancelGuard, cancelled: bool) -> String {
        let logprob_ids: Option<Vec<i64>> = self.tokens.iter().map(|(id, _)| *id).collect();
        let bytes: Option<Vec<&[u8]>> = self
            .tokens
            .iter()
            .map(|(_, bytes)| bytes.as_deref())
            .collect();
        let output_count = guard
            .server_usage
            .as_ref()
            .and_then(|u| u["completion_tokens"].as_u64());
        let logged = logprob_ids.as_deref().filter(|ids| !ids.is_empty());
        let (ids, source) = match self.native_ids.as_deref() {
            Some(native) => (Some(native), Some("__verbose.tokens")),
            None => (logged, logged.map(|_| "choices.logprobs.content")),
        };
        let bytes_aligned = !self.tokens.is_empty() && (self.native_ids.is_none() || logged == ids);
        json!({
            "submission_index": self.submission_index, "request_body": self.request_body,
            "prior_request_bodies": self.prior_request_bodies, "content": self.content, "reasoning": self.reasoning,
            "finish_reason": guard.finish_reason, "server_usage": guard.server_usage, "cancelled": cancelled,
            "generated_token_ids": ids, "generated_token_bytes": bytes.filter(|_| bytes_aligned),
            "token_id_source": source, "server_rendered_text": self.server_rendered_text,
            "token_count_matches_usage": ids.zip(output_count).map(|(tokens, count)| count == tokens.len() as u64),
        }).to_string()
    }
}

struct CancelGuard {
    tl: Timeline,
    slot: String,
    stream: bool,
    chunks: i64,
    first_delta_s: Option<f64>,
    last_delta_s: Option<f64>,
    request_id: Option<String>,
    response_id: Option<String>,
    finish_reason: Option<String>,
    server_usage: Option<Value>,
    stream_error: Option<String>,
    done_seen: bool,
    done: bool,
    capture: Option<OutputCapture>,
}

impl CancelGuard {
    fn new(tl: Timeline, slot: String, stream: bool) -> Self {
        Self {
            tl,
            slot,
            stream,
            chunks: 0,
            first_delta_s: None,
            last_delta_s: None,
            request_id: None,
            response_id: None,
            finish_reason: None,
            server_usage: None,
            stream_error: None,
            done_seen: false,
            done: false,
            capture: None,
        }
    }
    fn read_headers(&mut self, response: &reqwest::Response) {
        self.request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .map(str::to_owned);
    }
    fn read_response(&mut self, v: &Value) {
        if let Some(capture) = &mut self.capture {
            capture.observe(v);
        }
        if self.response_id.is_none() {
            self.response_id = v["id"].as_str().map(str::to_owned);
        }
        if v["usage"].is_object() {
            self.server_usage = Some(v["usage"].clone());
        }
        if !v["error"].is_null() {
            self.stream_error = Some(trunc(&v["error"].to_string(), 300));
        }
    }
    fn note(&self, cancelled: bool, error: Option<&str>) -> String {
        let error = error.or(self.stream_error.as_deref());
        let usage_source = if self
            .server_usage
            .as_ref()
            .and_then(|u| u["completion_tokens"].as_i64())
            .is_some()
        {
            "server_usage"
        } else {
            "unavailable"
        };
        let status = if cancelled {
            "cancelled"
        } else if error.is_some() || self.finish_reason.as_deref() == Some("error") {
            "error"
        } else if self.stream && !self.done_seen && self.finish_reason.is_none() {
            "missing_terminal"
        } else {
            "completed"
        };
        json!({
            "status": status, "finish_reason": self.finish_reason, "error": error,
            "request_id": self.request_id, "response_id": self.response_id,
            "usage_source": usage_source, "server_usage": self.server_usage, "observed_chunks": self.stream.then_some(self.chunks),
            "first_delta_s": self.first_delta_s, "last_delta_s": self.last_delta_s, "done_seen": self.stream.then_some(self.done_seen),
            "terminal_unix_s": SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs_f64()),
        }).to_string()
    }
    fn finish(mut self, stats: &Stats, r: CompletionResult) -> CompletionResult {
        self.done = true;
        if r.ok() {
            stats.calls.fetch_add(1, Ordering::SeqCst);
            stats
                .prompt_tokens
                .fetch_add(r.prompt_tokens, Ordering::SeqCst);
            stats
                .completion_tokens
                .fetch_add(r.completion_tokens, Ordering::SeqCst);
            stats
                .reasoning_tokens
                .fetch_add(r.reasoning_tokens, Ordering::SeqCst);
            stats
                .cached_tokens
                .fetch_add(r.cached_tokens, Ordering::SeqCst);
        }
        self.tl.emit(
            "req_end",
            &self.slot,
            r.completion_tokens,
            &self.note(false, r.raw_error.as_deref()),
        );
        if let Some(capture) = &self.capture {
            self.tl
                .emit("req_output", &self.slot, 0, &capture.note(&self, false));
        }
        r
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if !self.done {
            let reported = self
                .server_usage
                .as_ref()
                .and_then(|u| u["completion_tokens"].as_i64())
                .unwrap_or(0);
            self.tl
                .emit("req_cancel", &self.slot, reported, &self.note(true, None));
            if let Some(capture) = &self.capture {
                self.tl
                    .emit("req_output", &self.slot, 0, &capture.note(self, true));
            }
        }
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;

    #[test]
    fn captures_sampled_ids_and_bytes_independently_of_rendered_text() {
        let mut guard = CancelGuard::new(Timeline::new(), "role".into(), false);
        guard.capture = Some(OutputCapture::new(2, json!({"stream": false})));
        guard.read_response(
            &json!({"choices":[{"message":{"content":"é","reasoning":"a\n "},
            "logprobs":{"content":[{"id":11,"bytes":[195],"top_logprobs":[{"id":99}]},
                                   {"id":12,"bytes":[169]},{"id":13,"bytes":[]}]}}],
            "usage":{"completion_tokens":3}}),
        );
        let note: Value =
            serde_json::from_str(&guard.capture.as_ref().unwrap().note(&guard, false)).unwrap();
        assert_eq!(note["generated_token_ids"], json!([11, 12, 13]));
        assert_eq!(note["generated_token_bytes"], json!([[195], [169], []]));
        assert_eq!(note["content"], "é");
        assert_eq!(note["reasoning"], "a\n ");
        assert_eq!(note["token_count_matches_usage"], true);
        guard.read_response(
            &json!({"__verbose":{"tokens":[11, 12, 13, 14], "content":"é"},
            "usage":{"completion_tokens":4}}),
        );
        let native: Value =
            serde_json::from_str(&guard.capture.as_ref().unwrap().note(&guard, false)).unwrap();
        assert_eq!(native["generated_token_ids"], json!([11, 12, 13, 14]));
        assert!(native["generated_token_bytes"].is_null());
        assert_eq!(native["server_rendered_text"], "é");
        assert_eq!(native["token_id_source"], "__verbose.tokens");
        assert_eq!(native["token_count_matches_usage"], true);
    }
}
