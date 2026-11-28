//! Per-problem event timeline + phase breakdown.
//!
//! Client events partition a problem's wall time into phases by priority:
//!   gen_decode  : at least one request is streaming tokens (after its first token)
//!   gen_prefill : requests in flight but none has produced a token yet (TTFT / queue)
//!   verify      : no request in flight, at least one verifier job running
//!   orch_busy   : nothing in flight, orchestrator process burning CPU (process CPU delta)
//!   idle_wait   : nothing in flight and the orchestrator is not on CPU
//! `orch_busy + idle_wait` is the orchestration critical path ("gap").

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub t: f64,
    pub ev: String,
    #[serde(default)]
    pub slot: String,
    #[serde(default)]
    pub cpu: f64,
    #[serde(default)]
    pub tokens: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub note: String,
}

pub fn process_cpu_s() -> f64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[derive(Clone)]
pub struct Timeline {
    t0: Instant,
    events: Arc<Mutex<Vec<Event>>>,
    capture_index: Arc<AtomicUsize>,
}

impl Default for Timeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Timeline {
    pub fn new() -> Self {
        Timeline {
            t0: Instant::now(),
            events: Arc::new(Mutex::new(Vec::new())),
            capture_index: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn next_capture_index(&self) -> usize {
        self.capture_index.fetch_add(1, Ordering::Relaxed)
    }
    pub fn now(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }
    pub fn emit(&self, ev: &str, slot: &str, tokens: i64, note: &str) {
        let e = Event {
            t: self.now(),
            ev: ev.to_string(),
            slot: slot.to_string(),
            cpu: process_cpu_s(),
            tokens,
            note: note.to_string(),
        };
        self.events.lock().unwrap().push(e);
    }
    pub fn events(&self) -> Vec<Event> {
        let mut v = self.events.lock().unwrap().clone();
        v.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        v
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Breakdown {
    pub wall_s: f64,
    pub gen_prefill_s: f64,
    pub gen_decode_s: f64,
    pub verify_s: f64,
    pub orch_busy_s: f64,
    pub idle_wait_s: f64,
    /// verify time including the part hidden under generation
    pub verify_total_s: f64,
    pub orchestrator_cpu_s: f64,
    pub requests: i64,
    pub cancelled_requests: i64,
    pub reported_completion_tokens: i64,
    pub cancelled_stream_chunks: i64,
    /// consensus -> last cancelled stream closed (None when no early abort)
    pub cancel_latency_s: Option<f64>,
    pub consensus_t: Option<f64>,
}

/// Compute the phase partition from a problem's events (times relative to problem start).
pub fn breakdown(events: &[Event]) -> Breakdown {
    let mut evs: Vec<&Event> = events.iter().collect();
    evs.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    let start = evs
        .iter()
        .find(|e| e.ev == "problem_start")
        .map(|e| (e.t, e.cpu));
    let end = evs
        .iter()
        .rev()
        .find(|e| e.ev == "problem_end")
        .map(|e| (e.t, e.cpu));
    let (Some((t_start, cpu_start)), Some((t_end, cpu_end))) = (start, end) else {
        return Breakdown::default();
    };
    let mut b = Breakdown {
        wall_s: t_end - t_start,
        orchestrator_cpu_s: cpu_end - cpu_start,
        ..Default::default()
    };

    let mut inflight: HashSet<String> = HashSet::new();
    let mut decoding: HashSet<String> = HashSet::new();
    let mut verifying: i64 = 0;
    let mut prev_t = t_start;
    let mut prev_cpu = cpu_start;
    for e in evs.iter().filter(|e| e.t >= t_start && e.t <= t_end) {
        let dt = (e.t - prev_t).max(0.0);
        let dcpu = (e.cpu - prev_cpu).max(0.0);
        if !decoding.is_empty() {
            b.gen_decode_s += dt;
        } else if !inflight.is_empty() {
            b.gen_prefill_s += dt;
        } else if verifying > 0 {
            b.verify_s += dt;
        } else {
            let busy = dcpu.min(dt);
            b.orch_busy_s += busy;
            b.idle_wait_s += dt - busy;
        }
        if verifying > 0 {
            b.verify_total_s += dt;
        }
        match e.ev.as_str() {
            "req_start" => {
                inflight.insert(e.slot.clone());
                b.requests += 1;
            }
            "first_token" => {
                decoding.insert(e.slot.clone());
            }
            "req_end" => {
                inflight.remove(&e.slot);
                decoding.remove(&e.slot);
                b.reported_completion_tokens += e.tokens;
            }
            "req_cancel" => {
                inflight.remove(&e.slot);
                decoding.remove(&e.slot);
                b.cancelled_requests += 1;
                b.reported_completion_tokens += e.tokens;
                if let Ok(note) = serde_json::from_str::<serde_json::Value>(&e.note) {
                    b.cancelled_stream_chunks += note["observed_chunks"].as_i64().unwrap_or(0);
                }
                if let Some(c) = b.consensus_t {
                    let lat = (e.t - t_start) - c;
                    b.cancel_latency_s = Some(b.cancel_latency_s.map_or(lat, |x: f64| x.max(lat)));
                }
            }
            "verify_start" => verifying += 1,
            "verify_end" => verifying -= 1,
            "consensus" => {
                if b.consensus_t.is_none() {
                    b.consensus_t = Some(e.t - t_start);
                }
            }
            _ => {}
        }
        prev_t = e.t;
        prev_cpu = e.cpu;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ev(t: f64, ev: &str, slot: &str, cpu: f64, tokens: i64) -> Event {
        Event {
            t,
            ev: ev.into(),
            slot: slot.into(),
            cpu,
            tokens,
            note: String::new(),
        }
    }
    #[test]
    fn partition_sums_to_wall() {
        let e = vec![
            ev(0.0, "problem_start", "", 0.0, 0),
            ev(0.1, "req_start", "a", 0.01, 0),
            ev(0.3, "first_token", "a", 0.01, 0),
            ev(1.0, "req_end", "a", 0.02, 50),
            ev(1.0, "verify_start", "a", 0.02, 0),
            ev(1.5, "verify_end", "a", 0.03, 0),
            ev(2.0, "problem_end", "", 0.13, 0),
        ];
        let b = breakdown(&e);
        let sum = b.gen_prefill_s + b.gen_decode_s + b.verify_s + b.orch_busy_s + b.idle_wait_s;
        assert!((sum - b.wall_s).abs() < 1e-9);
        assert!((b.gen_prefill_s - 0.2).abs() < 1e-9);
        assert!((b.gen_decode_s - 0.7).abs() < 1e-9);
        assert!((b.verify_s - 0.5).abs() < 1e-9);
        assert!((b.orch_busy_s - 0.11).abs() < 1e-9);
        assert_eq!(b.reported_completion_tokens, 50);
    }
}
