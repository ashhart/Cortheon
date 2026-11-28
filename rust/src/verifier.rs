//! Pool of persistent Python verifier workers (see verifier/worker.py).
//!
//! Python, SymPy and Z3 stay the checkers: each worker imports Cortheon's own
//! extraction / normalization / sandbox code once (so no per-check interpreter +
//! SymPy import cost on the critical path), and runs model scripts in Cortheon's
//! `run_python` sandbox (bwrap on Linux, rlimits, wall-clock timeout).
//! Rust bounds concurrency and timeouts, cancels abandoned sandbox work, and
//! replaces workers that stop responding.

use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{oneshot, Semaphore};

#[derive(Clone, Debug)]
pub struct VerifierConfig {
    pub python: String,
    pub worker_script: PathBuf,
    pub cortheon_src: Option<PathBuf>,
    pub workers: usize,
}

struct Worker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

// Kill an unresponsive import if startup times out or the spawning task disappears.
// Ready workers instead clean up their sandbox children on stdin EOF, including
// when the Tokio runtime shuts down before an in-flight cancellation is drained.
struct StartupGuard(Option<Child>);

impl Drop for StartupGuard {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            let _ = child.start_kill();
        }
    }
}

impl Worker {
    async fn spawn(cfg: &VerifierConfig) -> Result<Worker> {
        let mut cmd = Command::new(&cfg.python);
        cmd.arg(&cfg.worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(src) = &cfg.cortheon_src {
            cmd.env("CORTHEON_SRC", src);
        }
        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning verifier worker {:?}", cfg.worker_script))?;
        let stdin = child.stdin.take().unwrap();
        let mut stdout = BufReader::new(child.stdout.take().unwrap());
        let mut startup = StartupGuard(Some(child));
        let mut line = String::new();
        tokio::time::timeout(Duration::from_secs(60), stdout.read_line(&mut line))
            .await
            .map_err(|_| anyhow!("verifier worker did not become ready in 60s"))??;
        if serde_json::from_str::<Value>(&line)?["ready"] != true {
            return Err(anyhow!("verifier worker failed to start: {line}"));
        }
        Ok(Worker {
            child: startup.0.take().unwrap(),
            stdin,
            stdout,
        })
    }

    async fn roundtrip(
        &mut self,
        req: &Value,
        caller: &mut oneshot::Sender<Result<Value>>,
    ) -> Result<Value> {
        let mut s = serde_json::to_string(req)?;
        s.push('\n');
        self.stdin.write_all(s.as_bytes()).await?;
        self.stdin.flush().await?;
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line);
        tokio::pin!(read);
        let n = tokio::select! {
            biased;
            n = &mut read => n?,
            _ = caller.closed(), if req["op"] == "exec" => {
                // The pipe preserves ordering even if cancellation races startup
                // or completion. Do not reuse this worker until its reply is read.
                self.stdin.write_all(b"{\"op\":\"cancel\"}\n").await?;
                self.stdin.flush().await?;
                read.await?
            }
        };
        if n == 0 {
            return Err(anyhow!("verifier worker exited"));
        }
        Ok(serde_json::from_str(&line)?)
    }

    async fn stop(mut self) {
        // EOF asks the control loop to clean up its sandbox process group, which
        // upstream starts in a separate session. Give it time before killing it.
        drop(self.stdin);
        if tokio::time::timeout(Duration::from_secs(2), self.child.wait())
            .await
            .is_err()
        {
            let _ = self.child.kill().await;
        }
    }
}

#[derive(Clone)]
pub struct VerifierPool {
    cfg: VerifierConfig,
    idle: Arc<Mutex<Vec<Worker>>>,
    sem: Arc<Semaphore>,
}

impl VerifierPool {
    /// Workers are created on their first call. Integer-only voting needs none.
    pub(crate) fn lazy(cfg: VerifierConfig) -> Self {
        let n = cfg.workers.max(1);
        Self {
            cfg,
            idle: Arc::new(Mutex::new(Vec::new())),
            sem: Arc::new(Semaphore::new(n)),
        }
    }

    pub async fn start(cfg: VerifierConfig) -> Result<Self> {
        let n = cfg.workers.max(1);
        let spawns = (0..n).map(|_| Worker::spawn(&cfg));
        let workers: Vec<Worker> = futures::future::try_join_all(spawns).await?;
        Ok(VerifierPool {
            idle: Arc::new(Mutex::new(workers)),
            sem: Arc::new(Semaphore::new(n)),
            cfg,
        })
    }

    /// Queued calls leave the semaphore queue when dropped. Once a worker is
    /// assigned, an independent task cancels an abandoned exec and drains its reply.
    pub async fn call(&self, req: Value, timeout: Duration) -> Result<Value> {
        let permit = self.sem.clone().acquire_owned().await?;
        let worker = self.idle.lock().unwrap().pop();
        let mut worker = match worker {
            Some(worker) => worker,
            None => Worker::spawn(&self.cfg).await?,
        };
        let idle = self.idle.clone();
        let (mut sender, receiver) = oneshot::channel();
        tokio::spawn(async move {
            let _permit = permit;
            if sender.is_closed() {
                idle.lock().unwrap().push(worker);
                return;
            }
            let result = tokio::time::timeout(timeout, worker.roundtrip(&req, &mut sender))
                .await
                .map_err(|_| anyhow!("verifier job exceeded {:.1}s", timeout.as_secs_f64()))
                .and_then(|result| result);
            if result.is_ok() {
                idle.lock().unwrap().push(worker);
            } else {
                worker.stop().await;
            }
            let _ = sender.send(result);
        });
        receiver.await?
    }

    /// Free worker slots right now (used by tests to prove abandoned checks are killed).
    pub fn available(&self) -> usize {
        self.sem.available_permits()
    }

    pub async fn extract(&self, content: &str) -> Result<String> {
        let v = self
            .call(
                json!({"op": "extract", "content": content}),
                Duration::from_secs(30),
            )
            .await?;
        Ok(v["code"].as_str().unwrap_or("").to_string())
    }
    pub async fn exec(&self, code: &str, timeout_s: f64) -> Result<Value> {
        self.call(
            json!({"op": "exec", "code": code, "timeout_s": timeout_s}),
            Duration::from_secs_f64(timeout_s + 15.0),
        )
        .await
    }
    pub async fn prose(&self, content: &str) -> Result<(String, String)> {
        let v = self
            .call(
                json!({"op": "prose", "content": content}),
                Duration::from_secs(30),
            )
            .await?;
        Ok((
            v["raw_answer"].as_str().unwrap_or("").into(),
            v["norm_answer"].as_str().unwrap_or("").into(),
        ))
    }
    pub async fn equiv(&self, a: &str, b: &str) -> Result<bool> {
        let v = self
            .call(
                json!({"op": "equiv", "a": a, "b": b}),
                Duration::from_secs(30),
            )
            .await?;
        Ok(v["eq"].as_bool().unwrap_or(false))
    }
    pub async fn normalize(&self, raw: &str) -> Result<String> {
        let v = self
            .call(
                json!({"op": "normalize", "raw": raw}),
                Duration::from_secs(30),
            )
            .await?;
        Ok(v["norm"].as_str().unwrap_or("").into())
    }
    pub async fn ping(&self) -> Result<Value> {
        self.call(json!({"op": "ping"}), Duration::from_secs(10))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn dropping_a_queued_call_releases_its_place_without_dispatch() {
        let pool = VerifierPool::lazy(VerifierConfig {
            python: "unused".into(),
            worker_script: PathBuf::new(),
            cortheon_src: None,
            workers: 1,
        });
        let occupied = pool.sem.acquire().await.unwrap();
        let mut queued = Box::pin(pool.ping());
        assert!(futures::poll!(&mut queued).is_pending());
        tokio::task::yield_now().await;

        let mut next = Box::pin(pool.sem.acquire());
        assert!(futures::poll!(&mut next).is_pending());
        drop(queued);
        drop(occupied);
        // The next waiter must acquire synchronously. A detached caller still
        // in the semaphore queue would receive this permit and dispatch its job.
        assert!(futures::poll!(&mut next).is_ready());
    }
}
