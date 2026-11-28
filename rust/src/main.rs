use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cortheon_rs::client::Sampling;
use cortheon_rs::prompts::{self, load_aime2025, AimeItem, SkillPack};
use cortheon_rs::runner::{
    build_ctx_for_arms, run_one_with_timeline, summarize, write_events, BuildOpts,
};
use cortheon_rs::timeline::Timeline;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Parser)]
#[command(
    name = "cortheon-rs",
    version,
    about = "Rust orchestration core for Cortheon-style execution-verified parallel consensus"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run arms over a benchmark (or one --problem) against OpenAI-compatible endpoints
    Run {
        /// Comma-separated OpenAI-compatible base URLs. Slots are assigned round-robin.
        #[arg(long, value_delimiter = ',', required = true)]
        endpoints: Vec<String>,
        #[arg(long, default_value_t = 4)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        agree: usize,
        /// aime2025 | single
        #[arg(long, default_value = "aime2025")]
        bench: String,
        #[arg(long)]
        dataset: Option<PathBuf>,
        #[arg(long)]
        problem: Option<String>,
        #[arg(long, default_value_t = 0)]
        start: usize,
        #[arg(long, default_value_t = 30)]
        end: usize,
        /// Comma-separated arms: vanilla_b1, maj4_prose, maj8_prose, pal_b1, cortheon_w1, cortheon_full, maj, maj_early
        #[arg(long, default_value = "cortheon_w1")]
        arms: String,
        #[arg(long)]
        model: Option<String>,
        /// Model sampling options applied to each request, as a JSON object
        #[arg(long, default_value = "{}")]
        sampling_json: Sampling,
        /// Override the output-token budget on each request
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        max_tokens: Option<u32>,
        #[arg(long, default_value_t = 360, value_parser = clap::value_parser!(u32).range(1..))]
        request_timeout_s: u32,
        /// Run attempts in slot order at temperature zero with this seed on every request
        #[arg(long)]
        deterministic_seed: Option<u32>,
        /// Save request bodies, raw answer and reasoning, and server-reported sampled tokens in the event log
        #[arg(long)]
        capture_output: bool,
        #[arg(long, env = "CORTHEON_API_KEY", hide_env_values = true)]
        api_key: Option<String>,
        #[arg(long, env = "CORTHEON_SRC")]
        cortheon_src: Option<PathBuf>,
        #[arg(long, env = "CORTHEON_PYTHON", default_value = "python3")]
        python: String,
        #[arg(long)]
        worker_script: Option<PathBuf>,
        #[arg(long)]
        exec_workers: Option<usize>,
        #[arg(long, default_value_t = 2)]
        light_workers: usize,
        #[arg(long, default_value_t = 20.0)]
        sandbox_timeout: f64,
        /// disable speculative verification of closed code blocks while a stream is still running
        #[arg(long)]
        no_speculative: bool,
        /// use non-streaming requests (like Cortheon's own client); disables TTFT + speculative
        #[arg(long)]
        no_stream: bool,
        #[arg(long)]
        deep: bool,
        #[arg(long, default_value = "rust_results.json")]
        out: PathBuf,
        #[arg(long, default_value = "rust_events.jsonl")]
        events: PathBuf,
        /// Write live ReasonForge progress events as JSON lines, with problem_id and arm.
        #[arg(long)]
        progress_jsonl: Option<PathBuf>,
    },
    /// Serve the mock OpenAI-compatible server
    Mock {
        #[arg(long, default_value_t = 18080)]
        port: u16,
        #[arg(long)]
        scenario: PathBuf,
    },
    /// Print the Wave-1 prompts the Rust core builds (JSON) for parity checks
    DumpPrompts {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, env = "CORTHEON_SRC")]
        cortheon_src: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        n: usize,
    },
}

fn checkout(src: Option<PathBuf>) -> PathBuf {
    src.unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crate inside checkout")
            .to_path_buf()
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Mock { port, scenario } => {
            let sc: cortheon_rs::mock::Scenario =
                serde_json::from_str(&std::fs::read_to_string(&scenario)?)?;
            let (url, h) = cortheon_rs::mock::spawn(sc, port).await?;
            eprintln!("mock server listening on {url}");
            h.await?;
        }
        Cmd::DumpPrompts {
            dataset,
            cortheon_src,
            n,
        } => {
            let skills = SkillPack::load(&checkout(cortheon_src))?;
            let items = load_aime2025(&dataset, 0, 0)?;
            let roles = prompts::wave1_roles(&skills, n, false)?;
            let mut out = vec![];
            for it in &items {
                let prefix = prompts::shared_user_prefix(&skills, &it.prompt);
                let msgs: Vec<_> = roles.iter().map(|r| serde_json::json!({"system": prompts::SHARED_SYS, "user": format!("{prefix}\n\nPROTOCOL ({}):\n{}", r.name, r.card)})).collect();
                out.push(serde_json::json!({"id": it.id, "prompt": it.prompt, "wave1": msgs}));
            }
            println!("{}", serde_json::to_string(&out)?);
        }
        Cmd::Run {
            endpoints,
            n,
            agree,
            bench,
            dataset,
            problem,
            start,
            end,
            arms,
            model,
            sampling_json,
            max_tokens,
            request_timeout_s,
            deterministic_seed,
            capture_output,
            api_key,
            cortheon_src,
            python,
            worker_script,
            exec_workers,
            light_workers,
            sandbox_timeout,
            no_speculative,
            no_stream,
            deep,
            out,
            events,
            progress_jsonl,
        } => {
            let cortheon_src = checkout(cortheon_src);
            let worker_script = worker_script.unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("verifier/worker.py")
            });
            let arms: Vec<String> = arms.split(',').map(|s| s.trim().to_owned()).collect();
            anyhow::ensure!(n > 0, "--n must be positive");
            anyhow::ensure!(agree > 0, "--agree must be positive");
            anyhow::ensure!(
                sandbox_timeout.is_finite() && sandbox_timeout > 0.0,
                "--sandbox-timeout must be positive and finite"
            );
            for arm in &arms {
                match arm.as_str() {
                    "cortheon_w1" | "cortheon_full" | "vanilla_b1" | "pal_b1" | "maj4_prose"
                    | "maj8_prose" | "maj" | "maj_early" => (),
                    other => {
                        anyhow::bail!("unknown arm {other:?}; see run --help for available arms")
                    }
                }
            }
            let items: Vec<AimeItem> = match bench.as_str() {
                "aime2025" => {
                    let ds = dataset
                        .unwrap_or_else(|| cortheon_src.join("bench/datasets/aime_2025.json"));
                    load_aime2025(&ds, start, end)?
                }
                "single" => vec![AimeItem {
                    id: "single".into(),
                    prompt: problem.context("--bench single needs --problem")?,
                    expected: String::new(),
                }],
                other => anyhow::bail!("unknown bench {other}"),
            };
            let opts = BuildOpts {
                endpoints,
                model,
                api_key,
                stream: !no_stream,
                cortheon_src,
                python,
                sampling: sampling_json,
                max_tokens,
                request_timeout_s,
                deterministic_seed,
                capture_output,
                worker_script,
                exec_workers: exec_workers.unwrap_or(n.min(4)),
                light_workers,
                sandbox_timeout_s: sandbox_timeout,
                agree,
                speculative: !no_speculative && !no_stream,
            };
            if let Some(path) = &progress_jsonl {
                anyhow::ensure!(
                    path != &out && path != &events,
                    "--progress-jsonl needs its own output file"
                );
            }
            let ctx = build_ctx_for_arms(&opts, &arms, n).await?;
            let progress = progress_jsonl
                .map(std::fs::File::create)
                .transpose()?
                .map(|file| Arc::new(Mutex::new(Ok(file))));
            let mut ev_file = std::io::BufWriter::new(std::fs::File::create(&events)?);
            let mut rows = vec![];
            for it in &items {
                for arm in &arms {
                    let mut timeline = Timeline::new();
                    if let Some(file) = progress.clone() {
                        let (arm, problem) = (arm.clone(), it.id.clone());
                        timeline = timeline.with_progress(move |event| {
                            let mut record = event.clone();
                            record["arm"] = serde_json::json!(arm);
                            record["problem_id"] = serde_json::json!(problem);
                            let mut writer = file.lock().unwrap();
                            if let Ok(f) = &mut *writer {
                                if let Err(err) = writeln!(f, "{record}") {
                                    *writer = Err(err);
                                }
                            }
                        });
                    }
                    let (row, evs, b) =
                        run_one_with_timeline(&ctx, arm, it, n, deep, timeline).await?;
                    eprintln!(
                        "[{arm}] {} pred={} correct={} wall={:.2}s gen={:.2}s verify={:.2}s orch={:.3}s idle={:.3}s | {}",
                        it.id, row["predicted"], row["correct"], b.wall_s, b.gen_prefill_s + b.gen_decode_s, b.verify_s, b.orch_busy_s, b.idle_wait_s,
                        row["detail"].as_str().unwrap_or("")
                    );
                    write_events(&mut ev_file, arm, &it.id, &evs)?;
                    rows.push(row);
                }
            }
            ev_file.flush()?;
            let summary = summarize(&rows);
            std::fs::write(
                &out,
                serde_json::to_string_pretty(
                    &serde_json::json!({"impl": "rust", "n": n, "agree": agree, "summary": summary, "rows": rows}),
                )?,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            if let Some(file) = progress {
                if let Err(e) = &*file.lock().unwrap() {
                    anyhow::bail!("writing the progress log failed: {e}");
                }
            }
        }
    }
    Ok(())
}
