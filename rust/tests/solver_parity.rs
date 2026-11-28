use std::path::Path;
use std::process::Command;

fn compare_with_python(script: &str, args: &[&str]) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let result =
        Command::new(std::env::var("CORTHEON_PYTHON").unwrap_or_else(|_| "python3".into()))
            .arg(root.join("tests").join(script))
            .arg(env!("CARGO_BIN_EXE_cortheon-rs"))
            .args(args)
            .current_dir(root.parent().unwrap())
            .output()
            .expect("run the current checkout's Python reference solver");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn shared_aime_arms_match_upstream_python() {
    compare_with_python("upstream_parity.py", &[]);
}

#[test]
#[ignore = "the Rust solver implements Cortheon 2.3.1; upstream uses ReasonForge v5"]
fn reasonforge_v5_solver_parity() {
    compare_with_python("solver_parity.py", &[]);
}

#[test]
#[ignore = "includes the ReasonForge arms whose strategy changed in v5"]
fn reasonforge_v5_all_aime_arms_parity() {
    compare_with_python("upstream_parity.py", &["--all-arms"]);
}

#[test]
fn cli_checkpoints_each_completed_row_and_exposes_log_failures() {
    compare_with_python("checkpoints.py", &[]);
}
