use std::path::Path;
use std::process::Command;

fn compare_with_python(script: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let result =
        Command::new(std::env::var("CORTHEON_PYTHON").unwrap_or_else(|_| "python3".into()))
            .arg(root.join("tests").join(script))
            .arg(env!("CARGO_BIN_EXE_cortheon-rs"))
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
fn solver_matches_upstream_python_on_the_same_mock_responses() {
    compare_with_python("solver_parity.py");
}

#[test]
fn all_official_aime_arms_match_upstream_python() {
    compare_with_python("upstream_parity.py");
}

#[test]
fn cli_checkpoints_each_completed_row_and_exposes_log_failures() {
    compare_with_python("checkpoints.py");
}
