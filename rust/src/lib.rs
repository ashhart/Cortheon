//! cortheon-rs: Rust/tokio orchestration core for Cortheon v2-style
//! execution-verified parallel consensus. Derived from Cortheon v2
//! (https://github.com/ashhart/Cortheon, MIT, (c) 2026 Cortheon contributors).
mod ablation;
pub mod client;
pub mod consensus;
pub mod early_vote;
pub mod mock;
pub mod orchestrator;
pub mod prompts;
pub mod runner;
pub mod timeline;
pub mod verifier;
