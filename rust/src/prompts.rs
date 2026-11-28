//! Prompt construction ported from flash_forge/skill_loader.py and
//! flash_forge/reason_forge.py (cards and exemplars are read from the Cortheon
//! checkout). Use `cortheon-rs dump-prompts` to inspect the rendered messages.

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub const SHARED_SYS: &str = "You are a precision reasoning engine running inside a verified neuro-symbolic sandbox with Python 3, sympy, z3, fractions, itertools, math, and numpy available.\nNever guess arithmetic, combinatorics, or constraint satisfaction in prose when Python can compute and verify it exactly.";

pub struct SkillPack {
    cards_dir: PathBuf,
    exemplars: Vec<Value>,
}

impl SkillPack {
    pub fn load(cortheon_src: &Path) -> Result<Self> {
        let skill = cortheon_src.join("flash_forge").join("_skill");
        let ex_path = skill.join("exemplars.json");
        let exemplars = std::fs::read_to_string(&ex_path)
            .ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        Ok(SkillPack {
            cards_dir: skill.join("cards"),
            exemplars,
        })
    }

    pub fn card(&self, name: &str) -> Result<String> {
        let p = self.cards_dir.join(format!("{name}.md"));
        let text = std::fs::read_to_string(&p)
            .with_context(|| format!("Missing skill micro-card: {p:?}"))?;
        // Python's read_text() applies universal newline translation to protocol cards.
        Ok(text.trim().replace("\r\n", "\n").replace('\r', "\n"))
    }

    /// Port of SkillPack.match_exemplars (stable sort, descending score).
    pub fn match_exemplars(&self, phase: &str, query: &str, limit: usize) -> String {
        static TOKENS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9_]{3,}").unwrap());
        let q = query.to_lowercase();
        let q_tokens: HashSet<&str> = TOKENS.find_iter(&q).map(|m| m.as_str()).collect();
        let mut scored: Vec<(usize, &Value)> = vec![];
        for ex in &self.exemplars {
            if ex.get("phase").and_then(|p| p.as_str()) != Some(phase) {
                continue;
            }
            let tags: HashSet<String> = ex
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(|s| s.to_lowercase()))
                        .collect()
                })
                .unwrap_or_default();
            let score = q_tokens.iter().filter(|t| tags.contains(**t)).count();
            scored.push((score, ex));
        }
        if scored.is_empty() {
            return String::new();
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored
            .iter()
            .take(limit)
            .map(|(_, ex)| {
                format!(
                    "EXAMPLE ({}):\nInput: {}\nVerified Approach:\n{}",
                    ex.get("title").and_then(|x| x.as_str()).unwrap_or(phase),
                    ex.get("input").and_then(|x| x.as_str()).unwrap_or(""),
                    ex.get("output").and_then(|x| x.as_str()).unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[derive(Clone, Debug)]
pub struct Role {
    pub name: String,
    pub kind: String, // "pal" | "analytical"
    pub card: String,
    pub temp: f64,
    pub seed: i64,
    pub think: i64,
}

pub fn shared_user_prefix(skills: &SkillPack, problem: &str) -> String {
    let ex = skills.match_exemplars("pal_verify", problem, 1);
    let mut s = format!("PROBLEM:\n{}", problem.trim());
    if !ex.is_empty() {
        s.push_str("\n\n");
        s.push_str(&ex);
    }
    s
}

pub fn wave1_roles(skills: &SkillPack, n: usize, deep: bool) -> Result<Vec<Role>> {
    let think_pal = if deep { 1200 } else { 700 };
    let think_ana = if deep { 2000 } else { 1300 };
    let all = [
        ("pal_exact", "pal", "pal_exact", 0.25, 101, think_pal),
        ("pal_alt_z3", "pal", "pal_alt_z3", 0.60, 202, think_pal),
        ("pal_skeptic", "pal", "pal_skeptic", 0.45, 303, think_pal),
        (
            "analytical",
            "analytical",
            "reason_analytical",
            0.40,
            404,
            think_ana,
        ),
    ];
    all.iter()
        .take(n)
        .map(|(name, kind, card, t, s, tb)| {
            Ok(Role {
                name: name.to_string(),
                kind: kind.to_string(),
                card: skills.card(card)?,
                temp: *t,
                seed: *s,
                think: *tb,
            })
        })
        .collect()
}

pub struct AimeItem {
    pub id: String,
    pub prompt: String,
    pub expected: String,
}

const AIME_SUFFIX: &str = "\n\nNote: Every AIME answer is an integer from 000 to 999.\nEnd your final response with `ANSWER: <integer>`.";

pub(crate) fn aime_prompt(problem: &str) -> String {
    let problem = problem.trim();
    if problem.ends_with(AIME_SUFFIX) {
        problem.to_owned()
    } else {
        format!("{problem}{AIME_SUFFIX}")
    }
}

/// Port of bench/run_aime_ablation.load_aime2025.
pub fn load_aime2025(path: &Path, start: usize, end: usize) -> Result<Vec<AimeItem>> {
    let raw: Vec<Value> = serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| format!("reading {path:?}"))?,
    )?;
    let mut rows = Vec::with_capacity(raw.len());
    for (i, row) in raw.iter().enumerate() {
        let idx = as_i64(&row["problem_idx"])
            .with_context(|| format!("row {i}: missing or invalid problem_idx"))?;
        let answer = as_i64(&row["answer"])
            .filter(|a| (0..=999).contains(a))
            .with_context(|| format!("problem {idx}: expected an AIME answer from 0 to 999"))?;
        let problem = row["problem"]
            .as_str()
            .filter(|p| !p.trim().is_empty())
            .with_context(|| format!("problem {idx}: missing problem text"))?;
        rows.push((
            idx,
            AimeItem {
                id: format!("AIME25_{idx:02}"),
                prompt: aime_prompt(problem),
                expected: answer.to_string(),
            },
        ));
    }
    rows.sort_by_key(|(idx, _)| *idx);
    let items: Vec<_> = rows.into_iter().map(|(_, item)| item).collect();
    let e = if end == 0 {
        items.len()
    } else {
        end.min(items.len())
    };
    let s = start.min(e);
    Ok(items.into_iter().skip(s).take(e - s).collect())
}

fn as_i64(v: &Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
}
