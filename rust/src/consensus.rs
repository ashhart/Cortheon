//! Answer clustering, early-abort rule and weighting, ported from
//! flash_forge/reason_forge.py. Integer / plain-rational equivalence runs natively;
//! anything that needs SymPy goes to a Python worker running Cortheon's own
//! `_answers_equivalent`, so results match Cortheon exactly.

use crate::verifier::VerifierPool;
use regex::Regex;
use std::sync::OnceLock;

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub name: String,
    pub wave: i32,
    pub kind: String,
    pub raw_content: String,
    pub reasoning: String,
    pub code: String,
    /// Some(ok) when the script was executed
    pub exec_ok: Option<bool>,
    pub exec_stdout: String,
    pub raw_answer: String,
    pub norm_answer: String,
    pub verified_constraints: bool,
    pub error: String,
    pub latency_s: f64,
    pub exec_elapsed_s: f64,
    pub completion_tokens: i64,
    pub endpoint: String,
}

fn re(cell: &'static OnceLock<Regex>, pat: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pat).unwrap())
}

static INT_RE: OnceLock<Regex> = OnceLock::new();
static RAT_RE: OnceLock<Regex> = OnceLock::new();
static PFRAC_RE: OnceLock<Regex> = OnceLock::new();
static INT_EXPECTED: OnceLock<Regex> = OnceLock::new();

pub fn is_int(s: &str) -> bool {
    re(&INT_RE, r"^-?\d+$").is_match(s)
}

/// Port of reason_forge._answer_satisfies_prompt_format.
pub fn format_ok(problem: &str, norm: &str) -> bool {
    if norm.is_empty() {
        return false;
    }
    let r = re(
        &INT_EXPECTED,
        r"(?i)(?:ANSWER\s*[:=]\s*<\s*int(?:eger)?\s*>|exact\s+integer|is\s+an\s+integer\b|integer\s+from\s+\d+\s+to\s+\d+)",
    );
    if r.is_match(problem) {
        return is_int(norm.trim());
    }
    true
}

/// Port of reason_forge._branch_weight.
pub fn branch_weight(o: &Outcome) -> f64 {
    if o.kind == "pal" {
        if o.exec_ok == Some(true) && !o.norm_answer.is_empty() {
            return 3.0 + if o.verified_constraints { 0.25 } else { 0.0 };
        }
        return 0.0;
    }
    if !o.norm_answer.is_empty() {
        1.25
    } else {
        0.0
    }
}

fn parse_rat(s: &str) -> Option<(i128, i128)> {
    let t: String = s.chars().filter(|c| *c != ' ').collect();
    match t.split_once('/') {
        Some((a, b)) => Some((a.parse().ok()?, b.parse().ok()?)),
        None => Some((t.parse().ok()?, 1)),
    }
}

/// Native subset of `_answers_equivalent`. Returns None when SymPy is required.
pub fn fast_equiv(a: &str, b: &str) -> Option<bool> {
    if a.is_empty() || b.is_empty() {
        return Some(false);
    }
    if a.contains('\\') || a.contains('$') || b.contains('\\') || b.contains('$') {
        return None;
    }
    let prep = |s: &str| -> Option<String> {
        let t = s.trim();
        if re(&PFRAC_RE, r"^\(\s*(-?\d+)\s*\)\s*/\s*\(\s*(-?\d+)\s*\)$").is_match(t) {
            return None;
        }
        if is_int(t) {
            return t.parse::<i128>().ok().map(|v| v.to_string());
        }
        Some(t.replace('^', "**"))
    };
    let pa = prep(a)?;
    let pb = prep(b)?;
    if pa.is_empty() || pb.is_empty() {
        return Some(false);
    }
    if pa.to_lowercase() == pb.to_lowercase() {
        return Some(true);
    }
    let rat = re(&RAT_RE, r"^-?\d+(?:\s*/\s*-?\d+)?$");
    if rat.is_match(&pa) && rat.is_match(&pb) {
        let (an, ad) = parse_rat(&pa)?;
        let (bn, bd) = parse_rat(&pb)?;
        if ad == 0 || bd == 0 {
            return Some(false);
        }
        return Some(an.checked_mul(bd)? == bn.checked_mul(ad)?);
    }
    None
}

pub async fn equiv(light: &VerifierPool, a: &str, b: &str) -> bool {
    match fast_equiv(a, b) {
        Some(x) => x,
        None => light.equiv(a, b).await.unwrap_or(false),
    }
}

/// Port of reason_forge._cluster_answers. Returns (representative, member indices).
pub async fn cluster(light: &VerifierPool, outs: &[&Outcome]) -> Vec<(String, Vec<usize>)> {
    let mut clusters: Vec<(String, Vec<usize>)> = vec![];
    for (i, o) in outs.iter().enumerate() {
        if o.norm_answer.is_empty() {
            continue;
        }
        let mut placed = false;
        for c in clusters.iter_mut() {
            if equiv(light, &c.0, &o.norm_answer).await {
                c.1.push(i);
                if (o.norm_answer.contains('/') || is_int(&o.norm_answer)) && c.0.contains('.') {
                    c.0 = o.norm_answer.clone();
                }
                placed = true;
                break;
            }
        }
        if !placed {
            clusters.push((o.norm_answer.clone(), vec![i]));
        }
    }
    let key = |c: &(String, Vec<usize>)| {
        (
            c.1.len(),
            c.1.iter().map(|&i| branch_weight(outs[i])).sum::<f64>(),
        )
    };
    // Python: sort(key=(len, weight), reverse=True) -- stable, descending
    clusters.sort_by(|x, y| {
        let (kx, ky) = (key(x), key(y));
        ky.0.cmp(&kx.0)
            .then(ky.1.partial_cmp(&kx.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fast_equiv_matches_python_rules() {
        assert_eq!(fast_equiv("42", "042"), Some(true));
        assert_eq!(fast_equiv("1/2", "2/4"), Some(true));
        assert_eq!(fast_equiv("1/2", "1/3"), Some(false));
        assert_eq!(fast_equiv("7", "8"), Some(false));
        assert_eq!(fast_equiv("Alice", "alice"), Some(true));
        assert_eq!(fast_equiv("1/0", "1"), Some(false));
        assert_eq!(fast_equiv("sqrt(2)", "2**(1/2)"), None);
        assert_eq!(fast_equiv("\\frac{1}{2}", "1/2"), None);
        assert_eq!(fast_equiv("", "1"), Some(false));
    }
    #[test]
    fn format_rule() {
        let p = "Note: Every AIME answer is an integer from 000 to 999.";
        assert!(format_ok(p, "42"));
        assert!(!format_ok(p, "sqrt(2)"));
        assert!(format_ok("free form", "sqrt(2)"));
    }
}
