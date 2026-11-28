//! Strict, unweighted AIME voting with cancellation at the first irreversible lead.

use crate::client::{CallSpec, Endpoint, Stats};
use crate::prompts::aime_prompt;
use crate::timeline::Timeline;
use anyhow::{ensure, Result};
use futures::future::{join_all, AbortHandle, Abortable};
use regex::Regex;
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock};
use tokio::sync::mpsc;

static ANSWER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i-u)^(?:\*{2})?ANSWER:[ \t]*(?:\*{2})?[ \t]*([0-9]{1,3})(?:\*{2})?[.]?$")
        .unwrap()
});

fn answer(content: &str, reason: &str) -> Option<u16> {
    if reason != "stop" {
        return None;
    }
    let last = content
        .trim_matches([' ', '\t', '\r', '\n'])
        .rsplit('\n')
        .next()?
        .trim_matches([' ', '\t', '\r']);
    ANSWER.captures(last)?[1].parse().ok()
}

fn irreversible(votes: &BTreeMap<u16, usize>, remaining: usize) -> Option<u16> {
    let (&leader, &count) = votes.iter().max_by_key(|(_, n)| *n)?;
    let second = votes
        .iter()
        .filter(|(a, _)| **a != leader)
        .map(|(_, &n)| n)
        .max()
        .unwrap_or(0);
    (count > second + remaining).then_some(leader)
}

#[derive(Debug, Serialize)]
pub struct Vote {
    pub winner: Option<u16>,
    pub vote_counts: BTreeMap<u16, usize>,
    /// Indices of terminal choices considered, in arrival order, including abstentions.
    pub received: Vec<usize>,
    pub abstentions: usize,
    pub unresolved_at_decision: usize,
    pub error: Option<String>,
}

impl Vote {
    pub fn status(&self) -> &'static str {
        if self.error.is_some() {
            "invalid_stream"
        } else if self.winner.is_some() {
            "winner"
        } else if self.vote_counts.is_empty() {
            "no_valid_answers"
        } else {
            "tie"
        }
    }
}

/// Slot i uses endpoint i modulo the number of endpoints and seed 101 * (i + 1).
/// A counted stream must supply usage and [DONE] even after its ballot decided the vote.
pub async fn run(
    endpoints: &[Arc<Endpoint>],
    stats: &Stats,
    tl: &Timeline,
    problem: &str,
    n: usize,
) -> Result<Vote> {
    ensure!(
        (1..=16).contains(&n),
        "maj_early needs --n between 1 and 16"
    );
    ensure!(
        !endpoints.is_empty(),
        "maj_early needs at least one endpoint"
    );
    ensure!(
        endpoints
            .iter()
            .all(|ep| ep.stream && ep.deterministic_seed.is_none()),
        "maj_early requires streaming and distinct seeds"
    );

    let prompt = aime_prompt(problem);
    let (tx, mut rx) = mpsc::unbounded_channel();
    let mut requests = Vec::with_capacity(n);
    let mut handles = Vec::with_capacity(n);
    for i in 0..n {
        let ep = &endpoints[i % endpoints.len()];
        let spec = CallSpec {
            messages: vec![("user".into(), prompt.clone())],
            tag: format!("maj_early_s{i}"),
            seed: Some(101 * (i as i64 + 1)),
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            min_p: Some(0.0),
            template_reasoning_effort: Some("low".into()),
            loop_guard: false,
            ..Default::default()
        };
        let tx = tx.clone();
        let (handle, registration) = AbortHandle::new_pair();
        handles.push(handle);
        let request = async move {
            let result = ep
                .complete_ballot(&spec, tl, stats, |content, reason| {
                    let _ = tx.send((i, Ok(answer(content, reason))));
                })
                .await;
            if let Some(error) = &result.raw_error {
                let _ = tx.send((i, Err(error.clone())));
            }
            result.raw_error
        };
        requests.push(Abortable::new(request, registration));
    }
    drop(tx);

    let tally = async {
        let mut vote = Vote {
            winner: None,
            vote_counts: BTreeMap::new(),
            received: Vec::with_capacity(n),
            abstentions: 0,
            unresolved_at_decision: n,
            error: None,
        };
        while vote.unresolved_at_decision > 0 {
            let Some((i, ballot)) = rx.recv().await else {
                vote.error = Some("all readers exited before the vote completed".into());
                break;
            };
            let ballot = match ballot {
                Ok(ballot) => ballot,
                Err(e) => {
                    vote.error = Some(format!("sample {i}: {e}"));
                    break;
                }
            };
            vote.received.push(i);
            vote.unresolved_at_decision -= 1;
            if let Some(a) = ballot {
                *vote.vote_counts.entry(a).or_default() += 1;
            } else {
                vote.abstentions += 1;
            }
            if let Some(winner) = irreversible(&vote.vote_counts, vote.unresolved_at_decision) {
                vote.winner = Some(winner);
                tl.emit("consensus", "", 0, &format!("maj_early {winner}"));
                break;
            }
        }
        for (i, handle) in handles.iter().enumerate() {
            if !vote.received.contains(&i) || vote.error.is_some() {
                handle.abort();
            }
        }
        vote
    };
    let (outcomes, mut vote) = tokio::join!(join_all(requests), tally);
    if vote.error.is_none() {
        for &i in &vote.received {
            if let Ok(Some(e)) = &outcomes[i] {
                vote.error = Some(format!("sample {i}: {e}"));
                vote.winner = None;
                break;
            }
        }
    }
    Ok(vote)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_complete_final_answer_line_votes() {
        for (text, expected) in [
            ("work\n **ANSWER:** 007. \r\n", Some(7)),
            ("**aNsWeR: 42**", Some(42)),
            ("ANSWER: 0", Some(0)),
            ("ANSWER: 1234", None),
            ("ANSWER: 3.0", None),
            ("ANSWER: 2\nmore", None),
            ("thinking 3", None),
            ("ANSWER: ７", None),
            ("ANSWER: -1", None),
            ("ANSWER: 7 or 8", None),
        ] {
            assert_eq!(answer(text, "stop"), expected, "{text}");
        }
        for reason in ["length", "", "content_filter"] {
            assert_eq!(answer("ANSWER: 7", reason), None);
        }
    }

    #[test]
    fn a_tie_or_a_possible_tie_has_no_winner() {
        assert_eq!(irreversible(&BTreeMap::new(), 0), None);
        assert_eq!(irreversible(&BTreeMap::from([(1, 2)]), 2), None);
        let votes = BTreeMap::from([(3, 3), (7, 2), (8, 2)]);
        assert_eq!(irreversible(&votes, 1), None);
        assert_eq!(irreversible(&votes, 0), Some(3));
        assert_eq!(irreversible(&BTreeMap::from([(1, 2), (7, 2)]), 0), None);
    }
}
