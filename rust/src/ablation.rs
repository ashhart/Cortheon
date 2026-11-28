//! Single-sample baselines from bench/run_aime_ablation.py.

use crate::client::CallSpec;
use crate::orchestrator::{execute_pal, extract_prose_candidate, mk_outcome, Ctx};
use crate::prompts::{shared_user_prefix, AimeItem, SHARED_SYS};
use crate::timeline::Timeline;
use anyhow::Result;
use std::sync::Arc;

pub(crate) async fn single(
    ctx: &Arc<Ctx>,
    tl: &Timeline,
    arm: &str,
    item: &AimeItem,
) -> Result<(String, String)> {
    let endpoint = &ctx.endpoints[0];
    let tag = format!("abl_{arm}_{}", item.id);
    let spec = match arm {
        "vanilla_b1" => CallSpec {
            messages: vec![("user".into(), item.prompt.clone())],
            tag,
            max_tokens: 8192,
            temperature: 0.6,
            seed: Some(42),
            thinking_budget: Some(1500),
            ..Default::default()
        },
        "pal_b1" => {
            let prefix = shared_user_prefix(&ctx.skills, &item.prompt);
            let card = ctx.skills.card("pal_exact")?;
            CallSpec {
                messages: vec![
                    ("system".into(), SHARED_SYS.into()),
                    (
                        "user".into(),
                        format!("{prefix}\n\nPROTOCOL (pal_exact):\n{card}"),
                    ),
                ],
                tag,
                max_tokens: 2500,
                temperature: 0.25,
                seed: Some(101),
                thinking_budget: Some(500),
                reasoning_effort: Some("low".into()),
                ..Default::default()
            }
        }
        _ => anyhow::bail!("unknown single-sample baseline {arm}"),
    };
    let res = endpoint
        .complete(&spec, &spec.tag, tl, &ctx.stats, None)
        .await;
    if !res.ok() {
        return Ok((
            String::new(),
            format!(
                "{arm} | {}",
                res.raw_error.as_deref().unwrap_or(&res.finish_reason)
            ),
        ));
    }
    if arm == "vanilla_b1" {
        let answer = extract_prose_candidate(&ctx.light_pool, &res.content).await?;
        return Ok((answer, format!("{arm} | finish={}", res.finish_reason)));
    }
    let mut outcome = mk_outcome("pal_exact", 1, "pal", &res, &endpoint.base_url);
    // The official PAL control fixes its sandbox budget at 20 s.
    execute_pal(ctx, tl, &mut outcome, &spec.tag, 20.0, None).await?;
    if outcome.error.is_empty() {
        Ok((
            outcome.norm_answer,
            format!("{arm} | ok (assert={})", outcome.verified_constraints),
        ))
    } else {
        Ok((String::new(), format!("{arm} | exec_err={}", outcome.error)))
    }
}
