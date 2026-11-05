//! Bounded integration experiments and explicit file workflow. No generated native code.
use crate::{
    Authority, Effect, casework,
    curriculum::route,
    demand::genome::{BirthLimits, Genome, bounded_path},
    live_store::checksum,
    numeric::{NumericBudget, NumericTolerance, continuous::*},
    semantic::*,
    telemetry::{LabTelemetry, Value as V},
    text_skill::{publish_new, read_bounded},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, error::Error, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    shortage: String,
    cost: String,
}
fn render(template: &str, fields: &[(&str, u64)]) -> Result<String> {
    if template.len() > 512 {
        return Err("recipe bounds".into());
    }
    let mut s = template.to_owned();
    for (key, n) in fields {
        s = s.replace(&format!("{{{key}}}"), &n.to_string());
    }
    if s.contains(['{', '}']) {
        return Err("unknown recipe placeholder".into());
    }
    Ok(s)
}
fn quantity(n: f64) -> Result<u64> {
    if !n.is_finite() || n < 0.0 || n > 1e12 || n.fract() != 0.0 {
        return Err("nonnegative exact integer result required".into());
    }
    Ok(n as u64)
}
pub fn report(number: &Path, root: &Path, recipe: &Path, output: &Path) -> Result<()> {
    for p in [number, root, recipe, output] {
        bounded_path(p)?;
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let config: Recipe = serde_json::from_slice(&read_bounded(recipe, 4096)?)?;
    bounded_path(&root.join("orders.csv"))?;
    bounded_path(&root.join("stock.csv"))?;
    let orders = read_bounded(&root.join("orders.csv"), 1_048_576)?;
    let stock = read_bounded(&root.join("stock.csv"), 1_048_576)?;
    let orders_rows = casework::rows(&orders, &["item", "requested"])?;
    let stock_rows = casework::rows(&stock, &["item", "available", "unit_price_cents"])?;
    let mut inventory = BTreeMap::new();
    let item_ok = |s: &str| {
        !s.is_empty()
            && s.len() <= 64
            && s.as_bytes()[0].is_ascii_alphanumeric()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    };
    for row in stock_rows {
        if !item_ok(&row[0])
            || inventory
                .insert(
                    row[0].clone(),
                    (casework::number(&row[1])?, casework::number(&row[2])?),
                )
                .is_some()
        {
            return Err("duplicate or invalid stock item".into());
        }
    }
    let genome = Genome::open_number(number, &auth)?;
    fs::create_dir(output)?;
    let mut birth = genome.birth(&output.join("birth"), BirthLimits::default(), &auth)?;
    let runtime = birth.controller.runtime_mut();
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "item",
        "requested",
        "available",
        "needed",
        "unit_price_cents",
        "cost_cents",
    ])?;
    let mut seen = std::collections::BTreeSet::new();
    let mut receipts = Vec::new();
    for row in orders_rows {
        if !item_ok(&row[0]) || !seen.insert(row[0].clone()) {
            return Err("duplicate or invalid order item".into());
        }
        let requested = casework::number(&row[1])?;
        let (available, price) = *inventory.get(&row[0]).ok_or("missing stock item")?;
        let shortage = render(
            &config.shortage,
            &[("requested", requested), ("available", available)],
        )?;
        let binding = route(runtime, &shortage, &auth)?.ok_or("shortage needs clarification")?;
        // Zero purchases for surplus is a declared business rule, not learned cognition.
        let needed = quantity(binding.value.max(0.0))?;
        let sentence = render(&config.cost, &[("needed", needed), ("price", price)])?;
        let cost = route(runtime, &sentence, &auth)?.ok_or("cost needs clarification")?;
        let cents = quantity(cost.value)?;
        writer.write_record([
            row[0].clone(),
            requested.to_string(),
            available.to_string(),
            needed.to_string(),
            price.to_string(),
            cents.to_string(),
        ])?;
        receipts.push(serde_json::json!({"item":row[0],"shortage_request":shortage,"cost_request":sentence,"needed":needed,"cost_cents":cents,"shortage_module":binding.module,"cost_module":cost.module}));
    }
    let bytes = writer.into_inner()?;
    publish_new(&output.join("report.csv"), &bytes)?;
    let receipt = serde_json::json!({"schema":"axon-inventory-report/v1","workflow":"host-provided shortage then cost; learned language routing and numeric programs","orders_checksum":format!("{:016x}",checksum(&orders)),"stock_checksum":format!("{:016x}",checksum(&stock)),"rows":receipts,"loads":runtime.stats().loads,"active_modules":runtime.stats().active_modules,"external_purchases":0});
    publish_new(
        &output.join("audit-inventory-v1.json"),
        &serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{receipt}");
    Ok(())
}
fn demonstrations(arity: usize, label: &str, start: usize, count: usize) -> Vec<Demonstration> {
    (start..start + count)
        .map(|i| {
            let x: Vec<_> = (0..arity)
                .map(|j| ((i * 17 + j * 11 + 3) % 97) as f64 / 7.0 + 0.1)
                .collect();
            let mut roles: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(j, n)| format!("campo{j} {n}"))
                .collect();
            if i % 2 == 1 {
                roles.reverse();
            }
            let sentence = if i % 3 == 0 {
                format!("agora calcule {label} com {}", roles.join(" e "))
            } else if i % 3 == 1 {
                format!("{label} {} agora", roles.join(" com "))
            } else {
                format!("calcule {label} {}", roles.join(" e "))
            };
            let answer = x.iter().map(|n| n * n).sum::<f64>().sqrt();
            Demonstration {
                sentence,
                focus: Focus::default(),
                meaning: Meaning::Calculate {
                    arguments: x,
                    answer,
                },
            }
        })
        .collect()
}
#[derive(Serialize)]
struct Trial {
    generation: usize,
    window: u8,
    admitted: bool,
    admission_correct: usize,
    admission_total: usize,
    work: Option<u64>,
    bytes: Option<usize>,
    reason: Option<String>,
}
fn spaced(mut ds: Vec<Demonstration>, gap: usize) -> Vec<Demonstration> {
    for d in &mut ds {
        for role in 0..4 {
            d.sentence = d.sentence.replace(
                &format!("campo{role} "),
                &format!(
                    "campo{role} {}",
                    if gap == 2 { "valor " } else { "valor atual " }
                ),
            );
        }
    }
    ds
}
fn meta(output: &Path, telemetry: &mut LabTelemetry<std::io::Stdout>) -> Result<serde_json::Value> {
    let mut trials = Vec::new();
    let mut winner: Option<SemanticSeed> = None;
    for generation in 0..2 {
        let arity = generation + 3;
        let ds = spaced(demonstrations(arity, "descendente", 0, 12), generation + 2);
        let admission = spaced(
            demonstrations(arity, "descendente", 100, 12),
            generation + 2,
        );
        let mut candidates = Vec::new();
        for window in 1..=3 {
            let parent = if let Some(s) = &winner {
                s.clone()
            } else {
                SemanticSeed::new(vec![Rule::learn_relational(
                    &demonstrations(2, "ancestral", 0, 12),
                    NumericBudget::default(),
                    window,
                    &[],
                )?])?
            };
            let refs: Vec<_> = parent.rules().iter().collect();
            let budget = NumericBudget {
                max_cost: arity + 2,
                ..NumericBudget::default()
            };
            let learned = Rule::learn_relational(&ds, budget, window, &refs);
            let (admitted, correct, work, bytes, reason) = match learned {
                Ok(rule) => {
                    let work = rule.search_work();
                    let child = parent.merge(&SemanticSeed::new(vec![rule])?)?;
                    let correct = admission
                        .iter()
                        .filter(|d| {
                            let Meaning::Calculate { answer, .. } = d.meaning else {
                                return false;
                            };
                            Session::default()
                                .narrative(&child, &d.sentence)
                                .is_ok_and(|a| NumericTolerance::default().agrees(a.value, answer))
                        })
                        .count();
                    let bytes = child.to_bytes()?;
                    let admitted = correct == admission.len();
                    let name = format!("generation-{generation}-window-{window}.axsem");
                    publish_new(&output.join(name), &bytes)?;
                    if admitted {
                        candidates.push((bytes.len(), work, window, child));
                    }
                    (admitted, correct, Some(work), Some(bytes.len()), None)
                }
                Err(e) => (false, 0, None, None, Some(e.to_string())),
            };
            trials.push(Trial {
                generation,
                window,
                admitted,
                admission_correct: correct,
                admission_total: admission.len(),
                work,
                bytes,
                reason,
            });
            telemetry.emit(
                "validation",
                &[
                    ("method", V::Text("MetaLearning::admission")),
                    (
                        "name",
                        V::Text(&format!("generation-{generation}-window-{window}")),
                    ),
                    ("generation", V::Count(generation as u64)),
                    ("search_work", work.map(V::Count).unwrap_or(V::Null)),
                    (
                        "case",
                        V::Count((generation * 3 + window as usize - 1) as u64),
                    ),
                    ("correct", V::Count(correct as u64)),
                    ("checked", V::Count(admission.len() as u64)),
                    ("admitted", V::Flag(admitted)),
                ],
            )?;
        }
        candidates.sort_by_key(|(bytes, work, window, _)| (*bytes, *work, *window));
        winner = candidates.into_iter().next().map(|(_, _, _, s)| s);
        if winner.is_none() {
            break;
        }
    }
    let cold: Vec<_> = (3..=4)
        .map(|arity| {
            Rule::learn_relational(
                &spaced(demonstrations(arity, "descendente", 0, 12), arity - 1),
                NumericBudget {
                    max_cost: arity + 2,
                    ..NumericBudget::default()
                },
                (arity - 1) as u8,
                &[],
            )
            .is_ok()
        })
        .collect();
    let retained = winner.as_ref().map_or(0, |s| {
        demonstrations(2, "ancestral", 200, 12)
            .iter()
            .filter(|d| match &d.meaning {
                Meaning::Calculate { answer, .. } => Session::default()
                    .narrative(s, &d.sentence)
                    .is_ok_and(|a| NumericTolerance::default().agrees(a.value, *answer)),
                _ => false,
            })
            .count()
    });
    Ok(
        serde_json::json!({"policy_space":"anchor window 1..3 only; cost caps supplied","trials":trials,"cold_succeeded":cold,"retained_ancestral_cases":retained,"winner_rules":winner.as_ref().map(|s|s.rules().len())}),
    )
}
pub fn exercise(output: &Path) -> Result<()> {
    bounded_path(output)?;
    fs::create_dir(output)?;
    let mut telemetry = LabTelemetry::from_environment("integration-v1");
    telemetry.emit("lab_start", &[("method", V::Text("Integration::run"))])?;
    let meta = meta(output, &mut telemetry)?;
    let mut agent =
        ContinuousAgent::new("integration/sensor", 1, None, ContinuousPolicy::default())?;
    let identity_lessons = |label: &str| {
        demonstrations(1, label, 0, 12)
            .into_iter()
            .map(|mut d| {
                if let Meaning::Calculate { arguments, answer } = &mut d.meaning {
                    *answer = arguments[0];
                }
                d
            })
            .collect::<Vec<_>>()
    };
    let mut language = SemanticSeed::new(vec![Rule::learn_relational(
        &identity_lessons("sensor"),
        NumericBudget::default(),
        1,
        &[],
    )?])?;
    language.save_new(&output.join("stream-language-v1.axsem"))?;
    let unknown_before_teaching = language.bind_calculation("medida campo0 1").is_err();
    let mut records = Vec::new();
    for id in 1..=360u64 {
        let x = ((id * 17) % 97) as f64 / 7.0 - 7.0;
        let expected = if (121..=240).contains(&id) {
            -1.75 * x + 4.25
        } else {
            2.0 * x + 1.0
        };
        if id == 181 {
            language = language
                .teach_calculation(&identity_lessons("medida"), NumericBudget::default())?;
            language.save_new(&output.join("stream-language-v2.axsem"))?;
        }
        let label = if id <= 180 { "sensor" } else { "medida" };
        let sentence = if id % 2 == 0 {
            format!("campo0 {x} {label}")
        } else {
            format!("{label} campo0 {x} agora")
        };
        let input = language.bind_calculation(&sentence)?;
        let prediction = agent.observe(id, &input)?;
        let feedback = agent.feedback(id, expected)?;
        records.push(serde_json::json!({"id":id,"sentence":sentence,"input":input,"prediction":prediction.value,"expected":expected,"correct":feedback.correct,"scored":feedback.scored,"status":format!("{:?}",feedback.status)}));
        if let Some(job) = agent.take_learning_job() {
            agent.submit_learning_result(job.run(None));
        }
        if id % 30 == 0 {
            telemetry.emit(
                "continuous_progress",
                &[
                    ("method", V::Text("ContinuousAgent::feedback")),
                    ("observation", V::Count(id)),
                    ("correct", V::Count(agent.statistics().correct)),
                    ("scored", V::Count(agent.statistics().scored)),
                    ("drifts", V::Count(agent.statistics().drifts)),
                ],
            )?;
        }
    }
    let stats = agent.statistics();
    agent.save_checkpoint_new(&output.join("stream.axcont"))?;
    let old_language_retained = language
        .bind_calculation("sensor campo0 -3")
        .is_ok_and(|x| x == vec![-3.0]);
    let meta_passed = meta["winner_rules"].as_u64() == Some(3)
        && meta["retained_ancestral_cases"].as_u64() == Some(12)
        && meta["cold_succeeded"] == serde_json::json!([false, false]);
    let continuous_passed = stats.drifts == 2
        && stats.reuses >= 1
        && agent.phase() == ContinuousPhase::Active
        && unknown_before_teaching
        && old_language_retained
        && records.iter().rev().take(30).all(|r| r["correct"] == true);
    let goals = usize::from(meta_passed) + usize::from(continuous_passed);
    let result = serde_json::json!({"schema":"axon-integration-results/v1","meta":meta,"meta_passed":meta_passed,"continuous_passed":continuous_passed,"language":{"unknown_before_teaching":unknown_before_teaching,"old_binding_retained":old_language_retained,"updates":1},"continuous":{"observations":stats.observations,"scored":stats.scored,"correct":stats.correct,"drifts":stats.drifts,"reuses":stats.reuses,"search_work":stats.search_work,"archive_work":stats.archive_work,"regimes":"host-simulated affine sensor A -> B -> A","phase":format!("{:?}",agent.phase())},"records":records});
    publish_new(
        &output.join("audit-integration-v1.json"),
        &serde_json::to_vec_pretty(&result)?,
    )?;
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("Integration::result")),
            ("correct", V::Count(stats.correct)),
            ("scored", V::Count(stats.scored)),
            ("drifts", V::Count(stats.drifts)),
            ("reuses", V::Count(stats.reuses)),
            (
                "candidates",
                V::Count(result["meta"]["trials"].as_array().map_or(0, |t| t.len()) as u64),
            ),
            (
                "tasks",
                V::Count(result["meta"]["trials"].as_array().map_or(0, |t| t.len()) as u64),
            ),
            (
                "admitted",
                V::Count(
                    result["meta"]["trials"]
                        .as_array()
                        .map_or(0, |t| t.iter().filter(|t| t["admitted"] == true).count())
                        as u64,
                ),
            ),
            ("goal_tasks", V::Count(2)),
            ("goal_admitted", V::Count(goals as u64)),
            ("all_goals_passed", V::Flag(goals == 2)),
        ],
    )?;
    Ok(())
}
