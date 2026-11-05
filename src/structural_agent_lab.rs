//! Finite observation tables are the teacher; goal labels remain outside Agent.
use super::*;
use crate::structural::{
    Error as AgentError,
    agent::{Agent, Budget, Environment, Event, Interface},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Campaign {
    budget: Budget,
    episodes: Vec<Episode>,
}
#[derive(Default)]
pub(super) struct Summary {
    pub correct: u64,
    pub total: u64,
    pub portable: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Episode {
    id: String,
    interfaces: Vec<Teacher>,
    goals: Vec<Goal>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Teacher {
    name: String,
    epoch: u64,
    observations: Vec<Observation>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    input: Vec<i64>,
    output: i64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Goal {
    text: String,
    expected: Option<i64>,
}
struct World<'a> {
    teachers: &'a [Teacher],
    closed: bool,
}
pub(super) fn environment(teachers: &[Teacher], closed: bool) -> impl Environment + '_ {
    World { teachers, closed }
}
impl Environment for World<'_> {
    fn describe(&self, name: &str) -> Option<Interface> {
        self.teachers
            .iter()
            .find(|t| t.name == name)
            .map(|t| Interface {
                arity: t.observations.first().map_or(0, |o| o.input.len()),
                epoch: t.epoch,
                probes: t.observations.iter().map(|o| o.input.clone()).collect(),
            })
    }
    fn query(
        &mut self,
        name: &str,
        epoch: u64,
        input: &[i64],
    ) -> std::result::Result<i64, AgentError> {
        if self.closed {
            return Err(AgentError::Invalid("portable child cannot consult teacher"));
        }
        self.teachers
            .iter()
            .find(|t| t.name == name && t.epoch == epoch)
            .and_then(|t| t.observations.iter().find(|o| o.input == input))
            .map(|o| o.output)
            .ok_or(AgentError::Invalid(
                "unauthorized experiment or changed epoch",
            ))
    }
}
pub(super) fn run(
    campaign: &Campaign,
    seed: StructuralSeed,
    output: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<(StructuralSeed, Summary)> {
    if !(1..=200).contains(&campaign.episodes.len()) {
        return Err("agent episode quota".into());
    }
    let mut ids = BTreeSet::new();
    for e in &campaign.episodes {
        if e.id.is_empty()
            || e.id.len() > 64
            || !ids.insert(&e.id)
            || !(1..=16).contains(&e.interfaces.len())
            || !(1..=64).contains(&e.goals.len())
        {
            return Err("agent episode bounds".into());
        }
        let mut names = BTreeSet::new();
        for t in &e.interfaces {
            if !names.insert(&t.name) || !(7..=128).contains(&t.observations.len()) {
                return Err("agent teacher bounds".into());
            }
        }
        for g in &e.goals {
            crate::structural::agent::utterance(&g.text)?;
        }
    }
    let mut trained = None;
    let mut receipts = Vec::new();
    let mut receipt_bytes = 0usize;
    let mut summaries = Vec::new();
    let mut active_summary = Summary::default();
    for (variant, active, inherit) in [
        ("active", true, true),
        ("passive", false, true),
        ("no-library", true, false),
    ] {
        let mut student = Agent::new(
            seed.clone(),
            Budget {
                active,
                inherit,
                ..campaign.budget
            },
        )?;
        let mut correct = 0;
        let mut portable = 0;
        let mut total = 0;
        for episode in &campaign.episodes {
            let mut world = World {
                teachers: &episode.interfaces,
                closed: false,
            };
            for (i, goal) in episode.goals.iter().enumerate() {
                telemetry.emit(
                    "goal_start",
                    &[
                        ("method", V::Text("Agent::answer")),
                        ("variant", V::Text(variant)),
                        ("stage", V::Text(&episode.id)),
                        ("input", V::Text(&goal.text)),
                    ],
                )?;
                let before = student.statistics.queries;
                let mut events = Vec::<Event>::new();
                let actual = student.answer(&goal.text, &mut world, &mut |event| {
                    let input_json = serde_json::to_string(&event.input)
                        .map_err(|e| AgentError::Storage(e.to_string()))?;
                    let actual_text = event.actual.map(|n| n.to_string());
                    let proposed_text = event.proposed.map(|n| n.to_string());
                    telemetry
                        .emit(
                            "investigation",
                            &[
                                ("method", V::Text("Agent::investigate")),
                                ("variant", V::Text(variant)),
                                ("stage", V::Text(&episode.id)),
                                ("phase", V::Text(&event.phase)),
                                ("operation", V::Text(&event.operation)),
                                ("input_json", V::Text(&input_json)),
                                (
                                    "actual_text",
                                    actual_text.as_deref().map_or(V::Null, V::Text),
                                ),
                                (
                                    "proposed_text",
                                    proposed_text.as_deref().map_or(V::Null, V::Text),
                                ),
                                ("epoch", V::Count(event.epoch)),
                                ("work", V::Count(event.work)),
                                (
                                    "expression",
                                    event.expression.as_deref().map_or(V::Null, V::Text),
                                ),
                                (
                                    "actual",
                                    event.actual.map_or(V::Null, |n| V::Number(n as f64)),
                                ),
                            ],
                        )
                        .map_err(|e| AgentError::Storage(e.to_string()))?;
                    events.push(event.clone());
                    Ok(())
                });
                let ok = actual.as_ref().ok().copied() == goal.expected;
                correct += usize::from(ok);
                total += 1;
                let mut child = Agent::new(
                    StructuralSeed::from_bytes(&student.seed().to_bytes()?)?,
                    campaign.budget,
                )?;
                let child_actual = child.answer(
                    &goal.text,
                    &mut World {
                        teachers: &episode.interfaces,
                        closed: true,
                    },
                    &mut |_| Ok(()),
                );
                let child_ok = child_actual.as_ref().ok().copied() == goal.expected
                    && child.statistics.queries == 0;
                portable += usize::from(child_ok);
                receipts.push(serde_json::json!({"variant":variant,"episode":episode.id,"goal":i,"input":goal.text,"expected":goal.expected,"actual":actual.as_ref().ok(),"error":actual.as_ref().err().map(ToString::to_string),"correct":ok,"queries":student.statistics.queries-before,"events":events,"child_actual":child_actual.as_ref().ok(),"child_error":child_actual.as_ref().err().map(ToString::to_string),"child_correct":child_ok}));
                receipt_bytes +=
                    serde_json::to_vec(receipts.last().ok_or("missing agent receipt")?)?.len();
                if receipt_bytes > 16_000_000 {
                    return Err("aggregate agent receipt quota".into());
                }
                telemetry.emit(
                    "validation",
                    &[
                        ("method", V::Text("Agent::goal")),
                        ("case", V::Count(100000 + receipts.len() as u64)),
                        ("name", V::Text(&episode.id)),
                        ("variant", V::Text(variant)),
                        ("correct", V::Count(correct as u64)),
                        ("total", V::Count(total as u64)),
                        ("admitted", V::Flag(ok)),
                        ("queries", V::Count(student.statistics.queries as u64)),
                    ],
                )?;
            }
        }
        let summary = serde_json::json!({"variant":variant,"correct":correct,"total":total,"portable_correct":portable,"statistics":student.statistics,"model_bytes":student.seed().to_bytes()?.len()});
        telemetry.emit(
            "agent_result",
            &[
                ("method", V::Text("Agent::campaign")),
                ("case", V::Count(200000 + summaries.len() as u64)),
                ("variant", V::Text(variant)),
                ("correct", V::Count(correct as u64)),
                ("total", V::Count(total as u64)),
                ("portable_correct", V::Count(portable as u64)),
                ("queries", V::Count(student.statistics.queries as u64)),
                (
                    "refutations",
                    V::Count(student.statistics.refutations as u64),
                ),
                (
                    "acquisitions",
                    V::Count(student.statistics.acquisitions as u64),
                ),
                (
                    "work",
                    V::Count(
                        student.statistics.search_work
                            + student.statistics.probe_work
                            + student.statistics.language_work,
                    ),
                ),
                ("language_work", V::Count(student.statistics.language_work)),
            ],
        )?;
        summaries.push(summary);
        if variant == "active" {
            active_summary = Summary {
                correct: correct as u64,
                total: total as u64,
                portable: portable as u64,
            };
            trained = Some(student.into_seed());
        }
    }
    save(&output.join("agent-predictions.json"), &receipts)?;
    save(
        &output.join("agent-result.json"),
        &serde_json::json!({"variants":summaries,"limits":"Finite taught language, integer program grammar, finite observations, host-announced epochs; not general comprehension or a proof."}),
    )?;
    Ok((trained.ok_or("missing active agent")?, active_summary))
}
