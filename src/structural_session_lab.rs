//! Labels belong to the host evaluator, never to Session::turn.
use super::*;
use crate::structural::agent::{Environment, Interface};
use crate::structural::{
    agent::Budget,
    session::{MeasuredData, Quantity, Session},
};
use std::cell::{Cell, RefCell};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Campaign {
    episodes: Vec<Episode>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Episode {
    id: String,
    interfaces: Vec<agent_lab::Teacher>,
    turns: Vec<SessionTurn>,
    #[serde(default)]
    restart_at: Vec<usize>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionTurn {
    text: String,
    #[serde(default = "default_source")]
    source: String,
    expected: Expected,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Expected {
    status: String,
    value: Option<Quantity>,
}
#[derive(Default)]
pub(super) struct Summary {
    pub correct: u64,
    pub total: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Batch {
    interfaces: std::collections::BTreeMap<String, Interface>,
    turns: Vec<InputTurn>,
    #[serde(default)]
    files: std::collections::BTreeMap<String, InputFile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputFile {
    csv: String,
    unit: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputTurn {
    id: String,
    text: String,
    #[serde(default = "default_source")]
    source: String,
}
struct ClosedWorld {
    interfaces: std::collections::BTreeMap<String, Interface>,
    attempted: usize,
}
#[cfg(test)]
mod input_tests {
    use super::*;
    #[test]
    fn file_manifests_and_session_turns_do_not_accept_teacher_answers() {
        assert!(
            serde_json::from_value::<InputFile>(
                serde_json::json!({"csv":"x.csv","unit":"BRL","expected":7})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<InputTurn>(
                serde_json::json!({"id":"new","text":"some dados","expected":7})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<Batch>(
                serde_json::json!({"interfaces":{},"turns":[],"expected":7})
            )
            .is_err()
        );
    }
}
impl Environment for ClosedWorld {
    fn describe(&self, name: &str) -> Option<Interface> {
        self.interfaces.get(name).cloned()
    }
    fn query(
        &mut self,
        _: &str,
        _: u64,
        _: &[i64],
    ) -> std::result::Result<i64, crate::structural::Error> {
        self.attempted += 1;
        Err(crate::structural::Error::Invalid(
            "portable session has no teacher",
        ))
    }
}
pub(super) fn batch(
    seed: StructuralSeed,
    birth: &Path,
    input: &Path,
    receipt: &Path,
) -> Result<()> {
    let batch: Batch = serde_json::from_slice(&read_bounded(input, 16_000_000)?)?;
    if !(1..=16).contains(&batch.interfaces.len()) || !(1..=10000).contains(&batch.turns.len()) {
        return Err("session batch quota".into());
    }
    let mut ids = BTreeSet::new();
    for t in &batch.turns {
        if t.id.is_empty() || t.id.len() > 96 || !ids.insert(&t.id) {
            return Err("session batch identity".into());
        }
    }
    if batch.files.len() > 16 {
        return Err("session file quota".into());
    }
    let mut data = std::collections::BTreeMap::new();
    let mut provenance = Vec::new();
    for (name, file) in &batch.files {
        if !crate::structural::agent::valid_name(name) {
            return Err("session data name".into());
        }
        crate::demand::file_name(&file.csv)?;
        let path = input
            .parent()
            .ok_or("session input parent")?
            .join(&file.csv);
        bounded_path(&path)?;
        let raw = read_bounded(&path, 1_000_000)?;
        let value = MeasuredData {
            value: super::file_lab::mapped_csv(&seed, &raw)?,
            unit: file.unit.clone(),
        };
        value.validate()?;
        provenance.push(serde_json::json!({"name":name,"file":file.csv,"unit":file.unit,"bytes":raw.len(),"checksum":crate::live_store::checksum(&raw),"checksum_kind":"AXON noncryptographic checksum; host manifest grants access, not document text"}));
        data.insert(name.clone(), value);
    }
    let before = seed.to_bytes()?;
    let mut session = Session::open(seed, &birth.join("personal.axep"), Budget::default())?;
    let mut world = ClosedWorld {
        interfaces: batch.interfaces,
        attempted: 0,
    };
    let mut rows = Vec::new();
    let mut bytes = 0;
    for t in &batch.turns {
        let before_episodes = session.episodes();
        let actual = session.turn_with_data(
            &t.text,
            &t.source,
            &data,
            &mut world,
            &mut |_| Ok(()),
            &mut |_| Ok(()),
        );
        let row = serde_json::json!({"id":t.id,"text":t.text,"source":t.source,"actual":actual.as_ref().ok(),"error":actual.as_ref().err().map(ToString::to_string),"episodes_before":before_episodes,"episodes_after":session.episodes()});
        bytes += serde_json::to_vec(&row)?.len();
        if bytes > 16_000_000 {
            return Err("session batch receipt quota".into());
        }
        rows.push(row);
    }
    let unchanged = session.seed().to_bytes()? == before;
    save(receipt, &rows)?;
    println!(
        "{}",
        serde_json::json!({"turns":rows.len(),"denied_queries":world.attempted,"seed_unchanged":unchanged,"personal_episodes":session.episodes(),"training_required":false,"files":provenance,"scope":"Authorized local files, learned field mappings and pure procedures; units supplied by host. No general Portuguese or independent public wording acceptance implied."})
    );
    if world.attempted != 0 || !unchanged {
        return Err("portable session attempted new numeric training".into());
    }
    Ok(())
}

pub(super) fn run(
    c: &Campaign,
    seed: StructuralSeed,
    output: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<(StructuralSeed, Summary)> {
    if !(1..=128).contains(&c.episodes.len()) {
        return Err("session episode quota".into());
    }
    let mut ids = BTreeSet::new();
    for e in &c.episodes {
        crate::demand::file_name(&e.id)?;
        if e.id.len() > 64
            || !ids.insert(&e.id)
            || !(1..=16).contains(&e.interfaces.len())
            || !(1..=64).contains(&e.turns.len())
            || e.restart_at.iter().any(|&i| i >= e.turns.len())
        {
            return Err("session episode bounds".into());
        }
    }
    let mut rows = Vec::new();
    let mut bytes = 0;
    let mut summaries = Vec::new();
    let mut active = None;
    for (variant_index, (variant, memory, compose)) in [
        ("context", true, true),
        ("no-memory", false, true),
        ("no-composition", true, false),
    ]
    .into_iter()
    .enumerate()
    {
        let mut working = seed.clone();
        let mut correct = 0u64;
        let mut total = 0u64;
        let mut work = 0u64;
        let mut queries = 0usize;
        for e in &c.episodes {
            let path = output.join(format!("session-{variant}-{}.axep", e.id));
            let mut session = Some(Session::open(working.clone(), &path, Budget::default())?);
            session
                .as_mut()
                .expect("open")
                .set_controls(memory, compose);
            let mut world = agent_lab::environment(&e.interfaces, false);
            for (index, t) in e.turns.iter().enumerate() {
                let restarted = e.restart_at.contains(&index);
                if restarted {
                    let previous = session.take().expect("open");
                    queries += previous.queries();
                    working = previous.into_seed();
                    session = Some(Session::open(working.clone(), &path, Budget::default())?);
                    session
                        .as_mut()
                        .expect("open")
                        .set_controls(memory, compose);
                }
                let s = session.as_mut().expect("open");
                let before_queries = s.queries();
                let before_episodes = s.episodes();
                let measured = Cell::new(0usize);
                let actual = {
                    let sink = RefCell::new(&mut *telemetry);
                    s.turn(
                        &t.text,
                        &t.source,
                        &mut world,
                        &mut |decision| {
                            measured.set(decision.work);
                            let plans = serde_json::to_string(&decision.plans)
                                .map_err(|e| crate::structural::Error::Storage(e.to_string()))?;
                            sink.borrow_mut()
                                .emit(
                                    "session_interpretation",
                                    &[
                                        ("method", V::Text("Session::ground")),
                                        ("case", V::Count(810000 + variant_index as u64)),
                                        ("variant", V::Text(variant)),
                                        ("stage", V::Text(&e.id)),
                                        ("turn", V::Count(index as u64)),
                                        ("input", V::Text(&t.text)),
                                        ("plans_json", V::Text(&plans)),
                                        ("meanings", V::Count(decision.plans.len() as u64)),
                                        ("grounded", V::Count(decision.grounded as u64)),
                                        ("work", V::Count(decision.work as u64)),
                                        ("evidence", V::Counts(&decision.evidence)),
                                        ("actual_json", V::Null),
                                        (
                                            "error",
                                            decision.error.as_deref().map_or(V::Null, V::Text),
                                        ),
                                    ],
                                )
                                .map_err(|e| crate::structural::Error::Storage(e.to_string()))
                        },
                        &mut |event| {
                            sink.borrow_mut()
                                .emit(
                                    "investigation",
                                    &[
                                        ("method", V::Text("Session::numeric")),
                                        ("variant", V::Text(variant)),
                                        ("stage", V::Text(&e.id)),
                                        ("phase", V::Text(&event.phase)),
                                        ("operation", V::Text(&event.operation)),
                                        ("work", V::Count(event.work)),
                                        (
                                            "expression",
                                            event.expression.as_deref().map_or(V::Null, V::Text),
                                        ),
                                    ],
                                )
                                .map_err(|e| crate::structural::Error::Storage(e.to_string()))
                        },
                    )
                };
                let passed = actual
                    .as_ref()
                    .is_ok_and(|r| r.status == t.expected.status && r.value == t.expected.value);
                correct += u64::from(passed);
                total += 1;
                work += measured.get() as u64;
                let query_delta = s.queries() - before_queries;
                let row = serde_json::json!({"variant":variant,"session":e.id,"turn":index,"text":t.text,"source":t.source,"expected_status":t.expected.status,"expected_value":t.expected.value,"actual":actual.as_ref().ok(),"error":actual.as_ref().err().map(ToString::to_string),"correct":passed,"queries":query_delta,"episodes_before":before_episodes,"episodes_after":s.episodes(),"work":measured.get(),"restarted":restarted});
                bytes += serde_json::to_vec(&row)?.len();
                if bytes > 16_000_000 {
                    return Err("session receipt quota".into());
                }
                let actual_json = serde_json::to_string(&actual.as_ref().ok())?;
                telemetry.emit(
                    "session_turn",
                    &[
                        ("method", V::Text("Session::turn")),
                        ("case", V::Count(800000 + variant_index as u64)),
                        ("variant", V::Text(variant)),
                        ("stage", V::Text(&e.id)),
                        ("turn", V::Count(index as u64)),
                        ("input", V::Text(&t.text)),
                        ("actual_json", V::Text(&actual_json)),
                        ("correct", V::Count(correct)),
                        ("total", V::Count(total)),
                        ("queries", V::Count(query_delta as u64)),
                        ("work", V::Count(measured.get() as u64)),
                        ("restarted", V::Flag(restarted)),
                        ("admitted", V::Flag(passed)),
                    ],
                )?;
                rows.push(row);
            }
            let finished = session.take().expect("open");
            queries += finished.queries();
            working = finished.into_seed();
        }
        telemetry.emit(
            "session_result",
            &[
                ("method", V::Text("Session::control")),
                ("case", V::Count(950000 + variant_index as u64)),
                ("variant", V::Text(variant)),
                ("correct", V::Count(correct)),
                ("total", V::Count(total)),
                ("queries", V::Count(queries as u64)),
                ("work", V::Count(work)),
            ],
        )?;
        summaries.push(serde_json::json!({"variant":variant,"correct":correct,"total":total,"queries":queries,"work":work}));
        if variant == "context" {
            active = Some((working, Summary { correct, total }));
        }
    }
    save(&output.join("session-predictions.json"), &rows)?;
    save(
        &output.join("session-result.json"),
        &serde_json::json!({"controls":summaries,"scope":"Same-author contextual development, supplied quantity/unit interfaces and learned local fragments. Not independent-authorship language or M1 completion. Memory ablation disables grounding, not source ownership housekeeping or unit safety."}),
    )?;
    active.ok_or_else(|| "missing contextual variant".into())
}

#[cfg(test)]
mod tests {
    use super::Batch;
    #[test]
    fn portable_input_rejects_teacher_outputs_and_future_labels() {
        for value in [
            serde_json::json!({"interfaces":{"u":{"arity":2,"epoch":1,"probes":[[1,2]],"observations":[{"input":[1,2],"output":3}]}},"turns":[]}),
            serde_json::json!({"interfaces":{},"turns":[{"id":"x","text":"some 1 reais com 2 reais","expected":3}]}),
        ] {
            assert!(serde_json::from_value::<Batch>(value).is_err());
        }
    }
}
