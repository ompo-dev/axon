//! A JSON curriculum and receipts. Test labels never enter StructuralSeed::teach.
use crate::{
    Authority, Effect,
    demand::{
        Catalog, DemandRuntime, Module, ModuleKind, ModuleSpec, Seed,
        controller::CompositionBudget,
        genome::{BirthLimits, Genome, MAX_BYTES, bounded_path},
    },
    structural::{
        Example, Lesson, Prediction, StructuralSeed, Term,
        dialogue::{Dialogue, Reply},
    },
    telemetry::{LabTelemetry, Value as V},
    text_skill::{publish_new, read_bounded},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, error::Error, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[path = "structural_agent_lab.rs"]
mod agent_lab;
#[path = "algorithm_lab.rs"]
mod algorithm_lab;
#[path = "structural_dependency_lab.rs"]
pub mod dependency_lab;
#[path = "structural_encoder_lab.rs"]
pub mod encoder_lab;
#[path = "structural_file_lab.rs"]
mod file_lab;
#[path = "structural_grammar_lab.rs"]
pub mod grammar_lab;
#[path = "structural_joint_lab.rs"]
pub mod joint_lab;
#[path = "structural_math_lab.rs"]
pub mod math_lab;
#[path = "structural_relation_lab.rs"]
pub mod relation_lab;
#[path = "structural_relation_network_lab.rs"]
pub mod relation_network_lab;
#[path = "structural_segment_lab.rs"]
pub mod segment_lab;
#[path = "structural_sequence_lab.rs"]
pub mod sequence_lab;
#[path = "structural_session_lab.rs"]
mod session_lab;
#[path = "structural_synaptic_lab.rs"]
pub mod synaptic_lab;
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Teaching {
    #[serde(flatten)]
    lesson: Lesson,
    admission: Vec<Example>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub family: String,
    pub input: Term,
    pub expected: Option<Term>,
    #[serde(default)]
    pub forbid_heads: Vec<String>,
}
#[derive(serde::Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InterpretationInput {
    id: String,
    text: String,
}
pub fn file_request(number: &Path, birth: &Path, input: &Path, receipt: &Path) -> Result<()> {
    file_lab::run(number, birth, input, receipt)
}

pub fn interpretation_batch(
    number: &Path,
    birth: &Path,
    input: &Path,
    receipt: &Path,
) -> Result<()> {
    let inputs: Vec<InterpretationInput> =
        serde_json::from_slice(&read_bounded(input, 16_000_000)?)?;
    if !(1..=10000).contains(&inputs.len()) {
        return Err("interpretation batch quota".into());
    }
    let mut ids = BTreeSet::new();
    for row in &inputs {
        if row.id.is_empty() || row.id.len() > 96 || !ids.insert(&row.id) {
            return Err("interpretation identity".into());
        }
    }
    let seed = born_seed(number, birth)?;
    let before = seed.to_bytes()?;
    let mut rows = Vec::new();
    let mut bytes = 0;
    let mut total_work = 0usize;
    let mut covered = 0;
    let mut telemetry = LabTelemetry::from_environment("structural-interpretation-v1");
    for (index, row) in inputs.iter().enumerate() {
        let report = seed.measure_composition(&row.text, crate::structural::language::DEFAULT_WORK);
        total_work = total_work
            .checked_add(report.visits)
            .ok_or("interpretation work overflow")?;
        if total_work > 100_000_000 {
            return Err("aggregate interpretation work quota".into());
        }
        let outputs = report
            .result
            .as_ref()
            .ok()
            .map(|p| p.iter().map(|p| p.output.clone()).collect::<Vec<_>>())
            .unwrap_or_default();
        if !outputs.is_empty() {
            covered += 1;
        }
        let actual_json = serde_json::to_string(&outputs)?;
        telemetry.emit(
            "interpretation",
            &[
                ("method", V::Text("StructuralSeed::interpret")),
                ("case", V::Count(index as u64)),
                ("input", V::Text(&row.text)),
                ("actual_json", V::Text(&actual_json)),
                ("work", V::Count(report.visits as u64)),
            ],
        )?;
        let result = serde_json::json!({"id":row.id,"text":row.text,"meanings":outputs,"work":report.visits,"error":report.result.err().map(|e|e.to_string())});
        bytes += serde_json::to_vec(&result)?.len();
        if bytes > 16_000_000 {
            return Err("interpretation receipt quota".into());
        }
        rows.push(result);
    }
    let unchanged = seed.to_bytes()? == before;
    save(receipt, &rows)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"covered":covered,"work":total_work,"seed_unchanged":unchanged,"teacher_queries":0})
    );
    if !unchanged {
        return Err("interpretation mutated seed".into());
    }
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Turn {
    text: String,
    #[serde(default = "default_source")]
    source: String,
    expected: Reply,
}
fn default_source() -> String {
    "user".into()
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Conversation {
    id: String,
    turns: Vec<Turn>,
    #[serde(default)]
    restart_at: Vec<usize>,
    #[serde(default)]
    updates: Vec<Update>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Update {
    before_turn: usize,
    teaching: Teaching,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    version: u32,
    id: String,
    lessons: Vec<Teaching>,
    cases: Vec<Case>,
    dialogues: Vec<Conversation>,
    #[serde(default)]
    agent: Option<agent_lab::Campaign>,
    #[serde(default)]
    algorithms: Option<String>,
    #[serde(default)]
    sessions: Option<session_lab::Campaign>,
    #[serde(default)]
    consolidate_language: bool,
    #[serde(default)]
    consolidate_alternatives: bool,
    #[serde(default)]
    file_requests: Vec<FileCase>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileCase {
    id: String,
    request: String,
    status: String,
    expected: Option<crate::structural::algorithm::Value>,
}
#[derive(Serialize)]
struct Receipt<'a> {
    id: &'a str,
    family: &'a str,
    input: &'a Term,
    expected: &'a Option<Term>,
    actual: Option<Term>,
    correct: bool,
    rules: Vec<String>,
    visits: usize,
    error: Option<String>,
}
fn blocked(t: &Term, names: &[String]) -> bool {
    if let Term::Tree(v) = t {
        names
            .iter()
            .any(|s| v.first() == Some(&Term::Symbol(s.clone())))
            || v.iter().skip(1).any(|t| blocked(t, names))
    } else {
        false
    }
}
fn evaluate<'a>(seed: &StructuralSeed, c: &'a Case) -> Receipt<'a> {
    let result = seed.transform(&c.input, 2048);
    let (actual, rules, visits, error) = match result {
        Ok(Prediction {
            output,
            rules,
            visits,
        }) if !rules.is_empty() && !blocked(&output, &c.forbid_heads) => {
            (Some(output), rules, visits, None)
        }
        Ok(p) => (
            None,
            p.rules,
            p.visits,
            Some("unresolved or unlearned request".into()),
        ),
        Err(e) => (None, vec![], 0, Some(e.to_string())),
    };
    Receipt {
        id: &c.id,
        family: &c.family,
        input: &c.input,
        expected: &c.expected,
        correct: actual == c.expected,
        actual,
        rules,
        visits,
        error,
    }
}
fn save(path: &Path, value: &impl Serialize) -> Result<()> {
    publish_new(path, &serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn packed(seed: &StructuralSeed, output: &Path) -> Result<Genome> {
    packed_modules(&[("structural", seed)], output)
}
fn packed_modules(modules: &[(&str, &StructuralSeed)], output: &Path) -> Result<Genome> {
    let encoded = modules
        .iter()
        .map(|(_, seed)| seed.to_bytes())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let pages = modules
        .iter()
        .zip(&encoded)
        .map(|((id, _), bytes)| (*id, ModuleKind::Structural, bytes.as_slice()))
        .collect::<Vec<_>>();
    packed_payloads(&pages, output)
}
fn packed_payloads(pages: &[(&str, ModuleKind, &[u8])], output: &Path) -> Result<Genome> {
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut catalog = Catalog::new(output);
    for (id, kind, bytes) in pages {
        let extension = if *kind == ModuleKind::Structural {
            "axstruct"
        } else if matches!(
            kind,
            ModuleKind::RelationalPointer | ModuleKind::ContextPointer
        ) {
            "axrp"
        } else {
            "axseq"
        };
        let file = format!("{id}.{extension}");
        crate::demand::file_name(&file)?;
        publish_new(&output.join(&file), bytes)?;
        catalog.register(
            ModuleSpec {
                id,
                kind: *kind,
                file: &file,
                input: if matches!(
                    kind,
                    ModuleKind::RelationalPointer | ModuleKind::ContextPointer
                ) {
                    "position-candidates"
                } else if *kind == ModuleKind::RelationNetwork {
                    "paired-text"
                } else if matches!(
                    kind,
                    ModuleKind::Sequence
                        | ModuleKind::CompactSequence
                        | ModuleKind::EnhancedSequence
                        | ModuleKind::FactorizedSequence
                        | ModuleKind::JointSequence
                        | ModuleKind::LexicalEncoder
                        | ModuleKind::SegmentSequence
                        | ModuleKind::Dependency
                ) {
                    "text"
                } else {
                    "term"
                },
                output: if matches!(
                    kind,
                    ModuleKind::RelationalPointer | ModuleKind::ContextPointer
                ) {
                    "selection"
                } else if *kind == ModuleKind::RelationNetwork {
                    "empirical-relation"
                } else if *kind == ModuleKind::Dependency {
                    "dependency-tree"
                } else if matches!(
                    kind,
                    ModuleKind::Sequence
                        | ModuleKind::CompactSequence
                        | ModuleKind::EnhancedSequence
                        | ModuleKind::FactorizedSequence
                        | ModuleKind::JointSequence
                        | ModuleKind::LexicalEncoder
                        | ModuleKind::SegmentSequence
                ) {
                    "interpretation"
                } else {
                    "term"
                },
                cost: 1,
            },
            &auth,
        )?;
    }
    let mut runtime = DemandRuntime::new(
        catalog,
        Seed {
            active_modules: 1,
            active_bytes: 262144,
            ..Seed::default()
        },
    )?;
    let genome = Genome::capture(
        "structural/curriculum-v1",
        &mut runtime,
        CompositionBudget::default(),
        &[],
        MAX_BYTES,
        &auth,
    )?;
    genome.save_number_new(&output.join("structural.axnum"), &auth)?;
    Ok(genome)
}
pub fn run(protocol: &Path, output: &Path) -> Result<()> {
    bounded_path(protocol)?;
    bounded_path(output)?;
    let bytes = read_bounded(protocol, 16_000_000)?;
    let config: Protocol = serde_json::from_slice(&bytes)?;
    let algorithms = if let Some(file) = &config.algorithms {
        crate::demand::file_name(file)?;
        let path = protocol.parent().unwrap_or(Path::new(".")).join(file);
        bounded_path(&path)?;
        Some(serde_json::from_slice::<algorithm_lab::Campaign>(
            &read_bounded(&path, 16_000_000)?,
        )?)
    } else {
        None
    };
    if config.version != 1
        || config.id.is_empty()
        || config.id.len() > 64
        || config.lessons.is_empty()
        || config.lessons.len() > 128
        || config.cases.is_empty()
        || config.cases.len() > 10000
        || config.dialogues.len() > 500
        || config.file_requests.len() > 16
    {
        return Err("structural protocol bounds".into());
    }
    let mut seen = BTreeSet::new();
    let mut train = BTreeSet::new();
    for t in &config.lessons {
        for e in &t.lesson.examples {
            train.insert(serde_json::to_string(&e.input)?);
        }
    }
    for c in &config.cases {
        c.input.validate()?;
        if let Some(t) = &c.expected {
            t.validate()?;
        }
        let encoded = serde_json::to_string(&c.input)?;
        if c.id.is_empty()
            || c.id.len() > 96
            || c.family.is_empty()
            || c.family.len() > 64
            || !seen.insert(c.id.clone())
            || train.contains(&encoded)
            || c.forbid_heads.len() > 16
        {
            return Err("duplicate/leaked or invalid structural case".into());
        }
    }
    for d in &config.dialogues {
        if d.id.is_empty()
            || d.id.len() > 96
            || !seen.insert(d.id.clone())
            || d.turns.is_empty()
            || d.turns.len() > 32
            || d.updates.len() > 8
            || d.updates
                .iter()
                .any(|u| u.before_turn == 0 || u.before_turn >= d.turns.len())
            || d.restart_at.iter().any(|i| *i == 0 || *i >= d.turns.len())
        {
            return Err("dialogue bounds".into());
        }
    }
    fs::create_dir(output)?;
    publish_new(&output.join("curriculum.json"), &bytes)?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment("structural-v1");
    telemetry.emit(
        "lab_start",
        &[
            ("method", V::Text("StructuralSeed::teach")),
            ("protocol", V::Text(&config.id)),
        ],
    )?;
    let mut seed = StructuralSeed::default();
    let mut acquired = 0;
    let mut failures = Vec::new();
    for (i, t) in config.lessons.into_iter().enumerate() {
        let id = t.lesson.id.clone();
        let result = seed.teach(t.lesson, &t.admission);
        let admitted = result.is_ok();
        let programs = if admitted {
            seed.rules().last().expect("new rule").derived_programs()
        } else {
            Vec::new()
        };
        let program_refs = programs.iter().map(String::as_str).collect::<Vec<_>>();
        if admitted {
            acquired += 1;
        } else {
            failures.push(serde_json::json!({"id":id,"error":result.unwrap_err().to_string()}));
        }
        telemetry.emit(
            "validation",
            &[
                ("method", V::Text("StructuralSeed::admission")),
                ("case", V::Count(i as u64)),
                ("stage", V::Text(&id)),
                ("admitted", V::Flag(admitted)),
                ("acquired", V::Count(acquired)),
                ("programs", V::Texts(&program_refs)),
            ],
        )?;
    }
    save(
        &output.join("teaching-result.json"),
        &serde_json::json!({"acquired":acquired,"failures":failures}),
    )?;
    if config.consolidate_language {
        let receipts = seed.consolidate_language()?;
        for (index, r) in receipts.iter().enumerate() {
            telemetry.emit(
                "consolidation",
                &[
                    ("method", V::Text("StructuralSeed::consolidate")),
                    ("case", V::Count(960000 + index as u64)),
                    ("stage", V::Text(&r.kept)),
                    ("detail", V::Text(&r.removed)),
                    ("before_bytes", V::Count(r.before_bytes as u64)),
                    ("after_bytes", V::Count(r.after_bytes as u64)),
                ],
            )?;
        }
        save(&output.join("language-consolidation.json"), &receipts)?;
    }
    if config.consolidate_alternatives {
        let receipts = seed.consolidate_alternatives()?;
        save(&output.join("alternative-consolidation.json"), &receipts)?;
    }
    let mut agent_summary = agent_lab::Summary::default();
    if let Some(campaign) = &config.agent {
        (seed, agent_summary) = agent_lab::run(campaign, seed, output, &mut telemetry)?;
    }
    if let Some(campaign) = &algorithms {
        seed = algorithm_lab::teach(campaign, seed, output, &mut telemetry)?;
    }
    let mut session_summary = session_lab::Summary::default();
    if let Some(campaign) = &config.sessions {
        (seed, session_summary) = session_lab::run(campaign, seed, output, &mut telemetry)?;
    }
    let genome = packed(&seed, output)?;
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut birth = genome.birth(&output.join("birth"), BirthLimits::default(), &auth)?;
    let initial_loads = birth.controller.runtime().stats().loads;
    let mut reborn = None;
    birth.controller.runtime_mut().with_module(
        "structural",
        crate::structural::CONTEXT,
        &auth,
        |m| {
            let Module::Structural(s) = m else {
                return Err(crate::demand::DemandError::Invalid(
                    "structural module kind",
                ));
            };
            reborn = Some((**s).clone());
            Ok(())
        },
    )?;
    let seed = reborn.ok_or("missing structural seed")?;
    let algorithm_summary = if let Some(campaign) = &algorithms {
        algorithm_lab::evaluate(campaign, &seed, output, &mut telemetry)?
    } else {
        algorithm_lab::Summary::default()
    };
    let mut file_correct = 0;
    let mut file_rows = Vec::new();
    for (index, case) in config.file_requests.iter().enumerate() {
        crate::demand::file_name(&case.id)?;
        crate::demand::file_name(&case.request)?;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("evals/fixtures")
            .join(&case.request);
        let actual = file_lab::evaluate(&seed, &path, &mut telemetry, 970000 + index as u64 * 3)?;
        let correct = actual["status"] == case.status
            && actual["actual"] == serde_json::to_value(&case.expected)?;
        file_correct += usize::from(correct);
        let row = serde_json::json!({"id":case.id,"request":case.request,"expected_status":case.status,"expected":case.expected,"actual":actual,"correct":correct});
        save(&output.join(format!("file-{}.json", case.id)), &row)?;
        telemetry.emit(
            "file_validation",
            &[
                ("method", V::Text("File::validation")),
                ("case", V::Count(970002 + index as u64 * 3)),
                ("correct", V::Count(file_correct as u64)),
                ("total", V::Count((index + 1) as u64)),
            ],
        )?;
        file_rows.push(row);
    }
    if !file_rows.is_empty() {
        save(&output.join("file-predictions.json"), &file_rows)?;
    }
    let mut rows = Vec::new();
    let mut receipt_bytes = 0;
    let mut running_correct = 0;
    for (i, c) in config.cases.iter().enumerate() {
        let r = evaluate(&seed, c);
        receipt_bytes += serde_json::to_vec(&r)?.len();
        if receipt_bytes > 8_000_000 {
            return Err("aggregate structural receipt quota".into());
        }
        if r.correct {
            running_correct += 1;
        }
        rows.push(r);
        if (i + 1) % 128 == 0 {
            telemetry.emit(
                "validation",
                &[
                    ("method", V::Text("StructuralSeed::progress")),
                    ("correct", V::Count(running_correct)),
                    ("checked", V::Count((i + 1) as u64)),
                ],
            )?;
        }
    }
    let correct = rows.iter().filter(|r| r.correct).count();
    let mut family = BTreeSet::new();
    for r in &rows {
        family.insert(r.family.to_owned());
    }
    let mut family_receipts = Vec::new();
    for (i, f) in family.into_iter().enumerate() {
        let total = rows.iter().filter(|r| r.family == f).count();
        let ok = rows.iter().filter(|r| r.family == f && r.correct).count();
        family_receipts.push(serde_json::json!({"family":f,"total":total,"correct":ok}));
        telemetry.emit(
            "validation",
            &[
                ("method", V::Text("StructuralSeed::transfer")),
                ("case", V::Count(128 + i as u64)),
                ("stage", V::Text(&f)),
                ("correct", V::Count(ok as u64)),
                ("total", V::Count(total as u64)),
                ("admitted", V::Flag(ok == total)),
            ],
        )?;
    }
    save(&output.join("structural-predictions.json"), &rows)?;
    let mut turns = 0;
    let mut dialogue_correct = 0;
    let mut transcript = Vec::new();
    for (i, d) in config.dialogues.iter().enumerate() {
        let journal = output.join(format!("dialogue-{i}.axep"));
        let mut session = Some(Dialogue::open(&journal)?);
        let mut personal_seed = seed.clone();
        for (j, t) in d.turns.iter().enumerate() {
            for update in d.updates.iter().filter(|u| u.before_turn == j) {
                personal_seed.teach(update.teaching.lesson.clone(), &update.teaching.admission)?;
                personal_seed.save_new(&output.join(format!("dialogue-{i}-turn-{j}.axstruct")))?;
                telemetry.emit(
                    "teaching",
                    &[
                        ("method", V::Text("StructuralSeed::teach")),
                        ("dialogue", V::Count(i as u64)),
                        ("turn", V::Count(j as u64)),
                        ("rules", V::Count(personal_seed.rules().len() as u64)),
                    ],
                )?;
            }
            if d.restart_at.contains(&j) {
                session.take();
                session = Some(Dialogue::open(&journal)?);
                personal_seed = StructuralSeed::from_bytes(&personal_seed.to_bytes()?)?;
            }
            let before = session.as_ref().expect("open").episodes();
            let result = session
                .as_mut()
                .expect("open")
                .turn(&personal_seed, &t.text, &t.source);
            let (actual, error) = match result {
                Ok(r) => (Some(r), None),
                Err(e) => (None, Some(e.to_string())),
            };
            let ok = actual.as_ref() == Some(&t.expected);
            if ok {
                dialogue_correct += 1;
            }
            turns += 1;
            telemetry.emit(
                "dialogue_turn",
                &[
                    ("method", V::Text("Dialogue::turn")),
                    ("dialogue", V::Count(i as u64)),
                    ("turn", V::Count(j as u64)),
                    ("input", V::Text(&t.text)),
                    (
                        "outcome",
                        V::Text(actual.as_ref().map_or("error", |r| r.status.as_str())),
                    ),
                    (
                        "response",
                        actual
                            .as_ref()
                            .and_then(|r| r.text.as_deref())
                            .map_or(V::Null, V::Text),
                    ),
                    ("correct", V::Flag(ok)),
                    ("restarted", V::Flag(d.restart_at.contains(&j))),
                ],
            )?;
            transcript.push(serde_json::json!({"dialogue":d.id,"turn":j,"input":t.text,"source":t.source,"expected":t.expected,"actual":actual,"correct":ok,"error":error,"episodes_before":before,"episodes_after":session.as_ref().expect("open").episodes(),"restarted":d.restart_at.contains(&j)}));
        }
        telemetry.emit(
            "validation",
            &[
                ("method", V::Text("Dialogue::turn")),
                ("case", V::Count(256 + i as u64)),
                ("stage", V::Text(&d.id)),
                ("correct", V::Count(dialogue_correct)),
                ("total", V::Count(turns)),
            ],
        )?;
    }
    save(&output.join("dialogue-predictions.json"), &transcript)?;
    let programs = seed
        .rules()
        .iter()
        .filter_map(|r| {
            let p = r.derived_programs();
            if p.is_empty() {
                None
            } else {
                Some(serde_json::json!({"rule":r.id(),"programs":p}))
            }
        })
        .collect::<Vec<_>>();
    let derived_count = programs
        .iter()
        .map(|p| p["programs"].as_array().expect("program list").len())
        .sum::<usize>();
    save(
        &output.join("discovery-result.json"),
        &serde_json::json!({"derived_programs":derived_count,"rules":programs,"scope":"Bounded program synthesis from observed values, not a general proof or autonomous scientific discovery"}),
    )?;
    let passed = correct == rows.len()
        && dialogue_correct == turns
        && failures.is_empty()
        && agent_summary.correct == agent_summary.total
        && agent_summary.portable == agent_summary.total;
    let passed = passed
        && algorithm_summary.correct == algorithm_summary.total
        && session_summary.correct == session_summary.total
        && file_correct == config.file_requests.len();
    let result = serde_json::json!({"schema":"axon-structural-result/v1","protocol":config.id,"admitted":passed,"rules":acquired,"failed_lessons":failures,"families":family_receipts,"correct":correct,"total":rows.len(),"dialogue_correct":dialogue_correct,"dialogue_turns":turns,"initial_loads":initial_loads,"binary_bytes":genome.encoded_bytes(),"decimal_bytes":genome.number().len(),"elapsed_ns":started.elapsed().as_nanos().min(u64::MAX as u128) as u64,"priors":["typed JSON trees","anti-unification and repeated-variable matching","supervised input/output examples","explicit domain guards","functional episodic facts and bounded path lookup","whitespace tokenization and host dialogue actions"],"scope":"bounded taught grammar and symbolic transductions; not general conversation or open mathematical discovery"});
    let mut result = result;
    result["rules"] = serde_json::json!(seed.rules().len());
    result["derived_programs"] = serde_json::json!(derived_count);
    result["agent_correct"] = serde_json::json!(agent_summary.correct);
    result["agent_total"] = serde_json::json!(agent_summary.total);
    result["agent_portable"] = serde_json::json!(agent_summary.portable);
    result["algorithm_correct"] = serde_json::json!(algorithm_summary.correct);
    result["algorithm_total"] = serde_json::json!(algorithm_summary.total);
    result["session_correct"] = serde_json::json!(session_summary.correct);
    result["session_total"] = serde_json::json!(session_summary.total);
    result["file_correct"] = serde_json::json!(file_correct);
    result["file_total"] = serde_json::json!(config.file_requests.len());
    save(&output.join("structural-result.json"), &result)?;
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("Structural::result")),
            ("admitted", V::Flag(passed)),
            ("correct", V::Count(correct as u64)),
            ("total", V::Count(rows.len() as u64)),
            ("dialogue_correct", V::Count(dialogue_correct)),
            ("dialogue_turns", V::Count(turns)),
            ("acquired", V::Count(acquired)),
            ("total_rules", V::Count(seed.rules().len() as u64)),
            ("binary_bytes", V::Count(genome.encoded_bytes() as u64)),
            ("derived_programs", V::Count(derived_count as u64)),
            ("agent_correct", V::Count(agent_summary.correct)),
            ("agent_total", V::Count(agent_summary.total)),
            ("agent_portable", V::Count(agent_summary.portable)),
            ("algorithm_correct", V::Count(algorithm_summary.correct)),
            ("algorithm_total", V::Count(algorithm_summary.total)),
            ("session_correct", V::Count(session_summary.correct)),
            ("session_total", V::Count(session_summary.total)),
            ("file_correct", V::Count(file_correct as u64)),
            ("file_total", V::Count(config.file_requests.len() as u64)),
        ],
    )?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
fn born_seed(number: &Path, birth_dir: &Path) -> Result<StructuralSeed> {
    born_module(number, birth_dir, "structural")
}
fn born_module(number: &Path, birth_dir: &Path, id: &str) -> Result<StructuralSeed> {
    born_page(number, birth_dir, id, crate::structural::CONTEXT, |m| {
        if let Module::Structural(s) = m {
            Some(s.as_ref().clone())
        } else {
            None
        }
    })
}
fn born_page<T>(
    number: &Path,
    birth_dir: &Path,
    id: &str,
    context: &str,
    decode: impl Fn(&Module) -> Option<T>,
) -> Result<T> {
    bounded_path(number)?;
    bounded_path(birth_dir)?;
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let genome = Genome::open_number(number, &auth)?;
    let mut birth = genome.birth(birth_dir, BirthLimits::default(), &auth)?;
    let mut seed = None;
    birth
        .controller
        .runtime_mut()
        .with_module(id, context, &auth, |m| {
            seed = decode(m);
            if seed.is_none() {
                return Err(crate::demand::DemandError::Invalid("requested module type"));
            }
            Ok(())
        })?;
    seed.ok_or_else(|| "missing born module".into())
}
pub fn algorithm_batch(
    number: &Path,
    birth_dir: &Path,
    cases: &Path,
    receipt: &Path,
) -> Result<()> {
    bounded_path(cases)?;
    bounded_path(receipt)?;
    let seed = born_seed(number, birth_dir)?;
    algorithm_lab::batch(seed, cases, receipt)
}
pub fn algorithm_external(
    number: &Path,
    birth_dir: &Path,
    cases: &Path,
    receipt: &Path,
) -> Result<()> {
    bounded_path(cases)?;
    bounded_path(receipt)?;
    algorithm_lab::external(born_seed(number, birth_dir)?, cases, receipt)
}
pub fn session_batch(number: &Path, birth_dir: &Path, cases: &Path, receipt: &Path) -> Result<()> {
    bounded_path(cases)?;
    bounded_path(receipt)?;
    session_lab::batch(born_seed(number, birth_dir)?, birth_dir, cases, receipt)
}
pub fn batch(number: &Path, birth_dir: &Path, cases: &Path, receipt: &Path) -> Result<()> {
    bounded_path(cases)?;
    bounded_path(receipt)?;
    let seed = born_seed(number, birth_dir)?;
    let cases: Vec<Case> = serde_json::from_slice(&read_bounded(cases, 16_000_000)?)?;
    if cases.is_empty() || cases.len() > 10000 {
        return Err("batch case bounds".into());
    }
    let rows = cases.iter().map(|c| evaluate(&seed, c)).collect::<Vec<_>>();
    save(receipt, &rows)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"correct":rows.iter().filter(|r|r.correct).count(),"training_required":false})
    );
    Ok(())
}
pub fn talk(seed: &Path, journal: &Path, text: &str) -> Result<()> {
    bounded_path(seed)?;
    bounded_path(journal)?;
    let seed = StructuralSeed::open(seed)?;
    let mut dialogue = Dialogue::open(journal)?;
    println!(
        "{}",
        serde_json::to_string(&dialogue.turn(&seed, text, "user")?)?
    );
    Ok(())
}
