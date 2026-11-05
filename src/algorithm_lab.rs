//! Configured I/O lessons, policy evolution and prospective collection requests.
use super::*;
use crate::structural::Rule;
use crate::{
    nursery::{Assessment, Nursery, SearchConfig},
    structural::{
        agent::{Agent, Budget},
        algorithm::{self, Policy, SearchBudget, Value},
    },
};
use std::collections::BTreeMap;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Campaign {
    version: u32,
    lessons: Vec<Task>,
    requests: Vec<Request>,
    #[serde(default)]
    changes: Vec<Change>,
    #[serde(default)]
    policy_trials: usize,
    #[serde(default)]
    file_demo: Option<FileDemo>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    family: String,
    examples: Vec<algorithm::Example>,
    admission: Vec<algorithm::Example>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inquiry: Option<InquiryTask>,
    #[serde(default, skip_serializing_if = "algorithm::Representation::is_default")]
    representation: algorithm::Representation,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InquiryTask {
    pool: Vec<algorithm::Example>,
    max_queries: usize,
    max_hypotheses: usize,
}
impl InquiryTask {
    fn plan(&self, passive: bool) -> algorithm::InquiryPlan {
        algorithm::InquiryPlan {
            inputs: self.pool.iter().map(|e| e.input.clone()).collect(),
            max_queries: self.max_queries,
            max_hypotheses: self.max_hypotheses,
            strategy: if passive {
                algorithm::InquiryStrategy::Passive
            } else {
                algorithm::InquiryStrategy::Disagreement
            },
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Change {
    operation: String,
    input: Value,
    actual: Value,
    teaching: Task,
    requests: Vec<Request>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    id: String,
    family: String,
    text: String,
    bindings: BTreeMap<String, Value>,
    expected: Option<Value>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FileDemo {
    csv: String,
    text: String,
    parameter: i64,
    expected: Value,
}
#[derive(Default)]
pub(super) struct Summary {
    pub correct: u64,
    pub total: u64,
}
pub(super) fn batch(seed: StructuralSeed, cases: &Path, receipt: &Path) -> Result<()> {
    let requests: Vec<Request> = serde_json::from_slice(&read_bounded(cases, 16_000_000)?)?;
    if requests.is_empty() || requests.len() > 5000 {
        return Err("algorithm batch bounds".into());
    }
    let agent = Agent::new(seed, Budget::default())?;
    let mut ids = BTreeSet::new();
    let mut rows = Vec::new();
    for r in requests {
        if r.id.is_empty() || r.id.len() > 96 || !ids.insert(r.id.clone()) {
            return Err("algorithm batch identity".into());
        }
        if let Some(v) = &r.expected {
            v.validate()?;
        }
        let actual = agent.answer_value(&r.text, &r.bindings);
        rows.push(serde_json::json!({"id":r.id,"family":r.family,"text":r.text,"bindings":r.bindings,"expected":r.expected,"actual":actual.as_ref().ok(),"error":actual.as_ref().err().map(ToString::to_string),"correct":actual.as_ref().ok()==r.expected.as_ref()}));
    }
    save(receipt, &rows)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"correct":rows.iter().filter(|r|r["correct"]==true).count(),"training_required":false})
    );
    Ok(())
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalTask {
    id: String,
    train: Vec<algorithm::Example>,
    inputs: Vec<Value>,
    #[serde(default)]
    inquiry: Option<InquiryTask>,
    #[serde(default)]
    passive: bool,
    #[serde(default = "inherit_by_default")]
    inherit: bool,
    #[serde(default)]
    representation: algorithm::Representation,
}
fn inherit_by_default() -> bool {
    true
}
pub(super) fn external(seed: StructuralSeed, cases: &Path, receipt: &Path) -> Result<()> {
    let tasks: Vec<ExternalTask> = serde_json::from_slice(&read_bounded(cases, 16_000_000)?)?;
    if tasks.is_empty() || tasks.len() > 500 {
        return Err("external task bounds".into());
    }
    let inherited_libraries = seed.algorithm_library_windows();
    let no_libraries = vec![vec![]];
    let mut ids = BTreeSet::new();
    let mut rows = Vec::new();
    for task in tasks {
        let libraries = if task.inherit {
            &inherited_libraries
        } else {
            &no_libraries
        };
        let library = &libraries[0];
        if task.id.is_empty()
            || task.id.len() > 96
            || !ids.insert(task.id.clone())
            || task.inputs.is_empty()
            || task.inputs.len() > 16
        {
            return Err("external identity/input bounds".into());
        }
        for input in &task.inputs {
            input.validate()?;
        }
        if task.representation == algorithm::Representation::Polynomial && task.inquiry.is_some() {
            return Err("polynomial proposal does not support pool inquiry".into());
        }
        if let Some(inquiry) = &task.inquiry {
            inquiry.plan(task.passive).validate(&task.train)?;
            for e in &inquiry.pool {
                e.output.validate()?;
                if task.inputs.contains(&e.input) {
                    return Err("external query overlaps future inputs".into());
                }
            }
        } else if task.passive {
            return Err("external passive mode needs inquiry".into());
        }
        // Only authorized teaching answers are available. Future outputs never enter this process.
        let mut budget = SearchBudget {
            policy: seed.algorithm_policy(),
            ..SearchBudget::default()
        };
        let mut inquiry_receipt = None;
        let mut questions = Vec::new();
        let mut work = 0;
        let mut candidates = 0;
        let mut library_windows = 0;
        let result = (|| -> std::result::Result<algorithm::Program, crate::structural::Error> {
            let mut teaching = task.train.clone();
            if let Some(inquiry) = &task.inquiry {
                let receipt = algorithm::inquire(
                    &teaching,
                    library,
                    budget,
                    &inquiry.plan(task.passive),
                    &mut |input| {
                        inquiry
                            .pool
                            .iter()
                            .find(|e| &e.input == input)
                            .map(|e| e.output.clone())
                            .ok_or(crate::structural::Error::Invalid(
                                "unauthorized external inquiry",
                            ))
                    },
                    &mut |event| {
                        work = event.work;
                        candidates = event.candidates;
                        questions.push(event.clone());
                        Ok(())
                    },
                )?;
                work = receipt.work;
                candidates = receipt.candidates;
                budget.work = budget.work.saturating_sub(work);
                budget.candidates = budget.candidates.saturating_sub(candidates as usize);
                teaching = receipt.examples.clone();
                inquiry_receipt = Some(receipt);
            }
            let search =
                algorithm::learn_represented(&teaching, libraries, budget, task.representation);
            library_windows += search.windows;
            match search.result {
                Ok(r) => {
                    work += r.work;
                    candidates += r.candidates;
                    Ok(r.program)
                }
                Err(f) => {
                    work += f.work;
                    candidates += f.candidates;
                    Err(f.error)
                }
            }
        })();
        let (program, error) = match result {
            Ok(program) => (Some(program), None),
            Err(error) => (None, Some(error.to_string())),
        };
        let predictions: Vec<_> = task
            .inputs
            .iter()
            .map(|input| {
                program
                    .as_ref()
                    .and_then(|p| p.evaluate(input, algorithm::DEFAULT_FUEL).ok())
                    .map(|e| e.value)
            })
            .collect();
        rows.push(serde_json::json!({"id":task.id,"predictions":predictions,"work":work,"candidates":candidates,"error":error,"program":program,"inquiry":inquiry_receipt,"questions":questions,"inherit":task.inherit,"library_programs":library.len(),"library_windows":library_windows}));
    }
    save(receipt, &rows)?;
    println!(
        "{}",
        serde_json::json!({"tasks":rows.len(),"test_labels_available":false,"promoted":false})
    );
    Ok(())
}
fn map_error(e: impl std::fmt::Display) -> crate::structural::Error {
    crate::structural::Error::Storage(e.to_string())
}
fn validate(c: &Campaign) -> Result<()> {
    let lesson_limit = match c.version {
        1 => 32,
        2 => 64,
        _ => return Err("algorithm campaign version".into()),
    };
    if !(1..=lesson_limit).contains(&c.lessons.len())
        || c.requests.is_empty()
        || c.requests.len() > 5000
        || c.changes.len() > 16
        || c.policy_trials > 128
    {
        return Err("algorithm campaign bounds".into());
    }
    let mut ids = BTreeSet::new();
    for r in c
        .requests
        .iter()
        .chain(c.changes.iter().flat_map(|c| &c.requests))
    {
        if r.id.is_empty()
            || r.id.len() > 96
            || r.family.is_empty()
            || r.family.len() > 64
            || !ids.insert(&r.id)
            || r.bindings.len() > 16
            || r.text.len() > 4096
        {
            return Err("algorithm request identity/bounds".into());
        }
        for v in r.bindings.values() {
            v.validate()?;
        }
        if let Some(v) = &r.expected {
            v.validate()?;
        }
    }
    let mut lessons = BTreeSet::new();
    for t in &c.lessons {
        if !lessons.insert(&t.id) {
            return Err("duplicate algorithm lesson".into());
        }
    }
    for t in c
        .lessons
        .iter()
        .chain(c.changes.iter().map(|c| &c.teaching))
    {
        if t.representation == algorithm::Representation::Polynomial && t.inquiry.is_some() {
            return Err("polynomial proposal does not support pool inquiry".into());
        }
        if t.id.is_empty()
            || t.id.len() > 24
            || !t
                .id
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            || t.family.is_empty()
            || t.family.len() > 64
            || !(3..=24).contains(&t.examples.len())
            || !(2..=8).contains(&t.admission.len())
        {
            return Err("algorithm lesson metadata".into());
        }
        let mut inputs = BTreeSet::new();
        for ex in t.examples.iter().chain(&t.admission) {
            ex.input.validate()?;
            ex.output.validate()?;
            if !inputs.insert(&ex.input) {
                return Err("overlapping algorithm demonstrations".into());
            }
        }
        if let Some(inquiry) = &t.inquiry {
            inquiry.plan(false).validate(&t.examples)?;
            for e in &inquiry.pool {
                e.output.validate()?;
                if !inputs.insert(&e.input) {
                    return Err("algorithm inquiry overlaps admission".into());
                }
            }
        }
    }
    for change in &c.changes {
        if change.operation != change.teaching.id
            || !lessons.contains(&change.operation)
            || change.requests.is_empty()
        {
            return Err("algorithm change identity".into());
        }
        change.input.validate()?;
        change.actual.validate()?;
    }
    Ok(())
}
struct Training {
    result: std::result::Result<
        crate::structural::agent::AlgorithmAcquisition,
        crate::structural::Error,
    >,
    work: u64,
    candidates: u64,
}
fn teach_task(
    agent: &mut Agent,
    t: &Task,
    inherit: bool,
    budget: SearchBudget,
    passive: bool,
    emit: &mut impl FnMut(
        &crate::structural::agent::AlgorithmEvent,
    ) -> std::result::Result<(), crate::structural::Error>,
) -> std::result::Result<crate::structural::agent::AlgorithmAcquisition, crate::structural::Error> {
    if let Some(inquiry) = &t.inquiry {
        agent.learn_algorithm_inquiring(
            crate::structural::agent::AlgorithmLesson {
                name: &t.id,
                examples: &t.examples,
                admission: &t.admission,
            },
            &inquiry.plan(passive),
            budget,
            inherit,
            &mut |input| {
                inquiry
                    .pool
                    .iter()
                    .find(|e| &e.input == input)
                    .map(|e| e.output.clone())
                    .ok_or(crate::structural::Error::Invalid(
                        "unauthorized algorithm inquiry",
                    ))
            },
            emit,
        )
    } else {
        agent.learn_algorithm_represented(
            crate::structural::agent::AlgorithmLesson {
                name: &t.id,
                examples: &t.examples,
                admission: &t.admission,
            },
            t.representation,
            budget,
            inherit,
            emit,
        )
    }
}
fn train(
    agent: &mut Agent,
    t: &Task,
    inherit: bool,
    budget: SearchBudget,
    variant: &str,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Training {
    let mut work = 0;
    let mut candidates = 0;
    let mut last_inquiry = None;
    let result = teach_task(agent, t, inherit, budget, variant == "passive", &mut |e| {
        work = e.work;
        candidates = e.candidates;
        let expression = e
            .program
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(map_error)?;
        let inquiry = e
            .inquiry
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(map_error)?;
        if let Some(query) = &e.inquiry
            && (query.input.is_some() || last_inquiry.is_none())
        {
            last_inquiry = Some(query.clone());
        }
        let latest = last_inquiry
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(map_error)?;
        telemetry
            .emit(
                "algorithm_investigation",
                &[
                    ("method", V::Text("Agent::learn_algorithm")),
                    ("variant", V::Text(variant)),
                    ("operation", V::Text(&e.operation)),
                    ("phase", V::Text(&e.phase)),
                    ("work", V::Count(e.work)),
                    ("evaluations", V::Count(e.candidates)),
                    ("templates", V::Count(e.templates as u64)),
                    ("mining_work", V::Count(e.mining_work)),
                    ("library_programs", V::Count(e.library_programs as u64)),
                    ("library_windows", V::Count(e.library_windows as u64)),
                    ("expression", expression.as_deref().map_or(V::Null, V::Text)),
                    ("inquiry", inquiry.as_deref().map_or(V::Null, V::Text)),
                    ("last_inquiry", latest.as_deref().map_or(V::Null, V::Text)),
                    (
                        "query_count",
                        last_inquiry
                            .as_ref()
                            .map_or(V::Null, |q| V::Count(q.queries as u64)),
                    ),
                    (
                        "query_hypotheses",
                        last_inquiry
                            .as_ref()
                            .map_or(V::Null, |q| V::Count(q.hypotheses as u64)),
                    ),
                    (
                        "query_survivors",
                        last_inquiry
                            .as_ref()
                            .map_or(V::Null, |q| V::Count(q.survivors as u64)),
                    ),
                ],
            )
            .map_err(map_error)
    });
    Training {
        result,
        work,
        candidates,
    }
}
pub(super) fn teach(
    c: &Campaign,
    base: StructuralSeed,
    output: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<StructuralSeed> {
    validate(c)?;
    save(&output.join("algorithm-curriculum.json"), c)?;
    let mut policy = Policy::default();
    if c.policy_trials > 0 {
        let scope = format!(
            "algorithm-order-inquiry-v3-50000-30000000/{:016x}/{:016x}/{:016x}",
            crate::live_store::checksum(&serde_json::to_vec(&c.lessons)?),
            crate::live_store::checksum(&base.to_bytes()?),
            crate::live_store::checksum(&serde_json::to_vec(&[
                include_str!("algorithm_search.rs"),
                include_str!("algorithm_enumeration.rs"),
                include_str!("algorithm_polynomial.rs"),
                include_str!("algorithm_inquiry.rs"),
                include_str!("algorithm_abstraction.rs"),
                include_str!("algorithm_agent.rs"),
                include_str!("structural_algorithm.rs"),
                include_str!("structural_dependencies.rs"),
                include_str!("structural.rs"),
                include_str!("algorithm_lab.rs"),
                include_str!("nursery.rs"),
            ])?),
        );
        let mut nursery = Nursery::open(
            &output.join("algorithm-nursery.axep"),
            &scope,
            SearchConfig {
                maximum: 719,
                population: 8,
                patience: 1,
                max_candidates: c.policy_trials,
                max_encoded_bytes: 262144,
                max_evaluation_ns: 10_000_000_000,
            },
            314159,
        )?;
        while let Some(p) = nursery.propose_optimization()? {
            let started = Instant::now();
            let mut seed = base.clone();
            seed.set_algorithm_policy(Policy {
                order: p.dna as u16,
            })?;
            let mut learner = Agent::new(seed, Budget::default())?;
            let mut correct = 0;
            let mut work = 0;
            for t in &c.lessons {
                let mut lesson_work = 0;
                if teach_task(
                    &mut learner,
                    t,
                    true,
                    SearchBudget::default(),
                    false,
                    &mut |event| {
                        lesson_work = event.work;
                        Ok(())
                    },
                )
                .is_ok()
                {
                    correct += 1;
                }
                work += lesson_work;
            }
            let ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let bytes = learner.seed().to_bytes()?.len();
            nursery.commit(
                p,
                Assessment {
                    checked: c.lessons.len() as u64,
                    correct,
                    wrong: 0,
                    encoded_bytes: bytes as u64,
                    elapsed_ns: ns,
                },
            )?;
            telemetry.emit(
                "seed_candidate",
                &[
                    ("method", V::Text("AlgorithmPolicy::selection")),
                    ("candidate", V::Count(p.dna)),
                    ("generation", V::Count(nursery.completed() as u64)),
                    ("genotype", V::Text(&p.dna.to_string())),
                    ("correct", V::Count(correct)),
                    ("total", V::Count(c.lessons.len() as u64)),
                    ("work", V::Count(work)),
                    ("capsule_bytes", V::Count(bytes as u64)),
                ],
            )?;
        }
        let best = nursery.best().ok_or("no policy assessed")?;
        policy = Policy {
            order: best.dna as u16,
        };
        save(
            &output.join("algorithm-policy-result.json"),
            &serde_json::json!({"selected":best.dna,"unique":nursery.seen(),"completed":nursery.completed(),"growths":nursery.growths(),"parent":best.parent,"correct":best.assessment.correct,"total":best.assessment.checked,"scope":"Only search-construction order is evolved; external requests/labels were not used in selection."}),
        )?;
    }
    let mut summaries = Vec::new();
    let mut promoted = None;
    let mut failures = Vec::new();
    let mut variants = vec![
        ("inherited", true),
        ("no-library", false),
        ("default-order", true),
    ];
    if c.lessons.iter().any(|t| t.inquiry.is_some()) {
        variants.push(("passive", true));
    }
    for (variant, inherit) in variants {
        let mut seed = base.clone();
        seed.set_algorithm_policy(if variant == "default-order" {
            Policy::default()
        } else {
            policy
        })?;
        let mut agent = Agent::new(seed, Budget::default())?;
        let mut acquired = 0;
        let mut work = 0;
        for (i, t) in c.lessons.iter().enumerate() {
            let r = train(
                &mut agent,
                t,
                inherit,
                SearchBudget::default(),
                variant,
                telemetry,
            );
            work += r.work;
            let (admitted, inherited, error) = match &r.result {
                Ok(r) => {
                    acquired += 1;
                    (true, r.inherited, None)
                }
                Err(e) => (false, false, Some(e.to_string())),
            };
            if !admitted {
                failures
                    .push(serde_json::json!({"variant":variant,"operation":t.id,"error":error}));
            }
            telemetry.emit(
                "validation",
                &[
                    ("method", V::Text("Algorithm::admission")),
                    (
                        "case",
                        V::Count(300000 + summaries.len() as u64 * 1000 + i as u64),
                    ),
                    ("stage", V::Text(&t.id)),
                    ("variant", V::Text(variant)),
                    ("admitted", V::Flag(admitted)),
                    ("work", V::Count(r.work)),
                    ("evaluations", V::Count(r.candidates)),
                    ("borrowed", V::Flag(inherited)),
                ],
            )?;
        }
        let mut correct = 0;
        for r in &c.requests {
            let actual = agent.answer_value(&r.text, &r.bindings);
            correct += usize::from(actual.ok() == r.expected);
        }
        agent
            .seed()
            .save_new(&output.join(format!("algorithm-{variant}.axstruct")))?;
        summaries.push(serde_json::json!({"variant":variant,"acquired":acquired,"lessons":c.lessons.len(),"work_total_logged":work,"correct":correct,"total":c.requests.len()}));
        telemetry.emit(
            "algorithm_result",
            &[
                ("method", V::Text("Algorithm::control")),
                ("case", V::Count(399000 + summaries.len() as u64)),
                ("variant", V::Text(variant)),
                ("correct", V::Count(correct as u64)),
                ("total", V::Count(c.requests.len() as u64)),
                ("work", V::Count(work)),
                ("acquired", V::Count(acquired)),
            ],
        )?;
        if variant == "inherited" {
            promoted = Some(agent.into_seed());
        }
    }
    save(
        &output.join("algorithm-teaching-result.json"),
        &serde_json::json!({"controls":summaries,"failures":failures,"policy":policy}),
    )?;
    promoted.ok_or_else(|| "missing algorithm learner".into())
}
pub(super) fn evaluate(
    c: &Campaign,
    seed: &StructuralSeed,
    output: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<Summary> {
    let mut agent = Agent::new(seed.clone(), Budget::default())?;
    let mut summary = Summary::default();
    let mut rows = Vec::new();
    let mut bytes = 0;
    fn request(
        agent: &Agent,
        r: &Request,
        summary: &mut Summary,
        rows: &mut Vec<serde_json::Value>,
        bytes: &mut usize,
    ) -> Result<()> {
        let actual = agent.answer_value(&r.text, &r.bindings);
        let correct = actual.as_ref().ok() == r.expected.as_ref();
        summary.total += 1;
        summary.correct += u64::from(correct);
        let row = serde_json::json!({"id":r.id,"family":r.family,"text":r.text,"bindings":r.bindings,"expected":r.expected,"actual":actual.as_ref().ok(),"error":actual.as_ref().err().map(ToString::to_string),"correct":correct});
        *bytes += serde_json::to_vec(&row)?.len();
        if *bytes > 16_000_000 {
            return Err("algorithm result quota".into());
        }
        rows.push(row);
        Ok(())
    }
    for (i, r) in c.requests.iter().enumerate() {
        request(&agent, r, &mut summary, &mut rows, &mut bytes)?;
        if (i + 1) % 128 == 0 {
            telemetry.emit(
                "validation",
                &[
                    ("method", V::Text("Algorithm::transfer")),
                    ("case", V::Count(400000 + i as u64)),
                    ("correct", V::Count(summary.correct)),
                    ("total", V::Count(summary.total)),
                ],
            )?;
        }
    }
    for (i, change) in c.changes.iter().enumerate() {
        let previous = agent
            .seed()
            .rules()
            .iter()
            .find(|r| r.id() == format!("algorithm-{}", change.operation))
            .and_then(Rule::algorithm_program)
            .and_then(|p| p.evaluate(&change.input, algorithm::DEFAULT_FUEL).ok())
            .map(|e| e.value);
        let affected =
            agent.algorithm_feedback_affected(&change.operation, &change.input, &change.actual)?;
        let suspended = !affected.is_empty();
        let relearn = train(
            &mut agent,
            &change.teaching,
            true,
            SearchBudget::default(),
            "unannounced-change",
            telemetry,
        );
        telemetry.emit(
            "algorithm_drift",
            &[
                ("method", V::Text("Agent::algorithm_feedback")),
                ("operation", V::Text(&change.operation)),
                ("suspended", V::Flag(suspended)),
                ("invalidated", V::Count(affected.len() as u64)),
                ("admitted", V::Flag(relearn.result.is_ok())),
            ],
        )?;
        save(
            &output.join(format!("algorithm-change-{i}-result.json")),
            &serde_json::json!({"previous_prediction":previous,"actual":change.actual,"suspended":suspended,"invalidated":affected,"relearned":relearn.result.is_ok(),"error":relearn.result.as_ref().err().map(ToString::to_string)}),
        )?;
        for r in &change.requests {
            request(&agent, r, &mut summary, &mut rows, &mut bytes)?;
        }
    }
    if let Some(d) = &c.file_demo {
        crate::demand::file_name(&d.csv)?;
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("evals/fixtures")
            .join(&d.csv);
        bounded_path(&path)?;
        let raw = read_bounded(&path, 1_000_000)?;
        let mut reader = csv::Reader::from_reader(raw.as_slice());
        let mut records = Vec::new();
        for row in reader.records() {
            let row = row?;
            records.push(Value::List(
                row.iter()
                    .map(|s| {
                        s.parse::<i64>()
                            .map(Value::Integer)
                            .unwrap_or_else(|_| Value::Text(s.into()))
                    })
                    .collect(),
            ));
            if records.len() > 1024 {
                return Err("file record quota".into());
            }
        }
        let bindings = BTreeMap::from([(
            "dados".into(),
            Value::List(vec![Value::List(records), Value::Integer(d.parameter)]),
        )]);
        let actual = agent.answer_value(&d.text, &bindings);
        let correct = actual.as_ref().ok() == Some(&d.expected);
        summary.total += 1;
        summary.correct += u64::from(correct);
        publish_new(&output.join("algorithm-input.csv"), &raw)?;
        save(
            &output.join("algorithm-file-result.json"),
            &serde_json::json!({"input_file":d.csv,"parameter":d.parameter,"input_sha256_format":"AXON checksum, not cryptographic hash","checksum":crate::live_store::checksum(&raw),"actual":actual.as_ref().ok(),"expected":d.expected,"correct":correct,"scope":"Actual CSV IO; synthetic records; learned program, not a supplied workflow."}),
        )?;
    }
    save(&output.join("algorithm-predictions.json"), &rows)?;
    if !c.changes.is_empty() {
        let folder = output.join("algorithm-adapted");
        fs::create_dir(&folder)?;
        packed(agent.seed(), &folder)?;
    }
    save(
        &output.join("algorithm-result.json"),
        &serde_json::json!({"correct":summary.correct,"total":summary.total,"rules":seed.rules().iter().filter(|r|r.algorithm_program().is_some()).count(),"scope":"Bounded typed program construction from examples; declared grammar recipes; no free-language, general vision or AGI claim."}),
    )?;
    telemetry.emit(
        "algorithm_result",
        &[
            ("method", V::Text("Algorithm::portable")),
            ("case", V::Count(499999)),
            ("correct", V::Count(summary.correct)),
            ("total", V::Count(summary.total)),
        ],
    )?;
    Ok(summary)
}
