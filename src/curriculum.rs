//! JSON campaign adapter. Teachers and host routing are not inherited cognition.
use crate::{
    Authority, Effect, TextBudget, TextExample, TextSkill, TextValuePrediction,
    demand::{
        Catalog, DemandError, DemandRuntime, Module, ModuleKind, ModuleSpec, Seed,
        controller::CompositionBudget,
        genome::{BirthLimits, Genome, MAX_BYTES, bounded_path},
    },
    live_store::checksum,
    numeric::{
        NumericBudget, NumericExample, NumericSkill, NumericTolerance,
        evolution::{EvolvedSkill, constructive::*},
    },
    semantic::{Demonstration, Focus, Meaning, Rule, SemanticSeed, Session},
    telemetry::{LabTelemetry, Value as V},
    text_skill::{publish_new, read_bounded},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Numbers(Vec<f64>),
    Number(f64),
    Text(String),
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Example {
    pub input: Value,
    pub output: Value,
    #[serde(default)]
    pub arguments: Vec<f64>,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Numeric,
    Composed,
    Text,
    WordMath,
    Relational,
}
impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Numeric => "numeric",
            Self::Composed => "composed",
            Self::Text => "text",
            Self::WordMath => "word_math",
            Self::Relational => "relational",
        }
    }
    fn module(self) -> ModuleKind {
        match self {
            Self::Numeric => ModuleKind::Evolved,
            Self::Composed => ModuleKind::Numeric,
            Self::Text => ModuleKind::Text,
            Self::WordMath | Self::Relational => ModuleKind::Semantic,
        }
    }
    fn extension(self) -> &'static str {
        match self {
            Self::Numeric => "axevo",
            Self::Composed => "axmath",
            Self::Text => "axtext",
            Self::WordMath | Self::Relational => "axsem",
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub id: String,
    pub family: String,
    pub kind: Kind,
    pub required: bool,
    #[serde(default)]
    pub parents: Vec<String>,
    pub train: Vec<Example>,
    pub admission: Vec<Example>,
    pub test: Vec<Example>,
    #[serde(default)]
    pub transfer: Vec<Example>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub candidates_per_stage: usize,
    pub total_candidates: usize,
    pub work_per_stage: u64,
    pub total_numeric_work: u64,
    pub text_work_per_stage: u64,
    pub semantic_work_per_stage: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prepared {
    pub version: u32,
    pub id: String,
    pub source_sha256: String,
    pub budget: Budget,
    pub stages: Vec<Stage>,
}
fn identifier(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"_-".contains(&b))
}
impl Prepared {
    pub fn validate(&self) -> Result<()> {
        let b = &self.budget;
        if self.version != 1
            || !identifier(&self.id)
            || self.source_sha256.len() != 64
            || !self.source_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || self.stages.is_empty()
            || !self.stages.iter().any(|s| s.required)
            || self.stages.len() > 1000
            || !(1..=20000).contains(&b.candidates_per_stage)
            || b.total_candidates < b.candidates_per_stage
            || b.total_candidates > 1_000_000
            || !(1000..=100_000_000).contains(&b.work_per_stage)
            || b.total_numeric_work < b.work_per_stage
            || b.total_numeric_work > 2_000_000_000
            || !(1000..=2_000_000).contains(&b.text_work_per_stage)
            || !(1000..=20_000_000).contains(&b.semantic_work_per_stage)
        {
            return Err("invalid curriculum identity, dimensions or budget".into());
        }
        let mut seen = BTreeSet::new();
        let mut numeric_ids = BTreeSet::new();
        let mut relational_ids = BTreeSet::new();
        for stage in &self.stages {
            if !identifier(&stage.id)
                || !identifier(&stage.family)
                || seen.contains(&stage.id)
                || stage.parents.len() > 8
                || stage.parents.iter().any(|p| {
                    if stage.kind == Kind::Relational {
                        !relational_ids.contains(p)
                    } else {
                        !numeric_ids.contains(p)
                    }
                })
                || stage.parents.iter().collect::<BTreeSet<_>>().len() != stage.parents.len()
                || (!matches!(
                    stage.kind,
                    Kind::Numeric | Kind::Composed | Kind::Relational
                ) && !stage.parents.is_empty())
                || !(4..=32).contains(&stage.train.len())
                || !(4..=32).contains(&stage.admission.len())
                || stage.test.is_empty()
                || stage.test.len() > 512
                || stage.transfer.len() > 128
            {
                return Err(format!("invalid stage {} or forward dependency", stage.id).into());
            }
            if matches!(stage.kind, Kind::Numeric | Kind::Composed) {
                numeric_ids.insert(stage.id.clone());
            }
            if stage.kind == Kind::Relational {
                relational_ids.insert(stage.id.clone());
            }
            let mut inputs = BTreeSet::new();
            let mut arity = None;
            for row in stage
                .train
                .iter()
                .chain(&stage.admission)
                .chain(&stage.test)
                .chain(&stage.transfer)
            {
                if !inputs.insert(serde_json::to_string(&row.input)?) {
                    return Err(format!("duplicate/leaked input in {}", stage.id).into());
                }
                match (&row.input, &row.output, stage.kind) {
                    (Value::Numbers(x), Value::Number(y), Kind::Numeric | Kind::Composed)
                        if (1..=4).contains(&x.len())
                            && y.is_finite()
                            && x.iter().all(|v| v.is_finite()) =>
                    {
                        if arity.is_some_and(|n| n != x.len()) {
                            return Err("numeric arity drift".into());
                        }
                        arity = Some(x.len());
                        if !row.arguments.is_empty() {
                            return Err("numeric examples cannot carry semantic roles".into());
                        }
                    }
                    (Value::Text(x), Value::Text(y), Kind::Text)
                        if x.len() <= 1024 && y.len() <= 256 && row.arguments.is_empty() => {}
                    (Value::Text(x), Value::Number(y), Kind::WordMath | Kind::Relational)
                        if x.len() <= 1024
                            && y.is_finite()
                            && (1..=4).contains(&row.arguments.len())
                            && row.arguments.iter().all(|v| v.is_finite()) => {}
                    _ => return Err(format!("wrong typed example in {}", stage.id).into()),
                }
            }
            seen.insert(stage.id.clone());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub id: String,
    pub family: String,
    pub kind: Kind,
    pub required: bool,
    pub admitted: bool,
    pub reason: String,
    pub test_correct: usize,
    pub test_total: usize,
    pub transfer_correct: usize,
    pub transfer_total: usize,
    pub new_candidates: usize,
    pub total_candidates: usize,
    pub numeric_work: u64,
    pub search_work: Option<u64>,
    pub elapsed_ns: u64,
    pub model_bytes: Option<u64>,
    pub model_checksum: Option<u64>,
    pub artifact: Option<String>,
    pub cached: bool,
}
fn context(id: &str) -> String {
    format!("curriculum/{id}")
}
fn predict(module: &Module, input: &Value) -> Option<Value> {
    match (module, input) {
        (Module::Evolved(s), Value::Numbers(x)) => s.program().evaluate(x).ok().map(Value::Number),
        (Module::Numeric(s), Value::Numbers(x)) => s.program().evaluate(x).ok().map(Value::Number),
        (Module::Text(s), Value::Text(x)) => match s.predict_value(s.context(), x).ok()? {
            TextValuePrediction::Agreed(y) => Some(Value::Text(y)),
            _ => None,
        },
        (Module::Semantic(s), Value::Text(x)) => Session::default()
            .narrative(s, x)
            .ok()
            .map(|a| Value::Number(a.value)),
        _ => None,
    }
}
#[derive(Serialize)]
pub struct RoutedAnswer {
    pub module: String,
    pub value: f64,
    pub expression: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    module: Option<String>,
    input: Value,
    expected: Option<Value>,
}
fn batch(number: &Path, birth_dir: &Path, cases: &Path, receipt: &Path) -> Result<()> {
    for p in [number, birth_dir, cases, receipt] {
        bounded_path(p)?;
    }
    let probes: Vec<Probe> = serde_json::from_slice(&read_bounded(cases, 16_000_000)?)?;
    if probes.is_empty()
        || probes.len() > 10_000
        || probes.iter().any(|p| {
            p.module.as_ref().is_some_and(|s| !identifier(s))
                || match &p.input {
                    Value::Numbers(x) => {
                        x.len() > 4 || x.is_empty() || x.iter().any(|n| !n.is_finite())
                    }
                    Value::Text(s) => s.len() > 1024,
                    Value::Number(_) => true,
                }
        })
    {
        return Err("probe bounds".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let genome = Genome::open_number(number, &auth)?;
    let mut birth = genome.birth(birth_dir, BirthLimits::default(), &auth)?;
    let runtime = birth.controller.runtime_mut();
    let mut records = Vec::new();
    let mut correct = 0;
    for p in probes {
        let actual = if let Some(id) = &p.module {
            let ctx = runtime
                .catalog()
                .get(id)
                .ok_or("probe module unavailable")?
                .context
                .clone();
            runtime.with_module(id, &ctx, &auth, |m| Ok(predict(m, &p.input)))?
        } else {
            let Value::Text(s) = &p.input else {
                return Err("routing requires text".into());
            };
            route(runtime, s, &auth)?.map(|a| Value::Number(a.value))
        };
        let pass = match &p.expected {
            Some(v) => agrees(actual.as_ref(), v),
            None => actual.is_none(),
        };
        correct += usize::from(pass);
        records.push(serde_json::json!({"module":p.module,"input":p.input,"expected":p.expected,"actual":actual,"correct":pass}));
    }
    let result = serde_json::json!({"schema":"axon-portable-probes/v1","correct":correct,"checked":records.len(),"records":records,"active_modules":runtime.stats().active_modules,"loads":runtime.stats().loads,"training_required":false});
    json_new(receipt, &result)?;
    println!(
        "{}",
        serde_json::json!({"correct":correct,"checked":result["checked"],"active_modules":runtime.stats().active_modules})
    );
    Ok(())
}
pub fn route(
    runtime: &mut DemandRuntime,
    input: &str,
    auth: &Authority,
) -> Result<Option<RoutedAnswer>> {
    let modules: Vec<_> = runtime
        .catalog()
        .entries()
        .iter()
        .filter(|m| m.kind == ModuleKind::Semantic)
        .map(|m| (m.id.clone(), m.context.clone()))
        .collect();
    let mut answers = Vec::new();
    let mut unsafe_match = false;
    for (id, ctx) in modules {
        runtime.with_module(&id, &ctx, auth, |m| {
            if let Module::Semantic(s) = m {
                match Session::default().narrative(s, input) {
                    Ok(a) => answers.push(RoutedAnswer {
                        module: id.clone(),
                        value: a.value,
                        expression: a.expression,
                    }),
                    Err(
                        crate::semantic::SemanticError::Ambiguous
                        | crate::semantic::SemanticError::Domain,
                    ) => unsafe_match = true,
                    Err(_) => {}
                }
            }
            Ok(())
        })?;
    }
    Ok(if answers.len() == 1 && !unsafe_match {
        answers.pop()
    } else {
        None
    })
}
fn agrees(actual: Option<&Value>, expected: &Value) -> bool {
    match (actual, expected) {
        (Some(Value::Number(a)), Value::Number(b)) => NumericTolerance::default().agrees(*a, *b),
        (Some(Value::Text(a)), Value::Text(b)) => a == b,
        _ => false,
    }
}
fn load(kind: Kind, path: &Path, id: &str) -> Result<Module> {
    Ok(match kind {
        Kind::Numeric => Module::Evolved(Box::new(EvolvedSkill::open(path, &context(id))?)),
        Kind::Composed => Module::Numeric(Box::new(NumericSkill::from_bytes(&read_bounded(
            path, 32768,
        )?)?)),
        Kind::Text => Module::Text(TextSkill::load(path)?),
        Kind::WordMath | Kind::Relational => Module::Semantic(Box::new(SemanticSeed::open(path)?)),
    })
}
fn score(module: Option<&Module>, rows: &[Example]) -> usize {
    rows.iter()
        .filter(|e| {
            agrees(
                module.and_then(|m| predict(m, &e.input)).as_ref(),
                &e.output,
            )
        })
        .count()
}
fn number_rows(rows: &[Example]) -> Vec<NumericExample> {
    rows.iter()
        .map(|e| {
            let (Value::Numbers(x), Value::Number(y)) = (&e.input, &e.output) else {
                unreachable!("validated numeric stage")
            };
            NumericExample::new(x.clone(), *y)
        })
        .collect()
}
fn json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    publish_new(path, &serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
struct Stored {
    kind: Kind,
    id: String,
    path: PathBuf,
}
struct Remaining {
    candidates: usize,
    work: u64,
}
#[derive(Default)]
struct Fit {
    module: Option<Module>,
    reason: String,
    new: usize,
    total: usize,
    numeric_work: u64,
    search_work: Option<u64>,
}
fn fit(
    stage: &Stage,
    config: &Prepared,
    directory: &Path,
    ancestors: &[Stored],
    rng: u64,
    remaining: Remaining,
    mut observe: impl FnMut(ConstructiveProgress),
) -> Result<Fit> {
    let b = &config.budget;
    match stage.kind {
        Kind::Composed => {
            let modules: Vec<_> = ancestors
                .iter()
                .filter(|p| stage.parents.contains(&p.id))
                .map(|p| load(p.kind, &p.path, &p.id))
                .collect::<Result<_>>()?;
            let programs: Vec<_> = modules
                .iter()
                .filter_map(|m| match m {
                    Module::Evolved(s) => Some((s.program(), s.fingerprint())),
                    Module::Numeric(s) => Some((s.program(), s.fingerprint())),
                    _ => None,
                })
                .collect();
            match NumericSkill::learn_from_programs(
                &stage.id,
                context(&stage.id),
                number_rows(&stage.train),
                &programs,
                NumericBudget {
                    max_cost: 12,
                    max_work: b.semantic_work_per_stage,
                    ..NumericBudget::default()
                },
                NumericTolerance::default(),
            ) {
                Ok(s) => {
                    let work = s.search().scalar_work;
                    Ok(Fit {
                        module: Some(Module::Numeric(Box::new(s))),
                        reason: "composed-from-examples".into(),
                        search_work: Some(work),
                        ..Fit::default()
                    })
                }
                Err(e) => Ok(Fit {
                    reason: e.to_string(),
                    ..Fit::default()
                }),
            }
        }
        Kind::Numeric => {
            if remaining.candidates == 0 || remaining.work < 1000 {
                return Ok(Fit {
                    reason: "campaign-numeric-budget".into(),
                    search_work: Some(0),
                    ..Fit::default()
                });
            }
            let parents: Vec<_> = ancestors
                .iter()
                .filter(|p| p.kind == Kind::Numeric && stage.parents.contains(&p.id))
                .map(|p| EvolvedSkill::open(&p.path, &context(&p.id)))
                .collect::<std::result::Result<_, _>>()?;
            let refs: Vec<_> = parents.iter().collect();
            let mut search = ConstructiveSearch::open_with_policy(
                &directory.join("search.axep"),
                number_rows(&stage.train),
                &refs,
                ConstructiveBudget {
                    initial_genes: 4,
                    max_candidates: b.candidates_per_stage,
                    max_work: b.work_per_stage,
                    ..ConstructiveBudget::default()
                },
                NumericTolerance::default(),
                rng,
                ConstructivePolicy::Diverse,
            )?;
            let before = search.progress();
            // Configuration fixes the stage ceiling; the global remainder bounds each batch too.
            let batch = remaining
                .candidates
                .saturating_sub(before.seen)
                .min(b.candidates_per_stage);
            let new = if b.work_per_stage > remaining.work || batch == 0 {
                0
            } else {
                search.run(batch, &mut observe)?
            };
            let p = search.progress();
            let result = search.certify(
                &stage.id,
                &context(&stage.id),
                number_rows(&stage.admission),
            );
            let admission_work = result.as_ref().map_or(0, |s| s.statistics().admission_work);
            let (module, reason) = match result {
                Ok(s) => (Some(Module::Evolved(Box::new(s))), "admitted".into()),
                Err(e) => (None, e.to_string()),
            };
            Ok(Fit {
                module,
                reason,
                new,
                total: p.seen,
                numeric_work: p.work + admission_work,
                search_work: Some(p.work),
            })
        }
        Kind::Text => {
            let rows = stage
                .train
                .iter()
                .map(|e| {
                    let (Value::Text(x), Value::Text(y)) = (&e.input, &e.output) else {
                        unreachable!()
                    };
                    TextExample::new(x, y)
                })
                .collect();
            match TextSkill::learn(
                &stage.id,
                context(&stage.id),
                rows,
                TextBudget {
                    max_work: b.text_work_per_stage,
                    ..TextBudget::default()
                },
            ) {
                Ok(s) => {
                    let work = s.program().search_work();
                    Ok(Fit {
                        module: Some(Module::Text(s)),
                        reason: "fitted".into(),
                        search_work: Some(work),
                        ..Fit::default()
                    })
                }
                Err(e) => Ok(Fit {
                    reason: e.to_string(),
                    ..Fit::default()
                }),
            }
        }
        Kind::WordMath | Kind::Relational => {
            let examples: Vec<_> = stage
                .train
                .iter()
                .map(|e| {
                    let (Value::Text(x), Value::Number(y)) = (&e.input, &e.output) else {
                        unreachable!()
                    };
                    Demonstration {
                        sentence: x.clone(),
                        focus: Focus::default(),
                        meaning: Meaning::Calculate {
                            arguments: e.arguments.clone(),
                            answer: *y,
                        },
                    }
                })
                .collect();
            let budget = NumericBudget {
                max_work: b.semantic_work_per_stage,
                ..NumericBudget::default()
            };
            let parents: Vec<_> = ancestors
                .iter()
                .filter(|p| p.kind == Kind::Relational && stage.parents.contains(&p.id))
                .map(|p| SemanticSeed::open(&p.path))
                .collect::<std::result::Result<_, _>>()?;
            let refs: Vec<_> = parents.iter().flat_map(|s| s.rules()).collect();
            let learned = if stage.kind == Kind::Relational {
                Rule::learn_relational(&examples, budget, 1, &refs)
            } else {
                Rule::learn(&examples, budget)
            };
            match learned {
                Ok(rule) => {
                    let work = rule.search_work();
                    Ok(Fit {
                        module: Some(Module::Semantic(Box::new(SemanticSeed::new(vec![rule])?))),
                        reason: "fitted".into(),
                        search_work: Some(work),
                        ..Fit::default()
                    })
                }
                Err(e) => Ok(Fit {
                    reason: e.to_string(),
                    ..Fit::default()
                }),
            }
        }
    }
}
fn encoded(module: &Module) -> Result<Vec<u8>> {
    Ok(match module {
        Module::Evolved(s) => s.to_bytes(),
        Module::Numeric(s) => s.to_bytes(),
        Module::Text(s) => s.to_bytes()?,
        Module::Semantic(s) => s.to_bytes()?,
        _ => return Err("unsupported curriculum module".into()),
    })
}
pub fn run(prepared: &Path, output: &Path, rng: u64) -> Result<()> {
    bounded_path(prepared)?;
    bounded_path(output)?;
    let bytes = read_bounded(prepared, 16_000_000)?;
    let config: Prepared = serde_json::from_slice(&bytes)?;
    config.validate()?;
    let registry = std::env::var_os("AXON_CURRICULUM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".axon-memory/curriculum")
        });
    bounded_path(&registry)?;
    fs::create_dir_all(&registry)?;
    let scope = registry.join(format!("{}-{rng}", config.id));
    bounded_path(&scope)?;
    fs::create_dir_all(&scope)?;
    let lock = scope.join("campaign.lock");
    let _lock = CampaignLock::new(lock)?;
    let mut telemetry = LabTelemetry::from_environment("curriculum-v1");
    let started = Instant::now();
    telemetry.emit(
        "lab_start",
        &[
            ("method", V::Text("Curriculum::run")),
            ("tasks", V::Count(config.stages.len() as u64)),
        ],
    )?;
    let mut outcomes = Vec::new();
    let mut acquired: Vec<Stored> = Vec::new();
    let (mut used, mut work, mut total) = (0usize, 0u64, 0usize);
    for (index, stage) in config.stages.iter().enumerate() {
        let now = Instant::now();
        let mut identity = serde_json::to_vec(&(stage, &config.budget))?;
        for parent in acquired.iter().filter(|p| stage.parents.contains(&p.id)) {
            let data = read_bounded(&parent.path, 262_144)?;
            identity.extend((data.len() as u64).to_le_bytes());
            identity.extend(data);
        }
        let directory = scope.join(format!("{}-{:016x}", stage.id, checksum(&identity)));
        bounded_path(&directory)?;
        fs::create_dir_all(&directory)?;
        let identity_path = directory.join("scope.bin");
        if identity_path.exists() {
            if read_bounded(&identity_path, 16_000_000)? != identity {
                return Err("exact curriculum scope mismatch".into());
            }
        } else {
            publish_new(&identity_path, &identity)?;
        }
        let result_path = directory.join("outcome.json");
        let artifact = format!("{}.{}", stage.id, stage.kind.extension());
        let model = directory.join(&artifact);
        let mut outcome;
        if result_path.exists() {
            outcome = serde_json::from_slice::<Outcome>(&read_bounded(&result_path, 16_384)?)?;
            if outcome.id != stage.id
                || outcome.kind != stage.kind
                || outcome.family != stage.family
                || outcome.required != stage.required
                || outcome.test_total != stage.test.len()
                || outcome.transfer_total != stage.transfer.len()
                || outcome.test_correct > outcome.test_total
                || outcome.transfer_correct > outcome.transfer_total
                || (outcome.admitted
                    && (outcome.artifact.as_deref() != Some(&artifact)
                        || outcome.model_bytes.is_none()
                        || outcome.model_checksum.is_none()))
                || (!outcome.admitted
                    && (outcome.test_correct != 0
                        || outcome.transfer_correct != 0
                        || outcome.artifact.is_some()
                        || outcome.model_bytes.is_some()
                        || outcome.model_checksum.is_some()))
            {
                return Err("cached outcome identity".into());
            }
            if outcome.admitted {
                if Some(checksum(&read_bounded(&model, 262_144)?)) != outcome.model_checksum {
                    return Err("cached model checksum".into());
                }
                let module = load(stage.kind, &model, &stage.id)?;
                if score(Some(&module), &stage.train) != stage.train.len()
                    || score(Some(&module), &stage.admission) != stage.admission.len()
                    || score(Some(&module), &stage.test) != outcome.test_correct
                    || score(Some(&module), &stage.transfer) != outcome.transfer_correct
                {
                    return Err("cached curriculum verdict mismatch".into());
                }
            }
            outcome.cached = true;
            outcome.new_candidates = 0;
        } else {
            if model.exists() || directory.join("predictions.json").exists() {
                return Err(format!(
                    "incomplete stage {} is quarantined; preserve its evidence",
                    stage.id
                )
                .into());
            }
            telemetry.emit(
                "stage",
                &[
                    ("method", V::Text("Curriculum::fit")),
                    ("case", V::Count(index as u64)),
                    ("name", V::Text(&stage.id)),
                    ("domain", V::Text(stage.kind.name())),
                ],
            )?;
            let mut error = None;
            let Fit {
                mut module,
                mut reason,
                new,
                total: stage_total,
                numeric_work,
                search_work,
            } = fit(
                stage,
                &config,
                &directory,
                &acquired,
                rng ^ checksum(stage.id.as_bytes()),
                Remaining {
                    candidates: config.budget.total_candidates.saturating_sub(total),
                    work: config.budget.total_numeric_work.saturating_sub(work),
                },
                |p| {
                    if error.is_none() {
                        error = telemetry
                            .emit(
                                "evolution_progress",
                                &[
                                    ("method", V::Text("ConstructiveSearch::run")),
                                    ("case", V::Count(index as u64)),
                                    ("evaluations", V::Count(p.completed as u64)),
                                    ("correct", V::Count(p.correct as u64)),
                                    ("work", V::Count(p.work)),
                                    ("genes", V::Count(p.genes as u64)),
                                ],
                            )
                            .err();
                    }
                },
            )?;
            if let Some(e) = error {
                return Err(e.into());
            }
            used += new;
            let mut admitted = module.as_ref().is_some_and(|m| {
                score(Some(m), &stage.train) == stage.train.len()
                    && score(Some(m), &stage.admission) == stage.admission.len()
            });
            if !admitted {
                if module.is_some() {
                    reason = "refuted-by-admission".into();
                }
                module = None;
            }
            if let Some(m) = &module
                && (m.retained_bytes() > 262_144 || encoded(m)?.len() > 180_000)
            {
                admitted = false;
                reason = "module-exceeds-portable-page-budget".into();
                module = None;
            }
            // Post-selection labels never choose a replacement or reward an operator.
            let test_correct = score(module.as_ref(), &stage.test);
            let transfer_correct = score(module.as_ref(), &stage.transfer);
            let mut predictions = Vec::new();
            for (phase, rows) in [("test", &stage.test), ("transfer", &stage.transfer)] {
                for (row, e) in rows.iter().enumerate() {
                    let actual = module.as_ref().and_then(|m| predict(m, &e.input));
                    predictions.push(serde_json::json!({"phase":phase,"row":row,"actual":actual,"correct":agrees(actual.as_ref(),&e.output)}));
                }
            }
            json_new(&directory.join("predictions.json"), &predictions)?;
            let (model_bytes, model_checksum) = if let Some(m) = &module {
                publish_new(&model, &encoded(m)?)?;
                let data = fs::read(&model)?;
                (Some(data.len() as u64), Some(checksum(&data)))
            } else {
                (None, None)
            };
            outcome = Outcome {
                id: stage.id.clone(),
                family: stage.family.clone(),
                kind: stage.kind,
                required: stage.required,
                admitted,
                reason,
                test_correct,
                test_total: stage.test.len(),
                transfer_correct,
                transfer_total: stage.transfer.len(),
                new_candidates: new,
                total_candidates: stage_total,
                numeric_work,
                search_work,
                elapsed_ns: now.elapsed().as_nanos() as u64,
                model_bytes,
                model_checksum,
                artifact: admitted.then_some(artifact.clone()),
                cached: false,
            };
            json_new(&result_path, &outcome)?;
        }
        total += outcome.total_candidates;
        work += outcome.numeric_work;
        if total > config.budget.total_candidates || work > config.budget.total_numeric_work {
            return Err("campaign accounting exceeds budget".into());
        }
        if outcome.admitted {
            let copy = output.join(&artifact);
            publish_new(&copy, &fs::read(&model)?)?;
            acquired.push(Stored {
                kind: stage.kind,
                id: stage.id.clone(),
                path: copy,
            });
        }
        json_new(&output.join(format!("{}-result.json", stage.id)), &outcome)?;
        publish_new(
            &output.join(format!("{}-predictions.json", stage.id)),
            &read_bounded(&directory.join("predictions.json"), 256_000)?,
        )?;
        telemetry.emit(
            "validation",
            &[
                ("method", V::Text("Curriculum::validate")),
                ("case", V::Count(index as u64)),
                ("name", V::Text(&stage.id)),
                ("domain", V::Text(stage.kind.name())),
                ("family", V::Text(&stage.family)),
                (
                    "phase",
                    V::Text(if outcome.cached {
                        "development-recheck"
                    } else {
                        "post-selection"
                    }),
                ),
                ("correct", V::Count(outcome.test_correct as u64)),
                ("checked", V::Count(outcome.test_total as u64)),
                (
                    "transfer_correct",
                    V::Count(outcome.transfer_correct as u64),
                ),
                ("transfer_total", V::Count(outcome.transfer_total as u64)),
                ("admitted", V::Flag(outcome.admitted)),
                ("detail", V::Text(&outcome.reason)),
                ("new_candidates", V::Count(outcome.new_candidates as u64)),
                (
                    "search_work",
                    outcome.search_work.map(V::Count).unwrap_or(V::Null),
                ),
            ],
        )?;
        outcomes.push(outcome);
    }
    let pages = pack(&acquired, &config, output, &mut telemetry)?;
    let result = serde_json::json!({"schema":"axon-curriculum-results/v1","id":config.id,"stages":outcomes,"new_candidates":used,"numeric_work":work,"pages":pages,"process_elapsed_ns":started.elapsed().as_nanos() as u64,"RSS":null,"energy":null,"personal_episodes":0});
    json_new(&output.join("curriculum-results.json"), &result)?;
    let admitted = outcomes.iter().filter(|o| o.admitted).count();
    let passed = outcomes
        .iter()
        .filter(|o| {
            o.required
                && o.admitted
                && o.test_correct == o.test_total
                && o.transfer_correct == o.transfer_total
        })
        .count();
    let required = outcomes.iter().filter(|o| o.required).count();
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("Curriculum::result")),
            ("admitted", V::Count(admitted as u64)),
            ("tasks", V::Count(outcomes.len() as u64)),
            ("goal_tasks", V::Count(required as u64)),
            ("goal_admitted", V::Count(passed as u64)),
            ("all_goals_passed", V::Flag(passed == required)),
            ("new_candidates", V::Count(used as u64)),
            ("elapsed_ns", V::Count(started.elapsed().as_nanos() as u64)),
        ],
    )?;
    Ok(())
}
struct CampaignLock {
    _file: fs::File,
}
impl CampaignLock {
    fn new(path: PathBuf) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.try_lock()?;
        Ok(Self { _file: file })
    }
}
fn pack(
    models: &[Stored],
    config: &Prepared,
    output: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
) -> Result<serde_json::Value> {
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut pages = Vec::new();
    let mut at = 0;
    while at < models.len() {
        let mut catalog = Catalog::new(output);
        let start = at;
        let mut bytes = 0;
        while at < models.len() && at - start < 128 {
            let m = &models[at];
            let size = fs::metadata(&m.path)?.len() as usize;
            if at > start && bytes + size > 180_000 {
                break;
            }
            if size > 180_000 {
                return Err("single curriculum module exceeds page budget".into());
            }
            let file = m
                .path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("module filename")?;
            catalog.register(
                ModuleSpec {
                    id: &m.id,
                    kind: m.kind.module(),
                    file,
                    input: if matches!(m.kind, Kind::Numeric | Kind::Composed) {
                        "numbers"
                    } else {
                        "text"
                    },
                    output: if m.kind == Kind::Text {
                        "text"
                    } else {
                        "number"
                    },
                    cost: 1,
                },
                &auth,
            )?;
            bytes += size;
            at += 1;
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
            "curriculum/v1",
            &mut runtime,
            CompositionBudget::default(),
            &[],
            MAX_BYTES,
            &auth,
        )?;
        let name = format!("page-{}.axnum", pages.len());
        genome.save_number_new(&output.join(&name), &auth)?;
        let mut birth = genome.birth(
            &output.join(format!("birth-{}", pages.len())),
            BirthLimits::default(),
            &auth,
        )?;
        let initial = birth.controller.runtime().stats().loads;
        let mut retention = 0;
        let mut post_selection_rechecks = 0;
        for m in &models[start..at] {
            let stage = config
                .stages
                .iter()
                .find(|s| s.id == m.id)
                .ok_or("retention stage")?;
            let ctx = if matches!(m.kind, Kind::WordMath | Kind::Relational) {
                "axon/relational-v1".into()
            } else {
                context(&m.id)
            };
            birth
                .controller
                .runtime_mut()
                .with_module(&m.id, &ctx, &auth, |module| {
                    for example in stage.train.iter().chain(&stage.admission) {
                        if !agrees(predict(module, &example.input).as_ref(), &example.output) {
                            return Err(DemandError::Invalid("curriculum birth retention"));
                        }
                        retention += 1;
                    }
                    let receipt: Outcome = serde_json::from_slice(
                        &read_bounded(&output.join(format!("{}-result.json", m.id)), 16_384)
                            .map_err(|_| DemandError::Invalid("curriculum receipt"))?,
                    )
                    .map_err(|_| DemandError::Invalid("curriculum receipt"))?;
                    if score(Some(module), &stage.test) != receipt.test_correct
                        || score(Some(module), &stage.transfer) != receipt.transfer_correct
                    {
                        return Err(DemandError::Invalid("curriculum post-selection retention"));
                    }
                    post_selection_rechecks += stage.test.len() + stage.transfer.len();
                    Ok(())
                })?;
        }
        let stats = birth.controller.runtime().stats();
        pages.push(serde_json::json!({"file":name,"checksum":format!("{:016x}",checksum(&fs::read(output.join(&name))?)),"encoded_bytes":genome.encoded_bytes(),"modules":models[start..at].iter().map(|m|&m.id).collect::<Vec<_>>(),"initial_loads":initial,"loads":stats.loads,"active_modules":stats.active_modules,"retention_checks":retention,"post_selection_rechecks":post_selection_rechecks}));
    }
    json_new(
        &output.join("model-index.json"),
        &serde_json::json!({"schema":"axon-curriculum-model-index/v1","pages":pages,"routing":"explicit stage id supplied by host; not free-language intent recognition"}),
    )?;
    telemetry.emit(
        "birth",
        &[
            ("method", V::Text("Genome::birth")),
            ("modules", V::Count(models.len() as u64)),
            ("pages", V::Count(pages.len() as u64)),
        ],
    )?;
    Ok(serde_json::Value::Array(pages))
}
pub fn command() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("structural-talk") if args.len()==4 => crate::structural_lab::talk(Path::new(&args[1]),Path::new(&args[2]),&args[3]),
        Some("structure") if args.len()==3 => crate::structural_lab::run(Path::new(&args[1]),Path::new(&args[2])),
        Some("structural-batch") if args.len()==5 => crate::structural_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("algorithm-batch") if args.len()==5 => crate::structural_lab::algorithm_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("algorithm-external") if args.len()==5 => crate::structural_lab::algorithm_external(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("session-batch") if args.len()==5 => crate::structural_lab::session_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("joint-train") if args.len()==5 => crate::structural_lab::joint_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("encoder-train") if args.len()==5 => crate::structural_lab::encoder_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("encoder-refine") if args.len()==5 => crate::structural_lab::encoder_lab::refine(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("encoder-batch") if args.len()==5 => crate::structural_lab::encoder_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-pack") if args.len()==5 => crate::structural_lab::math_lab::pack(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-batch") if args.len()==5 => crate::structural_lab::math_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-search-batch") if args.len()==5 => crate::structural_lab::math_lab::search_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-actor-pack") if args.len()==5 => crate::structural_lab::math_lab::actor::pack(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-actor-batch") if args.len()==5 => crate::structural_lab::math_lab::actor::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-actor-focused-pack") if args.len()==5 => crate::structural_lab::math_lab::actor::focused_pack(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("math-actor-focused-batch") if args.len()==5 => crate::structural_lab::math_lab::actor::focused_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("bdh-seed") if args.len()==5 => crate::structural_lab::synaptic_lab::seed(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("bdh-session") if args.len()==5 => crate::structural_lab::synaptic_lab::session(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("pointer-train") if args.len()==5 => crate::structural_lab::math_lab::pointer::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("pointer-pack") if args.len()==5 => crate::structural_lab::math_lab::pointer_pack(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("pointer-batch") if args.len()==5 => crate::structural_lab::math_lab::pointer::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("pointer-frames-batch") if args.len()==5 => crate::structural_lab::math_lab::pointer::frames_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some(command) if args.len()==5 && crate::structural_lab::math_lab::pointer::handles(command) => crate::structural_lab::math_lab::pointer::run(command,Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some(command @ ("math-actor-frames-batch"|"math-actor-focused-frames-batch")) if args.len()==5 => crate::structural_lab::math_lab::actor::frames_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4]),command=="math-actor-focused-frames-batch"),
        Some("segment-train") if args.len()==5 => crate::structural_lab::segment_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("segment-batch") if args.len()==5 => crate::structural_lab::encoder_lab::segment_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("joint-batch") if args.len()==5 => crate::structural_lab::joint_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("search-batch") if args.len()==5 => crate::structural_lab::joint_lab::probe(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("interpret-batch") if args.len()==5 => crate::structural_lab::interpretation_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("file-request") if args.len()==5 => crate::structural_lab::file_request(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("sequence-train") if args.len()==5 => crate::structural_lab::sequence_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("sequence-refine") if args.len()==5 => crate::structural_lab::sequence_lab::refine(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("sequence-batch") if args.len()==5 => crate::structural_lab::sequence_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("sequence-compression-probe") if args.len()==3 => crate::structural_lab::sequence_lab::compression_probe(Path::new(&args[1]),Path::new(&args[2])),
        Some("compact-train") if args.len()==5 => crate::structural_lab::sequence_lab::compact_train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("compact-refine") if args.len()==5 => crate::structural_lab::sequence_lab::compact_refine(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("compact-batch") if args.len()==5 => crate::structural_lab::sequence_lab::compact_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("enhanced-train") if args.len()==5 => crate::structural_lab::sequence_lab::enhanced_train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("enhanced-refine") if args.len()==5 => crate::structural_lab::sequence_lab::enhanced_refine(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("enhanced-batch") if args.len()==5 => crate::structural_lab::sequence_lab::enhanced_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("factorized-train") if args.len()==5 => crate::structural_lab::sequence_lab::factorized_train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("factorized-refine") if args.len()==5 => crate::structural_lab::sequence_lab::factorized_refine(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("factorized-batch") if args.len()==5 => crate::structural_lab::sequence_lab::factorized_batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("dependency-train") if args.len()==5 => crate::structural_lab::dependency_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("dependency-batch") if args.len()==5 => crate::structural_lab::dependency_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-network-train") if args.len()==5 => crate::structural_lab::relation_network_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-network-batch") if args.len()==5 => crate::structural_lab::relation_network_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("grammar-train") if args.len()==5 => crate::structural_lab::grammar_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("grammar-batch") if args.len()==5 => crate::structural_lab::grammar_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-train") if args.len()==5 => crate::structural_lab::relation_lab::train(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-batch") if args.len()==5 => crate::structural_lab::relation_lab::batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-train-edit") if args.len()==5 => crate::structural_lab::relation_lab::train_aligned(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-batch-edit") if args.len()==5 => crate::structural_lab::relation_lab::batch_aligned(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-train-memory") if args.len()==5 => crate::structural_lab::relation_lab::train_memory(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-batch-memory") if args.len()==5 => crate::structural_lab::relation_lab::batch_memory(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-train-topology") if args.len()==5 => crate::structural_lab::relation_lab::train_topology(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("relation-batch-topology") if args.len()==5 => crate::structural_lab::relation_lab::batch_topology(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("batch") if args.len()==5 => batch(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("ask") if args.len()==4 => {
            let auth=Authority::from([Effect::ReadData,Effect::WriteData]);
            let genome=Genome::open_number(Path::new(&args[1]),&auth)?;
            let mut birth=genome.birth(Path::new(&args[2]),BirthLimits::default(),&auth)?;
            let input:String=serde_json::from_str(&args[3])?;
            let answer=route(birth.controller.runtime_mut(),&input,&auth)?;
            println!("{}",serde_json::json!({"answer":answer,"loads":birth.controller.runtime().stats().loads,"active_modules":birth.controller.runtime().stats().active_modules,"training_required":false}));Ok(())
        }
        Some("report") if args.len()==5 => crate::curriculum_integration::report(Path::new(&args[1]),Path::new(&args[2]),Path::new(&args[3]),Path::new(&args[4])),
        Some("exercise") if args.len()==2 => crate::curriculum_integration::exercise(Path::new(&args[1])),
        Some("run") if args.len()==4 => run(Path::new(&args[1]),Path::new(&args[2]),args[3].parse()?),
        Some("predict") if args.len()==5 => {
            let auth=Authority::from([Effect::ReadData,Effect::WriteData]);
            let genome=Genome::open_number(Path::new(&args[1]),&auth)?;
            let mut birth=genome.birth(Path::new(&args[2]),BirthLimits::default(),&auth)?;
            let input:Value=serde_json::from_str(&args[4])?;
            let ctx=birth.controller.runtime().catalog().get(&args[3]).ok_or("unknown stage")?.context.clone();
            let mut answer=None;
            birth.controller.runtime_mut().with_module(&args[3],&ctx,&auth,|module|{answer=predict(module,&input);Ok(())})?;
            println!("{}",serde_json::json!({"answer":answer,"training_required":false,"personal_episodes":0}));Ok(())
        }
        _=>Err("usage: axon-uic-curriculum-lab run <prepared.json> <existing-output> <rng> | predict <page.axnum> <new-birth-directory> <stage-id> <JSON-input>".into()),
    }
}
