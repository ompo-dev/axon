//! Typed, read-only planning over live files. Learned role extraction/shortfall; provided integer optimizer.
use crate::{
    Authority, Effect, EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue,
    cumulative::StudyError,
    demand::{DemandError, DemandRuntime, Module, ModuleKind, genome::bounded_path},
    live_store::checksum,
    text_skill::read_bounded,
};
use microlp::{
    ComparisonOp, OptimizationDirection, Problem, SolutionStatus, SolveOptions, SolveOutcome,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::Path,
    time::Duration,
};

#[derive(Debug)]
pub enum CaseError {
    Invalid(&'static str),
    Clarify(&'static str),
    Stale,
    Infeasible,
    Unproven,
    Demand(DemandError),
    Study(StudyError),
    Storage(crate::TextSkillError),
    Episode(crate::EpisodeError),
    Io(std::io::Error),
    Optimizer(microlp::Error),
}
impl fmt::Display for CaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid case: {s}"),
            Self::Clarify(s) => write!(f, "clarification required: {s}"),
            Self::Stale => f.write_str("input files changed; no plan committed"),
            Self::Infeasible => {
                f.write_str("no feasible purchase plan within supplied constraints")
            }
            Self::Unproven => f.write_str("optimizer quota ended without an optimality result"),
            Self::Demand(e) => e.fmt(f),
            Self::Study(e) => e.fmt(f),
            Self::Storage(e) => e.fmt(f),
            Self::Episode(e) => e.fmt(f),
            Self::Io(e) => e.fmt(f),
            Self::Optimizer(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for CaseError {}
impl From<DemandError> for CaseError {
    fn from(e: DemandError) -> Self {
        Self::Demand(e)
    }
}
impl From<StudyError> for CaseError {
    fn from(e: StudyError) -> Self {
        Self::Study(e)
    }
}
impl From<crate::TextSkillError> for CaseError {
    fn from(e: crate::TextSkillError) -> Self {
        Self::Storage(e)
    }
}
impl From<crate::EpisodeError> for CaseError {
    fn from(e: crate::EpisodeError) -> Self {
        Self::Episode(e)
    }
}
impl From<std::io::Error> for CaseError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Choice {
    pub supplier: String,
    pub packs: u64,
}
#[derive(Clone, Debug)]
pub struct PlanReceipt {
    pub item: String,
    pub needed: i64,
    pub units: u64,
    pub cost_cents: u64,
    pub budget_cents: u64,
    pub choices: Vec<Choice>,
    pub source_stock: u64,
    pub source_offers: u64,
    pub model: u64,
    pub frame: (u64, u64),
    pub episode: u64,
    pub duplicate: bool,
    pub nodes: u64,
    pub iterations: u64,
}
struct Offer {
    supplier: String,
    units: u64,
    price: u64,
    available: u64,
}
pub(crate) fn number(s: &str) -> Result<u64, CaseError> {
    let n = s
        .parse::<u64>()
        .map_err(|_| CaseError::Invalid("nonnegative integer field"))?;
    if n > 1000000 {
        return Err(CaseError::Invalid("integer range"));
    }
    Ok(n)
}
pub(crate) fn rows(bytes: &[u8], headers: &[&str]) -> Result<Vec<Vec<String>>, CaseError> {
    let mut csv = csv::ReaderBuilder::new().from_reader(bytes);
    if csv
        .headers()
        .map_err(|_| CaseError::Invalid("CSV header"))?
        .iter()
        .ne(headers.iter().copied())
    {
        return Err(CaseError::Clarify("file columns/unit contract"));
    }
    let mut rows = Vec::new();
    for row in csv.records() {
        let row = row.map_err(|_| CaseError::Invalid("CSV record"))?;
        if rows.len() == 128
            || row
                .iter()
                .any(|s| s.len() > 128 || s.trim() != s || s.chars().any(char::is_control))
        {
            return Err(CaseError::Invalid("CSV bounds"));
        }
        rows.push(row.iter().map(str::to_owned).collect());
    }
    Ok(rows)
}
fn scoped(root: &Path, name: &str) -> Result<std::path::PathBuf, CaseError> {
    let path = root.join(name);
    bounded_path(&path)?;
    let path = std::fs::canonicalize(path)?;
    if !path.starts_with(root) {
        return Err(CaseError::Invalid("file outside scope"));
    }
    Ok(path)
}
fn require(a: &Authority, e: Effect) -> Result<(), CaseError> {
    if a.allows(e) {
        Ok(())
    } else {
        Err(DemandError::Unauthorized(e).into())
    }
}
pub fn plan_files(
    runtime: &mut DemandRuntime,
    root: &Path,
    sentence: &str,
    stamp: EpisodeStamp,
    a: &Authority,
) -> Result<PlanReceipt, CaseError> {
    plan_files_observed(runtime, root, sentence, stamp, a, || {})
}
/// The host hook is for observed checkpoints; it is not untrusted generated code.
pub fn plan_files_observed(
    runtime: &mut DemandRuntime,
    root: &Path,
    sentence: &str,
    stamp: EpisodeStamp,
    a: &Authority,
    before_commit: impl FnOnce(),
) -> Result<PlanReceipt, CaseError> {
    require(a, Effect::ReadData)?;
    require(a, Effect::WriteData)?;
    bounded_path(root)?;
    let root = std::fs::canonicalize(root)?;
    let study = runtime
        .catalog()
        .entries()
        .iter()
        .filter(|e| e.kind == ModuleKind::Study)
        .collect::<Vec<_>>();
    if study.len() != 1 {
        return Err(CaseError::Clarify("one explicit study binding required"));
    }
    let study_id = study[0].id.clone();
    let study_context = study[0].context.clone();
    let parsed=runtime.with_module(&study_id,&study_context,a,|module|{
        let Module::Study(seed)=module else{return Err(DemandError::Invalid("study module"));};
        let candidates=seed.frames().iter().filter(|f|f.accepts(sentence)).collect::<Vec<_>>();
        if candidates.len()!=1{return Err(DemandError::Invalid("sentence requires clarification"));}
        let frame=candidates[0];let item=frame.read_word(sentence).map_err(|_|DemandError::Invalid("item extraction"))?;let values=frame.read_values(sentence).map_err(|_|DemandError::Invalid("quantity extraction"))?;Ok((item,values,frame.identities()))
    }).map_err(|e|if matches!(e,DemandError::Invalid(_)){CaseError::Clarify("item, kit quantity, pieces per kit, budget in cents; use a demonstrated sentence shape")}else{e.into()})?;
    let (item, values, frame) = parsed;
    if item.is_empty()
        || item.len() > 64
        || values.len() != 3
        || values
            .iter()
            .any(|v| v.fract() != 0. || *v < 0. || *v > 1000000.)
    {
        return Err(CaseError::Clarify(
            "bounded nonnegative integer quantities and cents",
        ));
    }
    let stock_path = scoped(&root, "stock.csv")?;
    let offers_path = scoped(&root, "offers.csv")?;
    let stock_bytes = read_bounded(&stock_path, 1048576)?;
    let offer_bytes = read_bounded(&offers_path, 1048576)?;
    let stock_rows = rows(&stock_bytes, &["item", "quantity", "unit"])?;
    let mut stock = None;
    for row in stock_rows {
        if row[0] == item {
            if row[2] != "pecas" {
                return Err(CaseError::Clarify(
                    "stock unit must match demonstrated pieces",
                ));
            }
            if stock.replace(number(&row[1])?).is_some() {
                return Err(CaseError::Clarify(
                    "conflicting/duplicate stock observations",
                ));
            }
        }
    }
    let stock = stock.ok_or(CaseError::Clarify("stock observation for item"))?;
    let mut offers = Vec::new();
    let mut suppliers = BTreeSet::new();
    for row in rows(
        &offer_bytes,
        &[
            "supplier",
            "item",
            "units",
            "price_cents",
            "available",
            "unit",
        ],
    )? {
        if row[1] == item {
            if row[5] != "pecas" || !suppliers.insert(row[0].clone()) || row[0].is_empty() {
                return Err(CaseError::Clarify("offer units or supplier identity"));
            }
            let units = number(&row[2])?;
            if units == 0 {
                return Err(CaseError::Invalid("empty pack"));
            }
            offers.push(Offer {
                supplier: row[0].clone(),
                units,
                price: number(&row[3])?,
                available: number(&row[4])?,
            });
        }
    }
    if offers.is_empty() || offers.len() > 16 {
        return Err(CaseError::Clarify("1..16 supplied offers for item"));
    }
    let models = runtime
        .catalog()
        .entries()
        .iter()
        .filter(|e| {
            e.kind == ModuleKind::Numeric
                && e.input == "casework/count,count,count"
                && e.output == "casework/shortfall"
        })
        .collect::<Vec<_>>();
    if models.len() != 1 {
        return Err(CaseError::Clarify("shortfall learned-model binding"));
    }
    let id = models[0].id.clone();
    let context = models[0].context.clone();
    let (needed, model) = runtime.with_module(&id, &context, a, |module| {
        let Module::Numeric(s) = module else {
            return Err(DemandError::Invalid("numeric binding"));
        };
        let value = s
            .predict(&context, &[values[0], values[1], stock as f64])
            .map_err(|e| DemandError::Module(e.to_string()))?;
        if value.fract() != 0. || value.abs() > 1000000000000. {
            return Err(DemandError::Invalid("shortfall integer domain"));
        }
        Ok((value as i64, s.fingerprint()))
    })?;
    let mut problem = Problem::new(OptimizationDirection::Minimize);
    let vars = offers
        .iter()
        .map(|o| problem.add_integer_var(o.price as f64, (0, o.available as i32)))
        .collect::<Vec<_>>();
    problem.add_constraint(
        vars.iter().zip(&offers).map(|(&v, o)| (v, o.units as f64)),
        ComparisonOp::Ge,
        needed as f64,
    );
    problem.add_constraint(
        vars.iter().zip(&offers).map(|(&v, o)| (v, o.price as f64)),
        ComparisonOp::Le,
        values[2],
    );
    let mut options = SolveOptions::default();
    options.time_limit = Some(Duration::from_secs(1));
    options.node_limit = Some(10000);
    let outcome = problem.solve_with(options).map_err(|e| {
        if matches!(e, microlp::Error::Infeasible) {
            CaseError::Infeasible
        } else {
            CaseError::Optimizer(e)
        }
    })?;
    let SolveOutcome::Solution(solution) = outcome else {
        return Err(CaseError::Unproven);
    };
    if solution.status() != SolutionStatus::Optimal {
        return Err(CaseError::Unproven);
    }
    let mut units = 0u64;
    let mut cost = 0u64;
    let mut choices = Vec::new();
    for (&var, o) in vars.iter().zip(&offers) {
        let n = solution.var_value(var);
        if !n.is_finite() || n.fract() != 0. || n < 0. || n > o.available as f64 {
            return Err(CaseError::Invalid("optimizer assignment"));
        }
        let n = n as u64;
        units = units
            .checked_add(
                n.checked_mul(o.units)
                    .ok_or(CaseError::Invalid("units overflow"))?,
            )
            .ok_or(CaseError::Invalid("units overflow"))?;
        cost = cost
            .checked_add(
                n.checked_mul(o.price)
                    .ok_or(CaseError::Invalid("cost overflow"))?,
            )
            .ok_or(CaseError::Invalid("cost overflow"))?;
        if n != 0 {
            choices.push(Choice {
                supplier: o.supplier.clone(),
                packs: n,
            });
        }
    }
    if (units as i128) < i128::from(needed)
        || cost > values[2] as u64
        || solution.objective() != cost as f64
    {
        return Err(CaseError::Invalid("exact post-solve constraints"));
    }
    before_commit();
    if scoped(&root, "stock.csv")? != stock_path || scoped(&root, "offers.csv")? != offers_path {
        return Err(CaseError::Stale);
    }
    if read_bounded(&stock_path, 1048576)? != stock_bytes
        || read_bounded(&offers_path, 1048576)? != offer_bytes
    {
        return Err(CaseError::Stale);
    }
    let journal = root.join("individual.axep");
    bounded_path(&journal)?;
    let (mut store, _) = EpisodeStore::open(journal)?;
    let mut fields = BTreeMap::from([
        ("item".into(), EpisodeValue::Text(item.clone())),
        ("needed".into(), EpisodeValue::Integer(needed)),
        ("units".into(), EpisodeValue::Integer(units as i64)),
        ("cost_cents".into(), EpisodeValue::Integer(cost as i64)),
        (
            "budget_cents".into(),
            EpisodeValue::Integer(values[2] as i64),
        ),
        (
            "stock".into(),
            EpisodeValue::Text(format!("{:016X}", checksum(&stock_bytes))),
        ),
        (
            "offers".into(),
            EpisodeValue::Text(format!("{:016X}", checksum(&offer_bytes))),
        ),
        ("model".into(), EpisodeValue::Text(format!("{model:016X}"))),
        (
            "frame".into(),
            EpisodeValue::Text(format!("{:016X}/{:016X}", frame.0, frame.1)),
        ),
    ]);
    for (i, c) in choices.iter().enumerate() {
        fields.insert(
            format!("choice-{i}"),
            EpisodeValue::Text(format!("{}:{}", c.supplier, c.packs)),
        );
    }
    let receipt = store.record(
        stamp,
        EpisodeData::Trace {
            context: "casework/v1".into(),
            channel: "plan".into(),
            fields,
        },
    )?;
    let stats = solution.stats();
    Ok(PlanReceipt {
        item,
        needed,
        units,
        cost_cents: cost,
        budget_cents: values[2] as u64,
        choices,
        source_stock: checksum(&stock_bytes),
        source_offers: checksum(&offer_bytes),
        model,
        frame,
        episode: receipt.id,
        duplicate: receipt.duplicate,
        nodes: stats.nodes_solved,
        iterations: stats.lp_iterations,
    })
}
