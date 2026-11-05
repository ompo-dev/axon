//! Bounded example-driven composition of existing modules, not a general cognitive engine.
use super::*;
use crate::{behavior::Resettable, reactive::ReactiveCursor};
use std::{cmp::Reverse, collections::BinaryHeap};

const MAGIC: &[u8; 8] = b"AXGP0001";
const FILE_LIMIT: u64 = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoalExample {
    pub input: String,
    pub output: i64,
}
pub type GoalTrace = Vec<GoalExample>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompositionBudget {
    pub max_steps: usize,
    pub max_candidates: usize,
    pub max_work: u64,
    pub max_cost: u64,
}
impl Default for CompositionBudget {
    fn default() -> Self {
        Self {
            max_steps: 6,
            max_candidates: 4096,
            max_work: 100_000,
            max_cost: 256,
        }
    }
}
impl CompositionBudget {
    pub fn validate(self) -> Result<(), DemandError> {
        if !(1..=8).contains(&self.max_steps)
            || !(1..=4096).contains(&self.max_candidates)
            || !(1..=1_000_000).contains(&self.max_work)
            || !(1..=10000).contains(&self.max_cost)
        {
            return Err(DemandError::Invalid("composition budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct CompositionStatistics {
    pub candidates: usize,
    /// Candidate removals and executed stages, not CPU instructions or scalar-search work.
    pub work: u64,
}
impl CompositionStatistics {
    fn charge(&mut self, limit: u64) -> Result<(), DemandError> {
        if self.work >= limit {
            return Err(DemandError::Budget("composition work"));
        }
        self.work += 1;
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositionStop {
    Admitted,
    Refuted,
    Exhausted,
}
pub struct CompositionRun {
    pub procedure: Option<Procedure>,
    pub stop: CompositionStop,
    pub statistics: CompositionStatistics,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcedureStep {
    Module { id: String, identity: u64 },
    DecodeScalars,
    ExactInteger,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Text,
    Scalars,
    Scalar,
    Integer,
}
enum Value {
    Text(String),
    Scalars(Vec<f64>),
    Scalar(f64),
    Integer(i64),
}
#[derive(Clone, Debug)]
pub struct Procedure {
    identity: u64,
    context: String,
    steps: Vec<ProcedureStep>,
    training: Vec<GoalTrace>,
    admission: Vec<GoalTrace>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcedureCursor {
    identity: u64,
    states: Vec<(String, ReactiveCursor)>,
    suspended: bool,
    frames: u64,
}
impl ProcedureCursor {
    pub fn frames(&self) -> u64 {
        self.frames
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.states.capacity() * size_of::<(String, ReactiveCursor)>()
            + self
                .states
                .iter()
                .map(|(id, _)| id.capacity())
                .sum::<usize>()
    }
}
fn evidence(traces: &[GoalTrace]) -> Result<(), DemandError> {
    if traces.is_empty()
        || traces.len() > 8
        || traces.iter().map(Vec::len).sum::<usize>() > 128
        || traces.iter().any(|t| {
            t.is_empty()
                || t.len() > 32
                || t.iter().any(|e| {
                    e.input.is_empty()
                        || e.input.len() > 1024
                        || e.input.contains(['\n', '\r', '\0'])
                })
        })
    {
        return Err(DemandError::Invalid("goal evidence bounds"));
    }
    Ok(())
}
fn signature(step: &ProcedureStep, catalog: &Catalog) -> Result<(Kind, Kind), DemandError> {
    Ok(match step {
        ProcedureStep::DecodeScalars => (Kind::Text, Kind::Scalars),
        ProcedureStep::ExactInteger => (Kind::Scalar, Kind::Integer),
        ProcedureStep::Module { id, identity } => {
            let e = catalog
                .get(id)
                .ok_or_else(|| DemandError::Missing(id.clone()))?;
            if e.identity != *identity {
                return Err(DemandError::Identity);
            }
            match e.kind {
                ModuleKind::Text => (Kind::Text, Kind::Text),
                ModuleKind::Evolved | ModuleKind::Numeric => (Kind::Scalars, Kind::Scalar),
                ModuleKind::Reactive => (Kind::Integer, Kind::Integer),
                _ => return Err(DemandError::Invalid("non-transform module in procedure")),
            }
        }
    })
}
fn execute(
    steps: &[ProcedureStep],
    runtime: &mut DemandRuntime,
    input: &str,
    cursor: &mut ProcedureCursor,
    authority: &Authority,
    stats: &mut CompositionStatistics,
    limit: u64,
) -> Result<i64, DemandError> {
    require(authority, Effect::ReadData)?;
    if cursor.suspended {
        return Err(DemandError::Suspended);
    }
    if input.is_empty() || input.len() > 1024 || input.contains(['\n', '\r', '\0']) {
        return Err(DemandError::Invalid("goal input"));
    }
    let mut next = cursor.clone();
    let mut value = Value::Text(input.into());
    for step in steps {
        stats.charge(limit)?;
        value = match (step, value) {
            (ProcedureStep::DecodeScalars, Value::Text(text)) => {
                let mut numbers = Vec::new();
                for part in text.split(',') {
                    if numbers.len() >= 4 {
                        return Err(DemandError::Invalid("scalar arity"));
                    }
                    let v: f64 = part
                        .trim()
                        .parse()
                        .map_err(|_| DemandError::Invalid("scalar input"))?;
                    if !v.is_finite() {
                        return Err(DemandError::Invalid("finite scalar"));
                    }
                    numbers.push(v);
                }
                Value::Scalars(numbers)
            }
            (ProcedureStep::ExactInteger, Value::Scalar(v)) => {
                if !v.is_finite() || v.fract() != 0.0 || v.abs() > 9_007_199_254_740_992.0 {
                    return Err(DemandError::Invalid("exact integer bridge"));
                }
                Value::Integer(v as i64)
            }
            (ProcedureStep::Module { id, identity }, value) => {
                let e = runtime
                    .catalog()
                    .get(id)
                    .ok_or_else(|| DemandError::Missing(id.clone()))?;
                if e.identity != *identity {
                    return Err(DemandError::Identity);
                }
                let context = e.context.clone();
                runtime.with_module(id, &context, authority, |module| match (module, value) {
                    (Module::Text(s), Value::Text(text)) => {
                        match s.predict_value(&context, &text).map_err(storage)? {
                            TextValuePrediction::Agreed(v) => Ok(Value::Text(v)),
                            _ => Err(DemandError::NeedsExample {
                                row: 0,
                                module: id.clone(),
                                input: text,
                            }),
                        }
                    }
                    (Module::Evolved(s), Value::Scalars(v)) => s
                        .predict(&context, &v)
                        .map(Value::Scalar)
                        .map_err(|e| DemandError::Module(e.to_string())),
                    (Module::Numeric(s), Value::Scalars(v)) => s
                        .predict(&context, &v)
                        .map(Value::Scalar)
                        .map_err(|e| DemandError::Module(e.to_string())),
                    (Module::Reactive(pair), Value::Integer(v)) => {
                        let i = if let Some(i) = next.states.iter().position(|(key, _)| key == id) {
                            i
                        } else {
                            next.states.push((id.clone(), pair.1.cursor()));
                            next.states.len() - 1
                        };
                        pair.1
                            .step(&mut next.states[i].1, &context, &[v])
                            .map(Value::Integer)
                            .map_err(|e| DemandError::Module(e.to_string()))
                    }
                    _ => Err(DemandError::Invalid("procedure port type")),
                })?
            }
            _ => return Err(DemandError::Invalid("procedure bridge type")),
        };
    }
    let Value::Integer(output) = value else {
        return Err(DemandError::Invalid("procedure output type"));
    };
    next.frames = next
        .frames
        .checked_add(1)
        .ok_or(DemandError::Budget("goal frames"))?;
    *cursor = next;
    Ok(output)
}
fn fits(
    steps: &[ProcedureStep],
    traces: &[GoalTrace],
    runtime: &mut DemandRuntime,
    authority: &Authority,
    stats: &mut CompositionStatistics,
    limit: u64,
) -> Result<bool, DemandError> {
    for trace in traces {
        let mut cursor = ProcedureCursor {
            identity: 0,
            states: Vec::new(),
            suspended: false,
            frames: 0,
        };
        for row in trace {
            match execute(
                steps,
                runtime,
                &row.input,
                &mut cursor,
                authority,
                stats,
                limit,
            ) {
                Ok(value) if value == row.output => {}
                Err(e @ DemandError::Budget(_))
                | Err(e @ DemandError::Unauthorized(_))
                | Err(e @ DemandError::Identity)
                | Err(e @ DemandError::Io(_)) => return Err(e),
                _ => return Ok(false),
            }
        }
    }
    Ok(true)
}
impl Procedure {
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn steps(&self) -> &[ProcedureStep] {
        &self.steps
    }
    pub fn cursor(&self) -> ProcedureCursor {
        ProcedureCursor {
            identity: self.fingerprint(),
            states: Vec::new(),
            suspended: false,
            frames: 0,
        }
    }
    pub fn discover(
        context: &str,
        training: Vec<GoalTrace>,
        admission: Vec<GoalTrace>,
        runtime: &mut DemandRuntime,
        authority: &Authority,
        budget: CompositionBudget,
    ) -> Result<CompositionRun, DemandError> {
        name(context)?;
        evidence(&training)?;
        evidence(&admission)?;
        budget.validate()?;
        require(authority, Effect::ReadData)?;
        if training.iter().any(|a| admission.contains(a)) {
            return Err(DemandError::Invalid("overlapping goal traces"));
        }
        let mut choices = vec![
            (ProcedureStep::DecodeScalars, 1),
            (ProcedureStep::ExactInteger, 1),
        ];
        for e in runtime.catalog().entries() {
            if matches!(
                e.kind,
                ModuleKind::Text | ModuleKind::Evolved | ModuleKind::Reactive | ModuleKind::Numeric
            ) && e.retained_bytes <= runtime.seed().active_bytes
            {
                choices.push((
                    ProcedureStep::Module {
                        id: e.id.clone(),
                        identity: e.identity,
                    },
                    u64::from(e.cost),
                ));
            }
        }
        if choices.len() > 18 {
            return Err(DemandError::Budget("composition library"));
        }
        let mut frontier =
            BinaryHeap::from([Reverse((0u64, Vec::<ProcedureStep>::new(), Kind::Text))]);
        let mut statistics = CompositionStatistics::default();
        while let Some(Reverse((cost, steps, kind))) = frontier.pop() {
            if statistics.candidates >= budget.max_candidates {
                break;
            }
            statistics.charge(budget.max_work)?;
            statistics.candidates += 1;
            if kind == Kind::Integer
                && fits(
                    &steps,
                    &training,
                    runtime,
                    authority,
                    &mut statistics,
                    budget.max_work,
                )?
            {
                let admitted = fits(
                    &steps,
                    &admission,
                    runtime,
                    authority,
                    &mut statistics,
                    budget.max_work,
                )?;
                let procedure = if admitted {
                    let mut p = Self {
                        identity: 0,
                        context: context.into(),
                        steps,
                        training,
                        admission,
                    };
                    p.validate(runtime.catalog())?;
                    p.identity = checksum(&p.to_bytes());
                    Some(p)
                } else {
                    None
                };
                return Ok(CompositionRun {
                    procedure,
                    stop: if admitted {
                        CompositionStop::Admitted
                    } else {
                        CompositionStop::Refuted
                    },
                    statistics,
                });
            }
            if steps.len() >= budget.max_steps {
                continue;
            }
            for (step, charge) in &choices {
                if steps.contains(step) || signature(step, runtime.catalog())?.0 != kind {
                    continue;
                }
                let next_cost = cost
                    .checked_add(*charge)
                    .ok_or(DemandError::Budget("composition cost"))?;
                if next_cost > budget.max_cost {
                    continue;
                }
                if frontier.len() >= 4096 {
                    return Err(DemandError::Budget("composition frontier"));
                }
                let mut next = steps.clone();
                next.push(step.clone());
                frontier.push(Reverse((
                    next_cost,
                    next,
                    signature(step, runtime.catalog())?.1,
                )));
            }
        }
        Ok(CompositionRun {
            procedure: None,
            stop: CompositionStop::Exhausted,
            statistics,
        })
    }
    fn validate(&self, catalog: &Catalog) -> Result<(), DemandError> {
        name(&self.context)?;
        evidence(&self.training)?;
        evidence(&self.admission)?;
        if self.steps.is_empty()
            || self.steps.len() > 8
            || self.to_bytes().len() as u64 > FILE_LIMIT
            || self.training.iter().any(|t| self.admission.contains(t))
        {
            return Err(DemandError::Invalid("procedure schema/evidence"));
        }
        let mut kind = Kind::Text;
        for (i, step) in self.steps.iter().enumerate() {
            let (input, output) = signature(step, catalog)?;
            if input != kind || self.steps[..i].contains(step) {
                return Err(DemandError::Invalid("procedure topology"));
            }
            kind = output;
        }
        if kind != Kind::Integer {
            return Err(DemandError::Invalid("procedure readout"));
        }
        Ok(())
    }
    pub fn predict(
        &self,
        context: &str,
        runtime: &mut DemandRuntime,
        cursor: &mut ProcedureCursor,
        input: &str,
        authority: &Authority,
    ) -> Result<i64, DemandError> {
        if context != self.context {
            return Err(DemandError::Context);
        }
        if cursor.identity != self.fingerprint() {
            return Err(DemandError::Identity);
        }
        execute(
            &self.steps,
            runtime,
            input,
            cursor,
            authority,
            &mut CompositionStatistics::default(),
            8,
        )
    }
    pub fn replay(
        &self,
        runtime: &mut DemandRuntime,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        self.validate(runtime.catalog())?;
        let mut stats = CompositionStatistics::default();
        if !fits(
            &self.steps,
            &self.training,
            runtime,
            authority,
            &mut stats,
            4096,
        )? || !fits(
            &self.steps,
            &self.admission,
            runtime,
            authority,
            &mut stats,
            4096,
        )? {
            return Err(DemandError::Invalid("procedure evidence replay"));
        }
        Ok(())
    }
    pub fn to_bytes(&self) -> Vec<u8> {
        fn text(b: &mut Vec<u8>, s: &str) {
            b.extend((s.len() as u16).to_le_bytes());
            b.extend(s.as_bytes());
        }
        let mut b = MAGIC.to_vec();
        text(&mut b, &self.context);
        b.push(self.steps.len() as u8);
        for step in &self.steps {
            match step {
                ProcedureStep::Module { id, identity } => {
                    b.push(0);
                    text(&mut b, id);
                    b.extend(identity.to_le_bytes());
                }
                ProcedureStep::DecodeScalars => b.push(1),
                ProcedureStep::ExactInteger => b.push(2),
            }
        }
        for traces in [&self.training, &self.admission] {
            b.push(traces.len() as u8);
            for trace in traces {
                b.push(trace.len() as u8);
                for row in trace {
                    text(&mut b, &row.input);
                    b.extend(row.output.to_le_bytes());
                }
            }
        }
        b.extend(checksum(&b).to_le_bytes());
        b
    }
    pub fn fingerprint(&self) -> u64 {
        self.identity
    }
    pub fn save_new(&self, path: &Path, authority: &Authority) -> Result<(), DemandError> {
        require(authority, Effect::WriteData)?;
        if self.to_bytes().len() as u64 > FILE_LIMIT {
            return Err(DemandError::Budget("procedure bytes"));
        }
        publish_new(path, &self.to_bytes()).map_err(storage)
    }
    pub fn open(
        path: &Path,
        context: &str,
        runtime: &mut DemandRuntime,
        authority: &Authority,
    ) -> Result<Self, DemandError> {
        require(authority, Effect::ReadData)?;
        let p = Self::from_bytes(
            &read_bounded(path, FILE_LIMIT).map_err(storage)?,
            runtime.catalog(),
        )?;
        if p.context != context {
            return Err(DemandError::Context);
        }
        p.replay(runtime, authority)?;
        Ok(p)
    }
    pub fn from_bytes(b: &[u8], catalog: &Catalog) -> Result<Self, DemandError> {
        if b.len() < 16
            || b.len() as u64 > FILE_LIMIT
            || &b[..8] != MAGIC
            || checksum(&b[..b.len() - 8])
                != u64::from_le_bytes(b[b.len() - 8..].try_into().expect("width"))
        {
            return Err(DemandError::Invalid("procedure header/checksum"));
        }
        let mut r = Reader(&b[8..b.len() - 8]);
        let context = r.text()?;
        let count = r.take(1)?[0] as usize;
        if count > 8 {
            return Err(DemandError::Budget("procedure steps"));
        }
        let mut steps = Vec::new();
        for _ in 0..count {
            steps.push(match r.take(1)?[0] {
                0 => ProcedureStep::Module {
                    id: r.text()?,
                    identity: r.u64()?,
                },
                1 => ProcedureStep::DecodeScalars,
                2 => ProcedureStep::ExactInteger,
                _ => return Err(DemandError::Invalid("procedure opcode")),
            });
        }
        let mut traces = || -> Result<Vec<GoalTrace>, DemandError> {
            let count = r.take(1)?[0] as usize;
            if count > 8 {
                return Err(DemandError::Budget("goal traces"));
            }
            let mut traces = Vec::new();
            for _ in 0..count {
                let rows = r.take(1)?[0] as usize;
                if rows > 32 {
                    return Err(DemandError::Budget("goal rows"));
                }
                let mut trace = Vec::new();
                for _ in 0..rows {
                    let n = r.u16()? as usize;
                    if n > 1024 {
                        return Err(DemandError::Budget("goal input bytes"));
                    }
                    let input = String::from_utf8(r.take(n)?.to_vec())
                        .map_err(|_| DemandError::Invalid("goal UTF-8"))?;
                    trace.push(GoalExample {
                        input,
                        output: r.u64()? as i64,
                    });
                }
                traces.push(trace);
            }
            Ok(traces)
        };
        let training = traces()?;
        let admission = traces()?;
        if !r.0.is_empty() {
            return Err(DemandError::Invalid("procedure trailing data"));
        }
        let mut p = Self {
            identity: 0,
            context,
            steps,
            training,
            admission,
        };
        p.validate(catalog)?;
        p.identity = checksum(&p.to_bytes());
        Ok(p)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GoalDecision {
    pub id: u64,
    pub value: i64,
    pub procedure: u64,
}
struct GoalState {
    procedure: Procedure,
    cursor: ProcedureCursor,
}
struct Pending {
    context: String,
    decision: GoalDecision,
    dispatched: bool,
}
pub struct GoalController {
    runtime: DemandRuntime,
    goals: Vec<GoalState>,
    pending: Vec<Pending>,
    last_id: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchStatus {
    Reached,
    Unreachable,
    Refuted,
    Uncertain,
}
#[derive(Debug)]
pub struct DispatchReceipt {
    pub decision: u64,
    pub goal: u16,
    pub attempted: Vec<u8>,
    pub observations: Vec<(u8, u16)>,
    pub status: DispatchStatus,
}
impl GoalController {
    pub fn new(runtime: DemandRuntime) -> Self {
        Self {
            runtime,
            goals: Vec::new(),
            pending: Vec::new(),
            last_id: 0,
        }
    }
    pub fn runtime(&self) -> &DemandRuntime {
        &self.runtime
    }
    pub fn runtime_mut(&mut self) -> &mut DemandRuntime {
        &mut self.runtime
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    /// Existing goal cursors survive only when all referenced module identities still match.
    pub fn rebind(
        &mut self,
        mut runtime: DemandRuntime,
        authority: &Authority,
    ) -> Result<(), DemandError> {
        for goal in &self.goals {
            goal.procedure.replay(&mut runtime, authority)?;
        }
        self.runtime = runtime;
        Ok(())
    }
    pub fn install(&mut self, p: Procedure, authority: &Authority) -> Result<(), DemandError> {
        if self.goals.len() >= 8 || self.goals.iter().any(|s| s.procedure.context == p.context) {
            return Err(DemandError::Invalid("goal slots/context"));
        }
        p.replay(&mut self.runtime, authority)?;
        let cursor = p.cursor();
        self.goals.push(GoalState {
            procedure: p,
            cursor,
        });
        Ok(())
    }
    pub fn propose(
        &mut self,
        context: &str,
        id: u64,
        input: &str,
        authority: &Authority,
    ) -> Result<GoalDecision, DemandError> {
        if id <= self.last_id || self.pending.len() >= 64 {
            return Err(DemandError::Budget("goal IDs/pending feedback"));
        }
        let state = self
            .goals
            .iter_mut()
            .find(|g| g.procedure.context == context)
            .ok_or(DemandError::Context)?;
        let value = state.procedure.predict(
            context,
            &mut self.runtime,
            &mut state.cursor,
            input,
            authority,
        )?;
        let decision = GoalDecision {
            id,
            value,
            procedure: state.procedure.fingerprint(),
        };
        self.pending.push(Pending {
            context: context.into(),
            decision,
            dispatched: false,
        });
        self.last_id = id;
        Ok(decision)
    }
    /// Delayed feedback is tied to the original goal and prediction; mismatch quarantines that goal.
    pub fn feedback(&mut self, context: &str, id: u64, actual: i64) -> Result<bool, DemandError> {
        let i = self
            .pending
            .iter()
            .position(|p| p.context == context && p.decision.id == id)
            .ok_or(DemandError::Context)?;
        let decision = self.pending.remove(i).decision;
        if decision.value != actual {
            self.goals
                .iter_mut()
                .find(|g| g.procedure.context == context)
                .expect("installed")
                .cursor
                .suspended = true;
            self.pending.retain(|p| p.context != context);
            return Ok(false);
        }
        Ok(true)
    }
    /// Host-supplied resettable adapters are permitted only under explicit simulation authority.
    /// A failed external step is not rolled back or silently retried.
    pub fn dispatch<A: Resettable>(
        &mut self,
        decision: u64,
        device: (&str, &str),
        cursor: &mut DormantCursor,
        adapter: &mut A,
        forbidden: &BTreeSet<u16>,
        authority: &Authority,
    ) -> Result<DispatchReceipt, DemandError> {
        require(authority, Effect::Simulate)?;
        let pending = self
            .pending
            .iter()
            .position(|p| p.decision.id == decision)
            .ok_or(DemandError::Context)?;
        let request = &self.pending[pending];
        if request.dispatched {
            return Err(DemandError::Invalid("decision already dispatched"));
        }
        if self
            .goals
            .iter()
            .find(|g| g.procedure.context == request.context)
            .expect("installed")
            .cursor
            .suspended
        {
            return Err(DemandError::Suspended);
        }
        let goal = u16::try_from(request.decision.value)
            .map_err(|_| DemandError::Invalid("finite device goal"))?;
        let mut receipt = DispatchReceipt {
            decision,
            goal,
            attempted: Vec::new(),
            observations: Vec::new(),
            status: DispatchStatus::Unreachable,
        };
        let Some(actions) = self
            .runtime
            .plan_behavior(device.0, device.1, cursor, goal, forbidden, authority)?
        else {
            return Ok(receipt);
        };
        if actions.len() > 48 {
            return Err(DemandError::Budget("dispatch actions"));
        }
        self.pending[pending].dispatched = true;
        for action in actions {
            receipt.attempted.push(action);
            let output = match adapter.step(action) {
                Ok(v) => v,
                Err(_) => {
                    cursor.suspended = true;
                    receipt.status = DispatchStatus::Uncertain;
                    return Ok(receipt);
                }
            };
            receipt.observations.push((action, output));
            if let Err(error) = self
                .runtime
                .observe_behavior(device.0, device.1, cursor, action, output, authority)
            {
                cursor.suspended = true;
                receipt.status = if matches!(error, DemandError::Drift { .. }) {
                    DispatchStatus::Refuted
                } else {
                    DispatchStatus::Uncertain
                };
                return Ok(receipt);
            }
        }
        receipt.status = DispatchStatus::Reached;
        Ok(receipt)
    }
}
