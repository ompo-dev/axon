//! Output projections share observations; their product is evaluated only when needed.
use crate::{
    Authority, Effect,
    behavior::{BehaviorError, BehaviorModel, Reply, SequenceOracle, Word},
    demand::{Catalog, DemandError, DemandRuntime, Module, ModuleKind, ModuleSpec, Seed},
    live_store::checksum,
    transfer::{TransferError, discover},
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt, fs,
    path::Path,
};

pub const MAX_FACTORS: usize = 8;
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Frame {
    dimensions: u8,
    values: [u16; MAX_FACTORS],
}
impl Frame {
    pub fn new(values: &[u16]) -> Result<Self, BehaviorError> {
        if values.is_empty() || values.len() > MAX_FACTORS {
            return Err(BehaviorError::Invalid("frame dimensions"));
        }
        let mut frame = Self {
            dimensions: values.len() as u8,
            values: [0; MAX_FACTORS],
        };
        frame.values[..values.len()].copy_from_slice(values);
        Ok(frame)
    }
    pub fn values(&self) -> &[u16] {
        &self.values[..self.dimensions as usize]
    }
}
pub type FrameReply = Vec<Frame>;
pub trait FrameOracle {
    fn query(&mut self, word: &[u8]) -> Result<FrameReply, BehaviorError>;
}
pub trait FrameResettable {
    fn reset(&mut self) -> Result<(), BehaviorError>;
    fn step(&mut self, action: u8) -> Result<Frame, BehaviorError>;
}
#[derive(Clone, Copy, Debug)]
pub struct FrameBudget {
    pub max_queries: usize,
    pub max_steps: usize,
    pub max_cache_nodes: usize,
}
impl Default for FrameBudget {
    fn default() -> Self {
        Self {
            max_queries: 32768,
            max_steps: 800000,
            max_cache_nodes: 200000,
        }
    }
}
/// Shared prefix cache; not an episodic journal or a process sandbox.
pub struct FrameCache<E> {
    environment: E,
    actions: u8,
    dimensions: usize,
    authority: Authority,
    budget: FrameBudget,
    cache: BTreeMap<Word, Frame>,
    queries: usize,
    steps: usize,
    suspended: bool,
}
impl<E: FrameResettable> FrameCache<E> {
    pub fn new(
        environment: E,
        actions: u8,
        dimensions: usize,
        budget: FrameBudget,
        authority: Authority,
    ) -> Result<Self, BehaviorError> {
        if !(1..=16).contains(&actions)
            || !(1..=MAX_FACTORS).contains(&dimensions)
            || budget.max_queries == 0
            || budget.max_queries > 32768
            || budget.max_steps == 0
            || budget.max_steps > 800000
            || budget.max_cache_nodes == 0
            || budget.max_cache_nodes > 200000
        {
            return Err(BehaviorError::Invalid("frame cache configuration"));
        }
        Ok(Self {
            environment,
            actions,
            dimensions,
            authority,
            budget,
            cache: BTreeMap::new(),
            queries: 0,
            steps: 0,
            suspended: false,
        })
    }
    pub fn queries(&self) -> usize {
        self.queries
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    pub fn cache_nodes(&self) -> usize {
        self.cache.len()
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    fn validate(&self, word: &[u8]) -> Result<(), BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        if word.len() > 48 || word.iter().any(|&a| a >= self.actions) {
            return Err(BehaviorError::Invalid("frame query word"));
        }
        Ok(())
    }
    pub fn fresh(&mut self, word: &[u8]) -> Result<FrameReply, BehaviorError> {
        self.validate(word)?;
        if word.is_empty() {
            return Ok(Vec::new());
        }
        if !self.authority.allows(Effect::Simulate) {
            return Err(BehaviorError::Unauthorized(Effect::Simulate));
        }
        let missing = (1..=word.len())
            .filter(|&i| !self.cache.contains_key(&word[..i]))
            .count();
        if self.queries >= self.budget.max_queries
            || self.steps + word.len() > self.budget.max_steps
            || self.cache.len() + missing > self.budget.max_cache_nodes
        {
            return Err(BehaviorError::Limit("frame experiments"));
        }
        self.queries += 1;
        if let Err(e) = self.environment.reset() {
            self.suspended = true;
            return Err(e);
        }
        let mut reply = Vec::with_capacity(word.len());
        for (i, &action) in word.iter().enumerate() {
            self.steps += 1;
            let actual = match self.environment.step(action) {
                Ok(frame) => frame,
                Err(e) => {
                    self.suspended = true;
                    return Err(e);
                }
            };
            if actual.values().len() != self.dimensions {
                self.suspended = true;
                return Err(BehaviorError::Invalid("frame reply shape"));
            }
            if self
                .cache
                .get(&word[..=i])
                .is_some_and(|&known| known != actual)
            {
                self.suspended = true;
                return Err(BehaviorError::NonDeterministic);
            }
            reply.push(actual);
        }
        for (i, &frame) in reply.iter().enumerate() {
            self.cache.insert(word[..=i].to_vec(), frame);
        }
        Ok(reply)
    }
}
impl<E: FrameResettable> FrameOracle for FrameCache<E> {
    fn query(&mut self, word: &[u8]) -> Result<FrameReply, BehaviorError> {
        self.validate(word)?;
        if let Some(reply) = (1..=word.len())
            .map(|i| self.cache.get(&word[..i]).copied())
            .collect::<Option<FrameReply>>()
        {
            return Ok(reply);
        }
        self.fresh(word)
    }
}

#[derive(Debug)]
pub enum FactorError {
    Invalid(&'static str),
    Limit(&'static str),
    Suspended,
    Drift {
        dimension: usize,
        expected: u16,
        actual: u16,
    },
    Behavior(BehaviorError),
    Transfer(TransferError),
    Demand(DemandError),
    Io(std::io::Error),
}
impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "factor model: {self:?}")
    }
}
impl std::error::Error for FactorError {}
impl From<BehaviorError> for FactorError {
    fn from(e: BehaviorError) -> Self {
        Self::Behavior(e)
    }
}
impl From<TransferError> for FactorError {
    fn from(e: TransferError) -> Self {
        Self::Transfer(e)
    }
}
impl From<DemandError> for FactorError {
    fn from(e: DemandError) -> Self {
        Self::Demand(e)
    }
}
impl From<std::io::Error> for FactorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

struct Projection<'a, O> {
    oracle: &'a mut O,
    dimensions: usize,
    dimension: usize,
}
impl<O: FrameOracle> SequenceOracle for Projection<'_, O> {
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        let frames = self.oracle.query(word)?;
        if frames.len() != word.len() || frames.iter().any(|f| f.values().len() != self.dimensions)
        {
            return Err(BehaviorError::Invalid("projected reply shape"));
        }
        Ok(frames.iter().map(|f| f.values[self.dimension]).collect())
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorModel {
    context: String,
    factors: Vec<BehaviorModel>,
}
impl FactorModel {
    pub fn new(context: &str, factors: Vec<BehaviorModel>) -> Result<Self, FactorError> {
        if context.is_empty()
            || context.len() > 220
            || context.trim() != context
            || context.chars().any(char::is_control)
            || factors.is_empty()
            || factors.len() > MAX_FACTORS
        {
            return Err(FactorError::Invalid("factor context/dimensions"));
        }
        let actions = factors[0].actions();
        for (i, factor) in factors.iter().enumerate() {
            if factor.actions() != actions || factor.context() != format!("{context}/factor-{i}") {
                return Err(FactorError::Invalid("factor interface/context"));
            }
        }
        Ok(Self {
            context: context.into(),
            factors,
        })
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn actions(&self) -> u8 {
        self.factors[0].actions()
    }
    pub fn dimensions(&self) -> usize {
        self.factors.len()
    }
    pub fn factors(&self) -> &[BehaviorModel] {
        &self.factors
    }
    pub fn factor_states(&self) -> usize {
        self.factors.iter().map(BehaviorModel::states).sum()
    }
    /// Upper bound; reachable/minimal joint states may be much fewer.
    pub fn product_states(&self) -> u64 {
        self.factors.iter().map(|m| m.states() as u64).product()
    }
    pub fn encoded_model_bytes(&self) -> usize {
        self.factors.iter().map(BehaviorModel::encoded_bytes).sum()
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(
            &self
                .factors
                .iter()
                .flat_map(|m| m.fingerprint().to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
    pub fn transition(
        &self,
        states: &[u16; MAX_FACTORS],
        action: u8,
    ) -> Result<([u16; MAX_FACTORS], Frame), FactorError> {
        let mut next = [0; MAX_FACTORS];
        let mut output = Frame {
            dimensions: self.dimensions() as u8,
            values: [0; MAX_FACTORS],
        };
        for (i, model) in self.factors.iter().enumerate() {
            let edge = model.edge(states[i], action)?;
            next[i] = edge.next;
            output.values[i] = edge.output;
        }
        Ok((next, output))
    }
    pub fn predict(&self, word: &[u8]) -> Result<FrameReply, FactorError> {
        if word.len() > 48 {
            return Err(FactorError::Limit("prediction word"));
        }
        let mut states = [0; MAX_FACTORS];
        let mut reply = Vec::with_capacity(word.len());
        for &action in word {
            let (next, frame) = self.transition(&states, action)?;
            states = next;
            reply.push(frame);
        }
        Ok(reply)
    }
    pub fn goal(&self, constraints: &[(usize, u16)]) -> Result<FactorGoal, FactorError> {
        let mut target = [None; MAX_FACTORS];
        if constraints.is_empty() {
            return Err(FactorError::Invalid("empty factor goal"));
        }
        for &(dimension, token) in constraints {
            if dimension >= self.dimensions() || target[dimension].is_some() {
                return Err(FactorError::Invalid("factor goal dimension"));
            }
            target[dimension] = Some(token);
        }
        Ok(FactorGoal {
            identity: self.fingerprint(),
            target,
        })
    }
    pub fn plan(
        &self,
        history: &[u8],
        goal: &FactorGoal,
        budget: PlanBudget,
    ) -> Result<Option<Word>, FactorError> {
        if budget.max_nodes == 0
            || budget.max_nodes > 65536
            || budget.max_depth == 0
            || budget.max_depth > 48
            || history.len() > 48
            || goal.identity != self.fingerprint()
        {
            return Err(FactorError::Invalid("plan configuration/identity"));
        }
        let mut root = [0; MAX_FACTORS];
        let mut last = None;
        for &action in history {
            let (next, frame) = self.transition(&root, action)?;
            root = next;
            last = Some(frame);
        }
        if last.is_some_and(|f| goal.matches(f)) {
            return Ok(Some(Vec::new()));
        }
        let mut visited = BTreeSet::from([root]);
        let mut queue = VecDeque::from([(root, Vec::new())]);
        let mut truncated = false;
        while let Some((state, word)) = queue.pop_front() {
            for action in 0..self.actions() {
                let (next, frame) = self.transition(&state, action)?;
                let mut extension = word.clone();
                extension.push(action);
                if goal.matches(frame) {
                    return Ok(Some(extension));
                }
                if visited.contains(&next) {
                    continue;
                }
                if extension.len() >= budget.max_depth {
                    truncated = true;
                    continue;
                }
                if visited.len() >= budget.max_nodes {
                    return Err(FactorError::Limit("plan nodes"));
                }
                visited.insert(next);
                queue.push_back((next, extension));
            }
        }
        if truncated {
            Err(FactorError::Limit("plan depth"))
        } else {
            Ok(None)
        }
    }
    pub fn save_bundle_new(
        &self,
        directory: &Path,
        authority: &Authority,
    ) -> Result<(), FactorError> {
        for effect in [Effect::ReadData, Effect::WriteData] {
            if !authority.allows(effect) {
                return Err(BehaviorError::Unauthorized(effect).into());
            }
        }
        fs::create_dir(directory)?;
        let mut catalog = Catalog::new(directory);
        for (i, model) in self.factors.iter().enumerate() {
            let file = format!("factor-{i}.axbehavior");
            model.save_new(directory.join(&file))?;
            catalog.register(
                ModuleSpec {
                    id: &format!("factor-{i}"),
                    kind: ModuleKind::Behavior,
                    file: &file,
                    input: "factor/events",
                    output: &format!("factor/output-{i}"),
                    cost: 1,
                },
                authority,
            )?;
        }
        catalog.save_new(directory.join("factors.axseed"), authority)?;
        Ok(())
    }
    pub fn load_bundle(
        path: &Path,
        context: &str,
        authority: &Authority,
    ) -> Result<Self, FactorError> {
        let catalog = Catalog::load(path, authority)?;
        let count = catalog.entries().len();
        if !(1..=MAX_FACTORS).contains(&count) {
            return Err(FactorError::Invalid("factor catalog dimensions"));
        }
        for i in 0..count {
            let descriptor = catalog
                .get(&format!("factor-{i}"))
                .ok_or(FactorError::Invalid("factor descriptor"))?;
            if descriptor.kind != ModuleKind::Behavior
                || descriptor.context != format!("{context}/factor-{i}")
                || descriptor.input != "factor/events"
                || descriptor.output != format!("factor/output-{i}")
            {
                return Err(FactorError::Invalid("factor descriptor contract"));
            }
        }
        let mut runtime = DemandRuntime::new(catalog, Seed::default())?;
        let mut factors = Vec::new();
        for i in 0..count {
            let model = runtime.with_module(
                &format!("factor-{i}"),
                &format!("{context}/factor-{i}"),
                authority,
                |m| match m {
                    Module::Behavior(model) => Ok(model.clone()),
                    _ => Err(DemandError::Invalid("factor kind")),
                },
            )?;
            factors.push(model);
        }
        Self::new(context, factors)
    }
    /// Preplans before effects, then verifies the entire frame before the next action.
    pub fn execute_goal(
        &self,
        history: &[u8],
        goal: &FactorGoal,
        environment: &mut impl FrameResettable,
        authority: &Authority,
        budget: PlanBudget,
    ) -> Result<Option<Word>, FactorError> {
        for effect in [Effect::Simulate, Effect::WriteData] {
            if !authority.allows(effect) {
                return Err(BehaviorError::Unauthorized(effect).into());
            }
        }
        let Some(plan) = self.plan(history, goal, budget)? else {
            return Ok(None);
        };
        if history.len() + plan.len() > 48 {
            return Err(FactorError::Limit("execution steps"));
        }
        let mut tracker = FactorTracker::new(self, self.context())?;
        environment.reset()?;
        for &action in history.iter().chain(&plan) {
            tracker.observe(action, environment.step(action)?)?;
        }
        Ok(Some(plan))
    }
}
pub fn learn(
    actions: u8,
    dimensions: usize,
    context: &str,
    oracle: &mut impl FrameOracle,
) -> Result<FactorModel, FactorError> {
    if !(1..=MAX_FACTORS).contains(&dimensions)
        || context.is_empty()
        || context.len() > 220
        || context.trim() != context
        || context.chars().any(char::is_control)
        || !(1..=16).contains(&actions)
    {
        return Err(FactorError::Invalid("factor learning configuration"));
    }
    let mut factors = Vec::new();
    for dimension in 0..dimensions {
        let mut projection = Projection {
            oracle,
            dimensions,
            dimension,
        };
        factors.push(discover(
            actions,
            &format!("{context}/factor-{dimension}"),
            &mut projection,
        )?);
    }
    FactorModel::new(context, factors)
}
#[derive(Clone, Copy, Debug)]
pub struct PlanBudget {
    pub max_nodes: usize,
    pub max_depth: usize,
}
impl Default for PlanBudget {
    fn default() -> Self {
        Self {
            max_nodes: 65536,
            max_depth: 48,
        }
    }
}
#[derive(Clone, Debug)]
pub struct FactorGoal {
    identity: u64,
    target: [Option<u16>; MAX_FACTORS],
}
impl FactorGoal {
    fn matches(&self, frame: Frame) -> bool {
        self.target
            .iter()
            .zip(frame.values())
            .all(|(want, &got)| want.is_none_or(|value| value == got))
    }
}
/// Fixed-size state. Monitoring does not learn or persist new observations.
pub struct FactorTracker<'a> {
    model: &'a FactorModel,
    states: [u16; MAX_FACTORS],
    observations: u64,
    suspended: bool,
}
impl<'a> FactorTracker<'a> {
    pub fn new(model: &'a FactorModel, context: &str) -> Result<Self, FactorError> {
        if model.context() != context {
            return Err(FactorError::Invalid("tracker context"));
        }
        Ok(Self {
            model,
            states: [0; MAX_FACTORS],
            observations: 0,
            suspended: false,
        })
    }
    pub fn observations(&self) -> u64 {
        self.observations
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn states(&self) -> &[u16; MAX_FACTORS] {
        &self.states
    }
    pub fn predict_next(&self, action: u8) -> Result<Frame, FactorError> {
        if self.suspended {
            return Err(FactorError::Suspended);
        }
        Ok(self.model.transition(&self.states, action)?.1)
    }
    pub fn observe(&mut self, action: u8, actual: Frame) -> Result<(), FactorError> {
        if self.suspended {
            return Err(FactorError::Suspended);
        }
        let (next, expected) = self.model.transition(&self.states, action)?;
        if actual.values().len() != self.model.dimensions() {
            self.suspended = true;
            return Err(FactorError::Invalid("tracker frame shape"));
        }
        for (dimension, (&expected, &actual)) in
            expected.values().iter().zip(actual.values()).enumerate()
        {
            if expected != actual {
                self.suspended = true;
                return Err(FactorError::Drift {
                    dimension,
                    expected,
                    actual,
                });
            }
        }
        self.observations = self
            .observations
            .checked_add(1)
            .ok_or(FactorError::Limit("observation count"))?;
        self.states = next;
        Ok(())
    }
}
