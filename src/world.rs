//! Bounded, interventional learning of deterministic Boolean control systems.
use crate::{
    Authority, Effect, EpisodeData, EpisodeError, EpisodeStamp, EpisodeStore, EpisodeValue,
};
use crate::{
    live_store::checksum,
    text_skill::{TextSkillError, publish_new, read_bounded},
};
use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    path::Path,
};

const CHANNEL: &str = "world.transition/v1";
const SOURCE: &str = "axon.world/v1";
const MAX_RECORDS: u16 = 512;
const MAX_WORK: usize = 2_000_000;

#[derive(Debug)]
pub enum WorldError {
    Invalid(&'static str),
    Unauthorized(Effect),
    Exhausted,
    Suspended,
    Contradiction,
    NotIdentified,
    Storage(EpisodeError),
    Artifact(TextSkillError),
    ActionOccurred { after: u8, reason: String },
}
impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "world: {self:?}")
    }
}
impl std::error::Error for WorldError {}
impl From<EpisodeError> for WorldError {
    fn from(e: EpisodeError) -> Self {
        Self::Storage(e)
    }
}
impl From<TextSkillError> for WorldError {
    fn from(e: TextSkillError) -> Self {
        Self::Artifact(e)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorldShape {
    bits: u8,
    actions: u8,
}
impl WorldShape {
    pub fn new(bits: u8, actions: u8) -> Result<Self, WorldError> {
        if !(1..=6).contains(&bits) || !(1..=8).contains(&actions) {
            return Err(WorldError::Invalid("shape"));
        }
        Ok(Self { bits, actions })
    }
    pub fn bits(self) -> u8 {
        self.bits
    }
    pub fn actions(self) -> u8 {
        self.actions
    }
    pub fn states(self) -> u8 {
        1 << self.bits
    }
    pub fn mask(self) -> u8 {
        self.states() - 1
    }
    pub fn all_states(self) -> u64 {
        if self.bits == 6 {
            u64::MAX
        } else {
            (1 << self.states()) - 1
        }
    }
    fn validate(self, state: u8, action: u8) -> Result<(), WorldError> {
        if state > self.mask() || action >= self.actions {
            return Err(WorldError::Invalid("state/action"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct TransitionRule {
    pub require_on: u8,
    pub require_off: u8,
    pub toggle: u8,
    pub set_mask: u8,
    pub set_value: u8,
}
impl TransitionRule {
    pub fn apply(self, state: u8) -> u8 {
        if state & self.require_on == self.require_on && state & self.require_off == 0 {
            ((state ^ self.toggle) & !self.set_mask) | self.set_value
        } else {
            state
        }
    }
    fn bytes(self) -> [u8; 5] {
        [
            self.require_on,
            self.require_off,
            self.toggle,
            self.set_mask,
            self.set_value,
        ]
    }
    fn complexity(self) -> u32 {
        self.require_on.count_ones()
            + self.require_off.count_ones()
            + self.toggle.count_ones()
            + self.set_mask.count_ones()
    }
    fn valid(self, shape: WorldShape) -> bool {
        let [on, off, toggle, set, value] = self.bytes();
        (on | off | toggle | set | value) & !shape.mask() == 0
            && on & off == 0
            && (on | off).count_ones() <= 2
            && value & !set == 0
            && ((toggle != 0 && toggle.count_ones() <= 2 && set == 0)
                || (toggle == 0 && set.count_ones() == 1)
                || (on | off | toggle | set | value == 0))
    }
}

#[derive(Clone, Debug)]
struct Catalog {
    rules: Vec<TransitionRule>,
    tables: Vec<[u8; 64]>,
}
impl Catalog {
    fn new(shape: WorldShape) -> Self {
        let mut effects = vec![(0, 0, 0)];
        for mask in 1..=shape.mask() {
            if mask.count_ones() <= 2 {
                effects.push((mask, 0, 0));
            }
            if mask.count_ones() == 1 {
                effects.push((0, mask, 0));
                effects.push((0, mask, mask));
            }
        }
        let mut by_behavior = BTreeMap::<[u8; 64], TransitionRule>::new();
        for on in 0..=shape.mask() {
            for off in 0..=shape.mask() {
                if on & off != 0 || (on | off).count_ones() > 2 {
                    continue;
                }
                for &(toggle, set_mask, set_value) in &effects {
                    if toggle == 0 && set_mask == 0 && on | off != 0 {
                        continue;
                    }
                    let rule = TransitionRule {
                        require_on: on,
                        require_off: off,
                        toggle,
                        set_mask,
                        set_value,
                    };
                    let mut table = [0; 64];
                    for state in 0..shape.states() {
                        table[state as usize] = rule.apply(state);
                    }
                    let existing = by_behavior.entry(table).or_insert(rule);
                    if (rule.complexity(), rule) < (existing.complexity(), *existing) {
                        *existing = rule;
                    }
                }
            }
        }
        let (tables, rules) = by_behavior.into_iter().unzip();
        Self { rules, tables }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationKind {
    Passive,
    Intervention,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    pub before: u8,
    pub action: u8,
    pub after: u8,
    pub kind: ObservationKind,
}

#[derive(Clone, Debug)]
pub struct ProbeScope {
    pub starts: u64,
    pub outcomes: u64,
    pub action_costs: Vec<u32>,
    pub remaining: u64,
}
impl ProbeScope {
    pub fn unrestricted(shape: WorldShape, remaining: u64) -> Self {
        Self {
            starts: shape.all_states(),
            outcomes: shape.all_states(),
            action_costs: vec![1; shape.actions as usize],
            remaining,
        }
    }
    fn validate(&self, shape: WorldShape) -> Result<(), WorldError> {
        if self.starts & !shape.all_states() != 0
            || self.outcomes & !shape.all_states() != 0
            || self.action_costs.len() != shape.actions as usize
            || self.action_costs.contains(&0)
        {
            return Err(WorldError::Invalid("probe scope"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Probe {
    pub state: u8,
    pub action: u8,
    pub cost: u32,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeChoice {
    Experiment(Probe),
    Identified,
    NoSafeAffordableExperiment,
}

#[derive(Clone, Debug)]
pub struct WorldLearner {
    shape: WorldShape,
    context: String,
    catalog: Catalog,
    candidates: Vec<Vec<usize>>,
    records: u16,
    interventions: u16,
    digest: u64,
    suspended: bool,
}
impl WorldLearner {
    pub fn new(shape: WorldShape, context: &str) -> Result<Self, WorldError> {
        validate_context(context)?;
        let catalog = Catalog::new(shape);
        let candidates = vec![(0..catalog.rules.len()).collect(); shape.actions as usize];
        Ok(Self {
            shape,
            context: context.to_owned(),
            catalog,
            candidates,
            records: 0,
            interventions: 0,
            digest: 0,
            suspended: false,
        })
    }
    pub fn shape(&self) -> WorldShape {
        self.shape
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn records(&self) -> u16 {
        self.records
    }
    pub fn interventions(&self) -> u16 {
        self.interventions
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn candidate_counts(&self) -> Vec<usize> {
        self.candidates.iter().map(Vec::len).collect()
    }
    pub fn identified(&self) -> bool {
        !self.suspended && self.candidates.iter().all(|x| x.len() == 1)
    }
    fn ready(&self) -> Result<(), WorldError> {
        if self.suspended {
            return Err(WorldError::Suspended);
        }
        if self.records >= MAX_RECORDS {
            return Err(WorldError::Exhausted);
        }
        Ok(())
    }
    pub fn observe(&mut self, t: Transition) -> Result<(), WorldError> {
        self.ready()?;
        self.shape.validate(t.before, t.action)?;
        if t.after > self.shape.mask() {
            self.suspended = true;
            return Err(WorldError::Invalid("outcome outside shape"));
        }
        self.records += 1;
        let mut bytes = self.digest.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[
            t.before,
            t.action,
            t.after,
            u8::from(t.kind == ObservationKind::Intervention),
        ]);
        self.digest = checksum(&bytes);
        if t.kind == ObservationKind::Intervention {
            self.interventions += 1;
            self.candidates[t.action as usize]
                .retain(|&i| self.catalog.tables[i][t.before as usize] == t.after);
            if self.candidates[t.action as usize].is_empty() {
                self.suspended = true;
                return Err(WorldError::Contradiction);
            }
        }
        Ok(())
    }
    pub fn predictions(&self, state: u8, action: u8) -> Result<u64, WorldError> {
        if self.suspended {
            return Err(WorldError::Suspended);
        }
        self.shape.validate(state, action)?;
        Ok(self.candidates[action as usize].iter().fold(0, |set, &i| {
            set | (1 << self.catalog.tables[i][state as usize])
        }))
    }
    pub fn choose_probe(&self, scope: &ProbeScope) -> Result<ProbeChoice, WorldError> {
        if self.suspended {
            return Err(WorldError::Suspended);
        }
        scope.validate(self.shape)?;
        if self.identified() {
            return Ok(ProbeChoice::Identified);
        }
        self.ready()?;
        let mut best: Option<(u64, u64, Probe)> = None;
        let mut work = 0;
        for (action, candidates) in self.candidates.iter().enumerate() {
            let cost = scope.action_costs[action];
            if candidates.len() <= 1 || u64::from(cost) > scope.remaining {
                continue;
            }
            for state in 0..self.shape.states() {
                if scope.starts & (1 << state) == 0 {
                    continue;
                }
                let mut buckets = [0_u64; 64];
                let mut safe = true;
                for &i in candidates {
                    work += 1;
                    if work > MAX_WORK {
                        return Err(WorldError::Exhausted);
                    }
                    let next = self.catalog.tables[i][state as usize];
                    if scope.outcomes & (1 << next) == 0 {
                        safe = false;
                        break;
                    }
                    buckets[next as usize] += 1;
                }
                if !safe {
                    continue;
                }
                let n = candidates.len() as u64;
                // Expected eliminated hypotheses, normalized by experiment cost.
                let gain = n * n - buckets.iter().map(|n| n * n).sum::<u64>();
                let divisor = n * u64::from(cost);
                if gain == 0 {
                    continue;
                }
                let probe = Probe {
                    state,
                    action: action as u8,
                    cost,
                };
                if best.as_ref().is_none_or(|(g, d, _)| {
                    u128::from(gain) * u128::from(*d) > u128::from(*g) * u128::from(divisor)
                }) {
                    best = Some((gain, divisor, probe));
                }
            }
        }
        Ok(
            best.map_or(ProbeChoice::NoSafeAffordableExperiment, |(_, _, p)| {
                ProbeChoice::Experiment(p)
            }),
        )
    }
    pub fn compile(&self) -> Result<WorldModel, WorldError> {
        if !self.identified() {
            return Err(WorldError::NotIdentified);
        }
        Ok(WorldModel {
            shape: self.shape,
            context: self.context.clone(),
            records: self.records,
            interventions: self.interventions,
            digest: self.digest,
            rules: self
                .candidates
                .iter()
                .map(|c| self.catalog.rules[c[0]])
                .collect(),
        })
    }
    pub fn plan(&self, start: u8, goal: Goal, states: u64) -> Result<PlanResult, WorldError> {
        if self.suspended {
            return Err(WorldError::Suspended);
        }
        plan(
            self.shape,
            start,
            goal,
            states,
            self.identified(),
            |s, a| self.predictions(s, a),
        )
    }
    pub fn restore(
        store: &EpisodeStore,
        shape: WorldShape,
        context: &str,
    ) -> Result<Self, WorldError> {
        let mut learner = Self::new(shape, context)?;
        for episode in store.traces(context, CHANNEL)? {
            if episode.stamp().source != SOURCE {
                return Err(WorldError::Invalid("trace source"));
            }
            let EpisodeData::Trace { fields, .. } = episode.data() else {
                unreachable!()
            };
            if fields.len() != 6
                || integer(fields, "bits")? != i64::from(shape.bits)
                || integer(fields, "actions")? != i64::from(shape.actions)
            {
                return Err(WorldError::Invalid("trace shape/schema"));
            }
            let number = |name| {
                u8::try_from(integer(fields, name)?)
                    .map_err(|_| WorldError::Invalid("trace number"))
            };
            let kind = match fields.get("intervention") {
                Some(EpisodeValue::Boolean(true)) => ObservationKind::Intervention,
                Some(EpisodeValue::Boolean(false)) => ObservationKind::Passive,
                _ => return Err(WorldError::Invalid("trace kind")),
            };
            let result = learner.observe(Transition {
                before: number("before")?,
                action: number("action")?,
                after: number("after")?,
                kind,
            });
            if matches!(result, Err(WorldError::Contradiction)) {
                continue;
            }
            result?;
        }
        Ok(learner)
    }
}

fn validate_context(context: &str) -> Result<(), WorldError> {
    if context.is_empty()
        || context.len() > 256
        || context.trim() != context
        || context.chars().any(char::is_control)
    {
        return Err(WorldError::Invalid("context"));
    }
    Ok(())
}
fn integer(fields: &BTreeMap<String, EpisodeValue>, name: &str) -> Result<i64, WorldError> {
    match fields.get(name) {
        Some(EpisodeValue::Integer(n)) => Ok(*n),
        _ => Err(WorldError::Invalid("trace field")),
    }
}
fn trace(shape: WorldShape, context: &str, t: Transition) -> EpisodeData {
    let mut fields = BTreeMap::new();
    for (name, value) in [
        ("bits", shape.bits),
        ("actions", shape.actions),
        ("before", t.before),
        ("action", t.action),
        ("after", t.after),
    ] {
        fields.insert(name.to_owned(), EpisodeValue::Integer(i64::from(value)));
    }
    fields.insert(
        "intervention".to_owned(),
        EpisodeValue::Boolean(t.kind == ObservationKind::Intervention),
    );
    EpisodeData::Trace {
        context: context.to_owned(),
        channel: CHANNEL.to_owned(),
        fields,
    }
}

/// This adapter is simulation-only. Authority is cooperative, not a sandbox.
pub struct WorldSession<'a> {
    learner: WorldLearner,
    store: &'a mut EpisodeStore,
    poisoned: bool,
}
impl<'a> WorldSession<'a> {
    pub fn restore(
        store: &'a mut EpisodeStore,
        shape: WorldShape,
        context: &str,
    ) -> Result<Self, WorldError> {
        let learner = WorldLearner::restore(store, shape, context)?;
        Ok(Self {
            learner,
            store,
            poisoned: false,
        })
    }
    pub fn learner(&self) -> &WorldLearner {
        &self.learner
    }
    pub fn record_passive(&mut self, before: u8, action: u8, after: u8) -> Result<(), WorldError> {
        self.check()?;
        self.learner.shape.validate(before, action)?;
        if after > self.learner.shape.mask() {
            return Err(WorldError::Invalid("passive outcome"));
        }
        let t = Transition {
            before,
            action,
            after,
            kind: ObservationKind::Passive,
        };
        let stamp = self.stamp();
        self.store
            .record(stamp, trace(self.learner.shape, &self.learner.context, t))?;
        self.learner.observe(t)
    }
    fn check(&self) -> Result<(), WorldError> {
        if self.poisoned {
            Err(WorldError::Suspended)
        } else {
            self.learner.ready()
        }
    }
    fn stamp(&self) -> EpisodeStamp {
        EpisodeStamp::new(
            SOURCE,
            format!("trial-{}", self.store.next_id()),
            self.store.next_id(),
        )
    }
    pub fn intervene<F>(
        &mut self,
        probe: Probe,
        scope: &mut ProbeScope,
        authority: &Authority,
        mut simulator: F,
    ) -> Result<u8, WorldError>
    where
        F: FnMut(u8, u8) -> u8,
    {
        self.check()?;
        if !authority.allows(Effect::Simulate) {
            return Err(WorldError::Unauthorized(Effect::Simulate));
        }
        if !authority.allows(Effect::WriteData) {
            return Err(WorldError::Unauthorized(Effect::WriteData));
        }
        scope.validate(self.learner.shape)?;
        self.learner.shape.validate(probe.state, probe.action)?;
        if probe.cost != scope.action_costs[probe.action as usize]
            || u64::from(probe.cost) > scope.remaining
            || scope.starts & (1 << probe.state) == 0
            || self.learner.predictions(probe.state, probe.action)? & !scope.outcomes != 0
        {
            return Err(WorldError::Invalid("unsafe/unaffordable probe"));
        }
        let mut t = Transition {
            before: probe.state,
            action: probe.action,
            after: 0,
            kind: ObservationKind::Intervention,
        };
        let stamp = self.stamp();
        self.store
            .check_record(&stamp, &trace(self.learner.shape, &self.learner.context, t))?;
        t.after = simulator(probe.state, probe.action);
        scope.remaining -= u64::from(probe.cost);
        if let Err(error) = self
            .store
            .record(stamp, trace(self.learner.shape, &self.learner.context, t))
        {
            self.poisoned = true;
            return Err(WorldError::ActionOccurred {
                after: t.after,
                reason: error.to_string(),
            });
        }
        let result = self.learner.observe(t);
        if t.after > self.learner.shape.mask()
            || scope.outcomes & (1_u64.checked_shl(u32::from(t.after)).unwrap_or(0)) == 0
        {
            self.poisoned = true;
            return Err(WorldError::ActionOccurred {
                after: t.after,
                reason: "outcome outside authorized scope".to_owned(),
            });
        }
        if let Err(error) = result {
            self.poisoned = true;
            return Err(WorldError::ActionOccurred {
                after: t.after,
                reason: error.to_string(),
            });
        }
        Ok(t.after)
    }
    pub fn execute<F>(
        &mut self,
        model: &WorldModel,
        plan: &WorldPlan,
        scope: &mut ProbeScope,
        authority: &Authority,
        mut simulator: F,
    ) -> Result<u8, WorldError>
    where
        F: FnMut(u8, u8) -> u8,
    {
        if self.learner.compile()?.fingerprint() != model.fingerprint() {
            return Err(WorldError::Invalid("model does not match memory"));
        }
        if model.plan(plan.start, plan.goal, plan.allowed_states)?
            != PlanResult::Found(plan.clone())
        {
            return Err(WorldError::Invalid("stale/modified plan"));
        }
        // Preflight the whole scope and budget; every observed transition is still checked.
        scope.validate(model.shape)?;
        let total: u64 = plan
            .steps
            .iter()
            .map(|step| u64::from(scope.action_costs[step.action as usize]))
            .sum();
        if total > scope.remaining {
            return Err(WorldError::Exhausted);
        }
        for step in &plan.steps {
            if scope.starts & (1 << step.before) == 0 || scope.outcomes & (1 << step.after) == 0 {
                return Err(WorldError::Invalid("plan outside scope"));
            }
        }
        let mut state = plan.start;
        for step in &plan.steps {
            let after = self.intervene(
                Probe {
                    state,
                    action: step.action,
                    cost: scope.action_costs[step.action as usize],
                },
                scope,
                authority,
                &mut simulator,
            )?;
            if after != step.after {
                self.poisoned = true;
                return Err(WorldError::ActionOccurred {
                    after,
                    reason: "model drift".to_owned(),
                });
            }
            state = after;
        }
        Ok(state)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Goal {
    pub on: u8,
    pub off: u8,
}
impl Goal {
    pub fn matches(self, state: u8) -> bool {
        state & self.on == self.on && state & self.off == 0
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanStep {
    pub before: u8,
    pub action: u8,
    pub after: u8,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldPlan {
    pub start: u8,
    pub goal: Goal,
    pub allowed_states: u64,
    pub steps: Vec<PlanStep>,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlanResult {
    Found(WorldPlan),
    NeedsExperiment,
    Unreachable,
}
fn plan<F>(
    shape: WorldShape,
    start: u8,
    goal: Goal,
    allowed: u64,
    identified: bool,
    mut predict: F,
) -> Result<PlanResult, WorldError>
where
    F: FnMut(u8, u8) -> Result<u64, WorldError>,
{
    if start > shape.mask()
        || goal.on & goal.off != 0
        || (goal.on | goal.off) & !shape.mask() != 0
        || allowed & !shape.all_states() != 0
        || allowed & (1 << start) == 0
    {
        return Err(WorldError::Invalid("planning scope/goal"));
    }
    let mut queue = VecDeque::from([start]);
    let mut seen = 1_u64 << start;
    let mut parents = [None; 64];
    while let Some(state) = queue.pop_front() {
        if goal.matches(state) {
            let mut steps = Vec::new();
            let mut cursor = state;
            while cursor != start {
                let (before, action) = parents[cursor as usize].expect("visited parent");
                steps.push(PlanStep {
                    before,
                    action,
                    after: cursor,
                });
                cursor = before;
            }
            steps.reverse();
            return Ok(PlanResult::Found(WorldPlan {
                start,
                goal,
                allowed_states: allowed,
                steps,
            }));
        }
        for action in 0..shape.actions {
            let outcomes = predict(state, action)?;
            if outcomes.count_ones() != 1 || outcomes & !allowed != 0 {
                continue;
            }
            let next = outcomes.trailing_zeros() as u8;
            if seen & (1 << next) == 0 {
                seen |= 1 << next;
                parents[next as usize] = Some((state, action));
                queue.push_back(next);
            }
        }
    }
    Ok(if identified {
        PlanResult::Unreachable
    } else {
        PlanResult::NeedsExperiment
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorldModel {
    shape: WorldShape,
    context: String,
    rules: Vec<TransitionRule>,
    records: u16,
    interventions: u16,
    digest: u64,
}
impl WorldModel {
    pub fn shape(&self) -> WorldShape {
        self.shape
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn rules(&self) -> &[TransitionRule] {
        &self.rules
    }
    pub fn interventions(&self) -> u16 {
        self.interventions
    }
    pub fn predict(&self, state: u8, action: u8) -> Result<u8, WorldError> {
        self.shape.validate(state, action)?;
        Ok(self.rules[action as usize].apply(state))
    }
    pub fn plan(&self, start: u8, goal: Goal, states: u64) -> Result<PlanResult, WorldError> {
        plan(self.shape, start, goal, states, true, |s, a| {
            Ok(1 << self.predict(s, a)?)
        })
    }
    fn bytes(&self) -> Vec<u8> {
        let mut bytes = b"AXWD0001".to_vec();
        bytes.extend_from_slice(&[self.shape.bits, self.shape.actions]);
        bytes.extend_from_slice(&(self.context.len() as u16).to_le_bytes());
        bytes.extend_from_slice(self.context.as_bytes());
        bytes.extend_from_slice(&self.records.to_le_bytes());
        bytes.extend_from_slice(&self.interventions.to_le_bytes());
        bytes.extend_from_slice(&self.digest.to_le_bytes());
        for rule in &self.rules {
            bytes.extend_from_slice(&rule.bytes());
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        bytes
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.bytes())
    }
    pub fn encoded_bytes(&self) -> usize {
        self.bytes().len()
    }
    pub fn save_new(&self, path: impl AsRef<Path>) -> Result<(), WorldError> {
        publish_new(path.as_ref(), &self.bytes())?;
        Ok(())
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, WorldError> {
        let bytes = read_bounded(path.as_ref(), 512)?;
        Self::from_bytes(&bytes)
    }
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, WorldError> {
        if bytes.len() > 512 {
            return Err(WorldError::Invalid("model size"));
        }
        if bytes.len() < 33 || &bytes[..8] != b"AXWD0001" {
            return Err(WorldError::Invalid("model header"));
        }
        let payload = bytes.len() - 8;
        if checksum(&bytes[..payload])
            != u64::from_le_bytes(bytes[payload..].try_into().expect("width"))
        {
            return Err(WorldError::Invalid("model checksum"));
        }
        let shape = WorldShape::new(bytes[8], bytes[9])?;
        let n = u16::from_le_bytes(bytes[10..12].try_into().expect("width")) as usize;
        if n > 256 || bytes.len() != 32 + n + shape.actions as usize * 5 {
            return Err(WorldError::Invalid("model length"));
        }
        let context = std::str::from_utf8(&bytes[12..12 + n])
            .map_err(|_| WorldError::Invalid("model UTF-8"))?
            .to_owned();
        validate_context(&context)?;
        let p = 12 + n;
        let records = u16::from_le_bytes(bytes[p..p + 2].try_into().expect("width"));
        let interventions = u16::from_le_bytes(bytes[p + 2..p + 4].try_into().expect("width"));
        if records > MAX_RECORDS
            || interventions > records
            || interventions < u16::from(shape.actions)
        {
            return Err(WorldError::Invalid("model evidence counts"));
        }
        let digest = u64::from_le_bytes(bytes[p + 4..p + 12].try_into().expect("width"));
        let mut rules = Vec::new();
        for r in bytes[p + 12..payload].as_chunks::<5>().0 {
            let rule = TransitionRule {
                require_on: r[0],
                require_off: r[1],
                toggle: r[2],
                set_mask: r[3],
                set_value: r[4],
            };
            if !rule.valid(shape) {
                return Err(WorldError::Invalid("model rule"));
            }
            rules.push(rule);
        }
        Ok(Self {
            shape,
            context,
            rules,
            records,
            interventions,
            digest,
        })
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.context.capacity()
            + self.rules.capacity() * std::mem::size_of::<TransitionRule>()
    }
}
