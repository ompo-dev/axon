//! Active learning of finite, resettable input/output systems with hidden state.
use crate::{
    Authority, Effect, EpisodeData, EpisodeError, EpisodeStamp, EpisodeStore, EpisodeValue,
};
use crate::{
    live_store::checksum,
    text_skill::{TextSkillError, publish_new, read_bounded},
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    path::Path,
};

pub type Word = Vec<u8>;
pub type Reply = Vec<u16>;
const SOURCE: &str = "axon.behavior/v1";
const CHANNEL: &str = "behavior.query/v1";
const ABORT_CHANNEL: &str = "behavior.abort/v1";

#[derive(Debug)]
pub enum BehaviorError {
    Invalid(&'static str),
    Limit(&'static str),
    Unauthorized(Effect),
    Suspended,
    NonDeterministic,
    UnexpectedOutput { expected: u16, actual: u16 },
    NotCounterexample,
    Storage(EpisodeError),
    Artifact(TextSkillError),
    QueryOccurred { reply: Reply, reason: String },
}
impl fmt::Display for BehaviorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "behavior: {self:?}")
    }
}
impl std::error::Error for BehaviorError {}
impl From<EpisodeError> for BehaviorError {
    fn from(e: EpisodeError) -> Self {
        Self::Storage(e)
    }
}
impl From<TextSkillError> for BehaviorError {
    fn from(e: TextSkillError) -> Self {
        Self::Artifact(e)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BehaviorBudget {
    pub max_states: usize,
    pub max_suffixes: usize,
    pub max_prefixes: usize,
    pub max_word: usize,
    pub max_queries: usize,
    pub max_steps: usize,
    pub max_table_work: usize,
}
impl Default for BehaviorBudget {
    fn default() -> Self {
        Self {
            max_states: 128,
            max_suffixes: 128,
            max_prefixes: 512,
            max_word: 48,
            max_queries: 32_768,
            max_steps: 800_000,
            max_table_work: 10_000_000,
        }
    }
}
impl BehaviorBudget {
    fn validate(self) -> Result<(), BehaviorError> {
        if self.max_states == 0
            || self.max_states > 128
            || self.max_suffixes == 0
            || self.max_suffixes > 128
            || self.max_prefixes == 0
            || self.max_prefixes > 512
            || self.max_word == 0
            || self.max_word > 48
            || self.max_queries == 0
            || self.max_queries > 32_768
            || self.max_steps == 0
            || self.max_steps > 800_000
            || self.max_table_work == 0
            || self.max_table_work > 10_000_000
        {
            return Err(BehaviorError::Invalid("budget"));
        }
        Ok(())
    }
}
fn context_valid(context: &str) -> Result<(), BehaviorError> {
    if context.is_empty()
        || context.len() > 256
        || context.trim() != context
        || context.chars().any(char::is_control)
    {
        return Err(BehaviorError::Invalid("context"));
    }
    Ok(())
}
fn word_valid(word: &[u8], actions: u8, maximum: usize) -> Result<(), BehaviorError> {
    if word.len() > maximum {
        return Err(BehaviorError::Limit("word length"));
    }
    if word.iter().any(|&a| a >= actions) {
        return Err(BehaviorError::Invalid("action"));
    }
    Ok(())
}
pub trait SequenceOracle {
    /// Each query begins in the same initial condition, not an arbitrary chosen state.
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError>;
}

pub trait Resettable {
    fn reset(&mut self) -> Result<(), BehaviorError>;
    fn step(&mut self, action: u8) -> Result<u16, BehaviorError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BehaviorEdge {
    pub next: u16,
    pub output: u16,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BehaviorModel {
    context: String,
    actions: u8,
    edges: Vec<Vec<BehaviorEdge>>,
}
impl BehaviorModel {
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn actions(&self) -> u8 {
        self.actions
    }
    pub fn states(&self) -> usize {
        self.edges.len()
    }
    pub fn edge(&self, state: u16, action: u8) -> Result<BehaviorEdge, BehaviorError> {
        self.edges
            .get(state as usize)
            .and_then(|row| row.get(action as usize))
            .copied()
            .ok_or(BehaviorError::Invalid("model state/action"))
    }
    pub fn predict(&self, word: &[u8]) -> Result<Reply, BehaviorError> {
        word_valid(word, self.actions, 48)?;
        let mut state = 0;
        let mut reply = Vec::new();
        for &action in word {
            let edge = self.edge(state, action)?;
            state = edge.next;
            reply.push(edge.output);
        }
        Ok(reply)
    }
    pub fn state_after(&self, history: &[u8]) -> Result<u16, BehaviorError> {
        word_valid(history, self.actions, 48)?;
        let mut state = 0;
        for &action in history {
            state = self.edge(state, action)?.next;
        }
        Ok(state)
    }
    pub fn access_sequences(&self) -> Vec<Word> {
        let mut words = vec![None; self.states()];
        words[0] = Some(Vec::new());
        let mut queue = VecDeque::from([0_usize]);
        while let Some(state) = queue.pop_front() {
            for action in 0..self.actions {
                let next = self.edges[state][action as usize].next as usize;
                if words[next].is_none() {
                    let mut word = words[state].clone().expect("visited");
                    word.push(action);
                    words[next] = Some(word);
                    queue.push_back(next);
                }
            }
        }
        words
            .into_iter()
            .map(|w| w.expect("validated reachable model"))
            .collect()
    }
    pub fn plan_output(
        &self,
        history: &[u8],
        goal: u16,
        forbidden: &BTreeSet<u16>,
    ) -> Result<Option<Word>, BehaviorError> {
        self.plan_from(self.state_after(history)?, goal, forbidden)
    }
    pub(crate) fn plan_from(
        &self,
        state: u16,
        goal: u16,
        forbidden: &BTreeSet<u16>,
    ) -> Result<Option<Word>, BehaviorError> {
        if forbidden.contains(&goal) {
            return Err(BehaviorError::Invalid("forbidden goal"));
        }
        let start = state as usize;
        if start >= self.states() {
            return Err(BehaviorError::Invalid("planning state"));
        }
        let mut paths = vec![None; self.states()];
        paths[start] = Some(Vec::<u8>::new());
        let mut queue = VecDeque::from([start]);
        while let Some(state) = queue.pop_front() {
            let prefix = paths[state].clone().expect("visited");
            for action in 0..self.actions {
                let edge = self.edges[state][action as usize];
                if forbidden.contains(&edge.output) {
                    continue;
                }
                let mut word = prefix.clone();
                word.push(action);
                if edge.output == goal {
                    return Ok(Some(word));
                }
                let next = edge.next as usize;
                if paths[next].is_none() {
                    paths[next] = Some(word);
                    queue.push_back(next);
                }
            }
        }
        Ok(None)
    }
    pub(crate) fn bytes(&self) -> Vec<u8> {
        let mut b = b"AXBH0001".to_vec();
        b.push(self.actions);
        b.extend_from_slice(&(self.states() as u16).to_le_bytes());
        b.extend_from_slice(&(self.context.len() as u16).to_le_bytes());
        b.extend_from_slice(self.context.as_bytes());
        for row in &self.edges {
            for edge in row {
                b.extend_from_slice(&edge.next.to_le_bytes());
                b.extend_from_slice(&edge.output.to_le_bytes());
            }
        }
        b.extend_from_slice(&checksum(&b).to_le_bytes());
        b
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.bytes())
    }
    pub fn encoded_bytes(&self) -> usize {
        self.bytes().len()
    }
    pub fn save_new(&self, path: impl AsRef<Path>) -> Result<(), BehaviorError> {
        publish_new(path.as_ref(), &self.bytes())?;
        Ok(())
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, BehaviorError> {
        let b = read_bounded(path.as_ref(), 16_384)?;
        Self::from_bytes(&b)
    }
    pub(crate) fn from_bytes(b: &[u8]) -> Result<Self, BehaviorError> {
        if b.len() > 16_384 {
            return Err(BehaviorError::Invalid("model size"));
        }
        if b.len() < 25 || &b[..8] != b"AXBH0001" {
            return Err(BehaviorError::Invalid("model header"));
        }
        let p = b.len() - 8;
        if checksum(&b[..p]) != u64::from_le_bytes(b[p..].try_into().expect("width")) {
            return Err(BehaviorError::Invalid("model checksum"));
        }
        let actions = b[8];
        let states = u16::from_le_bytes(b[9..11].try_into().expect("width")) as usize;
        let n = u16::from_le_bytes(b[11..13].try_into().expect("width")) as usize;
        if !(1..=16).contains(&actions)
            || !(1..=128).contains(&states)
            || n > 256
            || b.len() != 21 + n + states * actions as usize * 4
        {
            return Err(BehaviorError::Invalid("model dimensions"));
        }
        let context = std::str::from_utf8(&b[13..13 + n])
            .map_err(|_| BehaviorError::Invalid("model UTF-8"))?
            .to_owned();
        context_valid(&context)?;
        let mut edges = Vec::new();
        for row in b[13 + n..p].chunks(actions as usize * 4) {
            let mut r = Vec::new();
            for cell in row.as_chunks::<4>().0 {
                let next = u16::from_le_bytes(cell[..2].try_into().expect("width"));
                if next as usize >= states {
                    return Err(BehaviorError::Invalid("model target"));
                }
                r.push(BehaviorEdge {
                    next,
                    output: u16::from_le_bytes(cell[2..].try_into().expect("width")),
                });
            }
            edges.push(r);
        }
        let model = Self {
            context,
            actions,
            edges,
        };
        let mut visited = BTreeSet::from([0_u16]);
        let mut queue = VecDeque::from([0_u16]);
        while let Some(s) = queue.pop_front() {
            for edge in &model.edges[s as usize] {
                if visited.insert(edge.next) {
                    queue.push_back(edge.next);
                }
            }
        }
        if visited.len() != states {
            return Err(BehaviorError::Invalid("unreachable model state"));
        }
        Ok(model)
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.context.capacity()
            + self.edges.capacity() * std::mem::size_of::<Vec<BehaviorEdge>>()
            + self
                .edges
                .iter()
                .map(|r| r.capacity() * std::mem::size_of::<BehaviorEdge>())
                .sum::<usize>()
    }
    pub(crate) fn from_edges(
        context: &str,
        edges: Vec<Vec<BehaviorEdge>>,
    ) -> Result<Self, BehaviorError> {
        context_valid(context)?;
        let actions = edges.first().map_or(0, Vec::len);
        if !(1..=16).contains(&actions)
            || !(1..=128).contains(&edges.len())
            || edges
                .iter()
                .any(|r| r.len() != actions || r.iter().any(|e| e.next as usize >= edges.len()))
        {
            return Err(BehaviorError::Invalid("constructed model dimensions"));
        }
        let model = Self {
            context: context.into(),
            actions: actions as u8,
            edges,
        };
        Self::from_bytes(&model.bytes())
    }
}

/// Constant-size monitoring state; it does not journal or execute actions.
pub struct BehaviorTracker<'a> {
    model: &'a BehaviorModel,
    state: u16,
    observations: u64,
    suspended: bool,
}
impl<'a> BehaviorTracker<'a> {
    pub fn new(model: &'a BehaviorModel, context: &str) -> Result<Self, BehaviorError> {
        if model.context() != context {
            return Err(BehaviorError::Invalid("tracker context"));
        }
        Ok(Self {
            model,
            state: 0,
            observations: 0,
            suspended: false,
        })
    }
    pub fn state(&self) -> u16 {
        self.state
    }
    pub fn observations(&self) -> u64 {
        self.observations
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn predict_next(&self, action: u8) -> Result<u16, BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        Ok(self.model.edge(self.state, action)?.output)
    }
    pub fn observe(&mut self, action: u8, output: u16) -> Result<(), BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        let edge = self.model.edge(self.state, action)?;
        self.observations = self
            .observations
            .checked_add(1)
            .ok_or(BehaviorError::Limit("tracker observations"))?;
        if output != edge.output {
            self.suspended = true;
            return Err(BehaviorError::UnexpectedOutput {
                expected: edge.output,
                actual: output,
            });
        }
        self.state = edge.next;
        Ok(())
    }
    pub fn plan(
        &self,
        goal: u16,
        forbidden: &BTreeSet<u16>,
    ) -> Result<Option<Word>, BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        self.model.plan_from(self.state, goal, forbidden)
    }
}

pub struct BehaviorLearner {
    actions: u8,
    context: String,
    prefixes: Vec<Word>,
    suffixes: Vec<Word>,
    budget: BehaviorBudget,
}
impl BehaviorLearner {
    pub fn new(actions: u8, context: &str, budget: BehaviorBudget) -> Result<Self, BehaviorError> {
        budget.validate()?;
        context_valid(context)?;
        if !(1..=16).contains(&actions) || actions as usize > budget.max_suffixes {
            return Err(BehaviorError::Invalid("actions/suffix budget"));
        }
        Ok(Self {
            actions,
            context: context.to_owned(),
            prefixes: vec![Vec::new()],
            suffixes: (0..actions).map(|a| vec![a]).collect(),
            budget,
        })
    }
    pub fn suffixes(&self) -> &[Word] {
        &self.suffixes
    }
    pub fn prefix_count(&self) -> usize {
        self.prefixes.len()
    }
    pub fn hypothesize<O: SequenceOracle>(
        &mut self,
        oracle: &mut O,
    ) -> Result<BehaviorModel, BehaviorError> {
        let mut work = 0_usize;
        loop {
            let mut all = self.prefixes.clone();
            for s in &self.prefixes {
                for a in 0..self.actions {
                    let mut w = s.clone();
                    w.push(a);
                    all.push(w);
                }
            }
            all.sort();
            all.dedup();
            let mut rows = BTreeMap::<Word, Vec<Reply>>::new();
            for s in all {
                let mut row = Vec::new();
                for e in &self.suffixes {
                    let mut word = s.clone();
                    word.extend_from_slice(e);
                    word_valid(&word, self.actions, self.budget.max_word)?;
                    work = work
                        .checked_add(word.len())
                        .ok_or(BehaviorError::Limit("table work"))?;
                    if work > self.budget.max_table_work {
                        return Err(BehaviorError::Limit("table work"));
                    }
                    let reply = oracle.query(&word)?;
                    if reply.len() != word.len() {
                        return Err(BehaviorError::Invalid("oracle reply length"));
                    }
                    row.push(reply[s.len()..].to_vec());
                }
                rows.insert(s, row);
            }
            let mut distinguishing = None;
            'consistent: for (i, s) in self.prefixes.iter().enumerate() {
                for t in &self.prefixes[i + 1..] {
                    if rows[s] != rows[t] {
                        continue;
                    }
                    for a in 0..self.actions {
                        let mut sa = s.clone();
                        sa.push(a);
                        let mut ta = t.clone();
                        ta.push(a);
                        if let Some(k) = rows[&sa].iter().zip(&rows[&ta]).position(|(x, y)| x != y)
                        {
                            let mut e = vec![a];
                            e.extend_from_slice(&self.suffixes[k]);
                            distinguishing = Some(e);
                            break 'consistent;
                        }
                    }
                }
            }
            if let Some(e) = distinguishing {
                if self.suffixes.len() >= self.budget.max_suffixes {
                    return Err(BehaviorError::Limit("distinguishing suffixes"));
                }
                if self.suffixes.contains(&e) {
                    return Err(BehaviorError::Invalid("inconsistent oracle/table"));
                }
                self.suffixes.push(e);
                continue;
            }
            let mut unclosed = None;
            'closed: for s in &self.prefixes {
                for a in 0..self.actions {
                    let mut sa = s.clone();
                    sa.push(a);
                    if !self.prefixes.iter().any(|t| rows[t] == rows[&sa]) {
                        unclosed = Some(sa);
                        break 'closed;
                    }
                }
            }
            if let Some(sa) = unclosed {
                if self.prefixes.len() >= self.budget.max_prefixes {
                    return Err(BehaviorError::Limit("access prefixes"));
                }
                self.prefixes.push(sa);
                continue;
            }
            let mut representatives = Vec::<Word>::new();
            for s in &self.prefixes {
                if !representatives.iter().any(|t| rows[t] == rows[s]) {
                    representatives.push(s.clone());
                }
            }
            if representatives.len() > self.budget.max_states {
                return Err(BehaviorError::Limit("latent states"));
            }
            let mut edges = Vec::new();
            for s in &representatives {
                let mut row = Vec::new();
                for a in 0..self.actions {
                    let mut sa = s.clone();
                    sa.push(a);
                    let next = representatives
                        .iter()
                        .position(|t| rows[t] == rows[&sa])
                        .expect("closed") as u16;
                    // The first action-sized suffixes remain in alphabet order.
                    row.push(BehaviorEdge {
                        next,
                        output: rows[s][a as usize][0],
                    });
                }
                edges.push(row);
            }
            return Ok(BehaviorModel {
                context: self.context.clone(),
                actions: self.actions,
                edges,
            });
        }
    }
    pub fn refine<O: SequenceOracle>(
        &mut self,
        model: &BehaviorModel,
        word: &[u8],
        oracle: &mut O,
    ) -> Result<(), BehaviorError> {
        if model.context != self.context || model.actions != self.actions {
            return Err(BehaviorError::Invalid("counterexample context"));
        }
        word_valid(word, self.actions, self.budget.max_word)?;
        let actual = oracle.query(word)?;
        if actual.len() != word.len() {
            return Err(BehaviorError::Invalid("counterexample reply"));
        }
        if actual == model.predict(word)? {
            return Err(BehaviorError::NotCounterexample);
        }
        let additions: Vec<Word> = (1..=word.len())
            .map(|n| word[..n].to_vec())
            .filter(|w| !self.prefixes.contains(w))
            .collect();
        if self.prefixes.len() + additions.len() > self.budget.max_prefixes {
            return Err(BehaviorError::Limit("counterexample prefixes"));
        }
        self.prefixes.extend(additions);
        Ok(())
    }
    /// A bounded conformance heuristic, not a true equivalence oracle.
    pub fn counterexample<O: SequenceOracle>(
        &self,
        model: &BehaviorModel,
        oracle: &mut O,
        middle_depth: u8,
        max_tests: usize,
    ) -> Result<Option<Word>, BehaviorError> {
        if model.context != self.context
            || model.actions != self.actions
            || middle_depth > 3
            || max_tests == 0
            || max_tests > 65_536
        {
            return Err(BehaviorError::Invalid("conformance scope"));
        }
        let mut unresolved: Vec<(u16, u16)> = (0..model.states() as u16)
            .flat_map(|s| (s + 1..model.states() as u16).map(move |t| (s, t)))
            .collect();
        let mut signatures = Vec::new();
        for suffix in &self.suffixes {
            let mut outputs = Vec::new();
            for state in 0..model.states() as u16 {
                let mut current = state;
                let mut reply = Vec::new();
                for &action in suffix {
                    let edge = model.edge(current, action)?;
                    current = edge.next;
                    reply.push(edge.output);
                }
                outputs.push(reply);
            }
            signatures.push(outputs);
        }
        let mut characterization = vec![Vec::new()];
        let mut scoring_work = 0_usize;
        while !unresolved.is_empty() {
            let mut best = None;
            for (index, outputs) in signatures.iter().enumerate() {
                scoring_work += unresolved.len();
                if scoring_work > self.budget.max_table_work {
                    return Err(BehaviorError::Limit("characterization work"));
                }
                let gain = unresolved
                    .iter()
                    .filter(|&&(s, t)| outputs[s as usize] != outputs[t as usize])
                    .count();
                if best.is_none_or(|(_, old)| gain > old) {
                    best = Some((index, gain));
                }
            }
            let (index, gain) = best.expect("nonempty suffixes");
            if gain == 0 {
                return Err(BehaviorError::Invalid("indistinguishable candidate states"));
            }
            characterization.push(self.suffixes[index].clone());
            unresolved
                .retain(|&(s, t)| signatures[index][s as usize] == signatures[index][t as usize]);
        }
        let mut middles = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..middle_depth {
            let mut next = Vec::new();
            for p in frontier {
                for a in 0..self.actions {
                    let mut w = p.clone();
                    w.push(a);
                    next.push(w);
                }
            }
            if middles.len() + next.len() > 5_000 {
                return Err(BehaviorError::Limit("conformance middle words"));
            }
            middles.extend(next.clone());
            frontier = next;
        }
        let mut tests = 0;
        for access in model.access_sequences() {
            for middle in &middles {
                for suffix in &characterization {
                    tests += 1;
                    if tests > max_tests {
                        return Err(BehaviorError::Limit("conformance tests"));
                    }
                    let mut word = access.clone();
                    word.extend_from_slice(middle);
                    word.extend_from_slice(suffix);
                    word_valid(&word, self.actions, self.budget.max_word)?;
                    let reply = oracle.query(&word)?;
                    if reply.len() != word.len() {
                        return Err(BehaviorError::Invalid("conformance reply length"));
                    }
                    if reply != model.predict(&word)? {
                        return Ok(Some(word));
                    }
                }
            }
        }
        Ok(None)
    }
}

/// Durable query cache. Stored source/context names are not authentication.
pub struct BehaviorMemory<'a, F>
where
    F: Resettable,
{
    store: &'a mut EpisodeStore,
    context: String,
    actions: u8,
    cache: BTreeMap<Word, Reply>,
    prefix_outputs: BTreeMap<Word, u16>,
    budget: BehaviorBudget,
    queries: usize,
    steps: usize,
    suspended: bool,
    authority: Authority,
    adapter: F,
}
impl<'a, F> BehaviorMemory<'a, F>
where
    F: Resettable,
{
    pub fn restore(
        store: &'a mut EpisodeStore,
        context: &str,
        actions: u8,
        budget: BehaviorBudget,
        authority: Authority,
        adapter: F,
    ) -> Result<Self, BehaviorError> {
        budget.validate()?;
        context_valid(context)?;
        if !(1..=16).contains(&actions) {
            return Err(BehaviorError::Invalid("actions"));
        }
        let mut cache = BTreeMap::new();
        let mut prefix_outputs = BTreeMap::new();
        let mut queries = 0;
        let mut steps = 0;
        let mut suspended = false;
        for episode in store.traces(context, CHANNEL)? {
            if episode.stamp().source != SOURCE {
                return Err(BehaviorError::Invalid("trace source"));
            }
            let EpisodeData::Trace { fields, .. } = episode.data() else {
                unreachable!()
            };
            if fields.len() != 3
                || fields.get("actions") != Some(&EpisodeValue::Integer(i64::from(actions)))
            {
                return Err(BehaviorError::Invalid("trace schema"));
            }
            let input = field_text(fields, "input")?;
            let output = field_text(fields, "output")?;
            let word = decode_word(input)?;
            let reply = decode_reply(output)?;
            word_valid(&word, actions, budget.max_word)?;
            if word.is_empty() || word.len() != reply.len() {
                return Err(BehaviorError::Invalid("trace lengths"));
            }
            queries += 1;
            steps += word.len();
            if queries > budget.max_queries || steps > budget.max_steps {
                return Err(BehaviorError::Limit("replayed experience"));
            }
            for (i, &out) in reply.iter().enumerate() {
                if prefix_outputs
                    .insert(word[..=i].to_vec(), out)
                    .is_some_and(|old| old != out)
                {
                    suspended = true;
                }
            }
            cache.insert(word, reply);
        }
        for episode in store.traces(context, ABORT_CHANNEL)? {
            if episode.stamp().source != SOURCE {
                return Err(BehaviorError::Invalid("abort source"));
            }
            let EpisodeData::Trace { fields, .. } = episode.data() else {
                unreachable!()
            };
            if fields.len() != 4
                || fields.get("actions") != Some(&EpisodeValue::Integer(i64::from(actions)))
            {
                return Err(BehaviorError::Invalid("abort schema"));
            }
            let word = decode_word(field_text(fields, "input")?)?;
            word_valid(&word, actions, budget.max_word)?;
            if word.is_empty() {
                return Err(BehaviorError::Invalid("abort word"));
            }
            let output = field_text(fields, "output")?;
            if output != "none" && decode_reply(output)?.len() > word.len() {
                return Err(BehaviorError::Invalid("abort reply"));
            }
            if !matches!(fields.get("failure"), Some(EpisodeValue::Integer(1..=4))) {
                return Err(BehaviorError::Invalid("abort failure"));
            }
            queries += 1;
            steps += word.len();
            suspended = true;
        }
        if queries > budget.max_queries || steps > budget.max_steps {
            return Err(BehaviorError::Limit("replayed experience"));
        }
        Ok(Self {
            store,
            context: context.to_owned(),
            actions,
            cache,
            prefix_outputs,
            budget,
            queries,
            steps,
            suspended,
            authority,
            adapter,
        })
    }
    pub fn queries(&self) -> usize {
        self.queries
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    pub fn cached_words(&self) -> usize {
        self.cache.len()
    }
    pub fn prefix_trie_nodes(&self) -> usize {
        self.prefix_outputs.len() + 1
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    pub fn fresh(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        self.run(word, None)
    }
    fn run(&mut self, word: &[u8], expected: Option<&[u16]>) -> Result<Reply, BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        word_valid(word, self.actions, self.budget.max_word)?;
        if word.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(outputs) = expected {
            if outputs.len() != word.len() {
                return Err(BehaviorError::Invalid("expected reply length"));
            }
            if outputs.iter().enumerate().any(|(i, &out)| {
                self.prefix_outputs
                    .get(&word[..=i])
                    .is_some_and(|&old| old != out)
            }) {
                return Err(BehaviorError::Invalid(
                    "model contradicts stored experience",
                ));
            }
        }
        for effect in [Effect::Simulate, Effect::WriteData] {
            if !self.authority.allows(effect) {
                return Err(BehaviorError::Unauthorized(effect));
            }
        }
        if self.queries >= self.budget.max_queries
            || self.steps + word.len() > self.budget.max_steps
        {
            return Err(BehaviorError::Limit("query/step budget"));
        }
        let stamp = EpisodeStamp::new(
            SOURCE,
            format!("query-{}", self.store.next_id()),
            self.store.next_id(),
        );
        let placeholder = vec![0; word.len()];
        self.store.check_record(
            &stamp,
            &query_trace(&self.context, self.actions, word, &placeholder),
        )?;
        self.store.check_record(
            &stamp,
            &abort_trace(&self.context, self.actions, word, &placeholder, 4),
        )?;
        self.queries += 1;
        self.steps += word.len();
        let mut reply = Vec::new();
        let mut failure = self.adapter.reset().err().map(|e| (1, e.to_string()));
        if failure.is_none() {
            for (i, &action) in word.iter().enumerate() {
                match self.adapter.step(action) {
                    Ok(output) => reply.push(output),
                    Err(error) => {
                        failure = Some((2, error.to_string()));
                        break;
                    }
                }
                if expected.is_some_and(|outputs| outputs[i] != reply[i]) {
                    failure = Some((3, "model drift".into()));
                    break;
                }
                if self
                    .prefix_outputs
                    .get(&word[..=i])
                    .is_some_and(|&old| old != reply[i])
                {
                    failure = Some((4, "inconsistent prefix output".into()));
                    break;
                }
            }
        }
        if let Some((code, reason)) = failure {
            self.suspended = true;
            if let Err(error) = self.store.record(
                stamp,
                abort_trace(&self.context, self.actions, word, &reply, code),
            ) {
                return Err(BehaviorError::QueryOccurred {
                    reply,
                    reason: format!("{reason}; recording failed: {error}"),
                });
            }
            return Err(BehaviorError::QueryOccurred { reply, reason });
        }
        if let Err(e) = self.store.record(
            stamp,
            query_trace(&self.context, self.actions, word, &reply),
        ) {
            self.suspended = true;
            return Err(BehaviorError::QueryOccurred {
                reply,
                reason: e.to_string(),
            });
        }
        for (i, &out) in reply.iter().enumerate() {
            self.prefix_outputs.insert(word[..=i].to_vec(), out);
        }
        self.cache.insert(word.to_vec(), reply.clone());
        Ok(reply)
    }
    pub fn execute_goal(
        &mut self,
        model: &BehaviorModel,
        history: &[u8],
        goal: u16,
        forbidden: &BTreeSet<u16>,
    ) -> Result<Option<Reply>, BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        if model.context != self.context || model.actions != self.actions {
            return Err(BehaviorError::Invalid("execution context"));
        }
        let Some(plan) = model.plan_output(history, goal, forbidden)? else {
            return Ok(None);
        };
        let mut word = history.to_vec();
        word.extend(plan);
        word_valid(&word, self.actions, self.budget.max_word)?;
        let expected = model.predict(&word)?;
        if expected.iter().any(|o| forbidden.contains(o)) {
            return Err(BehaviorError::Invalid("forbidden history/output"));
        }
        let actual = self.run(&word, Some(&expected))?;
        Ok(Some(actual))
    }
}
impl<F> SequenceOracle for BehaviorMemory<'_, F>
where
    F: Resettable,
{
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        if self.suspended {
            return Err(BehaviorError::Suspended);
        }
        word_valid(word, self.actions, self.budget.max_word)?;
        if word.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(reply) = self.cache.get(word) {
            return Ok(reply.clone());
        }
        if let Some(reply) = (1..=word.len())
            .map(|n| self.prefix_outputs.get(&word[..n]).copied())
            .collect::<Option<Vec<_>>>()
        {
            return Ok(reply);
        }
        self.fresh(word)
    }
}
fn query_trace(context: &str, actions: u8, word: &[u8], reply: &[u16]) -> EpisodeData {
    let input = word.iter().map(|a| format!("{a:02x}")).collect::<String>();
    let output = reply.iter().map(|o| format!("{o:04x}")).collect::<String>();
    EpisodeData::Trace {
        context: context.to_owned(),
        channel: CHANNEL.to_owned(),
        fields: BTreeMap::from([
            (
                "actions".to_owned(),
                EpisodeValue::Integer(i64::from(actions)),
            ),
            ("input".to_owned(), EpisodeValue::Text(input)),
            ("output".to_owned(), EpisodeValue::Text(output)),
        ]),
    }
}
fn abort_trace(context: &str, actions: u8, word: &[u8], reply: &[u16], code: i64) -> EpisodeData {
    let EpisodeData::Trace {
        context,
        mut fields,
        ..
    } = query_trace(context, actions, word, reply)
    else {
        unreachable!()
    };
    if reply.is_empty() {
        fields.insert("output".into(), EpisodeValue::Text("none".into()));
    }
    fields.insert("failure".into(), EpisodeValue::Integer(code));
    EpisodeData::Trace {
        context,
        channel: ABORT_CHANNEL.into(),
        fields,
    }
}
fn field_text<'a>(
    fields: &'a BTreeMap<String, EpisodeValue>,
    key: &str,
) -> Result<&'a str, BehaviorError> {
    match fields.get(key) {
        Some(EpisodeValue::Text(t)) => Ok(t),
        _ => Err(BehaviorError::Invalid("trace field")),
    }
}
fn decode_word(s: &str) -> Result<Word, BehaviorError> {
    if !s.is_ascii() || !s.len().is_multiple_of(2) {
        return Err(BehaviorError::Invalid("trace word encoding"));
    }
    s.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| {
            u8::from_str_radix(std::str::from_utf8(b).expect("ASCII"), 16)
                .map_err(|_| BehaviorError::Invalid("trace word hex"))
        })
        .collect()
}
fn decode_reply(s: &str) -> Result<Reply, BehaviorError> {
    if !s.is_ascii() || !s.len().is_multiple_of(4) {
        return Err(BehaviorError::Invalid("trace reply encoding"));
    }
    s.as_bytes()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            u16::from_str_radix(std::str::from_utf8(b).expect("ASCII"), 16)
                .map_err(|_| BehaviorError::Invalid("trace reply hex"))
        })
        .collect()
}
