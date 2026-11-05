//! Independent local systems and auditors; never passed as latent state to the learner.
use crate::behavior::*;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::PathBuf,
};

pub const WORKFLOW_ACTIONS: u8 = 11;
pub const ACK: u16 = 100;
pub const PUBLISHED: u16 = 201;
pub const DENIED: u16 = 409;
pub struct DocumentWorkflow {
    phase: u8,
    authenticated: bool,
    pub mutant: bool,
    payload: Vec<u8>,
    output: Option<PathBuf>,
}
impl DocumentWorkflow {
    pub fn new(payload: Vec<u8>, output: Option<PathBuf>) -> Self {
        Self {
            phase: 0,
            authenticated: false,
            mutant: false,
            payload,
            output,
        }
    }
    pub fn hidden_state(&self) -> u16 {
        u16::from(self.phase) * 2 + u16::from(self.authenticated)
    }
    pub fn set_audit_state(&mut self, state: u16) {
        self.phase = (state / 2) as u8;
        self.authenticated = state % 2 == 1;
    }
    pub fn action_name(action: u8) -> &'static str {
        [
            "authenticate",
            "create",
            "ingest",
            "format-check",
            "size-check",
            "utf8-check",
            "review-mark",
            "seal-mark",
            "publish",
            "logout",
            "status",
        ][action as usize]
    }
}
impl Resettable for DocumentWorkflow {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.phase = 0;
        self.authenticated = false;
        Ok(())
    }
    fn step(&mut self, action: u8) -> Result<u16, BehaviorError> {
        match action {
            0 => {
                self.authenticated = true;
                Ok(ACK)
            }
            9 => {
                self.authenticated = false;
                Ok(ACK)
            }
            10 => Ok(match self.phase {
                0 => 404,
                8 => PUBLISHED,
                _ => 102,
            }),
            1..=8 if !self.authenticated => Ok(401),
            1 => {
                self.phase = 1;
                Ok(ACK)
            }
            2 => {
                if self.phase > 0 && !(self.mutant && self.phase == 7) {
                    self.phase = 2;
                }
                Ok(ACK)
            }
            3..=7 => {
                let valid_payload = self.payload.len() >= 20
                    && self.payload.len() <= 65_536
                    && self.payload.starts_with(b"# ")
                    && !self.payload.contains(&0)
                    && std::str::from_utf8(&self.payload).is_ok();
                if self.phase == action - 1 && valid_payload {
                    self.phase = action;
                }
                Ok(ACK)
            }
            8 if self.phase == 7 => {
                if let Some(path) = &self.output {
                    crate::text_skill::publish_new(path, &self.payload)?;
                }
                self.phase = 8;
                Ok(PUBLISHED)
            }
            8 => Ok(DENIED),
            _ => Err(BehaviorError::Invalid("workflow action")),
        }
    }
}
pub struct RawOracle<A: Resettable> {
    pub adapter: A,
    pub queries: usize,
    pub steps: usize,
}
impl<A: Resettable> RawOracle<A> {
    pub fn new(adapter: A) -> Self {
        Self {
            adapter,
            queries: 0,
            steps: 0,
        }
    }
}
impl<A: Resettable> SequenceOracle for RawOracle<A> {
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        self.adapter.reset()?;
        self.queries += 1;
        self.steps += word.len();
        word.iter().map(|&a| self.adapter.step(a)).collect()
    }
}
pub struct CachedOracle<A: Resettable> {
    pub raw: RawOracle<A>,
    cache: BTreeMap<Word, u16>,
    max_queries: usize,
    max_steps: usize,
}
impl<A: Resettable> CachedOracle<A> {
    pub fn new(adapter: A, budget: BehaviorBudget) -> Self {
        Self {
            raw: RawOracle::new(adapter),
            cache: BTreeMap::new(),
            max_queries: budget.max_queries,
            max_steps: budget.max_steps,
        }
    }
    pub fn trie_nodes(&self) -> usize {
        self.cache.len() + 1
    }
}
impl<A: Resettable> SequenceOracle for CachedOracle<A> {
    fn query(&mut self, word: &[u8]) -> Result<Reply, BehaviorError> {
        if let Some(reply) = (1..=word.len())
            .map(|n| self.cache.get(&word[..n]).copied())
            .collect::<Option<Vec<_>>>()
        {
            return Ok(reply);
        }
        if self.raw.queries >= self.max_queries {
            return Err(BehaviorError::Limit("lab queries"));
        }
        if self.raw.steps + word.len() > self.max_steps {
            return Err(BehaviorError::Limit("lab steps"));
        }
        let reply = self.raw.query(word)?;
        for (i, &out) in reply.iter().enumerate() {
            self.cache.insert(word[..=i].to_vec(), out);
        }
        Ok(reply)
    }
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn pick(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
#[derive(Clone)]
pub struct HiddenMachine {
    pub actions: u8,
    transitions: Vec<Vec<(usize, u16)>>,
    state: usize,
    decoration: usize,
}
impl HiddenMachine {
    pub fn generate(rng: &mut Rng) -> Self {
        let actions = (2 + rng.pick(3)) as u8;
        let states = 8 + rng.pick(25);
        let outputs = 2 + rng.pick(3);
        let mut transitions = Vec::new();
        for state in 0..states {
            let mut row = Vec::new();
            for action in 0..actions {
                // One input supplies a reachable cycle; other effects remain opaque.
                let next = if action == 0 {
                    (state + 1) % states
                } else {
                    rng.pick(states)
                };
                row.push((next, rng.pick(outputs) as u16));
            }
            transitions.push(row);
        }
        Self {
            actions,
            transitions,
            state: 0,
            decoration: 0,
        }
    }
    pub fn states(&self) -> usize {
        self.transitions.len()
    }
    pub fn transition(&self, state: usize, action: u8) -> (usize, u16) {
        self.transitions[state][action as usize]
    }
    pub fn minimal_states(&self) -> usize {
        let mut groups = vec![0; self.states()];
        loop {
            let mut signatures = BTreeMap::new();
            let mut next = Vec::new();
            for row in &self.transitions {
                let sig: Vec<_> = row.iter().map(|&(s, o)| (o, groups[s])).collect();
                let n = signatures.len();
                next.push(*signatures.entry(sig).or_insert(n));
            }
            if next == groups {
                return groups.iter().copied().collect::<BTreeSet<_>>().len();
            }
            groups = next;
        }
    }
}
impl Resettable for HiddenMachine {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        self.state = 0;
        self.decoration = 0;
        Ok(())
    }
    fn step(&mut self, action: u8) -> Result<u16, BehaviorError> {
        if action >= self.actions {
            return Err(BehaviorError::Invalid("hidden action"));
        }
        let (next, output) = self.transition(self.state, action);
        self.state = next;
        // Dormant, observationally irrelevant internal state must not be inferred as a concept.
        self.decoration = (self.decoration + 1 + action as usize) % 17;
        Ok(output)
    }
}

pub struct ProductAudit {
    pub checked_edges: usize,
    pub checked_pairs: usize,
    pub counterexample: Option<Word>,
}
pub fn audit_machine<F>(
    model: &BehaviorModel,
    actions: u8,
    initial: usize,
    mut truth: F,
) -> Result<ProductAudit, BehaviorError>
where
    F: FnMut(usize, u8) -> Result<(usize, u16), BehaviorError>,
{
    if actions != model.actions() {
        return Err(BehaviorError::Invalid("audit alphabet"));
    }
    let mut queue = VecDeque::from([(initial, 0_u16, Vec::<u8>::new())]);
    let mut visited = BTreeSet::from([(initial, 0_u16)]);
    let mut checked = 0;
    while let Some((hidden, learned, word)) = queue.pop_front() {
        for action in 0..actions {
            let (next_hidden, output) = truth(hidden, action)?;
            let predicted = model.edge(learned, action)?;
            checked += 1;
            let mut prefix = word.clone();
            prefix.push(action);
            if predicted.output != output {
                return Ok(ProductAudit {
                    checked_edges: checked,
                    checked_pairs: visited.len(),
                    counterexample: Some(prefix),
                });
            }
            if visited.insert((next_hidden, predicted.next)) {
                if visited.len() > 16_384 {
                    return Err(BehaviorError::Limit("product audit"));
                }
                queue.push_back((next_hidden, predicted.next, prefix));
            }
        }
    }
    Ok(ProductAudit {
        checked_edges: checked,
        checked_pairs: visited.len(),
        counterexample: None,
    })
}

pub fn discover<O: SequenceOracle>(
    actions: u8,
    context: &str,
    budget: BehaviorBudget,
    oracle: &mut O,
) -> Result<(BehaviorModel, Vec<usize>, usize), BehaviorError> {
    let mut learner = BehaviorLearner::new(actions, context, budget)?;
    let mut revisions = Vec::new();
    for _ in 0..64 {
        let model = learner.hypothesize(oracle)?;
        revisions.push(model.states());
        let mut counterexample = learner.counterexample(&model, oracle, 2, 65_536)?;
        if counterexample.is_none() {
            let mut rng = Rng::new(0x7198b302);
            for _ in 0..2_048 {
                let n = budget.max_word;
                let word: Word = (0..n).map(|_| rng.pick(actions as usize) as u8).collect();
                let actual = oracle.query(&word)?;
                if actual.len() != word.len() {
                    return Err(BehaviorError::Invalid("sample reply length"));
                }
                if let Some(i) = actual
                    .iter()
                    .zip(model.predict(&word)?)
                    .position(|(&x, y)| x != y)
                {
                    counterexample = Some(word[..=i].to_vec());
                    break;
                }
            }
        }
        match counterexample {
            Some(mut word) => {
                // Reduce a witnessed behavioral failure using new black-box queries only.
                let mut index = 0;
                while index < word.len() {
                    let mut shorter = word.clone();
                    shorter.remove(index);
                    let actual = oracle.query(&shorter)?;
                    if actual.len() != shorter.len() {
                        return Err(BehaviorError::Invalid("reduced reply length"));
                    }
                    if actual != model.predict(&shorter)? {
                        word = shorter;
                        index = 0;
                    } else {
                        index += 1;
                    }
                }
                learner.refine(&model, &word, oracle)?;
            }
            None => return Ok((model, revisions, learner.suffixes().len())),
        }
    }
    Err(BehaviorError::Limit("refinement rounds"))
}

/// Compare a learned contract with a fresh black box, without latent-state access.
pub fn regression_probe<O: SequenceOracle>(
    model: &BehaviorModel,
    oracle: &mut O,
    max_tests: usize,
) -> Result<Option<Word>, BehaviorError> {
    if max_tests == 0 || max_tests > 65_536 {
        return Err(BehaviorError::Invalid("regression budget"));
    }
    let mut tests = 0;
    for access in model.access_sequences() {
        for a in 0..model.actions() {
            for b in 0..model.actions() {
                tests += 1;
                if tests > max_tests {
                    return Err(BehaviorError::Limit("regression probes"));
                }
                let mut word = access.clone();
                word.extend([a, b]);
                let expected = model.predict(&word)?;
                let observed = oracle.query(&word)?;
                if observed.len() != word.len() {
                    return Err(BehaviorError::Invalid("regression reply length"));
                }
                if observed != expected {
                    return Ok(Some(word));
                }
            }
        }
    }
    Ok(None)
}
