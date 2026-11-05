//! Active structural transfer between finite resettable interfaces.
//! The prior asserts a family, not that the external system really belongs to it.
use crate::behavior::{BehaviorEdge, BehaviorError, BehaviorModel, Reply, SequenceOracle, Word};
use crate::live_store::checksum;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    fmt,
    mem::size_of,
};

const MAX_ACTIONS: usize = 6;
const MAX_SYMBOLS: usize = 32;
const UNBOUND: u32 = 65_536;

#[derive(Debug)]
pub enum TransferError {
    Invalid(&'static str),
    Limit(&'static str),
    Refuted,
    NoUsefulProbe,
    Suspended,
    Oracle(BehaviorError),
}
impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "structural transfer: {self:?}")
    }
}
impl std::error::Error for TransferError {}
impl From<BehaviorError> for TransferError {
    fn from(e: BehaviorError) -> Self {
        Self::Oracle(e)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TransferBudget {
    pub max_hypotheses: usize,
    pub max_queries: usize,
    pub max_steps: usize,
    pub max_word: usize,
    pub max_probes: usize,
    pub max_work: u64,
}
impl Default for TransferBudget {
    fn default() -> Self {
        Self {
            max_hypotheses: 65_536,
            max_queries: 128,
            max_steps: 4096,
            max_word: 48,
            max_probes: 1024,
            max_work: 100_000_000,
        }
    }
}
impl TransferBudget {
    fn validate(self) -> Result<(), TransferError> {
        if self.max_hypotheses == 0
            || self.max_hypotheses > 65_536
            || self.max_queries == 0
            || self.max_queries > 512
            || self.max_steps == 0
            || self.max_steps > 32_768
            || self.max_word == 0
            || self.max_word > 48
            || self.max_probes == 0
            || self.max_probes > 4096
            || self.max_work == 0
            || self.max_work > 200_000_000
        {
            return Err(TransferError::Invalid("budget"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct TransferStats {
    pub initial_hypotheses: usize,
    pub surviving_hypotheses: usize,
    pub queries: usize,
    pub steps: usize,
    pub work: u64,
    pub peak_hypothesis_bytes: usize,
}
struct Prototype<'a> {
    model: &'a BehaviorModel,
    symbols: Vec<u16>,
    transitions: Vec<(u16, u8)>,
}
#[derive(Clone, Copy)]
struct Hypothesis {
    prototype: usize,
    initial: u16,
    actions: [u8; MAX_ACTIONS],
    outputs: [u32; MAX_SYMBOLS],
}
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
struct Signature {
    length: u8,
    tokens: [u32; 48],
}

pub struct TransferLearner<'a> {
    prototypes: Vec<Prototype<'a>>,
    hypotheses: Vec<Hypothesis>,
    context: String,
    actions: u8,
    budget: TransferBudget,
    stats: TransferStats,
    observations: BTreeMap<Word, Reply>,
    digest: u64,
    suspended: bool,
}
#[derive(Debug)]
pub struct TransferResult {
    pub model: BehaviorModel,
    pub source_fingerprints: Vec<u64>,
    pub stats: TransferStats,
    pub evidence_digest: u64,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferProbe {
    pub word: Word,
    pub eliminated_pairs: u64,
    pub novel_bindings: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeSelection {
    Compact,
    Reference,
}

impl<'a> TransferLearner<'a> {
    pub fn new(
        models: &[&'a BehaviorModel],
        actions: u8,
        context: &str,
        budget: TransferBudget,
    ) -> Result<Self, TransferError> {
        budget.validate()?;
        if models.len() > 16
            || actions == 0
            || actions > 16
            || context.is_empty()
            || context.len() > 256
            || context.trim() != context
            || context.chars().any(char::is_control)
        {
            return Err(TransferError::Invalid("atlas/interface/context"));
        }
        if models.is_empty() {
            return Err(TransferError::Refuted);
        }
        if actions as usize > MAX_ACTIONS {
            return Err(TransferError::Limit("transfer action alphabet"));
        }
        let mut prototypes = Vec::new();
        let mut fingerprints = BTreeSet::new();
        for &model in models {
            if model.actions() != actions {
                continue;
            }
            if !fingerprints.insert(model.fingerprint()) {
                continue;
            }
            let mut symbols = BTreeSet::new();
            for state in 0..model.states() {
                for action in 0..actions {
                    symbols.insert(model.edge(state as u16, action)?.output);
                }
            }
            if symbols.len() > MAX_SYMBOLS {
                return Err(TransferError::Limit("prototype output alphabet"));
            }
            let symbols = symbols.into_iter().collect::<Vec<_>>();
            let mut transitions = Vec::new();
            for state in 0..model.states() {
                for action in 0..actions {
                    let edge = model.edge(state as u16, action)?;
                    transitions.push((
                        edge.next,
                        symbols.binary_search(&edge.output).expect("alphabet") as u8,
                    ));
                }
            }
            prototypes.push(Prototype {
                model,
                symbols,
                transitions,
            });
        }
        if prototypes.is_empty() {
            return Err(TransferError::Refuted);
        }
        let mut factorial = 1_usize;
        for n in 1..=actions as usize {
            factorial *= n;
        }
        let count: usize = prototypes
            .iter()
            .map(|p| p.model.states() * factorial)
            .sum();
        if count > budget.max_hypotheses {
            return Err(TransferError::Limit("hypothesis expansion"));
        }
        let mut mappings = Vec::new();
        fn permutations(
            actions: u8,
            depth: usize,
            map: &mut [u8; MAX_ACTIONS],
            used: u8,
            output: &mut Vec<[u8; MAX_ACTIONS]>,
        ) {
            if depth == actions as usize {
                output.push(*map);
                return;
            }
            for a in 0..actions {
                if used & (1 << a) == 0 {
                    map[depth] = a;
                    permutations(actions, depth + 1, map, used | (1 << a), output);
                }
            }
        }
        permutations(actions, 0, &mut [0; MAX_ACTIONS], 0, &mut mappings);
        let mut hypotheses = Vec::with_capacity(count);
        for (prototype, p) in prototypes.iter().enumerate() {
            for initial in 0..p.model.states() {
                for &actions in &mappings {
                    hypotheses.push(Hypothesis {
                        prototype,
                        initial: initial as u16,
                        actions,
                        outputs: [UNBOUND; MAX_SYMBOLS],
                    });
                }
            }
        }
        Ok(Self {
            prototypes,
            hypotheses,
            context: context.into(),
            actions,
            budget,
            stats: TransferStats {
                initial_hypotheses: count,
                surviving_hypotheses: count,
                peak_hypothesis_bytes: count * size_of::<Hypothesis>(),
                ..TransferStats::default()
            },
            observations: BTreeMap::new(),
            digest: 0,
            suspended: false,
        })
    }
    pub fn stats(&self) -> TransferStats {
        self.stats
    }
    pub fn suspended(&self) -> bool {
        self.suspended
    }
    fn ready(&self) -> Result<(), TransferError> {
        if self.suspended {
            Err(TransferError::Suspended)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, amount: u64) -> Result<(), TransferError> {
        let work = self
            .stats
            .work
            .checked_add(amount)
            .ok_or(TransferError::Limit("work overflow"))?;
        if work > self.budget.max_work {
            return Err(TransferError::Limit("search work"));
        }
        self.stats.work = work;
        Ok(())
    }
    fn word(&self, word: &[u8]) -> Result<(), TransferError> {
        if word.is_empty()
            || word.len() > self.budget.max_word
            || word.iter().any(|&a| a >= self.actions)
        {
            return Err(TransferError::Invalid("probe word"));
        }
        Ok(())
    }
    pub fn preflight(&self, word: &[u8]) -> Result<(), TransferError> {
        self.ready()?;
        self.word(word)?;
        if self.observations.contains_key(word) {
            return Err(TransferError::Invalid("duplicate probe"));
        }
        if self.stats.queries >= self.budget.max_queries
            || self.stats.steps + word.len() > self.budget.max_steps
            || self.stats.work + (self.hypotheses.len() * word.len()) as u64 > self.budget.max_work
        {
            return Err(TransferError::Limit("experiment budget"));
        }
        Ok(())
    }
    pub fn observe(&mut self, word: &[u8], reply: &[u16]) -> Result<(), TransferError> {
        self.preflight(word)?;
        if reply.len() != word.len() {
            self.suspended = true;
            return Err(TransferError::Invalid("reply length"));
        }
        self.charge((self.hypotheses.len() * word.len()) as u64)?;
        self.stats.queries += 1;
        self.stats.steps += word.len();
        let mut digest = self.digest.to_le_bytes().to_vec();
        digest.extend_from_slice(&(word.len() as u16).to_le_bytes());
        digest.extend_from_slice(word);
        for o in reply {
            digest.extend_from_slice(&o.to_le_bytes());
        }
        self.digest = checksum(&digest);
        self.observations.insert(word.to_vec(), reply.to_vec());
        let prototypes = &self.prototypes;
        self.hypotheses.retain_mut(|h| {
            let p = &prototypes[h.prototype];
            let mut state = h.initial;
            for (&action, &actual) in word.iter().zip(reply) {
                let edge = p
                    .model
                    .edge(state, h.actions[action as usize])
                    .expect("validated mapping");
                let symbol = p
                    .symbols
                    .binary_search(&edge.output)
                    .expect("prototype alphabet");
                let value = u32::from(actual);
                if h.outputs[symbol] == UNBOUND {
                    if h.outputs[..p.symbols.len()].contains(&value) {
                        return false;
                    }
                    h.outputs[symbol] = value;
                } else if h.outputs[symbol] != value {
                    return false;
                }
                state = edge.next;
            }
            true
        });
        self.stats.surviving_hypotheses = self.hypotheses.len();
        if self.hypotheses.is_empty() {
            self.suspended = true;
            return Err(TransferError::Refuted);
        }
        if self.hypotheses.len() * 4 < self.hypotheses.capacity()
            && self.hypotheses.capacity() > 256
        {
            self.hypotheses.shrink_to_fit();
        }
        Ok(())
    }
    fn signature(&self, h: &Hypothesis, word: &[u8]) -> (Signature, u64) {
        let p = &self.prototypes[h.prototype];
        let mut state = h.initial;
        let mut fresh = [u8::MAX; MAX_SYMBOLS];
        let mut n = 0_u8;
        let mut sig = Signature {
            length: word.len() as u8,
            tokens: [0; 48],
        };
        for (i, &action) in word.iter().enumerate() {
            let edge = p
                .model
                .edge(state, h.actions[action as usize])
                .expect("valid");
            let symbol = p.symbols.binary_search(&edge.output).expect("alphabet");
            sig.tokens[i] = if h.outputs[symbol] != UNBOUND {
                h.outputs[symbol]
            } else {
                if fresh[symbol] == u8::MAX {
                    fresh[symbol] = n;
                    n += 1;
                }
                UNBOUND + u32::from(fresh[symbol])
            };
            state = edge.next;
        }
        (sig, u64::from(n))
    }
    fn probe_pool(&self) -> Result<Vec<Word>, TransferError> {
        let mut pool = BTreeSet::new();
        let mut layer = vec![Vec::new()];
        for _ in 0..3.min(self.budget.max_word) {
            let mut next = Vec::new();
            for prefix in layer {
                for action in 0..self.actions {
                    let mut word = prefix.clone();
                    word.push(action);
                    if pool.len() < self.budget.max_probes {
                        pool.insert(word.clone());
                    }
                    next.push(word);
                }
            }
            layer = next;
        }
        // Transition covers from one remaining hypothesis per prototype reach rare symbols.
        let mut covered = BTreeSet::new();
        for h in &self.hypotheses {
            if !covered.insert(h.prototype) {
                continue;
            }
            let p = &self.prototypes[h.prototype];
            let mut access = vec![None; p.model.states()];
            access[h.initial as usize] = Some(Vec::<u8>::new());
            let mut queue = VecDeque::from([h.initial]);
            while let Some(state) = queue.pop_front() {
                let prefix = access[state as usize].clone().expect("visited");
                for action in 0..self.actions {
                    let mut word = prefix.clone();
                    word.push(action);
                    if word.len() <= self.budget.max_word && pool.len() < self.budget.max_probes {
                        pool.insert(word.clone());
                    }
                    let edge = p.model.edge(state, h.actions[action as usize])?;
                    if access[edge.next as usize].is_none() && word.len() < self.budget.max_word {
                        access[edge.next as usize] = Some(word);
                        queue.push_back(edge.next);
                    }
                }
            }
        }
        Ok(pool
            .into_iter()
            .filter(|w| !self.observations.contains_key(w))
            .collect())
    }
    pub fn choose_probe(&mut self) -> Result<Option<TransferProbe>, TransferError> {
        self.select_probe(false)
    }
    /// Slow, exact reference for checking the compact selection implementation.
    pub fn choose_probe_reference(&mut self) -> Result<Option<TransferProbe>, TransferError> {
        self.select_probe(true)
    }
    fn compact_signature(&self, h: &Hypothesis, word: &[u8]) -> (u128, u64) {
        let p = &self.prototypes[h.prototype];
        let mut state = h.initial;
        let mut fresh = [u8::MAX; MAX_SYMBOLS];
        let mut n = 0_u8;
        let mut key = 0_u128;
        for (i, &action) in word.iter().enumerate() {
            let (next, symbol) = p.transitions
                [state as usize * self.actions as usize + h.actions[action as usize] as usize];
            let symbol = symbol as usize;
            let token = if h.outputs[symbol] != UNBOUND {
                h.outputs[symbol]
            } else {
                if fresh[symbol] == u8::MAX {
                    fresh[symbol] = n;
                    n += 1;
                }
                UNBOUND + u32::from(fresh[symbol])
            };
            key |= u128::from(token) << (17 * i);
            state = next;
        }
        (key, u64::from(n))
    }
    fn select_probe(&mut self, reference: bool) -> Result<Option<TransferProbe>, TransferError> {
        self.ready()?;
        if self.stats.queries >= self.budget.max_queries {
            return Err(TransferError::Limit("queries"));
        }
        let mut best: Option<TransferProbe> = None;
        let mut keys = Vec::with_capacity(self.hypotheses.len());
        for word in self.probe_pool()? {
            if self.stats.steps + word.len() > self.budget.max_steps {
                continue;
            }
            self.charge((self.hypotheses.len() * word.len()) as u64)?;
            let mut novelty = 0_u64;
            let squared = if !reference && word.len() <= 7 {
                // Seven 17-bit tokens fit losslessly in u128; sorting avoids per-group nodes.
                keys.clear();
                for h in &self.hypotheses {
                    let (key, n) = self.compact_signature(h, &word);
                    keys.push(key);
                    novelty += n;
                }
                keys.sort_unstable();
                let mut squared = 0_u64;
                let mut i = 0;
                while i < keys.len() {
                    let mut j = i + 1;
                    while j < keys.len() && keys[j] == keys[i] {
                        j += 1;
                    }
                    squared += ((j - i) as u64).pow(2);
                    i = j;
                }
                squared
            } else {
                let mut groups = BTreeMap::<Signature, u64>::new();
                for h in &self.hypotheses {
                    let (sig, n) = self.signature(h, &word);
                    *groups.entry(sig).or_default() += 1;
                    novelty += n;
                }
                groups.values().map(|n| n * n).sum()
            };
            let n = self.hypotheses.len() as u64;
            let gain = n * n - squared;
            if gain == 0 && novelty == 0 {
                continue;
            }
            let better = best.as_ref().is_none_or(|b| {
                let left = (
                    u128::from(gain) * b.word.len() as u128,
                    u128::from(novelty) * b.word.len() as u128,
                );
                let right = (
                    u128::from(b.eliminated_pairs) * word.len() as u128,
                    u128::from(b.novel_bindings) * word.len() as u128,
                );
                left > right || (left == right && word < b.word)
            });
            if better {
                best = Some(TransferProbe {
                    word,
                    eliminated_pairs: gain,
                    novel_bindings: novelty,
                });
            }
        }
        Ok(best)
    }
    fn realize(&self, h: &Hypothesis) -> Result<Option<BehaviorModel>, TransferError> {
        let p = &self.prototypes[h.prototype];
        let mut ids = vec![None; p.model.states()];
        ids[h.initial as usize] = Some(0_u16);
        let mut states = vec![h.initial];
        let mut rows = Vec::new();
        let mut i = 0;
        while i < states.len() {
            let state = states[i];
            let mut row = Vec::new();
            for action in 0..self.actions {
                let edge = p.model.edge(state, h.actions[action as usize])?;
                let symbol = p.symbols.binary_search(&edge.output).expect("alphabet");
                if h.outputs[symbol] == UNBOUND {
                    return Ok(None);
                }
                let next = if let Some(id) = ids[edge.next as usize] {
                    id
                } else {
                    let id = states.len() as u16;
                    ids[edge.next as usize] = Some(id);
                    states.push(edge.next);
                    id
                };
                row.push(BehaviorEdge {
                    next,
                    output: h.outputs[symbol] as u16,
                });
            }
            rows.push(row);
            i += 1;
        }
        Ok(Some(BehaviorModel::from_edges(&self.context, rows)?))
    }
    pub fn compile(&mut self) -> Result<Option<TransferResult>, TransferError> {
        self.ready()?;
        let mut model: Option<BehaviorModel> = None;
        let mut sources = BTreeSet::new();
        for i in 0..self.hypotheses.len() {
            let h = self.hypotheses[i];
            self.charge(
                (self.prototypes[h.prototype].model.states() * self.actions as usize) as u64,
            )?;
            let Some(candidate) = self.realize(&h)? else {
                return Ok(None);
            };
            if model.as_ref().is_some_and(|m| m != &candidate) {
                return Ok(None);
            }
            model = Some(candidate);
            sources.insert(self.prototypes[h.prototype].model.fingerprint());
        }
        Ok(model.map(|model| TransferResult {
            model,
            source_fingerprints: sources.into_iter().collect(),
            stats: self.stats,
            evidence_digest: self.digest,
        }))
    }
    pub fn learn(
        &mut self,
        oracle: &mut impl SequenceOracle,
    ) -> Result<TransferResult, TransferError> {
        self.learn_with_selection(oracle, ProbeSelection::Compact)
    }
    pub fn learn_with_selection(
        &mut self,
        oracle: &mut impl SequenceOracle,
        selection: ProbeSelection,
    ) -> Result<TransferResult, TransferError> {
        loop {
            if let Some(result) = self.compile()? {
                return Ok(result);
            }
            let probe = match selection {
                ProbeSelection::Compact => self.choose_probe()?,
                ProbeSelection::Reference => self.choose_probe_reference()?,
            }
            .ok_or(TransferError::NoUsefulProbe)?;
            self.preflight(&probe.word)?;
            let reply = match oracle.query(&probe.word) {
                Ok(r) => r,
                Err(e) => {
                    self.suspended = true;
                    return Err(e.into());
                }
            };
            self.observe(&probe.word, &reply)?;
        }
    }
}

/// Distinguishing suffixes computed from the candidate, never from the real environment.
fn characterization(model: &BehaviorModel) -> Result<BTreeSet<Word>, TransferError> {
    let mut suffixes = BTreeSet::from([Vec::new()]);
    let mut work = 0_u64;
    for left in 0..model.states() {
        for right in left + 1..model.states() {
            let mut queue = VecDeque::from([(left as u16, right as u16, Vec::<u8>::new())]);
            let mut visited = BTreeSet::from([(left as u16, right as u16)]);
            let mut truncated = false;
            let mut distinguished = false;
            'search: while let Some((l, r, prefix)) = queue.pop_front() {
                for a in 0..model.actions() {
                    work += 1;
                    if work > 2_000_000 {
                        return Err(TransferError::Limit("characterization work"));
                    }
                    let le = model.edge(l, a)?;
                    let re = model.edge(r, a)?;
                    let mut word = prefix.clone();
                    word.push(a);
                    if le.output != re.output {
                        suffixes.insert(word);
                        distinguished = true;
                        break 'search;
                    }
                    if visited.insert((le.next, re.next)) {
                        if word.len() < 48 {
                            queue.push_back((le.next, re.next, word));
                        } else {
                            truncated = true;
                        }
                    }
                }
            }
            if truncated && !distinguished {
                return Err(TransferError::Limit("characterization word"));
            }
        }
    }
    Ok(suffixes)
}

/// Bounded conformance, not an equivalence oracle or proof of family membership.
pub fn conformance_probe(
    model: &BehaviorModel,
    oracle: &mut impl SequenceOracle,
    max_tests: usize,
) -> Result<Option<Word>, TransferError> {
    if max_tests == 0 || max_tests > 65_536 {
        return Err(TransferError::Invalid("conformance tests"));
    }
    let suffixes = characterization(model)?;
    let accesses = model.access_sequences();
    let mut middles = vec![Vec::new()];
    for a in 0..model.actions() {
        middles.push(vec![a]);
        for b in 0..model.actions() {
            middles.push(vec![a, b]);
        }
    }
    let count = accesses
        .len()
        .checked_mul(middles.len())
        .and_then(|n| n.checked_mul(suffixes.len()))
        .ok_or(TransferError::Limit("conformance work"))?;
    let exploration_depth = if model.actions() <= 6 { 4 } else { 3 };
    let exploration_count = (model.actions() as usize).pow(exploration_depth);
    if count + exploration_count > max_tests {
        return Err(TransferError::Limit("conformance tests"));
    }
    for prefix in accesses {
        for middle in &middles {
            for suffix in &suffixes {
                let mut word = prefix.clone();
                word.extend_from_slice(middle);
                word.extend_from_slice(suffix);
                if word.len() > 48 {
                    return Err(TransferError::Limit("conformance word"));
                }
                let actual = oracle.query(&word)?;
                if actual.len() != word.len() {
                    return Err(TransferError::Invalid("conformance reply"));
                }
                if model.predict(&word)? != actual {
                    return Ok(Some(word));
                }
            }
        }
    }
    // A candidate may hide all undiscovered states behind a longer prefix. Apply the
    // same fixed-depth exploration to transfer and discovery, not only its own states.
    for mut index in 0..exploration_count {
        let mut word = vec![0; exploration_depth as usize];
        for a in word.iter_mut().rev() {
            *a = (index % model.actions() as usize) as u8;
            index /= model.actions() as usize;
        }
        let actual = oracle.query(&word)?;
        if actual.len() != word.len() {
            return Err(TransferError::Invalid("exploration reply"));
        }
        if model.predict(&word)? != actual {
            return Ok(Some(word));
        }
    }
    Ok(None)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptationMode {
    StructuralTransfer,
    Discovery,
}
pub struct Adaptation {
    pub model: BehaviorModel,
    pub mode: AdaptationMode,
    pub transfer_stats: TransferStats,
    pub transfer_failure: Option<String>,
    pub source_fingerprints: Vec<u64>,
    pub evidence_digest: u64,
}

/// A refuted/inconclusive prior changes the strategy; it does not redefine observations.
pub fn adapt(
    models: &[&BehaviorModel],
    actions: u8,
    context: &str,
    budget: TransferBudget,
    oracle: &mut impl SequenceOracle,
) -> Result<Adaptation, TransferError> {
    adapt_with_selection(
        models,
        actions,
        context,
        budget,
        oracle,
        ProbeSelection::Compact,
    )
}
pub fn adapt_with_selection(
    models: &[&BehaviorModel],
    actions: u8,
    context: &str,
    budget: TransferBudget,
    oracle: &mut impl SequenceOracle,
    selection: ProbeSelection,
) -> Result<Adaptation, TransferError> {
    let mut stats = TransferStats::default();
    let attempt = match TransferLearner::new(models, actions, context, budget) {
        Ok(mut learner) => {
            let result = learner.learn_with_selection(oracle, selection);
            stats = learner.stats();
            result
        }
        Err(e) => Err(e),
    };
    let failure = match attempt {
        Ok(result) => match conformance_probe(&result.model, oracle, 65_536)? {
            None => {
                return Ok(Adaptation {
                    model: result.model,
                    mode: AdaptationMode::StructuralTransfer,
                    transfer_stats: stats,
                    transfer_failure: None,
                    source_fingerprints: result.source_fingerprints,
                    evidence_digest: result.evidence_digest,
                });
            }
            Some(word) => format!("prior contradicted by conformance witness {word:?}"),
        },
        Err(
            e @ (TransferError::Refuted | TransferError::NoUsefulProbe | TransferError::Limit(_)),
        ) => e.to_string(),
        Err(e) => return Err(e),
    };
    let model = discover(actions, context, oracle)?;
    Ok(Adaptation {
        model,
        mode: AdaptationMode::Discovery,
        transfer_stats: stats,
        transfer_failure: Some(failure),
        source_fingerprints: Vec::new(),
        evidence_digest: 0,
    })
}
/// The same candidate-based conformance is used in the from-scratch baseline.
pub fn discover(
    actions: u8,
    context: &str,
    oracle: &mut impl SequenceOracle,
) -> Result<BehaviorModel, TransferError> {
    let mut learner = crate::behavior::BehaviorLearner::new(
        actions,
        context,
        crate::behavior::BehaviorBudget::default(),
    )?;
    for _ in 0..64 {
        let model = learner.hypothesize(oracle)?;
        match conformance_probe(&model, oracle, 65_536)? {
            None => return Ok(model),
            Some(word) => learner.refine(&model, &word, oracle)?,
        }
    }
    Err(TransferError::Limit("discovery rounds"))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoContact;
    impl SequenceOracle for NoContact {
        fn query(&mut self, _: &[u8]) -> Result<Reply, BehaviorError> {
            panic!("a truncated characterization must refuse before contact");
        }
    }
    #[test]
    fn depth_bound_is_not_mistaken_for_indistinguishable_states() {
        let edges = (0..52)
            .map(|state| {
                vec![BehaviorEdge {
                    next: (state + 1).min(51),
                    output: u16::from(state == 51),
                }]
            })
            .collect();
        let model = BehaviorModel::from_edges("long-chain", edges).unwrap();
        assert!(matches!(
            conformance_probe(&model, &mut NoContact, 65_536),
            Err(TransferError::Limit("characterization word"))
        ));
    }

    #[test]
    fn compact_tokens_pack_exactly_and_long_probes_keep_the_reference_path() {
        let rows = (0..32)
            .map(|state| {
                vec![
                    BehaviorEdge {
                        next: (state + 1) % 32,
                        output: state,
                    },
                    BehaviorEdge {
                        next: state,
                        output: state,
                    },
                ]
            })
            .collect();
        let model = BehaviorModel::from_edges("packing", rows).unwrap();
        let mut learner =
            TransferLearner::new(&[&model], 2, "packing-target", TransferBudget::default())
                .unwrap();
        learner.observe(&[0], &[65535]).unwrap();
        for h in &learner.hypotheses {
            for len in 1..=7 {
                let word = vec![0; len];
                let (signature, n) = learner.signature(h, &word);
                let expected = signature.tokens[..len]
                    .iter()
                    .enumerate()
                    .fold(0_u128, |key, (i, &t)| key | (u128::from(t) << (17 * i)));
                assert_eq!(learner.compact_signature(h, &word), (expected, n));
            }
        }
        assert!(
            learner
                .probe_pool()
                .unwrap()
                .iter()
                .any(|word| word.len() > 7)
        );
        assert_eq!(
            learner.choose_probe().unwrap(),
            learner.choose_probe_reference().unwrap()
        );
    }
}
