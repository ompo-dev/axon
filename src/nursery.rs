//! Persistent, non-repeating evolutionary search. More digits expand an interpreted
//! genotype space, not the instruction set of the underlying learner.
use crate::{EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue, behavior_lab::Rng};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::Path,
};

const CONTEXT: &str = "axon/nursery-v1";
#[derive(Debug)]
pub enum NurseryError {
    Invalid(&'static str),
    Storage(crate::EpisodeError),
}
impl fmt::Display for NurseryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => f.write_str(s),
            Self::Storage(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NurseryError {}
impl From<crate::EpisodeError> for NurseryError {
    fn from(e: crate::EpisodeError) -> Self {
        Self::Storage(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchConfig {
    pub maximum: u64,
    pub population: usize,
    pub patience: usize,
    pub max_candidates: usize,
    pub max_encoded_bytes: u64,
    pub max_evaluation_ns: u64,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            maximum: 32767,
            population: 64,
            patience: 3,
            max_candidates: 40000,
            max_encoded_bytes: 131072,
            max_evaluation_ns: 1_000_000_000,
        }
    }
}
impl SearchConfig {
    fn validate(self) -> Result<(), NurseryError> {
        if self.maximum > 9_999_999_999_999_999_999
            || !(2..=256).contains(&self.population)
            || !(1..=100).contains(&self.patience)
            || !(1..=40000).contains(&self.max_candidates)
            || self.max_encoded_bytes == 0
            || self.max_evaluation_ns == 0
            || self.max_encoded_bytes > i64::MAX as u64
            || self.max_evaluation_ns > i64::MAX as u64
        {
            return Err(NurseryError::Invalid("nursery limits"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assessment {
    pub checked: u64,
    pub correct: u64,
    pub wrong: u64,
    pub encoded_bytes: u64,
    pub elapsed_ns: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Individual {
    pub dna: u64,
    pub parent: Option<u64>,
    pub assessment: Assessment,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Proposal {
    pub dna: u64,
    pub parent: Option<u64>,
    pub digits: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchStop {
    Winner,
    BatchLimit,
    CandidateLimit,
    SpaceExhausted,
    UnconfirmedExhaustion,
    HistoricalExhaustion,
}
impl SearchStop {
    pub fn name(self) -> &'static str {
        match self {
            Self::Winner => "winner",
            Self::BatchLimit => "batch-limit",
            Self::CandidateLimit => "candidate-limit",
            Self::SpaceExhausted => "space-exhausted",
            Self::UnconfirmedExhaustion => "space-exhausted-with-unfinished-evaluations",
            Self::HistoricalExhaustion => "space-exhausted-with-imported-history",
        }
    }
}

pub struct Nursery {
    store: EpisodeStore,
    config: SearchConfig,
    seen: BTreeSet<u64>,
    results: BTreeMap<u64, Individual>,
    digits: u32,
    stale: usize,
    rng: Rng,
    pending: Option<Proposal>,
    growths: usize,
    checked: Option<u64>,
    best: Option<Individual>,
    front: Vec<Individual>,
    imported: BTreeSet<u64>,
}
fn integer(n: u64) -> Result<EpisodeValue, NurseryError> {
    Ok(EpisodeValue::Integer(i64::try_from(n).map_err(|_| {
        NurseryError::Invalid("record integer overflow")
    })?))
}
fn text(s: impl ToString) -> EpisodeValue {
    EpisodeValue::Text(s.to_string())
}
fn get_u(f: &BTreeMap<String, EpisodeValue>, name: &str) -> Result<u64, NurseryError> {
    match f.get(name) {
        Some(EpisodeValue::Integer(n)) if *n >= 0 => Ok(*n as u64),
        Some(EpisodeValue::Text(n)) => n
            .parse()
            .map_err(|_| NurseryError::Invalid("registry number")),
        _ => Err(NurseryError::Invalid("registry field")),
    }
}
fn capacity(digits: u32, maximum: u64) -> u64 {
    10u64
        .checked_pow(digits)
        .unwrap_or(u64::MAX)
        .saturating_sub(1)
        .min(maximum)
}
fn power(digits: u32) -> u64 {
    10u64.checked_pow(digits).unwrap_or(u64::MAX)
}
impl Nursery {
    /// Scope must describe the exact decoder, challenge suite and scoring contract.
    /// Same journal + different scope/config is refused instead of reusing stale fitness.
    pub fn open(
        path: &Path,
        scope: &str,
        config: SearchConfig,
        rng: u64,
    ) -> Result<Self, NurseryError> {
        config.validate()?;
        if scope.is_empty() || scope.len() > 3500 {
            return Err(NurseryError::Invalid("registry scope"));
        }
        let (mut store, _) = EpisodeStore::open(path)?;
        let identity = format!("scope={scope};config={config:?}");
        let meta = EpisodeData::Trace {
            context: CONTEXT.into(),
            channel: "metadata".into(),
            fields: BTreeMap::from([("identity".into(), text(identity))]),
        };
        if store.is_empty() {
            store.record(EpisodeStamp::new("nursery-meta", "v1", 0), meta.clone())?;
        }
        if store.episode(1)?.data() != &meta {
            return Err(NurseryError::Invalid(
                "registry belongs to another decoder, suite or budget",
            ));
        }
        let mut nursery = Self {
            store,
            config,
            seen: BTreeSet::new(),
            results: BTreeMap::new(),
            digits: 1,
            stale: 0,
            rng: Rng::new(rng),
            pending: None,
            growths: 0,
            checked: None,
            best: None,
            front: Vec::new(),
            imported: BTreeSet::new(),
        };
        for id in 2..=nursery.store.len() as u64 {
            let episode = nursery.store.episode(id)?;
            let EpisodeData::Trace {
                context,
                channel,
                fields,
            } = episode.data()
            else {
                return Err(NurseryError::Invalid("foreign registry record"));
            };
            if context != CONTEXT {
                return Err(NurseryError::Invalid("registry context"));
            }
            match channel.as_str() {
                "import" => {
                    let dna = get_u(fields, "dna")?;
                    if dna > config.maximum || !nursery.seen.insert(dna) {
                        return Err(NurseryError::Invalid("invalid historical DNA"));
                    }
                    nursery.imported.insert(dna);
                }
                "reservation" => {
                    let dna = get_u(fields, "dna")?;
                    let digits = get_u(fields, "digits")? as u32;
                    if dna > capacity(digits, nursery.config.maximum)
                        || digits != nursery.digits
                        || !nursery.seen.insert(dna)
                    {
                        return Err(NurseryError::Invalid("duplicate or invalid reserved DNA"));
                    }
                }
                "result" => {
                    let dna = get_u(fields, "dna")?;
                    let parent = match fields.get("parent") {
                        Some(EpisodeValue::Text(s)) if s == "_" => None,
                        _ => Some(get_u(fields, "parent")?),
                    };
                    let a = Assessment {
                        checked: get_u(fields, "checked")?,
                        correct: get_u(fields, "correct")?,
                        wrong: get_u(fields, "wrong")?,
                        encoded_bytes: get_u(fields, "bytes")?,
                        elapsed_ns: get_u(fields, "ns")?,
                    };
                    nursery.validate_assessment(dna, parent, a)?;
                    nursery.install(Individual {
                        dna,
                        parent,
                        assessment: a,
                    });
                }
                "growth" => {
                    let digits = get_u(fields, "digits")? as u32;
                    if digits != nursery.digits + 1
                        || capacity(nursery.digits, nursery.config.maximum)
                            == nursery.config.maximum
                    {
                        return Err(NurseryError::Invalid("invalid growth record"));
                    }
                    nursery.digits = digits;
                    nursery.growths += 1;
                    nursery.stale = 0;
                }
                _ => return Err(NurseryError::Invalid("registry channel")),
            }
        }
        Ok(nursery)
    }
    fn eligible(&self, a: Assessment) -> bool {
        a.encoded_bytes <= self.config.max_encoded_bytes
            && a.elapsed_ns <= self.config.max_evaluation_ns
    }
    fn quality(a: Assessment) -> (u64, std::cmp::Reverse<u64>) {
        (a.correct, std::cmp::Reverse(a.wrong))
    }
    fn validate_assessment(
        &self,
        dna: u64,
        parent: Option<u64>,
        a: Assessment,
    ) -> Result<(), NurseryError> {
        if !self.seen.contains(&dna)
            || self.imported.contains(&dna)
            || self.results.contains_key(&dna)
            || a.checked == 0
            || a.checked > 1_000_000
            || a.correct > a.checked
            || a.wrong > a.checked - a.correct
            || a.encoded_bytes == 0
            || a.encoded_bytes > i64::MAX as u64
            || a.elapsed_ns > i64::MAX as u64
            || self.checked.is_some_and(|n| n != a.checked)
            || parent.is_some_and(|p| !self.results.contains_key(&p))
        {
            return Err(NurseryError::Invalid(
                "invalid, stale or duplicate assessment",
            ));
        }
        Ok(())
    }
    fn install(&mut self, individual: Individual) {
        let before = self
            .best()
            .filter(|i| self.eligible(i.assessment))
            .map(|i| Self::quality(i.assessment));
        if self.eligible(individual.assessment)
            && before.is_none_or(|q| Self::quality(individual.assessment) > q)
        {
            self.stale = 0;
            self.front.clear();
        } else {
            self.stale += 1;
        }
        if self.eligible(individual.assessment)
            && before.is_none_or(|q| Self::quality(individual.assessment) >= q)
        {
            let a = individual.assessment;
            if !self.front.iter().any(|i| {
                i.assessment.encoded_bytes <= a.encoded_bytes
                    && i.assessment.elapsed_ns <= a.elapsed_ns
            }) {
                self.front.retain(|i| {
                    !(a.encoded_bytes <= i.assessment.encoded_bytes
                        && a.elapsed_ns <= i.assessment.elapsed_ns)
                });
                self.front.push(individual);
                self.front
                    .sort_by_key(|i| (i.assessment.encoded_bytes, i.assessment.elapsed_ns, i.dna));
            }
        }
        let key = |i: Individual| {
            (
                self.eligible(i.assessment),
                Self::quality(i.assessment),
                std::cmp::Reverse(i.assessment.encoded_bytes),
                std::cmp::Reverse(i.assessment.elapsed_ns),
                std::cmp::Reverse(i.dna),
            )
        };
        if self.best.is_none_or(|i| key(individual) > key(i)) {
            self.best = Some(individual);
        }
        self.checked = Some(individual.assessment.checked);
        self.results.insert(individual.dna, individual);
    }
    pub fn best(&self) -> Option<Individual> {
        self.best
    }
    /// Non-dominated memory/time winners of the best observed correctness level.
    pub fn parents(&self) -> Vec<Individual> {
        if self.front.len() <= 8 {
            return self.front.clone();
        }
        self.front
            .iter()
            .take(7)
            .chain(self.front.last())
            .copied()
            .collect()
    }
    pub fn seen(&self) -> usize {
        self.seen.len()
    }
    pub fn completed(&self) -> usize {
        self.results.len()
    }
    pub fn unfinished(&self) -> usize {
        self.seen.len() - self.results.len() - self.imported.len()
    }
    pub fn imported(&self) -> usize {
        self.imported.len()
    }
    pub fn snapshot_new(&mut self, path: &Path) -> Result<(), NurseryError> {
        self.store.snapshot_new(path)?;
        Ok(())
    }
    /// Exclude provably identical historical genotypes without inventing fitness
    /// for a different challenge suite or executing those individuals again.
    pub fn import_seen(&mut self, dna: u64, origin: &str) -> Result<bool, NurseryError> {
        if dna > self.config.maximum {
            return Err(NurseryError::Invalid("historical DNA outside decoder"));
        }
        if self.seen.contains(&dna) {
            return Ok(false);
        }
        if self.seen.len() >= self.config.max_candidates {
            return Err(NurseryError::Invalid("historical registry candidate quota"));
        }
        self.record(
            "import",
            dna.to_string(),
            BTreeMap::from([("dna".into(), text(dna)), ("origin".into(), text(origin))]),
        )?;
        self.seen.insert(dna);
        self.imported.insert(dna);
        Ok(true)
    }
    pub fn digits(&self) -> u32 {
        self.digits
    }
    pub fn growths(&self) -> usize {
        self.growths
    }
    pub fn entries(&self) -> impl Iterator<Item = &Individual> {
        self.results.values()
    }
    pub fn stop(&self) -> Option<SearchStop> {
        if self.best().is_some_and(|i| {
            self.eligible(i.assessment)
                && i.assessment.correct == i.assessment.checked
                && i.assessment.wrong == 0
        }) {
            return Some(SearchStop::Winner);
        }
        self.exhaustion()
    }
    fn exhaustion(&self) -> Option<SearchStop> {
        if self.seen.len() as u128 == self.config.maximum as u128 + 1 {
            return Some(if self.unfinished() != 0 {
                SearchStop::UnconfirmedExhaustion
            } else if self.imported() > 0 {
                SearchStop::HistoricalExhaustion
            } else {
                SearchStop::SpaceExhausted
            });
        }
        if self.seen.len() >= self.config.max_candidates {
            return Some(SearchStop::CandidateLimit);
        }
        None
    }
    fn record(
        &mut self,
        channel: &str,
        event: String,
        fields: BTreeMap<String, EpisodeValue>,
    ) -> Result<(), NurseryError> {
        self.store.record(
            EpisodeStamp::new(format!("nursery-{channel}"), event, 0),
            EpisodeData::Trace {
                context: CONTEXT.into(),
                channel: channel.into(),
                fields,
            },
        )?;
        Ok(())
    }
    fn grow(&mut self) -> Result<(), NurseryError> {
        let next = self.digits + 1;
        self.record(
            "growth",
            next.to_string(),
            BTreeMap::from([("digits".into(), integer(next as u64)?)]),
        )?;
        self.digits = next;
        self.growths += 1;
        self.stale = 0;
        Ok(())
    }
    /// Reserve durably before executing. After a crash unfinished DNA stays consumed;
    /// it is never evaluated twice or silently promoted as a winner.
    pub fn propose(&mut self) -> Result<Option<Proposal>, NurseryError> {
        self.propose_inner(false)
    }
    /// Continue optimizing cost/coverage after a correct individual, within the same hard caps.
    pub fn propose_optimization(&mut self) -> Result<Option<Proposal>, NurseryError> {
        self.propose_inner(true)
    }
    fn propose_inner(&mut self, optimize: bool) -> Result<Option<Proposal>, NurseryError> {
        if self.pending.is_some() {
            return Err(NurseryError::Invalid(
                "complete the pending evaluation first",
            ));
        }
        if (if optimize {
            self.exhaustion()
        } else {
            self.stop()
        })
        .is_some()
        {
            return Ok(None);
        }
        let mut cap = capacity(self.digits, self.config.maximum);
        if cap < self.config.maximum
            && (self.stale >= self.config.patience * self.config.population
                || self.seen.range(..=cap).count() as u128 == cap as u128 + 1)
        {
            self.grow()?;
            cap = capacity(self.digits, self.config.maximum);
        }
        let parents = self.parents();
        let mut chosen = None;
        for _ in 0..64 {
            let parent = if parents.is_empty() {
                None
            } else {
                Some(parents[self.rng.pick(parents.len())].dna)
            };
            let (n, parent) = if let Some(p) = parent.filter(|_| self.rng.pick(4) != 0) {
                let at = self.rng.pick(self.digits as usize) as u32;
                let place = power(at);
                let digit = self.rng.pick(10) as u64;
                (p - (p / place % 10) * place + digit * place, Some(p))
            } else {
                (self.rng.next_u64() % (cap + 1), None)
            };
            if n <= cap && !self.seen.contains(&n) {
                chosen = Some((n, parent));
                break;
            }
        }
        // Exhaustive fallback prevents duplicate rejection from falsely declaring stagnation.
        let (dna, parent) = chosen.unwrap_or_else(|| {
            (
                (0..=cap)
                    .find(|n| !self.seen.contains(n))
                    .expect("unseen capacity"),
                None,
            )
        });
        self.record(
            "reservation",
            dna.to_string(),
            BTreeMap::from([
                ("dna".into(), text(dna)),
                ("digits".into(), integer(self.digits as u64)?),
            ]),
        )?;
        self.seen.insert(dna);
        let proposal = Proposal {
            dna,
            parent,
            digits: self.digits,
        };
        self.pending = Some(proposal);
        Ok(Some(proposal))
    }
    pub fn commit(&mut self, p: Proposal, a: Assessment) -> Result<(), NurseryError> {
        if self.pending != Some(p) {
            return Err(NurseryError::Invalid("unreserved evaluation"));
        }
        self.validate_assessment(p.dna, p.parent, a)?;
        self.record(
            "result",
            p.dna.to_string(),
            BTreeMap::from([
                ("dna".into(), text(p.dna)),
                (
                    "parent".into(),
                    text(
                        p.parent
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "_".into()),
                    ),
                ),
                ("checked".into(), integer(a.checked)?),
                ("correct".into(), integer(a.correct)?),
                ("wrong".into(), integer(a.wrong)?),
                ("bytes".into(), integer(a.encoded_bytes)?),
                ("ns".into(), integer(a.elapsed_ns)?),
            ]),
        )?;
        self.install(Individual {
            dna: p.dna,
            parent: p.parent,
            assessment: a,
        });
        self.pending = None;
        Ok(())
    }
}
