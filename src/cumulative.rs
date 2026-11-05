//! Bounded cumulative acquisition: inherited programs/policy, active probes and durable observations.
use crate::{
    Authority, Effect, EpisodeData, EpisodeError, EpisodeReceipt, EpisodeStamp, EpisodeStore,
    EpisodeValue,
    grounded_language::LanguageFrame,
    live_store::checksum,
    numeric::{library::*, *},
    text_skill::{publish_new, read_bounded},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    mem::size_of,
    path::Path,
};

const MAGIC: &[u8; 8] = b"AXST0001";
pub const MAX_BYTES: usize = 131072;
const CHANNEL: &str = "cumulative/observation-v1";
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuestionPolicy {
    Split,
    Random,
}
#[derive(Clone, Copy, Debug)]
pub struct StudyPolicy {
    pub questions: usize,
    pub question_policy: QuestionPolicy,
    pub search: NumericBudget,
    pub attempts: usize,
    pub total_work: u64,
}
impl Default for StudyPolicy {
    fn default() -> Self {
        Self {
            questions: 16,
            question_policy: QuestionPolicy::Split,
            search: NumericBudget {
                max_cost: 5,
                max_terms: 20000,
                max_work: 2000000,
            },
            attempts: 4,
            total_work: 4000000,
        }
    }
}
impl StudyPolicy {
    pub fn validate(self) -> Result<(), StudyError> {
        self.search.validate()?;
        if !(1..=32).contains(&self.questions)
            || !(1..=8).contains(&self.attempts)
            || self.total_work == 0
            || self.total_work > 100000000
            || self.search.max_work > self.total_work
        {
            return Err(StudyError::Invalid("policy limits"));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum StudyError {
    Invalid(&'static str),
    Budget(&'static str),
    Unauthorized(Effect),
    Ambiguous,
    NeedsExamples,
    Conflict,
    Evidence { search_work: u64 },
    Numeric(NumericError),
    Episode(EpisodeError),
    Storage(crate::TextSkillError),
    Frame(String),
}
impl fmt::Display for StudyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid study: {s}"),
            Self::Budget(s) => write!(f, "study budget: {s}"),
            Self::Unauthorized(e) => write!(f, "study effect not authorized: {e:?}"),
            Self::Ambiguous => {
                f.write_str("multiple live hypotheses; ask a discriminating question")
            }
            Self::NeedsExamples => f.write_str("new mechanism needs demonstrations"),
            Self::Conflict => f.write_str("conflicting observations; start a new explicit epoch"),
            Self::Evidence { .. } => f.write_str(
                "induced program violates an observed value or domain; no acquisition admitted",
            ),
            Self::Numeric(e) => e.fmt(f),
            Self::Episode(e) => e.fmt(f),
            Self::Storage(e) => e.fmt(f),
            Self::Frame(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for StudyError {}
impl From<NumericError> for StudyError {
    fn from(e: NumericError) -> Self {
        Self::Numeric(e)
    }
}
impl From<EpisodeError> for StudyError {
    fn from(e: EpisodeError) -> Self {
        Self::Episode(e)
    }
}
impl From<crate::TextSkillError> for StudyError {
    fn from(e: crate::TextSkillError) -> Self {
        Self::Storage(e)
    }
}
fn require(a: &Authority, e: Effect) -> Result<(), StudyError> {
    if a.allows(e) {
        Ok(())
    } else {
        Err(StudyError::Unauthorized(e))
    }
}
fn label(s: &str) -> Result<(), StudyError> {
    if s.is_empty() || s.len() > 128 || s.trim() != s || s.chars().any(char::is_control) {
        Err(StudyError::Invalid("context"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StudySeed {
    context: String,
    revision: u32,
    parent: u64,
    policy: StudyPolicy,
    skills: Vec<NumericSkill>,
    frames: Vec<LanguageFrame>,
    library: Option<NumericLibrary>,
}
impl StudySeed {
    pub fn new(
        context: &str,
        skills: Vec<NumericSkill>,
        frames: Vec<LanguageFrame>,
        policy: StudyPolicy,
    ) -> Result<Self, StudyError> {
        label(context)?;
        policy.validate()?;
        if skills.len() > 32 || frames.len() > 4 {
            return Err(StudyError::Budget("seed components"));
        }
        let library = if skills.is_empty() {
            None
        } else {
            Some(NumericLibrary::mine(
                context,
                &skills.iter().collect::<Vec<_>>(),
                MiningBudget::default(),
            )?)
        };
        let seed = Self {
            context: context.into(),
            revision: 0,
            parent: 0,
            policy,
            skills,
            frames,
            library,
        };
        seed.validate()?;
        Ok(seed)
    }
    fn validate(&self) -> Result<(), StudyError> {
        label(&self.context)?;
        self.policy.validate()?;
        if self.skills.len() > 32
            || self.frames.len() > 4
            || self.revision > 1000000
            || (self.revision == 0) != (self.parent == 0)
            || self.to_bytes()?.len() > MAX_BYTES
        {
            return Err(StudyError::Invalid("seed bounds/lineage"));
        }
        let mut identities = BTreeSet::new();
        if self
            .skills
            .iter()
            .any(|s| !identities.insert(s.fingerprint()))
        {
            return Err(StudyError::Invalid("duplicate inherited skill"));
        }
        if self.skills.is_empty() != self.library.is_none() {
            return Err(StudyError::Invalid("library/source relation"));
        }
        if let Some(library) = &self.library {
            let replay = NumericLibrary::mine(
                &self.context,
                &self.skills.iter().collect::<Vec<_>>(),
                MiningBudget::default(),
            )?;
            if replay.fingerprint() != library.fingerprint() {
                return Err(StudyError::Invalid("library structural replay"));
            }
        }
        Ok(())
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn policy(&self) -> StudyPolicy {
        self.policy
    }
    pub fn revision(&self) -> u32 {
        self.revision
    }
    pub fn parent(&self) -> u64 {
        self.parent
    }
    pub fn skills(&self) -> &[NumericSkill] {
        &self.skills
    }
    pub fn frames(&self) -> &[LanguageFrame] {
        &self.frames
    }
    pub fn library(&self) -> Option<&NumericLibrary> {
        self.library.as_ref()
    }
    pub fn fingerprint(&self) -> u64 {
        checksum(&self.to_bytes().expect("validated components"))
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.context.capacity()
            + self.skills.capacity() * size_of::<NumericSkill>()
            + self
                .skills
                .iter()
                .map(|s| s.retained_bytes() - size_of::<NumericSkill>())
                .sum::<usize>()
            + self.frames.capacity() * size_of::<LanguageFrame>()
            + self
                .frames
                .iter()
                .map(|f| f.retained_bytes() - size_of::<LanguageFrame>())
                .sum::<usize>()
            + self
                .library
                .as_ref()
                .map_or(0, |l| l.retained_bytes() - size_of::<NumericLibrary>())
    }
    pub fn with_policy(&self, policy: StudyPolicy) -> Result<Self, StudyError> {
        policy.validate()?;
        let mut child = self.clone();
        child.policy = policy;
        child.revision = self
            .revision
            .checked_add(1)
            .ok_or(StudyError::Budget("revision"))?;
        child.parent = self.fingerprint();
        child.validate()?;
        Ok(child)
    }
    pub fn consolidate(&self, learned: &[NumericSkill]) -> Result<Self, StudyError> {
        let mut skills = self.skills.clone();
        for skill in learned {
            if !skills
                .iter()
                .any(|s| s.fingerprint() == skill.fingerprint())
            {
                skills.push(skill.clone());
            }
        }
        let mut child = Self::new(&self.context, skills, self.frames.clone(), self.policy)?;
        child.revision = self
            .revision
            .checked_add(1)
            .ok_or(StudyError::Budget("revision"))?;
        child.parent = self.fingerprint();
        child.validate()?;
        Ok(child)
    }
    pub fn offspring(&self, rng: u64) -> Result<Self, StudyError> {
        let mut child = self.clone();
        let mut random = crate::behavior_lab::Rng::new(rng);
        child.policy.question_policy = if random.pick(2) == 0 {
            QuestionPolicy::Split
        } else {
            QuestionPolicy::Random
        };
        child.policy.search.max_cost =
            (child.policy.search.max_cost as isize + random.pick(3) as isize - 1).clamp(3, 10)
                as usize;
        child.revision = self
            .revision
            .checked_add(1)
            .ok_or(StudyError::Budget("revision"))?;
        child.parent = self.fingerprint();
        child.validate()?;
        Ok(child)
    }
    pub fn check_descendant(&self, parent: &Self) -> Result<(), StudyError> {
        if self.context != parent.context
            || self.parent != parent.fingerprint()
            || self.revision != parent.revision + 1
            || self
                .frames
                .iter()
                .map(LanguageFrame::identities)
                .collect::<Vec<_>>()
                != parent
                    .frames
                    .iter()
                    .map(LanguageFrame::identities)
                    .collect::<Vec<_>>()
            || parent.skills.iter().any(|old| {
                !self
                    .skills
                    .iter()
                    .any(|new| new.fingerprint() == old.fingerprint())
            })
        {
            return Err(StudyError::Invalid("knowledge preservation/lineage"));
        }
        self.validate()
    }
    pub fn save_new(&self, path: &Path, a: &Authority) -> Result<(), StudyError> {
        require(a, Effect::WriteData)?;
        self.validate()?;
        publish_new(path, &self.to_bytes()?)?;
        Ok(())
    }
    pub fn open(path: &Path, a: &Authority) -> Result<Self, StudyError> {
        require(a, Effect::ReadData)?;
        Self::from_bytes(&read_bounded(path, MAX_BYTES as u64)?)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, StudyError> {
        let mut b = MAGIC.to_vec();
        section(&mut b, self.context.as_bytes());
        b.extend(self.revision.to_le_bytes());
        b.extend(self.parent.to_le_bytes());
        b.push(self.policy.questions as u8);
        b.push(u8::from(
            self.policy.question_policy == QuestionPolicy::Random,
        ));
        b.push(self.policy.search.max_cost as u8);
        b.extend((self.policy.search.max_terms as u32).to_le_bytes());
        b.extend(self.policy.search.max_work.to_le_bytes());
        b.push(self.policy.attempts as u8);
        b.extend(self.policy.total_work.to_le_bytes());
        b.push(self.skills.len() as u8);
        for skill in &self.skills {
            section(&mut b, &skill.to_bytes());
        }
        b.push(self.frames.len() as u8);
        for frame in &self.frames {
            section(&mut b, &frame.to_bytes().map_err(StudyError::Frame)?);
        }
        b.push(u8::from(self.library.is_some()));
        if let Some(l) = &self.library {
            section(&mut b, &l.to_bytes());
        }
        b.extend(checksum(&b).to_le_bytes());
        Ok(b)
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, StudyError> {
        if b.len() < 40 || b.len() > MAX_BYTES || &b[..8] != MAGIC {
            return Err(StudyError::Invalid("seed header/size"));
        }
        let body = b.len() - 8;
        if checksum(&b[..body]) != u64::from_le_bytes(b[body..].try_into().expect("width")) {
            return Err(StudyError::Invalid("seed checksum"));
        }
        let mut r = Reader(&b[8..body]);
        let context = std::str::from_utf8(r.section(128)?)
            .map_err(|_| StudyError::Invalid("context UTF8"))?
            .to_owned();
        let revision = r.u32()?;
        let parent = r.u64()?;
        let questions = r.byte()? as usize;
        let question_policy = match r.byte()? {
            0 => QuestionPolicy::Split,
            1 => QuestionPolicy::Random,
            _ => return Err(StudyError::Invalid("question policy")),
        };
        let policy = StudyPolicy {
            questions,
            question_policy,
            search: NumericBudget {
                max_cost: r.byte()? as usize,
                max_terms: r.u32()? as usize,
                max_work: r.u64()?,
            },
            attempts: r.byte()? as usize,
            total_work: r.u64()?,
        };
        policy.validate()?;
        let count = r.byte()? as usize;
        if count > 32 {
            return Err(StudyError::Budget("skills"));
        }
        let mut skills = Vec::new();
        for _ in 0..count {
            skills.push(NumericSkill::from_bytes(r.section(32768)?)?);
        }
        let count = r.byte()? as usize;
        if count > 4 {
            return Err(StudyError::Budget("frames"));
        }
        let mut frames = Vec::new();
        for _ in 0..count {
            frames
                .push(LanguageFrame::from_bytes(r.section(MAX_BYTES)?).map_err(StudyError::Frame)?);
        }
        let library = match r.byte()? {
            0 => None,
            1 => Some(NumericLibrary::from_bytes(r.section(MAX_BYTES)?)?),
            _ => return Err(StudyError::Invalid("library tag")),
        };
        if !r.0.is_empty() {
            return Err(StudyError::Invalid("seed trailing bytes"));
        }
        let seed = Self {
            context,
            revision,
            parent,
            policy,
            skills,
            frames,
            library,
        };
        seed.validate()?;
        Ok(seed)
    }
}
fn section(b: &mut Vec<u8>, v: &[u8]) {
    b.extend((v.len() as u32).to_le_bytes());
    b.extend(v);
}
struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], StudyError> {
        if n > self.0.len() {
            return Err(StudyError::Invalid("seed truncation"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn byte(&mut self) -> Result<u8, StudyError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, StudyError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("width")))
    }
    fn u64(&mut self) -> Result<u64, StudyError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("width")))
    }
    fn section(&mut self, max: usize) -> Result<&'a [u8], StudyError> {
        let n = self.u32()? as usize;
        if n > max {
            return Err(StudyError::Budget("seed section"));
        }
        self.take(n)
    }
}

pub struct Probe {
    pub input: Vec<f64>,
    pub disagreement: usize,
    pub work: u64,
}
pub struct Acquisition {
    pub skill: NumericSkill,
    pub borrowed: bool,
    pub search_work: u64,
    pub proposal_visits: u64,
}
pub struct Investigator<'a> {
    seed: &'a StudySeed,
    store: EpisodeStore,
    context: String,
    arity: usize,
    inherit: bool,
    observations: Vec<(Vec<f64>, Option<f64>)>,
    alive: Vec<usize>,
    conflict: bool,
}
impl<'a> Investigator<'a> {
    pub fn open(
        seed: &'a StudySeed,
        path: &Path,
        context: &str,
        arity: usize,
        inherit: bool,
        a: &Authority,
    ) -> Result<Self, StudyError> {
        require(a, Effect::ReadData)?;
        require(a, Effect::WriteData)?;
        label(context)?;
        if !(1..=4).contains(&arity) {
            return Err(StudyError::Invalid("arity"));
        }
        let (mut store, _) = EpisodeStore::open(path)?;
        let fields = BTreeMap::from([
            (
                "seed".into(),
                EpisodeValue::Text(format!("{:016X}", seed.fingerprint())),
            ),
            ("arity".into(), EpisodeValue::Integer(arity as i64)),
            ("inherit".into(), EpisodeValue::Boolean(inherit)),
        ]);
        let contracts = store.traces(context, "cumulative/contract-v1")?;
        if contracts.is_empty() {
            store.record(
                EpisodeStamp::new(context, "contract", 0),
                EpisodeData::Trace {
                    context: context.into(),
                    channel: "cumulative/contract-v1".into(),
                    fields,
                },
            )?;
        } else if contracts.len() != 1
            || !matches!(contracts[0].data(),EpisodeData::Trace{fields:old,..} if old==&fields)
        {
            return Err(StudyError::Invalid("journal seed/epoch contract"));
        }
        let mut agent = Self {
            seed,
            store,
            context: context.into(),
            arity,
            inherit,
            observations: Vec::new(),
            alive: Vec::new(),
            conflict: false,
        };
        for event in agent.store.traces(context, CHANNEL)? {
            let EpisodeData::Trace { fields, .. } = event.data() else {
                unreachable!()
            };
            if fields.len() != 2 {
                return Err(StudyError::Invalid("observation fields"));
            }
            let Some(EpisodeValue::Text(input)) = fields.get("input") else {
                return Err(StudyError::Invalid("input trace"));
            };
            let Some(EpisodeValue::Text(actual)) = fields.get("actual") else {
                return Err(StudyError::Invalid("actual trace"));
            };
            let input = parse_input(input, arity)?;
            let actual = if actual == "domain" {
                None
            } else {
                Some(
                    actual
                        .parse::<f64>()
                        .map_err(|_| StudyError::Invalid("actual scalar"))?,
                )
            };
            if actual.is_some_and(|v| !v.is_finite())
                || agent.observations.len() >= seed.policy.questions
            {
                return Err(StudyError::Invalid("observation bounds"));
            }
            agent.observations.push((input, actual));
        }
        agent.rebuild();
        Ok(agent)
    }
    fn rebuild(&mut self) {
        self.conflict = self.observations.iter().enumerate().any(|(i, (x, y))| {
            self.observations[..i]
                .iter()
                .any(|(old, actual)| old == x && !agrees(*actual, *y))
        });
        self.alive = if self.inherit {
            self.seed
                .skills
                .iter()
                .enumerate()
                .filter(|(_, s)| {
                    s.program().arity() == self.arity
                        && self
                            .observations
                            .iter()
                            .all(|(x, y)| agrees(s.predict(s.context(), x).ok(), *y))
                })
                .map(|(i, _)| i)
                .collect()
        } else {
            Vec::new()
        };
    }
    pub fn observations(&self) -> usize {
        self.observations.len()
    }
    pub fn survivors(&self) -> usize {
        self.alive.len()
    }
    pub fn conflicted(&self) -> bool {
        self.conflict
    }
    pub fn journal_bytes(&self) -> Result<u64, StudyError> {
        Ok(self.store.file_bytes()?)
    }
    pub fn next_probe(&self, points: &[Vec<f64>]) -> Result<Option<Probe>, StudyError> {
        if self.conflict {
            return Err(StudyError::Conflict);
        }
        if self.observations.len() >= self.seed.policy.questions {
            return Ok(None);
        }
        if points.len() > 512 {
            return Err(StudyError::Budget("probe pool"));
        }
        for p in points {
            check_input(p, self.arity)?;
        }
        let choices = points
            .iter()
            .filter(|p| !self.observations.iter().any(|(x, _)| x == *p))
            .collect::<Vec<_>>();
        if choices.is_empty() {
            return Ok(None);
        }
        if self.seed.policy.question_policy == QuestionPolicy::Random || self.alive.is_empty() {
            let mut rng = crate::behavior_lab::Rng::new(
                checksum(self.context.as_bytes()).wrapping_add(self.observations.len() as u64),
            );
            return Ok(Some(Probe {
                input: choices[rng.pick(choices.len())].clone(),
                disagreement: 0,
                work: 0,
            }));
        }
        let mut best = (0usize, 0usize);
        let mut work = 0;
        for (i, p) in choices.iter().enumerate() {
            let mut groups: Vec<(Option<f64>, usize)> = Vec::new();
            for &index in &self.alive {
                let s = &self.seed.skills[index];
                work += s.program().instructions() as u64;
                if work > 1000000 {
                    return Err(StudyError::Budget("probe scoring"));
                }
                let value = s.predict(s.context(), p).ok();
                if let Some(group) = groups.iter_mut().find(|(old, _)| agrees(*old, value)) {
                    group.1 += 1;
                } else {
                    groups.push((value, 1));
                }
            }
            let score = self.alive.len() * self.alive.len()
                - groups.iter().map(|(_, n)| n * n).sum::<usize>();
            if score > best.1 {
                best = (i, score);
            }
        }
        Ok(Some(Probe {
            input: choices[best.0].clone(),
            disagreement: best.1,
            work,
        }))
    }
    /// Observation is committed before candidate state changes. No external callback is replayed.
    pub fn observe(
        &mut self,
        stamp: EpisodeStamp,
        input: Vec<f64>,
        actual: Option<f64>,
        a: &Authority,
    ) -> Result<EpisodeReceipt, StudyError> {
        require(a, Effect::WriteData)?;
        check_input(&input, self.arity)?;
        if actual.is_some_and(|v| !v.is_finite()) {
            return Err(StudyError::Invalid("actual scalar"));
        }
        let fields = BTreeMap::from([
            (
                "input".into(),
                EpisodeValue::Text(
                    input
                        .iter()
                        .map(f64::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ),
            (
                "actual".into(),
                EpisodeValue::Text(actual.map_or_else(|| "domain".into(), |v| v.to_string())),
            ),
        ]);
        let data = EpisodeData::Trace {
            context: self.context.clone(),
            channel: CHANNEL.into(),
            fields,
        };
        for previous in self.store.traces(&self.context, CHANNEL)? {
            if previous.stamp().source == stamp.source && previous.stamp().event == stamp.event {
                if previous.stamp() == &stamp && previous.data() == &data {
                    return Ok(EpisodeReceipt {
                        id: previous.id(),
                        duplicate: true,
                    });
                }
                return Err(EpisodeError::EventCollision.into());
            }
        }
        if self.observations.len() >= self.seed.policy.questions {
            return Err(StudyError::Budget("observations"));
        }
        let receipt = self.store.record(stamp, data)?;
        if !receipt.duplicate {
            self.observations.push((input, actual));
            self.rebuild();
        }
        Ok(receipt)
    }
    pub fn acquire(&self) -> Result<Acquisition, StudyError> {
        if self.conflict {
            return Err(StudyError::Conflict);
        }
        if self.observations.is_empty() {
            return Err(StudyError::NeedsExamples);
        }
        if self.alive.len() == 1 {
            return self.checked_acquisition(Acquisition {
                skill: self.seed.skills[self.alive[0]].clone(),
                borrowed: true,
                search_work: 0,
                proposal_visits: 0,
            });
        }
        if self.alive.len() > 1 {
            return Err(StudyError::Ambiguous);
        }
        let examples = self
            .observations
            .iter()
            .filter_map(|(x, y)| y.map(|y| NumericExample::new(x.clone(), y)))
            .collect::<Vec<_>>();
        if examples.len() < 3 {
            return Err(StudyError::NeedsExamples);
        }
        if self.inherit
            && let Some(library) = &self.seed.library
        {
            // Try a small structurally useful portfolio before the mined library.
            // Primitive-equivalent and full-arity tools can crowd out deeper compositions.
            let mut preferred = self
                .seed
                .skills
                .iter()
                .filter(|s| s.program().arity() < self.arity)
                .filter_map(|s| {
                    s.program()
                        .instructions()
                        .checked_sub(s.program().arity() + 1)
                        .filter(|n| *n > 0)
                        .map(|gain| (gain, s))
                })
                .collect::<Vec<_>>();
            preferred.sort_by_key(|(gain, s)| (std::cmp::Reverse(*gain), s.fingerprint()));
            let preferred = preferred
                .into_iter()
                .take(2)
                .map(|(_, s)| s)
                .collect::<Vec<_>>();
            let whole = NumericSkill::learn(
                "acquired",
                &self.context,
                examples.clone(),
                &preferred,
                self.seed.policy.search,
                NumericTolerance::default(),
            );
            let spent = match whole {
                Ok(skill) => {
                    let search_work = skill.search().scalar_work;
                    return self.checked_acquisition(Acquisition {
                        skill,
                        borrowed: false,
                        search_work,
                        proposal_visits: 0,
                    });
                }
                Err(NumericError::Budget(s) | NumericError::NoProgram(s)) => s.scalar_work,
                Err(e) => return Err(e.into()),
            };
            let remaining = self
                .seed
                .policy
                .total_work
                .checked_sub(spent)
                .filter(|n| *n > 0)
                .ok_or(StudyError::Budget("acquisition work"))?;
            let budget = NumericBudget {
                max_work: self.seed.policy.search.max_work.min(remaining),
                ..self.seed.policy.search
            };
            let result = library.develop(
                "acquired",
                &self.context,
                examples,
                budget,
                DevelopmentBudget {
                    max_attempts: self.seed.policy.attempts,
                    max_work: remaining,
                    mining: MiningBudget::default(),
                },
                NumericTolerance::default(),
            );
            match result {
                Ok((skill, trace)) => self.checked_acquisition(Acquisition {
                    skill,
                    borrowed: false,
                    search_work: spent + trace.scalar_work,
                    proposal_visits: trace.proposal_visits,
                }),
                Err(NumericError::Development {
                    attempts,
                    scalar_work,
                    reason,
                }) => Err(NumericError::Development {
                    attempts,
                    scalar_work: spent + scalar_work,
                    reason,
                }
                .into()),
                Err(e) => Err(e.into()),
            }
        } else {
            let budget = NumericBudget {
                max_work: self.seed.policy.total_work,
                ..self.seed.policy.search
            };
            let skill = NumericSkill::learn(
                "acquired",
                &self.context,
                examples,
                &[],
                budget,
                NumericTolerance::default(),
            )?;
            let search_work = skill.search().scalar_work;
            self.checked_acquisition(Acquisition {
                skill,
                borrowed: false,
                search_work,
                proposal_visits: 0,
            })
        }
    }
    fn checked_acquisition(&self, result: Acquisition) -> Result<Acquisition, StudyError> {
        if self.observations.iter().any(|(input, actual)| {
            !agrees(
                result.skill.predict(result.skill.context(), input).ok(),
                *actual,
            )
        }) {
            return Err(StudyError::Evidence {
                search_work: result.search_work,
            });
        }
        Ok(result)
    }
}
fn check_input(x: &[f64], arity: usize) -> Result<(), StudyError> {
    if x.len() != arity || x.iter().any(|v| !v.is_finite() || v.abs() > 1000000.) {
        Err(StudyError::Invalid("input domain/arity"))
    } else {
        Ok(())
    }
}
fn parse_input(s: &str, arity: usize) -> Result<Vec<f64>, StudyError> {
    if s.len() > 256 {
        return Err(StudyError::Invalid("input text"));
    }
    let x = s
        .split(',')
        .map(str::parse::<f64>)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| StudyError::Invalid("input scalar"))?;
    check_input(&x, arity)?;
    Ok(x)
}
fn agrees(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => NumericTolerance::default().agrees(a, b),
        (None, None) => true,
        _ => false,
    }
}
