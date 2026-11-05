//! Bounded event-driven learning with delayed feedback and prospective model admission.
use super::{
    library::{DevelopmentBudget, NumericLibrary},
    *,
};
use std::{collections::VecDeque, mem::size_of};

#[path = "numeric_continuous_snapshot.rs"]
mod snapshot;

#[derive(Clone, Copy, Debug)]
pub struct ContinuousPolicy {
    pub training_window: usize,
    pub min_training: usize,
    pub validation_samples: usize,
    pub validation_distinct: usize,
    pub drift_window: usize,
    pub drift_errors: usize,
    pub pending_capacity: usize,
    pub recent_capacity: usize,
    pub max_label_delay: u64,
    pub archive_models: usize,
    pub archive_bytes: usize,
    pub retry_stride: u64,
    pub search: NumericBudget,
    pub development: DevelopmentBudget,
    pub tolerance: NumericTolerance,
}
impl Default for ContinuousPolicy {
    fn default() -> Self {
        Self {
            training_window: 16,
            min_training: 8,
            validation_samples: 6,
            validation_distinct: 4,
            drift_window: 8,
            drift_errors: 3,
            pending_capacity: 128,
            recent_capacity: 128,
            max_label_delay: 96,
            archive_models: 8,
            archive_bytes: 65536,
            retry_stride: 4,
            search: NumericBudget::default(),
            development: DevelopmentBudget::default(),
            tolerance: NumericTolerance::default(),
        }
    }
}
impl ContinuousPolicy {
    fn validate(self) -> Result<(), NumericError> {
        self.search.validate()?;
        self.tolerance.validate()?;
        self.development.mining.validate()?;
        if !(2..=32).contains(&self.training_window)
            || !(2..=self.training_window).contains(&self.min_training)
            || !(2..=16).contains(&self.validation_samples)
            || !(2..=self.validation_samples).contains(&self.validation_distinct)
            || self.validation_distinct > self.training_window
            || !(2..=32).contains(&self.drift_window)
            || !(2..=self.drift_window).contains(&self.drift_errors)
            || !(1..=1024).contains(&self.pending_capacity)
            || !(1..=1024).contains(&self.recent_capacity)
            || !(1..=4096).contains(&self.max_label_delay)
            || !(1..=64).contains(&self.archive_models)
            || !(1024..=1048576).contains(&self.archive_bytes)
            || !(1..=1024).contains(&self.retry_stride)
            || !(1..=16).contains(&self.development.max_attempts)
            || !(1..=100000000).contains(&self.development.max_work)
        {
            return Err(NumericError::Invalid("continuous policy"));
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ContinuousStatistics {
    pub observations: u64,
    pub predictions: u64,
    pub feedback: u64,
    pub scored: u64,
    pub correct: u64,
    pub expired: u64,
    pub late_for_learning: u64,
    pub drifts: u64,
    pub jobs: u64,
    pub failed_jobs: u64,
    pub stale_jobs: u64,
    pub rejected_candidates: u64,
    pub promotions: u64,
    pub reuses: u64,
    pub search_work: u64,
    pub archive_work: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinuousPhase {
    Collecting,
    Learning,
    Validating,
    Active,
    Suspended,
}
impl ContinuousPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Collecting => "collecting",
            Self::Learning => "learning",
            Self::Validating => "validating",
            Self::Active => "active",
            Self::Suspended => "suspended",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub id: u64,
    pub sequence: u64,
    pub epoch: u64,
    pub model: Option<u64>,
    pub value: Option<f64>,
    pub duplicate: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedbackStatus {
    Accepted,
    Duplicate,
    Expired,
    Historical,
    Drift,
    CandidateRejected,
    Promoted,
    ArchiveFull,
}
impl FeedbackStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Duplicate => "duplicate",
            Self::Expired => "expired",
            Self::Historical => "historical",
            Self::Drift => "drift",
            Self::CandidateRejected => "candidate_rejected",
            Self::Promoted => "promoted",
            Self::ArchiveFull => "archive_full",
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct FeedbackReceipt {
    pub id: u64,
    pub status: FeedbackStatus,
    pub scored: bool,
    pub correct: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct TemporalReceipt {
    pub prediction: Prediction,
    pub previous: Option<FeedbackReceipt>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobStatus {
    Candidate,
    Failed,
    Stale,
}
impl JobStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Failed => "failed",
            Self::Stale => "stale",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pending {
    prediction: Prediction,
    input: [f64; 4],
    guarded: bool,
    candidate_token: Option<u64>,
    candidate_value: Option<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Receipt {
    pending: Pending,
    label: Option<f64>,
    expired: bool,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct Labeled {
    sequence: u64,
    input: [f64; 4],
    output: f64,
}
#[derive(Clone, Debug)]
struct Model {
    skill: NumericSkill,
    identity: u64,
}
impl Model {
    fn new(skill: NumericSkill) -> Self {
        let identity = skill.fingerprint();
        Self { skill, identity }
    }
}
#[derive(Clone, Debug)]
struct Candidate {
    model: Model,
    token: u64,
    through: u64,
    correct: usize,
    distinct: Vec<[f64; 4]>,
    reused: bool,
}
#[derive(Clone, Debug)]
struct Archived {
    identity: u64,
    bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LearningJob {
    token: u64,
    epoch: u64,
    through: u64,
    context: String,
    arity: usize,
    library: Option<u64>,
    policy: ContinuousPolicy,
    examples: Vec<NumericExample>,
    archive: Vec<Archived>,
    parent: Option<(u32, u64)>,
}
#[derive(Debug)]
pub struct LearningResult {
    token: u64,
    epoch: u64,
    through: u64,
    model: Option<Model>,
    reused: bool,
    pub search_work: u64,
    pub archive_work: u64,
    pub elapsed_ns: u64,
    pub error: Option<String>,
}
impl LearningJob {
    pub fn token(&self) -> u64 {
        self.token
    }
    pub fn run(self, library: Option<&NumericLibrary>) -> LearningResult {
        let start = Instant::now();
        let mut result = LearningResult {
            token: self.token,
            epoch: self.epoch,
            through: self.through,
            model: None,
            reused: false,
            search_work: 0,
            archive_work: 0,
            elapsed_ns: 0,
            error: None,
        };
        let mut archive_work = 0;
        let mut execute = || -> Result<(NumericSkill, bool, u64, u64), NumericError> {
            if library.map(NumericLibrary::fingerprint) != self.library {
                return Err(NumericError::Context);
            }
            let mut cached = Vec::new();
            for artifact in self.archive.iter().rev() {
                let skill = NumericSkill::from_bytes(&artifact.bytes)?;
                if skill.program().arity() != self.arity {
                    return Err(NumericError::Invalid("archive arity"));
                }
                archive_work += self.examples.len() as u64 * skill.program().instructions() as u64;
                if self.examples.iter().all(|e| {
                    skill
                        .program()
                        .evaluate(&e.input)
                        .is_ok_and(|v| self.policy.tolerance.agrees(v, e.output))
                }) {
                    return Ok((skill, true, 0, archive_work));
                }
                cached.push(skill);
            }
            let (mut skill, work) = if let Some(library) = library {
                let (skill, trace) = library.develop(
                    "continuous",
                    &self.context,
                    self.examples.clone(),
                    self.policy.search,
                    self.policy.development,
                    self.policy.tolerance,
                )?;
                (skill, trace.scalar_work)
            } else {
                let tools: Vec<_> = cached.iter().take(16).collect();
                let skill = NumericSkill::learn(
                    "continuous",
                    &self.context,
                    self.examples.clone(),
                    &tools,
                    self.policy.search,
                    self.policy.tolerance,
                )?;
                let work = skill.search().scalar_work;
                (skill, work)
            };
            if let Some((revision, parent)) = self.parent {
                skill.revision = revision
                    .checked_add(1)
                    .ok_or(NumericError::Invalid("continuous lineage overflow"))?;
                skill.parent = Some(parent);
            }
            Ok((skill, false, work, archive_work))
        };
        match execute() {
            Ok((skill, reused, work, archive_work)) => {
                result.model = Some(Model::new(skill));
                result.reused = reused;
                result.search_work = work;
                result.archive_work = archive_work;
            }
            Err(error) => {
                result.search_work = match &error {
                    NumericError::Budget(s) | NumericError::NoProgram(s) => s.scalar_work,
                    NumericError::Development { scalar_work, .. } => *scalar_work,
                    _ => 0,
                };
                result.error = Some(error.to_string());
            }
        }
        result.archive_work = archive_work;
        result.elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        result
    }
}

/// Input/feedback operations never perform synthesis. The host runs owned jobs separately.
#[derive(Clone, Debug)]
pub struct ContinuousAgent {
    context: String,
    arity: usize,
    library: Option<u64>,
    policy: ContinuousPolicy,
    epoch: u64,
    last_id: u64,
    watermark: u64,
    job_token: u64,
    next_job_at: u64,
    failures: u32,
    active: Option<Model>,
    ready: bool,
    candidate: Option<Candidate>,
    inflight: Option<(u64, u64, u64)>,
    pending: Vec<Pending>,
    recent: VecDeque<Receipt>,
    window: VecDeque<Labeled>,
    errors: VecDeque<bool>,
    archive: Vec<Archived>,
    statistics: ContinuousStatistics,
    temporal_target: Option<usize>,
    temporal_last: Option<u64>,
}
impl ContinuousAgent {
    pub fn new(
        context: impl Into<String>,
        arity: usize,
        library: Option<u64>,
        policy: ContinuousPolicy,
    ) -> Result<Self, NumericError> {
        let context = context.into();
        identifier(&context)?;
        policy.validate()?;
        if !(1..=4).contains(&arity) {
            return Err(NumericError::Invalid("continuous arity"));
        }
        Ok(Self {
            context,
            arity,
            library,
            policy,
            epoch: 0,
            last_id: 0,
            watermark: 0,
            job_token: 0,
            next_job_at: 0,
            failures: 0,
            active: None,
            ready: false,
            candidate: None,
            inflight: None,
            pending: Vec::with_capacity(policy.pending_capacity),
            recent: VecDeque::with_capacity(policy.recent_capacity),
            window: VecDeque::with_capacity(policy.training_window),
            errors: VecDeque::with_capacity(policy.drift_window),
            archive: Vec::with_capacity(policy.archive_models),
            statistics: ContinuousStatistics::default(),
            temporal_target: None,
            temporal_last: None,
        })
    }
    pub fn context(&self) -> &str {
        &self.context
    }
    pub fn arity(&self) -> usize {
        self.arity
    }
    pub fn policy(&self) -> ContinuousPolicy {
        self.policy
    }
    pub fn statistics(&self) -> ContinuousStatistics {
        self.statistics
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn model_identity(&self) -> Option<u64> {
        self.active.as_ref().map(|m| m.identity)
    }
    pub fn active_skill(&self) -> Option<&NumericSkill> {
        self.active.as_ref().map(|m| &m.skill)
    }
    pub fn archived_models(&self) -> usize {
        self.archive.len()
    }
    pub fn archived_bytes(&self) -> usize {
        self.archive.iter().map(|m| m.bytes.len()).sum()
    }
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn window_count(&self) -> usize {
        self.window.len()
    }
    pub fn candidate_progress(&self) -> Option<(usize, usize)> {
        self.candidate
            .as_ref()
            .map(|c| (c.correct, c.distinct.len()))
    }
    pub fn phase(&self) -> ContinuousPhase {
        if self.ready {
            ContinuousPhase::Active
        } else if self.candidate.is_some() {
            ContinuousPhase::Validating
        } else if self.inflight.is_some() {
            ContinuousPhase::Learning
        } else if self.active.is_some() {
            ContinuousPhase::Suspended
        } else {
            ContinuousPhase::Collecting
        }
    }
    fn remember(&mut self, receipt: Receipt) {
        if self.recent.len() == self.policy.recent_capacity {
            self.recent.pop_front();
        }
        self.recent.push_back(receipt);
    }
    pub fn observe(&mut self, id: u64, input: &[f64]) -> Result<Prediction, NumericError> {
        if self.temporal_target.is_some() {
            return Err(NumericError::Invalid(
                "temporal session requires raw frames",
            ));
        }
        self.observe_inner(id, input)
    }
    pub fn advance_frame(
        &mut self,
        id: u64,
        input: &[f64],
        target: usize,
    ) -> Result<TemporalReceipt, NumericError> {
        if target >= self.arity || input.len() != self.arity || input.iter().any(|v| !v.is_finite())
        {
            return Err(NumericError::Domain("temporal frame contract"));
        }
        if self.temporal_target.is_some_and(|t| t != target)
            || (self.temporal_target.is_none() && self.statistics.observations != 0)
        {
            return Err(NumericError::Invalid(
                "cannot change temporal target or mix event modes",
            ));
        }
        if self.temporal_last == Some(id) {
            return Ok(TemporalReceipt {
                prediction: self.observe_inner(id, input)?,
                previous: None,
            });
        }
        if id <= self.last_id || self.statistics.observations == u64::MAX {
            return Err(NumericError::Invalid("temporal frame order"));
        }
        let previous = self
            .temporal_last
            .map(|old| self.feedback_inner(old, input[target]))
            .transpose()?;
        self.temporal_target = Some(target);
        let prediction = self.observe_inner(id, input)?;
        self.temporal_last = Some(id);
        Ok(TemporalReceipt {
            prediction,
            previous,
        })
    }
    fn observe_inner(&mut self, id: u64, input: &[f64]) -> Result<Prediction, NumericError> {
        if input.len() != self.arity || input.iter().any(|v| !v.is_finite()) {
            return Err(NumericError::Domain("continuous input"));
        }
        let mut values = [0.0; 4];
        values[..self.arity].copy_from_slice(input);
        let previous = self
            .pending
            .iter()
            .find(|p| p.prediction.id == id)
            .copied()
            .or_else(|| {
                self.recent
                    .iter()
                    .find(|r| r.pending.prediction.id == id)
                    .map(|r| r.pending)
            });
        if let Some(previous) = previous {
            if !same_input(&values, &previous.input) {
                return Err(NumericError::Invalid("observation id collision"));
            }
            return Ok(Prediction {
                duplicate: true,
                ..previous.prediction
            });
        }
        if id <= self.last_id {
            return Err(NumericError::Invalid(
                "observation ids must increase; old receipt expired",
            ));
        }
        let sequence = self
            .statistics
            .observations
            .checked_add(1)
            .ok_or(NumericError::Invalid("observation overflow"))?;
        let expired = self
            .pending
            .iter()
            .filter(|p| sequence - p.prediction.sequence >= self.policy.max_label_delay)
            .count();
        if self.pending.len() - expired >= self.policy.pending_capacity {
            return Err(NumericError::Invalid(
                "pending feedback capacity; apply backpressure",
            ));
        }
        let expiry_total = self
            .statistics
            .expired
            .checked_add(expired as u64)
            .ok_or(NumericError::Invalid("expiry overflow"))?;
        let mut index = 0;
        while index < self.pending.len() {
            if sequence - self.pending[index].prediction.sequence >= self.policy.max_label_delay {
                let pending = self.pending.remove(index);
                self.remember(Receipt {
                    pending,
                    label: None,
                    expired: true,
                });
            } else {
                index += 1;
            }
        }
        let (value, guarded) = if self.ready {
            match self
                .active
                .as_ref()
                .expect("ready model")
                .skill
                .program()
                .evaluate(input)
            {
                Ok(v) => (Some(v), false),
                Err(_) => (None, true),
            }
        } else {
            (None, false)
        };
        let (candidate_token, candidate_value) = self
            .candidate
            .as_ref()
            .map(|c| (Some(c.token), c.model.skill.program().evaluate(input).ok()))
            .unwrap_or((None, None));
        let prediction = Prediction {
            id,
            sequence,
            epoch: self.epoch,
            model: if self.ready {
                self.model_identity()
            } else {
                None
            },
            value,
            duplicate: false,
        };
        self.pending.push(Pending {
            prediction,
            input: values,
            guarded,
            candidate_token,
            candidate_value,
        });
        self.statistics.observations = sequence;
        self.statistics.expired = expiry_total;
        if value.is_some() {
            self.statistics.predictions += 1;
        }
        self.last_id = id;
        Ok(prediction)
    }
    pub fn feedback(&mut self, id: u64, actual: f64) -> Result<FeedbackReceipt, NumericError> {
        if self.temporal_target.is_some() {
            return Err(NumericError::Invalid(
                "temporal feedback comes only from the next frame",
            ));
        }
        self.feedback_inner(id, actual)
    }
    fn feedback_inner(&mut self, id: u64, actual: f64) -> Result<FeedbackReceipt, NumericError> {
        if !actual.is_finite() {
            return Err(NumericError::Domain("continuous feedback"));
        }
        if let Some(receipt) = self
            .recent
            .iter_mut()
            .find(|r| r.pending.prediction.id == id)
        {
            if let Some(label) = receipt.label {
                if label.to_bits() != actual.to_bits() {
                    return Err(NumericError::Invalid("feedback id collision"));
                }
                return Ok(FeedbackReceipt {
                    id,
                    status: FeedbackStatus::Duplicate,
                    scored: false,
                    correct: false,
                });
            }
            receipt.label = Some(actual);
            let scored = receipt.pending.prediction.value.is_some();
            let correct = receipt
                .pending
                .prediction
                .value
                .is_some_and(|v| self.policy.tolerance.agrees(v, actual));
            self.statistics.feedback += 1;
            self.statistics.scored += u64::from(scored);
            self.statistics.correct += u64::from(correct);
            self.statistics.late_for_learning += 1;
            return Ok(FeedbackReceipt {
                id,
                status: FeedbackStatus::Expired,
                scored,
                correct,
            });
        }
        let index = self
            .pending
            .iter()
            .position(|p| p.prediction.id == id)
            .ok_or(NumericError::Invalid("unknown or expired feedback id"))?;
        let pending = self.pending.remove(index);
        let scored = pending.prediction.value.is_some();
        let correct = pending
            .prediction
            .value
            .is_some_and(|v| self.policy.tolerance.agrees(v, actual));
        self.statistics.feedback += 1;
        self.statistics.scored += u64::from(scored);
        self.statistics.correct += u64::from(correct);
        self.remember(Receipt {
            pending,
            label: Some(actual),
            expired: false,
        });
        let mut receipt = FeedbackReceipt {
            id,
            status: FeedbackStatus::Accepted,
            scored,
            correct,
        };
        // Old/out-of-order labels score their original prediction but cannot rewrite a newer epoch.
        if pending.prediction.epoch != self.epoch || pending.prediction.sequence <= self.watermark {
            self.statistics.late_for_learning += 1;
            receipt.status = FeedbackStatus::Historical;
            return Ok(receipt);
        }
        self.watermark = pending.prediction.sequence;
        if self.ready && (scored || pending.guarded) {
            if self.errors.len() == self.policy.drift_window {
                self.errors.pop_front();
            }
            self.errors.push_back(!correct);
            if self.errors.iter().filter(|e| **e).count() >= self.policy.drift_errors {
                self.ready = false;
                self.candidate = None;
                self.inflight = None;
                self.epoch += 1;
                self.errors.clear();
                self.window.clear();
                self.failures = 0;
                self.next_job_at = 0;
                self.statistics.drifts += 1;
                receipt.status = FeedbackStatus::Drift;
            }
        }
        if self.window.len() == self.policy.training_window {
            self.window.pop_front();
        }
        self.window.push_back(Labeled {
            sequence: pending.prediction.sequence,
            input: pending.input,
            output: actual,
        });
        if let Some(candidate) = &mut self.candidate
            && pending.candidate_token == Some(candidate.token)
            && pending.prediction.sequence > candidate.through
        {
            if !pending
                .candidate_value
                .is_some_and(|v| self.policy.tolerance.agrees(v, actual))
            {
                self.candidate = None;
                self.statistics.rejected_candidates += 1;
                self.backoff();
                receipt.status = FeedbackStatus::CandidateRejected;
            } else {
                candidate.correct = candidate
                    .correct
                    .saturating_add(1)
                    .min(self.policy.validation_samples);
                if candidate.distinct.len() < self.policy.validation_distinct
                    && !candidate
                        .distinct
                        .iter()
                        .any(|x| same_input(x, &pending.input))
                {
                    candidate.distinct.push(pending.input);
                }
                if candidate.correct >= self.policy.validation_samples
                    && candidate.distinct.len() >= self.policy.validation_distinct
                {
                    let novel = !self
                        .archive
                        .iter()
                        .any(|m| m.identity == candidate.model.identity);
                    let bytes = novel.then(|| candidate.model.skill.to_bytes());
                    if novel
                        && (self.archive.len() >= self.policy.archive_models
                            || self.archived_bytes() + bytes.as_ref().expect("novel").len()
                                > self.policy.archive_bytes)
                    {
                        receipt.status = FeedbackStatus::ArchiveFull;
                    } else {
                        let candidate = self.candidate.take().expect("candidate");
                        if let Some(bytes) = bytes {
                            self.archive.push(Archived {
                                identity: candidate.model.identity,
                                bytes,
                            });
                        }
                        if candidate.reused {
                            self.statistics.reuses += 1;
                        }
                        self.active = Some(candidate.model);
                        self.ready = true;
                        self.epoch += 1;
                        self.window.clear();
                        self.errors.clear();
                        self.failures = 0;
                        self.statistics.promotions += 1;
                        receipt.status = FeedbackStatus::Promoted;
                    }
                }
            }
        }
        Ok(receipt)
    }
    fn backoff(&mut self) {
        self.failures = self.failures.saturating_add(1);
        self.next_job_at = self.watermark.saturating_add(
            self.policy
                .retry_stride
                .saturating_mul(1_u64 << self.failures.min(5)),
        );
    }
    pub fn take_learning_job(&mut self) -> Option<LearningJob> {
        if self.ready
            || self.candidate.is_some()
            || self.inflight.is_some()
            || self.window.len() < self.policy.min_training
            || self.watermark < self.next_job_at
        {
            return None;
        }
        let distinct = self
            .window
            .iter()
            .enumerate()
            .filter(|(i, x)| {
                !self
                    .window
                    .iter()
                    .take(*i)
                    .any(|y| same_input(&x.input, &y.input))
            })
            .count();
        if distinct < self.policy.validation_distinct {
            return None;
        }
        let token = self.job_token.checked_add(1)?;
        let examples = self
            .window
            .iter()
            .map(|e| NumericExample::new(e.input[..self.arity].to_vec(), e.output))
            .collect();
        let job = LearningJob {
            token,
            epoch: self.epoch,
            through: self.watermark,
            context: self.context.clone(),
            arity: self.arity,
            library: self.library,
            policy: self.policy,
            examples,
            archive: self.archive.clone(),
            parent: self
                .active
                .as_ref()
                .map(|m| (m.skill.revision(), m.identity)),
        };
        self.job_token = token;
        self.inflight = Some((token, self.epoch, self.watermark));
        self.statistics.jobs += 1;
        Some(job)
    }
    pub fn submit_learning_result(&mut self, result: LearningResult) -> JobStatus {
        self.statistics.search_work = self
            .statistics
            .search_work
            .saturating_add(result.search_work);
        self.statistics.archive_work = self
            .statistics
            .archive_work
            .saturating_add(result.archive_work);
        if self.inflight != Some((result.token, result.epoch, result.through))
            || result.epoch != self.epoch
        {
            self.statistics.stale_jobs += 1;
            return JobStatus::Stale;
        }
        self.inflight = None;
        let Some(model) = result.model else {
            self.statistics.failed_jobs += 1;
            self.backoff();
            return JobStatus::Failed;
        };
        if model.skill.context() != self.context
            || model.skill.program().arity() != self.arity
            || self.window.iter().any(|e| {
                !model
                    .skill
                    .program()
                    .evaluate(&e.input[..self.arity])
                    .is_ok_and(|v| self.policy.tolerance.agrees(v, e.output))
            })
        {
            self.statistics.stale_jobs += 1;
            self.backoff();
            return JobStatus::Stale;
        }
        self.candidate = Some(Candidate {
            model,
            token: result.token,
            through: result.through,
            correct: 0,
            distinct: Vec::with_capacity(self.policy.validation_distinct),
            reused: result.reused,
        });
        JobStatus::Candidate
    }
    pub fn retained_bytes(&self) -> usize {
        size_of::<Self>()
            + self.context.capacity()
            + self.pending.capacity() * size_of::<Pending>()
            + self.recent.capacity() * size_of::<Receipt>()
            + self.window.capacity() * size_of::<Labeled>()
            + self.errors.capacity() * size_of::<bool>()
            + self.archive.capacity() * size_of::<Archived>()
            + self
                .archive
                .iter()
                .map(|m| m.bytes.capacity())
                .sum::<usize>()
            + self.active.as_ref().map_or(0, |m| m.skill.retained_bytes())
            + self.candidate.as_ref().map_or(0, |c| {
                c.model.skill.retained_bytes() + c.distinct.capacity() * size_of::<[f64; 4]>()
            })
    }
}
fn same_input(a: &[f64; 4], b: &[f64; 4]) -> bool {
    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
