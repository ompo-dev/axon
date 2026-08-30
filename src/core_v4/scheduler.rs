use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Capability {
    SemanticLookup,
    EpisodicLookup,
    ProceduralCircuit,
    DeliberativeReasoning,
    Simulation,
    ExternalQuestion,
    Reframe,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CognitiveMode {
    Ignore,
    Remember,
    Reflex,
    Retrieve,
    Think,
    Simulate,
    Ask,
    Reframe,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CognitiveLevel {
    L0,
    L1,
    L2,
    L3,
    L4,
    L5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceUse {
    pub events_processed: u32,
    pub bytes_moved: u64,
    pub microjoules: u64,
}

impl ResourceUse {
    pub const fn new(events_processed: u32, bytes_moved: u64, microjoules: u64) -> Self {
        Self {
            events_processed,
            bytes_moved,
            microjoules,
        }
    }

    fn fits(&self, budget: ComputeBudget) -> bool {
        self.events_processed <= budget.max_events
            && self.bytes_moved <= budget.max_bytes_moved
            && self.microjoules <= budget.max_microjoules
    }

    fn pressure(&self, budget: ComputeBudget) -> f32 {
        let events = self.events_processed as f32 / budget.max_events as f32;
        let bytes = self.bytes_moved as f32 / budget.max_bytes_moved as f32;
        let energy = self.microjoules as f32 / budget.max_microjoules as f32;
        events.max(bytes).max(energy)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComputeBudget {
    pub max_events: u32,
    pub max_bytes_moved: u64,
    pub max_microjoules: u64,
}

impl ComputeBudget {
    pub fn new(
        max_events: u32,
        max_bytes_moved: u64,
        max_microjoules: u64,
    ) -> Result<Self, SchedulerError> {
        if max_events == 0 || max_bytes_moved == 0 || max_microjoules == 0 {
            return Err(SchedulerError::InvalidBudget);
        }
        Ok(Self {
            max_events,
            max_bytes_moved,
            max_microjoules,
        })
    }
}

/// Signals received by the scheduler, all normalized in the inclusive 0..=1 range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CognitiveSignals {
    pub surprise: f32,
    pub goal_relevance: f32,
    pub uncertainty: f32,
    pub risk: f32,
    pub novelty: f32,
    pub information_gain: f32,
}

impl CognitiveSignals {
    pub fn try_new(
        surprise: f32,
        goal_relevance: f32,
        uncertainty: f32,
        risk: f32,
        novelty: f32,
        information_gain: f32,
    ) -> Result<Self, SchedulerError> {
        let values = [
            surprise,
            goal_relevance,
            uncertainty,
            risk,
            novelty,
            information_gain,
        ];
        if values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(SchedulerError::InvalidSignal);
        }
        Ok(Self {
            surprise,
            goal_relevance,
            uncertainty,
            risk,
            novelty,
            information_gain,
        })
    }

    fn cognitive_value(&self) -> f32 {
        0.20 * self.surprise
            + 0.30 * self.goal_relevance
            + 0.15 * self.uncertainty
            + 0.20 * self.risk
            + 0.05 * self.novelty
            + 0.10 * self.information_gain
    }

    fn confidence_target(&self) -> f32 {
        (0.55 + 0.30 * self.risk + 0.15 * self.uncertainty).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReframeEvidence {
    pub structured_residual: f32,
    pub persistence: f32,
    pub novelty: f32,
    pub adaptation_gain: f32,
}

impl ReframeEvidence {
    pub fn try_new(
        structured_residual: f32,
        persistence: f32,
        novelty: f32,
        adaptation_gain: f32,
    ) -> Result<Self, SchedulerError> {
        let values = [structured_residual, persistence, novelty, adaptation_gain];
        if values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(SchedulerError::InvalidReframeEvidence);
        }
        Ok(Self {
            structured_residual,
            persistence,
            novelty,
            adaptation_gain,
        })
    }

    pub fn jump_pressure(&self) -> f32 {
        (self.structured_residual * self.persistence * self.novelty / (self.adaptation_gain + 0.05))
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveRequest {
    pub task_kind: String,
    pub signals: CognitiveSignals,
    pub available_capabilities: BTreeSet<Capability>,
    pub reframe_evidence: Option<ReframeEvidence>,
}

impl CognitiveRequest {
    pub fn new(
        task_kind: impl Into<String>,
        signals: CognitiveSignals,
        available_capabilities: BTreeSet<Capability>,
    ) -> Self {
        Self {
            task_kind: task_kind.into(),
            signals,
            available_capabilities,
            reframe_evidence: None,
        }
    }

    pub fn with_reframe_evidence(mut self, reframe_evidence: ReframeEvidence) -> Self {
        self.reframe_evidence = Some(reframe_evidence);
        self
    }
}

/// A strategy must explicitly declare both the capability it needs and its resource estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateStrategy {
    pub mode: CognitiveMode,
    pub level: CognitiveLevel,
    pub expected_confidence: f32,
    pub expected_information_gain: f32,
    pub estimated_use: ResourceUse,
    pub required_capabilities: BTreeSet<Capability>,
}

impl CandidateStrategy {
    pub fn try_new(
        mode: CognitiveMode,
        level: CognitiveLevel,
        expected_confidence: f32,
        expected_information_gain: f32,
        estimated_use: ResourceUse,
        required_capabilities: BTreeSet<Capability>,
    ) -> Result<Self, SchedulerError> {
        if !expected_confidence.is_finite() || !(0.0..=1.0).contains(&expected_confidence) {
            return Err(SchedulerError::InvalidCandidate);
        }
        if !expected_information_gain.is_finite()
            || !(0.0..=1.0).contains(&expected_information_gain)
        {
            return Err(SchedulerError::InvalidCandidate);
        }
        Ok(Self {
            mode,
            level,
            expected_confidence,
            expected_information_gain,
            estimated_use,
            required_capabilities,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledPlan {
    pub mode: CognitiveMode,
    pub level: CognitiveLevel,
    pub estimated_use: ResourceUse,
    pub score: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StrategyOutcome {
    pub verified: bool,
    pub actual_use: ResourceUse,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StrategyStats {
    attempts: u32,
    verified_successes: u32,
    total_pressure: f32,
}

/// Immutable scheduler state. Execution reports feed a small empirical bias by task/mode.
#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveScheduler {
    budget: ComputeBudget,
    history: BTreeMap<(String, CognitiveMode), StrategyStats>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    InvalidBudget,
    InvalidSignal,
    InvalidReframeEvidence,
    InvalidCandidate,
    NoViableStrategy,
}

impl Display for SchedulerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBudget => write!(f, "all compute budget dimensions must be positive"),
            Self::InvalidSignal => {
                write!(f, "cognitive signals must be finite values between 0 and 1")
            }
            Self::InvalidReframeEvidence => {
                write!(f, "reframe evidence must be finite values between 0 and 1")
            }
            Self::InvalidCandidate => write!(
                f,
                "strategy estimates must be finite values between 0 and 1"
            ),
            Self::NoViableStrategy => {
                write!(f, "no strategy is available within the current budget")
            }
        }
    }
}

impl Error for SchedulerError {}

impl CognitiveScheduler {
    pub fn new(budget: ComputeBudget) -> Self {
        Self {
            budget,
            history: BTreeMap::new(),
        }
    }

    pub fn plan(
        &self,
        request: &CognitiveRequest,
        candidates: &[CandidateStrategy],
    ) -> Result<ScheduledPlan, SchedulerError> {
        let reframe_required = request
            .reframe_evidence
            .is_some_and(|evidence| evidence.jump_pressure() >= 0.80);
        let viable = candidates
            .iter()
            .filter(|candidate| candidate.estimated_use.fits(self.budget))
            .filter(|candidate| {
                candidate
                    .required_capabilities
                    .is_subset(&request.available_capabilities)
            })
            .filter(|candidate| {
                candidate.mode == CognitiveMode::Ask
                    || (reframe_required && candidate.mode == CognitiveMode::Reframe)
                    || candidate.expected_confidence >= request.signals.confidence_target()
            })
            .filter(|candidate| {
                !reframe_required
                    || matches!(candidate.mode, CognitiveMode::Reframe | CognitiveMode::Ask)
            })
            .collect::<Vec<_>>();
        let selected = if reframe_required {
            let reframes = viable
                .iter()
                .copied()
                .filter(|candidate| candidate.mode == CognitiveMode::Reframe)
                .collect::<Vec<_>>();
            if reframes.is_empty() {
                viable
                    .into_iter()
                    .filter(|candidate| candidate.mode == CognitiveMode::Ask)
                    .collect()
            } else {
                reframes
            }
        } else {
            viable
        };
        selected
            .into_iter()
            .map(|candidate| ScheduledPlan {
                mode: candidate.mode,
                level: candidate.level,
                estimated_use: candidate.estimated_use,
                score: self.score(request, candidate),
            })
            .max_by(|left, right| {
                left.score
                    .total_cmp(&right.score)
                    .then_with(|| right.level.cmp(&left.level))
            })
            .ok_or(SchedulerError::NoViableStrategy)
    }

    pub fn record_outcome(
        &self,
        task_kind: impl Into<String>,
        mode: CognitiveMode,
        outcome: StrategyOutcome,
    ) -> Self {
        let task_kind = task_kind.into();
        let key = (task_kind, mode);
        let mut history = self.history.clone();
        let mut stats = history.get(&key).copied().unwrap_or_default();
        stats.attempts = stats.attempts.saturating_add(1);
        stats.verified_successes = stats
            .verified_successes
            .saturating_add(u32::from(outcome.verified));
        stats.total_pressure += outcome.actual_use.pressure(self.budget);
        history.insert(key, stats);
        Self {
            budget: self.budget,
            history,
        }
    }

    fn score(&self, request: &CognitiveRequest, candidate: &CandidateStrategy) -> f32 {
        let empirical = self.empirical_utility(&request.task_kind, candidate.mode);
        0.45 * candidate.expected_confidence
            + 0.20 * candidate.expected_information_gain
            + 0.20 * request.signals.cognitive_value()
            + 0.15 * empirical
            - 0.30 * candidate.estimated_use.pressure(self.budget)
            - 0.03 * candidate.level as u8 as f32
    }

    fn empirical_utility(&self, task_kind: &str, mode: CognitiveMode) -> f32 {
        let Some(stats) = self.history.get(&(task_kind.to_string(), mode)) else {
            return 0.0;
        };
        if stats.attempts < 3 {
            return 0.0;
        }
        let success_rate = stats.verified_successes as f32 / stats.attempts as f32;
        let average_pressure = stats.total_pressure / stats.attempts as f32;
        (success_rate - average_pressure).clamp(-1.0, 1.0)
    }
}
