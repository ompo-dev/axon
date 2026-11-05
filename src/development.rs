//! Bounded developmental training over the plastic AXON core.
//!
//! The controller owns curriculum order and evaluation boundaries. It trains
//! only from oracle-labelled tasks, prioritizes the largest current errors,
//! and never consumes the reserve set as learning data.

use crate::structural::sequence::lexical_encoder::synaptic::Model;
use crate::{CognitiveCycle, CognitiveCycleError, EpisodeStore};
use std::{collections::BTreeSet, fmt};

const MAX_TASKS: usize = 8_192;
const MAX_ID_BYTES: usize = 64;

#[derive(Debug)]
pub enum DevelopmentError {
    Cycle(CognitiveCycleError),
    Invalid(&'static str),
}

impl fmt::Display for DevelopmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cycle(error) => write!(f, "development cycle error: {error}"),
            Self::Invalid(reason) => write!(f, "invalid developmental run: {reason}"),
        }
    }
}

impl std::error::Error for DevelopmentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cycle(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<CognitiveCycleError> for DevelopmentError {
    fn from(error: CognitiveCycleError) -> Self {
        Self::Cycle(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentTask {
    id: String,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl DevelopmentTask {
    pub fn new(
        id: impl Into<String>,
        key: Vec<f32>,
        value: Vec<f32>,
    ) -> Result<Self, DevelopmentError> {
        let id = id.into();
        if id.is_empty()
            || id.len() > MAX_ID_BYTES
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-./".contains(&byte))
        {
            return Err(DevelopmentError::Invalid("task id"));
        }
        validate_vector(&key)?;
        validate_vector(&value)?;
        Ok(Self { id, key, value })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn key(&self) -> &[f32] {
        &self.key
    }

    pub fn value(&self) -> &[f32] {
        &self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevelopmentBudget {
    pub max_passes: usize,
    pub tolerance: f32,
    pub max_work: usize,
    pub max_live: usize,
}

impl Default for DevelopmentBudget {
    fn default() -> Self {
        Self {
            max_passes: 32,
            tolerance: 1e-4,
            max_work: 200_000,
            max_live: 262_144,
        }
    }
}

impl DevelopmentBudget {
    fn validate(self) -> Result<(), DevelopmentError> {
        if !(1..=1_024).contains(&self.max_passes)
            || !self.tolerance.is_finite()
            || !(0.0..=1e6).contains(&self.tolerance)
            || !(1..=2_000_000).contains(&self.max_work)
            || !(1..=64 * 1024 * 1024).contains(&self.max_live)
        {
            return Err(DevelopmentError::Invalid("development budget"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DevelopmentStop {
    Converged,
    PassLimit,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DevelopmentReport {
    pub stop: DevelopmentStop,
    pub converged: bool,
    pub passes: usize,
    pub training_tasks: usize,
    pub reserve_tasks: usize,
    pub training_steps: usize,
    pub reserve_checks: usize,
    pub initial_reserve_correct: usize,
    pub final_reserve_correct: usize,
    pub initial_reserve_mse: f64,
    pub final_reserve_mse: f64,
    pub state_position: usize,
}

pub struct DevelopmentRun {
    cycle: CognitiveCycle,
    report: DevelopmentReport,
}

impl DevelopmentRun {
    pub fn cycle(&self) -> &CognitiveCycle {
        &self.cycle
    }

    pub fn report(&self) -> &DevelopmentReport {
        &self.report
    }

    pub fn into_parts(self) -> (CognitiveCycle, DevelopmentReport) {
        (self.cycle, self.report)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Score {
    correct: usize,
    mse: f64,
}

/// Train the plastic state with active replay and an untouched reserve set.
///
/// The reserve is required to contain keys not present in training. A target is
/// an oracle label supplied by the caller; the controller never treats its own
/// prediction as truth. This is the boundary that prevents self-confirming
/// errors from becoming learning.
pub fn run_development(
    model: Model,
    scope: &str,
    context: &str,
    episodes: &mut EpisodeStore,
    training: &[DevelopmentTask],
    reserve: &[DevelopmentTask],
    budget: DevelopmentBudget,
) -> Result<DevelopmentRun, DevelopmentError> {
    budget.validate()?;
    if scope.is_empty()
        || context.is_empty()
        || training.is_empty()
        || reserve.is_empty()
        || training.len() > MAX_TASKS
        || reserve.len() > MAX_TASKS
    {
        return Err(DevelopmentError::Invalid("development inputs"));
    }
    let dimensions = model.genes().dimensions;
    let mut ids = BTreeSet::new();
    let mut train_keys = BTreeSet::new();
    validate_tasks(training, dimensions, &mut ids, &mut train_keys, false)?;
    let mut reserve_keys = BTreeSet::new();
    validate_tasks(reserve, dimensions, &mut ids, &mut reserve_keys, true)?;
    if train_keys.iter().any(|key| reserve_keys.contains(key)) {
        return Err(DevelopmentError::Invalid("training/reserve key overlap"));
    }

    let mut cycle = CognitiveCycle::new(model, scope, context)?;
    let initial = score(&cycle, reserve, budget)?;
    let mut final_score = initial;
    let mut passes = 0;
    let mut training_steps = 0;
    let mut reserve_checks = reserve.len();

    while passes < budget.max_passes && final_score.correct < reserve.len() {
        let mut order = training
            .iter()
            .map(|task| -> Result<_, DevelopmentError> {
                let prediction = cycle.predict(task.key(), budget.max_work, budget.max_live)?;
                Ok((mse(&prediction, task.value())?, task))
            })
            .collect::<Result<Vec<_>, _>>()?;
        order.sort_by(|(left, left_task), (right, right_task)| {
            right
                .total_cmp(left)
                .then_with(|| left_task.id().cmp(right_task.id()))
        });

        for (_, task) in order {
            let event = format!("pass-{passes:04}-{}-{}", task.id(), episodes.next_id());
            cycle.observe(
                episodes,
                "development-loop",
                &event,
                (passes + 1) as u64,
                task.key(),
                task.value(),
                true,
                0,
                budget.max_work,
                budget.max_live,
            )?;
            training_steps += 1;
        }
        passes += 1;
        final_score = score(&cycle, reserve, budget)?;
        reserve_checks += reserve.len();
    }

    let converged = final_score.correct == reserve.len();
    let report = DevelopmentReport {
        stop: if converged {
            DevelopmentStop::Converged
        } else {
            DevelopmentStop::PassLimit
        },
        converged,
        passes,
        training_tasks: training.len(),
        reserve_tasks: reserve.len(),
        training_steps,
        reserve_checks,
        initial_reserve_correct: initial.correct,
        final_reserve_correct: final_score.correct,
        initial_reserve_mse: initial.mse,
        final_reserve_mse: final_score.mse,
        state_position: cycle.state_position(),
    };
    Ok(DevelopmentRun { cycle, report })
}

fn validate_tasks(
    tasks: &[DevelopmentTask],
    dimensions: usize,
    ids: &mut BTreeSet<String>,
    keys: &mut BTreeSet<Vec<u32>>,
    reserve: bool,
) -> Result<(), DevelopmentError> {
    for task in tasks {
        if task.key.len() != dimensions || task.value.len() != dimensions {
            return Err(DevelopmentError::Invalid(if reserve {
                "reserve task dimensions"
            } else {
                "training task dimensions"
            }));
        }
        if !ids.insert(task.id.clone()) {
            return Err(DevelopmentError::Invalid("duplicate task id"));
        }
        let key = task.key.iter().map(|value| value.to_bits()).collect();
        if !keys.insert(key) {
            return Err(DevelopmentError::Invalid("duplicate task key"));
        }
    }
    Ok(())
}

fn validate_vector(vector: &[f32]) -> Result<(), DevelopmentError> {
    if vector.is_empty()
        || vector.len() > 32
        || vector
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1e6)
    {
        return Err(DevelopmentError::Invalid("task vector"));
    }
    Ok(())
}

fn score(
    cycle: &CognitiveCycle,
    tasks: &[DevelopmentTask],
    budget: DevelopmentBudget,
) -> Result<Score, DevelopmentError> {
    let mut total = 0.0;
    let mut correct = 0;
    for task in tasks {
        let prediction = cycle.predict(task.key(), budget.max_work, budget.max_live)?;
        let error = mse(&prediction, task.value())?;
        total += f64::from(error);
        correct += usize::from(error <= budget.tolerance);
    }
    Ok(Score {
        correct,
        mse: total / tasks.len() as f64,
    })
}

fn mse(predicted: &[f32], expected: &[f32]) -> Result<f32, DevelopmentError> {
    if predicted.len() != expected.len() {
        return Err(DevelopmentError::Invalid("prediction/task dimensions"));
    }
    let result = predicted
        .iter()
        .zip(expected)
        .map(|(actual, target)| {
            let delta = f64::from(*actual) - f64::from(*target);
            delta * delta
        })
        .sum::<f64>()
        / predicted.len() as f64;
    if !result.is_finite() || result > f64::from(f32::MAX) {
        return Err(DevelopmentError::Invalid("non-finite task error"));
    }
    Ok(result as f32)
}
