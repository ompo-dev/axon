/// Inputs to allocation. Importance can wake compute even if prediction is accurate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CognitiveValue {
    pub prediction_error: f32,
    pub goal_relevance: f32,
    pub uncertainty: f32,
    pub information_gain: f32,
    pub novelty: f32,
    pub computational_cost: f32,
}

impl CognitiveValue {
    pub fn try_new(
        prediction_error: f32,
        goal_relevance: f32,
        uncertainty: f32,
        information_gain: f32,
        novelty: f32,
        computational_cost: f32,
    ) -> Result<Self, SalienceError> {
        let inputs = [
            prediction_error,
            goal_relevance,
            uncertainty,
            information_gain,
            novelty,
            computational_cost,
        ];
        if inputs
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(SalienceError::InvalidValue);
        }
        Ok(Self {
            prediction_error,
            goal_relevance,
            uncertainty,
            information_gain,
            novelty,
            computational_cost,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SalienceError {
    InvalidValue,
}

impl Display for SalienceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cognitive-value inputs must be finite values between 0 and 1"
        )
    }
}

impl Error for SalienceError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SalienceGate {
    threshold: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SalienceDecision {
    pub score: f32,
    pub should_process: bool,
}

impl Default for SalienceGate {
    fn default() -> Self {
        Self { threshold: 0.20 }
    }
}

impl SalienceGate {
    pub fn evaluate(&self, value: &CognitiveValue) -> SalienceDecision {
        let score = 0.25 * value.prediction_error
            + 0.45 * value.goal_relevance
            + 0.10 * value.uncertainty
            + 0.10 * value.information_gain
            + 0.10 * value.novelty
            - 0.20 * value.computational_cost;
        SalienceDecision {
            score,
            should_process: score >= self.threshold,
        }
    }
}

use std::error::Error;
use std::fmt::{Display, Formatter};
