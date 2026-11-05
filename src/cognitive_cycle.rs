//! Bounded observe-predict-verify-learn cycle for the live AXON core.
//!
//! The hereditary model is immutable here. Only the private association state is
//! plastic, and it is committed after the corresponding observation is durably
//! recorded. This keeps genome, experience and plastic state as distinct layers
//! while giving them one executable causal loop.

use crate::structural::sequence::lexical_encoder::synaptic::{Mode, Model, State};
use crate::{Episode, EpisodeData, EpisodeError, EpisodeReceipt, EpisodeStamp, EpisodeStore};
use std::fmt;

#[derive(Debug)]
pub enum CognitiveCycleError {
    Episode(EpisodeError),
    Synaptic(crate::structural::Error),
    Invalid(&'static str),
}

impl fmt::Display for CognitiveCycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Episode(error) => write!(f, "episode cycle error: {error}"),
            Self::Synaptic(error) => write!(f, "synaptic cycle error: {error}"),
            Self::Invalid(reason) => write!(f, "invalid cognitive cycle: {reason}"),
        }
    }
}

impl std::error::Error for CognitiveCycleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Episode(error) => Some(error),
            Self::Synaptic(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<EpisodeError> for CognitiveCycleError {
    fn from(error: EpisodeError) -> Self {
        Self::Episode(error)
    }
}

impl From<crate::structural::Error> for CognitiveCycleError {
    fn from(error: crate::structural::Error) -> Self {
        Self::Synaptic(error)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveReceipt {
    pub episode: EpisodeReceipt,
    pub predicted: Vec<f32>,
    pub prediction_error: f32,
    pub validated: bool,
    pub learned: bool,
    pub synapses_written: usize,
    pub position: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayReceipt {
    pub replayed: usize,
    pub skipped: usize,
    pub position: usize,
}

pub struct CognitiveCycle {
    model: Model,
    state: State,
    context: String,
}

impl CognitiveCycle {
    pub fn new(model: Model, scope: &str, context: &str) -> Result<Self, CognitiveCycleError> {
        if scope.is_empty() || context.is_empty() {
            return Err(CognitiveCycleError::Invalid("scope/context"));
        }
        let state = model.begin(scope, Mode::Association)?;
        Ok(Self {
            model,
            state,
            context: context.to_owned(),
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn state_position(&self) -> usize {
        self.state.position()
    }

    pub fn state_snapshot(&self) -> Result<Vec<u8>, crate::structural::Error> {
        self.model.snapshot(&self.state)
    }

    pub fn predict(
        &self,
        key: &[f32],
        max_work: usize,
        max_live: usize,
    ) -> Result<Vec<f32>, CognitiveCycleError> {
        Ok(self
            .model
            .query(&self.state, key, max_work, max_live)?
            .output)
    }

    pub fn replay(
        &mut self,
        episodes: &EpisodeStore,
        max_work: usize,
        max_live: usize,
    ) -> Result<ReplayReceipt, CognitiveCycleError> {
        if self.state.position() != 0 {
            return Err(CognitiveCycleError::Invalid("replay requires fresh state"));
        }
        let mut candidate = self.state.clone();
        let mut replayed = 0;
        let mut skipped = 0;
        for episode in episodes.episodes()? {
            let Some((key, value)) = replay_vectors(episode, &self.context)? else {
                skipped += 1;
                continue;
            };
            self.model
                .associate(&mut candidate, &key, &value, max_work, max_live)?;
            replayed += 1;
        }
        self.state = candidate;
        Ok(ReplayReceipt {
            replayed,
            skipped,
            position: self.state.position(),
        })
    }

    /// Observe one validated or rejected experience.
    ///
    /// Prediction is computed from the current state. A validated observation
    /// is first applied to a clone, then the outcome is appended to the journal;
    /// only after that append succeeds does the plastic state become visible.
    /// Rejected observations remain evidence but do not train the state.
    pub fn observe(
        &mut self,
        episodes: &mut EpisodeStore,
        source: &str,
        event: &str,
        effective_at: u64,
        key: &[f32],
        value: &[f32],
        validated: bool,
        cost_ns: u64,
        max_work: usize,
        max_live: usize,
    ) -> Result<CognitiveReceipt, CognitiveCycleError> {
        if source.is_empty() || event.is_empty() || value.is_empty() {
            return Err(CognitiveCycleError::Invalid("source/event/value"));
        }
        let prediction = self.model.query(&self.state, key, max_work, max_live)?;
        let prediction_error = mean_squared_error(&prediction.output, value)?;

        let mut candidate = self.state.clone();
        let transition = if validated {
            Some(
                self.model
                    .associate(&mut candidate, key, value, max_work, max_live)?,
            )
        } else {
            None
        };
        let synapses_written = transition
            .as_ref()
            .map_or(0, |transition| transition.synapses_written);
        let position = transition
            .as_ref()
            .map_or(self.state.position(), |transition| transition.position);
        let detail = format!(
            "key={};value={};input={:016x};target={:016x};prediction={:016x};error={:.8};validated={validated};learned={}",
            encode_vector(key),
            encode_vector(value),
            hash_vector(key),
            hash_vector(value),
            hash_vector(&prediction.output),
            prediction_error,
            synapses_written > 0,
        );
        let receipt = episodes.record(
            EpisodeStamp::new(source, event, effective_at),
            EpisodeData::Outcome {
                context: self.context.clone(),
                action: format!(
                    "association/{}/{}",
                    if validated { "accept" } else { "reject" },
                    event
                ),
                success: validated,
                cost_ns,
                detail,
            },
        )?;
        let learned = validated && !receipt.duplicate && synapses_written > 0;
        if validated && !receipt.duplicate {
            self.state = candidate;
        }
        Ok(CognitiveReceipt {
            episode: receipt,
            predicted: prediction.output,
            prediction_error,
            validated,
            learned,
            synapses_written,
            position,
        })
    }
}

fn mean_squared_error(predicted: &[f32], expected: &[f32]) -> Result<f32, CognitiveCycleError> {
    if predicted.len() != expected.len() {
        return Err(CognitiveCycleError::Invalid("prediction/value dimensions"));
    }
    let error = predicted
        .iter()
        .zip(expected)
        .map(|(predicted, expected)| {
            let delta = predicted - expected;
            delta * delta
        })
        .sum::<f32>()
        / predicted.len() as f32;
    if !error.is_finite() {
        return Err(CognitiveCycleError::Invalid("non-finite prediction error"));
    }
    Ok(error)
}

fn hash_vector(values: &[f32]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, value| {
            hash.wrapping_mul(0x0000_0100_0000_01b3) ^ u64::from(value.to_bits())
        })
}

fn encode_vector(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect()
}

fn replay_vectors(
    episode: &Episode,
    context: &str,
) -> Result<Option<(Vec<f32>, Vec<f32>)>, CognitiveCycleError> {
    let EpisodeData::Outcome {
        context: episode_context,
        action,
        success,
        detail,
        ..
    } = episode.data()
    else {
        return Ok(None);
    };
    if episode_context != context || !*success || !action.starts_with("association/accept/") {
        return Ok(None);
    }
    let key = decode_field(detail, "key")?;
    let value = decode_field(detail, "value")?;
    Ok(Some((key, value)))
}

fn decode_field(detail: &str, name: &str) -> Result<Vec<f32>, CognitiveCycleError> {
    let prefix = format!("{name}=");
    let value = detail
        .split(';')
        .find_map(|field| field.strip_prefix(&prefix))
        .ok_or(CognitiveCycleError::Invalid("replay vector field"))?;
    if value.is_empty() || !value.len().is_multiple_of(8) || value.len() > 256 {
        return Err(CognitiveCycleError::Invalid("replay vector bounds"));
    }
    let mut result = Vec::with_capacity(value.len() / 8);
    for word in value.as_bytes().chunks_exact(8) {
        let word = std::str::from_utf8(word)
            .map_err(|_| CognitiveCycleError::Invalid("replay vector encoding"))?;
        let bits = u32::from_str_radix(word, 16)
            .map_err(|_| CognitiveCycleError::Invalid("replay vector encoding"))?;
        let scalar = f32::from_bits(bits);
        if !scalar.is_finite() || scalar.abs() > 1e6 {
            return Err(CognitiveCycleError::Invalid("replay vector value"));
        }
        result.push(scalar);
    }
    Ok(result)
}
