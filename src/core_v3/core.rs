use std::error::Error;
use std::fmt::{Display, Formatter};

use super::dynamic::{DynamicError, DynamicField, DynamicOutcome};
use super::episodic::EpisodicStore;
use super::event::Event;
use super::salience::{CognitiveValue, SalienceGate};
use super::semantic::SemanticMesh;

#[derive(Clone, Debug, PartialEq)]
pub struct CognitiveCore {
    salience: SalienceGate,
    semantic: SemanticMesh,
    dynamic: DynamicField,
    episodes: EpisodicStore,
    contradiction_streak: u32,
    reframe_after: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoreAction {
    Dormant,
    Adapt,
    Reframe,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoreStep {
    pub core: CognitiveCore,
    pub processed: bool,
    pub salience_score: f32,
    pub action: CoreAction,
    pub outcomes: Vec<DynamicOutcome>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CoreError {
    Dynamic(DynamicError),
}

impl Display for CoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dynamic(error) => write!(f, "{error}"),
        }
    }
}

impl Error for CoreError {}

impl From<DynamicError> for CoreError {
    fn from(value: DynamicError) -> Self {
        Self::Dynamic(value)
    }
}

impl Default for CognitiveCore {
    fn default() -> Self {
        Self {
            salience: SalienceGate::default(),
            semantic: SemanticMesh::default(),
            dynamic: DynamicField::empty(0.70).expect("valid built-in routing threshold"),
            episodes: EpisodicStore::default(),
            contradiction_streak: 0,
            reframe_after: 2,
        }
    }
}

impl CognitiveCore {
    pub fn semantic(&self) -> &SemanticMesh {
        &self.semantic
    }

    pub fn episodes(&self) -> &EpisodicStore {
        &self.episodes
    }

    pub fn learn_semantic(&self, fact: super::semantic::SemanticFact) -> Self {
        Self {
            salience: self.salience,
            semantic: self.semantic.bind(fact),
            dynamic: self.dynamic.clone(),
            episodes: self.episodes.clone(),
            contradiction_streak: self.contradiction_streak,
            reframe_after: self.reframe_after,
        }
    }

    pub fn observe(&self, event: Event, value: CognitiveValue) -> Result<CoreStep, CoreError> {
        let decision = self.salience.evaluate(&value);
        if !decision.should_process {
            return Ok(CoreStep {
                core: self.clone(),
                processed: false,
                salience_score: decision.score,
                action: CoreAction::Dormant,
                outcomes: Vec::new(),
            });
        }
        let dynamic = self.dynamic.process(&event)?;
        let has_large_residual = dynamic.outcomes.iter().any(|outcome| {
            matches!(outcome, DynamicOutcome::Activated { surprise, .. } if *surprise >= 0.80)
        });
        let contradiction_streak = if has_large_residual {
            self.contradiction_streak.saturating_add(1)
        } else {
            0
        };
        let action = if contradiction_streak >= self.reframe_after {
            CoreAction::Reframe
        } else {
            CoreAction::Adapt
        };
        let episodes = if value.goal_relevance >= 0.50 || value.novelty >= 0.50 {
            self.episodes
                .append(event.clone(), event.semantic_signature().clone())
        } else {
            self.episodes.clone()
        };
        Ok(CoreStep {
            core: Self {
                salience: self.salience,
                semantic: self.semantic.clone(),
                dynamic: dynamic.field,
                episodes,
                contradiction_streak,
                reframe_after: self.reframe_after,
            },
            processed: true,
            salience_score: decision.score,
            action,
            outcomes: dynamic.outcomes,
        })
    }
}
