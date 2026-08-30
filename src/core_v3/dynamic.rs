use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use super::event::Event;
use super::vector::{HyperVector, VectorError};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BranchKind {
    Semantic,
    Temporal,
    Causal,
    Visual,
    Linguistic,
}

impl BranchKind {
    const ALL: [Self; 5] = [
        Self::Semantic,
        Self::Temporal,
        Self::Causal,
        Self::Visual,
        Self::Linguistic,
    ];
}

#[derive(Clone, Debug, PartialEq)]
struct BranchState {
    state: HyperVector,
    prediction: HyperVector,
    uncertainty: f32,
    eligibility: f32,
    plasticity: f32,
}

/// Small top-down outcome signal. It is applied only where activity left an
/// eligibility trace, rather than broadcasting a global learning rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreditPacket {
    pub branch: BranchKind,
    pub reward: f32,
    pub prediction_error: f32,
    target_cell: Option<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CreditError {
    InvalidReward,
    InvalidPredictionError,
}

impl Display for CreditError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidReward => write!(f, "credit reward must be finite and between -1 and 1"),
            Self::InvalidPredictionError => {
                write!(
                    f,
                    "credit prediction error must be finite and between 0 and 1"
                )
            }
        }
    }
}

impl Error for CreditError {}

impl CreditPacket {
    pub fn try_new(
        branch: BranchKind,
        reward: f32,
        prediction_error: f32,
    ) -> Result<Self, CreditError> {
        if !reward.is_finite() || !(-1.0..=1.0).contains(&reward) {
            return Err(CreditError::InvalidReward);
        }
        if !prediction_error.is_finite() || !(0.0..=1.0).contains(&prediction_error) {
            return Err(CreditError::InvalidPredictionError);
        }
        Ok(Self {
            branch,
            reward,
            prediction_error,
            target_cell: None,
        })
    }

    pub fn for_cell(mut self, cell_id: u32) -> Self {
        self.target_cell = Some(cell_id);
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DendriticCell {
    pub id: u32,
    branches: BTreeMap<BranchKind, BranchState>,
    threshold: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DynamicOutcome {
    Silent,
    Activated {
        cell_id: u32,
        branch: BranchKind,
        surprise: f32,
        residual: HyperVector,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicStep {
    pub field: DynamicField,
    pub outcomes: Vec<DynamicOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicField {
    cells: Vec<DendriticCell>,
    routing_threshold: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DynamicError {
    InvalidThreshold,
    Vector(VectorError),
}

impl Display for DynamicError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidThreshold => {
                write!(f, "dynamic threshold must be finite and between 0 and 1")
            }
            Self::Vector(error) => write!(f, "{error}"),
        }
    }
}

impl Error for DynamicError {}

impl From<VectorError> for DynamicError {
    fn from(value: VectorError) -> Self {
        Self::Vector(value)
    }
}

impl DendriticCell {
    fn new(id: u32, dimension: usize, threshold: f32) -> Result<Self, DynamicError> {
        let zero = HyperVector::zeros(dimension)?;
        let branches = BranchKind::ALL
            .into_iter()
            .map(|kind| {
                (
                    kind,
                    BranchState {
                        state: zero.clone(),
                        prediction: zero.clone(),
                        uncertainty: 1.0,
                        eligibility: 0.0,
                        plasticity: 0.20,
                    },
                )
            })
            .collect();
        Ok(Self {
            id,
            branches,
            threshold,
        })
    }

    pub fn branch_active_dimensions(&self, branch: BranchKind) -> usize {
        self.branches
            .get(&branch)
            .map(|state| state.state.active_dimensions())
            .unwrap_or(0)
    }

    pub fn branch_plasticity(&self, branch: BranchKind) -> Option<f32> {
        self.branches.get(&branch).map(|state| state.plasticity)
    }

    pub fn branch_eligibility(&self, branch: BranchKind) -> Option<f32> {
        self.branches.get(&branch).map(|state| state.eligibility)
    }

    fn affinity(&self, event: &Event) -> Result<f32, DynamicError> {
        let branch = self
            .branches
            .get(&event.branch)
            .expect("all branch kinds are initialized");
        match branch.prediction.similarity(&event.signal) {
            Ok(prediction) => Ok(prediction.max(branch.state.similarity(&event.signal)?)),
            Err(VectorError::DimensionMismatch { .. }) => Ok(-1.0),
            Err(error) => Err(error.into()),
        }
    }

    fn observe(&self, event: &Event) -> Result<(Self, DynamicOutcome), DynamicError> {
        let current = self
            .branches
            .get(&event.branch)
            .expect("all branch kinds are initialized");
        let similarity = current.prediction.similarity(&event.signal)?;
        let surprise = 1.0 - similarity;
        if surprise < self.threshold {
            return Ok((self.clone(), DynamicOutcome::Silent));
        }

        let residual = event.signal.residual(&current.prediction)?;
        let adaptation_rate = (current.plasticity * current.eligibility).clamp(0.01, 0.50);
        let updated = BranchState {
            state: current.state.bundle(&event.signal)?,
            prediction: event.signal.clone(),
            uncertainty: (current.uncertainty * (1.0 - adaptation_rate)
                + surprise * adaptation_rate)
                .clamp(0.0, 1.0),
            eligibility: (current.eligibility * 0.95 + 0.05).clamp(0.0, 1.0),
            plasticity: current.plasticity,
        };
        let mut branches = self.branches.clone();
        branches.insert(event.branch, updated);
        Ok((
            Self {
                id: self.id,
                branches,
                threshold: self.threshold,
            },
            DynamicOutcome::Activated {
                cell_id: self.id,
                branch: event.branch,
                surprise,
                residual,
            },
        ))
    }

    fn apply_credit(&self, packet: CreditPacket) -> Self {
        if packet.target_cell.is_some_and(|cell_id| cell_id != self.id) {
            return self.clone();
        }
        let Some(current) = self.branches.get(&packet.branch) else {
            return self.clone();
        };
        if current.eligibility == 0.0 {
            return self.clone();
        }
        let plasticity = (current.plasticity
            + 0.25 * packet.reward * packet.prediction_error * current.eligibility)
            .clamp(0.01, 1.0);
        let mut branches = self.branches.clone();
        branches.insert(
            packet.branch,
            BranchState {
                state: current.state.clone(),
                prediction: current.prediction.clone(),
                uncertainty: current.uncertainty,
                eligibility: current.eligibility,
                plasticity,
            },
        );
        Self {
            id: self.id,
            branches,
            threshold: self.threshold,
        }
    }
}

impl DynamicField {
    pub fn empty(routing_threshold: f32) -> Result<Self, DynamicError> {
        if !routing_threshold.is_finite() || !(0.0..=1.0).contains(&routing_threshold) {
            return Err(DynamicError::InvalidThreshold);
        }
        Ok(Self {
            cells: Vec::new(),
            routing_threshold,
        })
    }

    pub fn cells(&self) -> &[DendriticCell] {
        &self.cells
    }

    /// Credit stays local to an active branch or an explicitly addressed cell.
    pub fn apply_credit(&self, packet: CreditPacket) -> Self {
        Self {
            cells: self
                .cells
                .iter()
                .map(|cell| cell.apply_credit(packet))
                .collect(),
            routing_threshold: self.routing_threshold,
        }
    }

    /// Only compatible local branches wake. A novel event grows one new cell.
    pub fn process(&self, event: &Event) -> Result<DynamicStep, DynamicError> {
        let active_ids: Vec<u32> = self
            .cells
            .iter()
            .map(|cell| Ok((cell.affinity(event)? >= self.routing_threshold).then_some(cell.id)))
            .collect::<Result<Vec<_>, DynamicError>>()?
            .into_iter()
            .flatten()
            .collect();
        if active_ids.is_empty() {
            let id = self
                .cells
                .iter()
                .map(|cell| cell.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            let (cell, outcome) =
                DendriticCell::new(id, event.signal.dimension(), 0.20)?.observe(event)?;
            let mut cells = self.cells.clone();
            cells.push(cell);
            return Ok(DynamicStep {
                field: Self {
                    cells,
                    routing_threshold: self.routing_threshold,
                },
                outcomes: vec![outcome],
            });
        }

        let mut cells = Vec::with_capacity(self.cells.len());
        let mut outcomes = Vec::new();
        for cell in &self.cells {
            if active_ids.contains(&cell.id) {
                let (next, outcome) = cell.observe(event)?;
                cells.push(next);
                outcomes.push(outcome);
            } else {
                cells.push(cell.clone());
            }
        }
        Ok(DynamicStep {
            field: Self {
                cells,
                routing_threshold: self.routing_threshold,
            },
            outcomes,
        })
    }
}
