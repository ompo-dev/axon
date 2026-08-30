use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VectorError {
    Empty,
    InvalidValue(i8),
    DimensionMismatch { left: usize, right: usize },
}

impl Display for VectorError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "hypervector dimension must be positive"),
            Self::InvalidValue(value) => {
                write!(f, "hypervector values must be ternary; got {value}")
            }
            Self::DimensionMismatch { left, right } => {
                write!(f, "hypervector dimensions differ: {left} != {right}")
            }
        }
    }
}

impl Error for VectorError {}

/// Sparse ternary state with VSA-style compositional operations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HyperVector {
    values: Vec<i8>,
}

impl HyperVector {
    pub fn zeros(dimension: usize) -> Result<Self, VectorError> {
        if dimension == 0 {
            return Err(VectorError::Empty);
        }
        Ok(Self {
            values: vec![0; dimension],
        })
    }

    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    pub fn active_dimensions(&self) -> usize {
        self.values.iter().filter(|value| **value != 0).count()
    }

    pub fn bind(&self, other: &Self) -> Result<Self, VectorError> {
        self.zip(other, |left, right| {
            if left == 0 || right == 0 {
                0
            } else {
                left * right
            }
        })
    }

    pub fn bundle(&self, other: &Self) -> Result<Self, VectorError> {
        self.zip(other, |left, right| (left + right).signum())
    }

    /// Signed residual includes both unexpected observations and absent predictions.
    pub fn residual(&self, prediction: &Self) -> Result<Self, VectorError> {
        self.zip(prediction, |observed, expected| {
            if observed == expected {
                0
            } else if observed == 0 {
                -expected
            } else {
                observed
            }
        })
    }

    pub fn similarity(&self, other: &Self) -> Result<f32, VectorError> {
        self.ensure_same_dimension(other)?;
        let (mut dot, mut left_energy, mut right_energy) = (0_i32, 0_i32, 0_i32);
        for (left, right) in self.values.iter().zip(&other.values) {
            dot += i32::from(*left) * i32::from(*right);
            left_energy += i32::from(*left) * i32::from(*left);
            right_energy += i32::from(*right) * i32::from(*right);
        }
        if left_energy == 0 || right_energy == 0 {
            return Ok(0.0);
        }
        Ok(dot as f32 / ((left_energy * right_energy) as f32).sqrt())
    }

    fn zip(&self, other: &Self, map: impl Fn(i8, i8) -> i8) -> Result<Self, VectorError> {
        self.ensure_same_dimension(other)?;
        Ok(Self {
            values: self
                .values
                .iter()
                .zip(&other.values)
                .map(|(left, right)| map(*left, *right))
                .collect(),
        })
    }

    fn ensure_same_dimension(&self, other: &Self) -> Result<(), VectorError> {
        if self.dimension() == other.dimension() {
            Ok(())
        } else {
            Err(VectorError::DimensionMismatch {
                left: self.dimension(),
                right: other.dimension(),
            })
        }
    }
}

impl TryFrom<Vec<i8>> for HyperVector {
    type Error = VectorError;

    fn try_from(values: Vec<i8>) -> Result<Self, Self::Error> {
        if values.is_empty() {
            return Err(VectorError::Empty);
        }
        if let Some(value) = values
            .iter()
            .copied()
            .find(|value| !(-1..=1).contains(value))
        {
            return Err(VectorError::InvalidValue(value));
        }
        Ok(Self { values })
    }
}
