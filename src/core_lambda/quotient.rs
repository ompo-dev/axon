//! LIFT/UNLIFT exato para populações discretas exchangeable.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiftedClass {
    pub representative: u64,
    pub multiplicity: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiftedPopulation {
    values: Vec<u64>,
    classes: Vec<LiftedClass>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuotientError {
    InvalidMember,
}

impl Display for QuotientError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "member is outside the lifted population")
    }
}

impl Error for QuotientError {}

impl LiftedPopulation {
    pub fn from_values(values: &[u64]) -> Self {
        let mut multiplicities = BTreeMap::<u64, usize>::new();
        for value in values {
            *multiplicities.entry(*value).or_default() += 1;
        }
        Self {
            values: values.to_vec(),
            classes: multiplicities
                .into_iter()
                .map(|(representative, multiplicity)| LiftedClass {
                    representative,
                    multiplicity,
                })
                .collect(),
        }
    }

    pub fn classes(&self) -> &[LiftedClass] {
        &self.classes
    }

    pub fn exact_sum(&self) -> u64 {
        self.values
            .iter()
            .fold(0_u64, |sum, value| sum.wrapping_add(*value))
    }

    pub fn lifted_sum(&self) -> u64 {
        self.classes.iter().fold(0_u64, |sum, class| {
            sum.wrapping_add(class.representative.wrapping_mul(class.multiplicity as u64))
        })
    }

    /// UNLIFT materializa apenas a contribuição individual solicitada, sem
    /// duplicar a população ou romper as outras classes equivalentes.
    pub fn unlift_value(&self, member: usize, replacement: u64) -> Result<u64, QuotientError> {
        let original = self
            .values
            .get(member)
            .copied()
            .ok_or(QuotientError::InvalidMember)?;
        Ok(self
            .lifted_sum()
            .wrapping_sub(original)
            .wrapping_add(replacement))
    }
}
