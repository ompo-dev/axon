/// A value and its legal changes. Operations return new values; callers retain prior state.
pub trait ChangeStructure {
    type Value: Eq;
    type Delta: Eq;

    fn zero(&self) -> Self::Delta;
    fn diff(&self, new: &Self::Value, old: &Self::Value) -> Result<Self::Delta, ChangeError>;
    fn apply(&self, value: &Self::Value, delta: &Self::Delta) -> Result<Self::Value, ChangeError>;
}

/// An incremental operator returns its output change and its next cache together.
pub trait IncrementalOp<I: ChangeStructure, O: ChangeStructure> {
    type Cache: Eq;

    fn full(&self, input: &I::Value) -> (O::Value, Self::Cache);
    fn delta(
        &self,
        change: &I::Delta,
        cache: &Self::Cache,
    ) -> Result<(O::Delta, Self::Cache), ChangeError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChangeError {
    DifferentLengths,
    NonCanonicalReplace,
    IndexOutOfBounds(usize),
    StaleOldValue(usize),
}

/// Modular scalar values form the additive group used by `SUM mod u64`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModularU64;

impl ChangeStructure for ModularU64 {
    type Value = u64;
    type Delta = u64;

    fn zero(&self) -> Self::Delta {
        0
    }

    fn diff(&self, new: &Self::Value, old: &Self::Value) -> Result<Self::Delta, ChangeError> {
        Ok(new.wrapping_sub(*old))
    }

    fn apply(&self, value: &Self::Value, delta: &Self::Delta) -> Result<Self::Value, ChangeError> {
        Ok(value.wrapping_add(*delta))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Replace {
    index: usize,
    old: u64,
    new: u64,
}

impl Replace {
    pub const fn new(index: usize, old: u64, new: u64) -> Self {
        Self { index, old, new }
    }

    pub const fn index(self) -> usize {
        self.index
    }

    pub const fn old(self) -> u64 {
        self.old
    }

    pub const fn new_value(self) -> u64 {
        self.new
    }
}

/// Canonical replacement stream: indices are strictly increasing. Shape validity belongs to apply.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaceDelta {
    changes: Vec<Replace>,
}

impl ReplaceDelta {
    pub fn try_new(changes: Vec<Replace>) -> Result<Self, ChangeError> {
        if changes
            .windows(2)
            .any(|pair| pair[0].index() >= pair[1].index())
        {
            return Err(ChangeError::NonCanonicalReplace);
        }
        Ok(Self { changes })
    }

    pub fn changes(&self) -> &[Replace] {
        &self.changes
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VectorU64;

impl ChangeStructure for VectorU64 {
    type Value = Vec<u64>;
    type Delta = ReplaceDelta;

    fn zero(&self) -> Self::Delta {
        ReplaceDelta {
            changes: Vec::new(),
        }
    }

    fn diff(&self, new: &Self::Value, old: &Self::Value) -> Result<Self::Delta, ChangeError> {
        if new.len() != old.len() {
            return Err(ChangeError::DifferentLengths);
        }
        let changes = new
            .iter()
            .zip(old)
            .enumerate()
            .filter_map(|(index, (&new, &old))| {
                (new != old).then_some(Replace::new(index, old, new))
            })
            .collect();
        ReplaceDelta::try_new(changes)
    }

    fn apply(&self, value: &Self::Value, delta: &Self::Delta) -> Result<Self::Value, ChangeError> {
        for change in delta.changes() {
            let current = value
                .get(change.index())
                .ok_or(ChangeError::IndexOutOfBounds(change.index()))?;
            if *current != change.old() {
                return Err(ChangeError::StaleOldValue(change.index()));
            }
        }
        let mut next = value.clone();
        for change in delta.changes() {
            next[change.index()] = change.new_value();
        }
        Ok(next)
    }
}

/// Declarative fold: cache is the current modular total, not a mutable hidden accumulator.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SumFold;

impl IncrementalOp<VectorU64, ModularU64> for SumFold {
    type Cache = u64;

    fn full(&self, input: &Vec<u64>) -> (u64, Self::Cache) {
        let total = input
            .iter()
            .copied()
            .fold(0_u64, |sum, value| sum.wrapping_add(value));
        (total, total)
    }

    fn delta(
        &self,
        change: &ReplaceDelta,
        cache: &Self::Cache,
    ) -> Result<(u64, Self::Cache), ChangeError> {
        let delta = change.changes().iter().fold(0_u64, |sum, change| {
            sum.wrapping_add(change.new_value().wrapping_sub(change.old()))
        });
        Ok((delta, cache.wrapping_add(delta)))
    }
}
