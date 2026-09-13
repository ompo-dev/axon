use std::{fmt, io};

use crate::{
    AverageCache, ChangeError, DeltaForge, DerivedArtifact, DerivedAveragePlan, ExactAverage,
    FoldSpec, ForgeError, ReplaceDelta, SemanticArtifact,
};

#[derive(Debug)]
pub enum LiveError {
    Change(ChangeError),
    Forge(ForgeError),
    InvalidArtifact,
    WrongSource,
    WrongVersion,
    WrongSequence,
    InvalidRecord,
    NeedsRecovery,
    ResourceLimit(&'static str),
    Io(io::Error),
}

impl fmt::Display for LiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "live average: {self:?}")
    }
}

impl std::error::Error for LiveError {}

impl From<io::Error> for LiveError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ForgeError> for LiveError {
    fn from(error: ForgeError) -> Self {
        Self::Forge(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AverageStrategy {
    Incremental,
    Full,
}

/// Ownership prevents source writes that bypass maintenance of the derived cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveAverage {
    source_id: u64,
    version: u64,
    last_event: u64,
    values: Vec<u64>,
    semantic: SemanticArtifact,
    plan: DerivedAveragePlan,
    cache: AverageCache,
    average: ExactAverage,
}

impl LiveAverage {
    pub fn materialize(
        source_id: u64,
        values: Vec<u64>,
        semantic: SemanticArtifact,
    ) -> Result<Self, LiveError> {
        semantic.verify().map_err(|_| LiveError::InvalidArtifact)?;
        if semantic.capability() != FoldSpec::AverageExactU64 {
            return Err(LiveError::InvalidArtifact);
        }
        let DerivedArtifact::Average(plan) =
            DeltaForge::synthesize_capability(semantic.capability())?
        else {
            return Err(LiveError::InvalidArtifact);
        };
        let (average, cache) = plan.full(&values)?;
        Ok(Self {
            source_id,
            version: 0,
            last_event: 0,
            values,
            semantic,
            plan,
            cache,
            average,
        })
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }

    pub const fn source_id(&self) -> u64 {
        self.source_id
    }

    pub const fn version(&self) -> u64 {
        self.version
    }

    pub const fn last_event(&self) -> u64 {
        self.last_event
    }

    pub const fn semantic(&self) -> SemanticArtifact {
        self.semantic
    }

    pub const fn query(&self) -> ExactAverage {
        self.average
    }

    pub fn apply(
        &mut self,
        source_id: u64,
        expected_version: u64,
        event: u64,
        delta: &ReplaceDelta,
    ) -> Result<ExactAverage, LiveError> {
        self.validate(source_id, expected_version, event, delta)?;
        let (average, cache) = self.plan.delta(delta, &self.cache)?;
        self.commit(event, delta, average, cache);
        Ok(average)
    }

    /// Both strategies enforce identical source/version/old-value guards.
    pub fn apply_with_strategy(
        &mut self,
        event: u64,
        delta: &ReplaceDelta,
        strategy: AverageStrategy,
    ) -> Result<ExactAverage, LiveError> {
        if strategy == AverageStrategy::Incremental || delta.changes().is_empty() {
            return self.apply(self.source_id, self.version, event, delta);
        }
        self.validate(self.source_id, self.version, event, delta)?;
        for change in delta.changes() {
            self.values[change.index()] = change.new_value();
        }
        // A nonempty slice of u64 with at most usize::MAX elements fits in u128.
        let (average, cache) = self
            .plan
            .full(&self.values)
            .expect("owned source fits u128");
        self.average = average;
        self.cache = cache;
        self.version += 1;
        self.last_event = event;
        Ok(average)
    }

    pub(crate) fn validate(
        &self,
        source_id: u64,
        expected_version: u64,
        event: u64,
        delta: &ReplaceDelta,
    ) -> Result<(), LiveError> {
        if source_id != self.source_id {
            return Err(LiveError::WrongSource);
        }
        if expected_version != self.version || self.version == u64::MAX {
            return Err(LiveError::WrongVersion);
        }
        if self.last_event.checked_add(1) != Some(event) {
            return Err(LiveError::WrongSequence);
        }
        for change in delta.changes() {
            let value = self.values.get(change.index()).ok_or(LiveError::Change(
                ChangeError::IndexOutOfBounds(change.index()),
            ))?;
            if *value != change.old() {
                return Err(LiveError::Change(ChangeError::StaleOldValue(
                    change.index(),
                )));
            }
        }
        Ok(())
    }

    fn commit(
        &mut self,
        event: u64,
        delta: &ReplaceDelta,
        average: ExactAverage,
        cache: AverageCache,
    ) {
        for change in delta.changes() {
            self.values[change.index()] = change.new_value();
        }
        self.average = average;
        self.cache = cache;
        self.version += 1;
        self.last_event = event;
    }

    pub(crate) fn restore_version(&mut self, version: u64, event: u64) -> Result<(), LiveError> {
        if version != event {
            return Err(LiveError::InvalidRecord);
        }
        self.version = version;
        self.last_event = event;
        Ok(())
    }
}
