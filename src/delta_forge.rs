use std::fmt;

use crate::{ChangeError, ChangeStructure, IncrementalOp, ReplaceDelta, SumFold, VectorU64};

pub const KERNEL_SEMANTICS_VERSION: u16 = 1;
pub const SEMANTIC_ARTIFACT_VERSION: u16 = 1;

/// Input grammar is intentionally small: an operator declaration, never an updater callback.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum FoldSpec {
    AddModU64,
    AverageExactU64,
    MinU64,
    PopulationVarianceExactU64,
}

impl FoldSpec {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddModU64 => "AddModU64",
            Self::AverageExactU64 => "AverageExactU64",
            Self::MinU64 => "MinU64",
            Self::PopulationVarianceExactU64 => "PopulationVarianceExactU64",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "AddModU64" => Some(Self::AddModU64),
            "AverageExactU64" => Some(Self::AverageExactU64),
            "MinU64" => Some(Self::MinU64),
            "PopulationVarianceExactU64" => Some(Self::PopulationVarianceExactU64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgebraicClass {
    CommutativeGroup,
    CommutativeGroupWithInvariantCardinality,
    IdempotentCommutativeMonoid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceState {
    ModularTotal,
    ExactSumAndCount,
    MinimumAndMultiplicity,
    ExactFirstAndSecondMoments,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateRule {
    SubtractOldThenAddNew,
    SubtractOldThenAddNewPreserveCount,
    AdjustWitnessOrRebuild,
    ReplaceExactMomentsPreserveCount,
}

/// Small declarative certificate checked against concrete state and changes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeltaCertificate {
    fold: FoldSpec,
    algebra: AlgebraicClass,
    maintenance_state: MaintenanceState,
    update_rule: UpdateRule,
}

impl DeltaCertificate {
    pub const fn fold(self) -> FoldSpec {
        self.fold
    }

    pub const fn algebra(self) -> AlgebraicClass {
        self.algebra
    }

    pub const fn maintenance_state(self) -> MaintenanceState {
        self.maintenance_state
    }

    pub const fn update_rule(self) -> UpdateRule {
        self.update_rule
    }

    pub const fn identifier(self) -> &'static str {
        match self.fold {
            FoldSpec::AddModU64 => "add_mod_replace_v1",
            FoldSpec::AverageExactU64 => "average_exact_replace_v1",
            FoldSpec::MinU64 => "min_witness_replace_v1",
            FoldSpec::PopulationVarianceExactU64 => "population_variance_exact_replace_v1",
        }
    }

    pub fn from_identifier(capability: FoldSpec, identifier: &str) -> Option<Self> {
        let certificate = DeltaForge::certificate_for(capability).ok()?;
        (certificate.identifier() == identifier).then_some(certificate)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ForgeError {
    UnsupportedFold(FoldSpec),
    StateBudgetExceeded {
        required_words: usize,
        available_words: usize,
    },
    ExactRebuildDisallowed(FoldSpec),
    InvalidChange(ChangeError),
    EmptyInput,
    ArithmeticOverflow,
    InvalidCache,
    CertificateViolation,
}

impl fmt::Display for ForgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedFold(spec) => write!(formatter, "unsupported fold {}", spec.as_str()),
            Self::StateBudgetExceeded {
                required_words,
                available_words,
            } => write!(
                formatter,
                "derived state needs {required_words} words, budget allows {available_words}"
            ),
            Self::ExactRebuildDisallowed(spec) => write!(
                formatter,
                "{} needs an exact rebuild when its compact witness is exhausted",
                spec.as_str()
            ),
            Self::InvalidChange(error) => write!(formatter, "invalid change: {error:?}"),
            Self::EmptyInput => formatter.write_str("aggregate requires at least one value"),
            Self::ArithmeticOverflow => formatter.write_str("artifact arithmetic overflowed"),
            Self::InvalidCache => {
                formatter.write_str("artifact cache is inconsistent with the change")
            }
            Self::CertificateViolation => formatter.write_str("artifact certificate was violated"),
        }
    }
}

impl std::error::Error for ForgeError {}

/// A semantic search budget expressed in machine-independent logical words.
/// Physical layout remains a backend concern.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SynthesisBudget {
    max_auxiliary_words: usize,
    allow_exact_rebuild: bool,
}

impl SynthesisBudget {
    pub const fn new(max_auxiliary_words: usize, allow_exact_rebuild: bool) -> Self {
        Self {
            max_auxiliary_words,
            allow_exact_rebuild,
        }
    }

    pub const fn max_auxiliary_words(self) -> usize {
        self.max_auxiliary_words
    }

    pub const fn allows_exact_rebuild(self) -> bool {
        self.allow_exact_rebuild
    }
}

impl Default for SynthesisBudget {
    fn default() -> Self {
        Self::new(3, true)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RebuildPolicy {
    Never,
    WhenWitnessExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateDecision {
    SemanticMismatch,
    StateBudgetExceeded,
    ExactRebuildDisallowed,
    Selected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CandidateAttempt {
    state: MaintenanceState,
    auxiliary_words: usize,
    rebuild: RebuildPolicy,
    decision: CandidateDecision,
}

impl CandidateAttempt {
    pub const fn state(self) -> MaintenanceState {
        self.state
    }

    pub const fn auxiliary_words(self) -> usize {
        self.auxiliary_words
    }

    pub const fn rebuild_policy(self) -> RebuildPolicy {
        self.rebuild
    }

    pub const fn decision(self) -> CandidateDecision {
        self.decision
    }
}

/// Complete trace of the current bounded maintenance-state grammar.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SynthesisTrace {
    spec: FoldSpec,
    attempts: [CandidateAttempt; 4],
    selected: Option<MaintenanceState>,
}

impl SynthesisTrace {
    pub const fn spec(self) -> FoldSpec {
        self.spec
    }

    pub const fn attempts(&self) -> &[CandidateAttempt; 4] {
        &self.attempts
    }

    pub const fn selected(self) -> Option<MaintenanceState> {
        self.selected
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticSemantics {
    ModularU64,
    ExactU64,
    ExactU128,
    ExactVarianceU128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChangeSemantics {
    ReplaceFinalState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuntimeGuards {
    arithmetic: ArithmeticSemantics,
    change: ChangeSemantics,
    requires_nonempty_input: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeGuardError {
    EmptyInput,
}

impl RuntimeGuards {
    pub const fn for_capability(capability: FoldSpec) -> Self {
        match capability {
            FoldSpec::AddModU64 => Self {
                arithmetic: ArithmeticSemantics::ModularU64,
                change: ChangeSemantics::ReplaceFinalState,
                requires_nonempty_input: false,
            },
            FoldSpec::AverageExactU64 => Self {
                arithmetic: ArithmeticSemantics::ExactU128,
                change: ChangeSemantics::ReplaceFinalState,
                requires_nonempty_input: true,
            },
            FoldSpec::MinU64 => Self {
                arithmetic: ArithmeticSemantics::ExactU64,
                change: ChangeSemantics::ReplaceFinalState,
                requires_nonempty_input: true,
            },
            FoldSpec::PopulationVarianceExactU64 => Self {
                arithmetic: ArithmeticSemantics::ExactVarianceU128,
                change: ChangeSemantics::ReplaceFinalState,
                requires_nonempty_input: true,
            },
        }
    }

    pub const fn arithmetic(self) -> ArithmeticSemantics {
        self.arithmetic
    }

    pub const fn change(self) -> ChangeSemantics {
        self.change
    }

    pub const fn identifier(self) -> &'static str {
        match (self.arithmetic, self.requires_nonempty_input) {
            (ArithmeticSemantics::ModularU64, false) => "modular_u64_replace_final_v1",
            (ArithmeticSemantics::ExactU64, true) => "exact_u64_replace_nonempty_final_v1",
            (ArithmeticSemantics::ExactU128, true) => "exact_u128_replace_nonempty_final_v1",
            (ArithmeticSemantics::ExactU128, false) => "exact_u128_replace_final_v1",
            (ArithmeticSemantics::ExactVarianceU128, true) => {
                "exact_population_variance_u128_replace_nonempty_final_v1"
            }
            (ArithmeticSemantics::ModularU64, true)
            | (ArithmeticSemantics::ExactU64, false)
            | (ArithmeticSemantics::ExactVarianceU128, false) => "unsupported",
        }
    }

    pub fn from_identifier(capability: FoldSpec, identifier: &str) -> Option<Self> {
        let guards = Self::for_capability(capability);
        (guards.identifier() == identifier).then_some(guards)
    }

    pub fn validate_values(self, values: &[u64]) -> Result<(), RuntimeGuardError> {
        if self.requires_nonempty_input && values.is_empty() {
            return Err(RuntimeGuardError::EmptyInput);
        }
        Ok(())
    }
}

impl fmt::Display for RuntimeGuardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => formatter.write_str("aggregate requires at least one value"),
        }
    }
}

impl std::error::Error for RuntimeGuardError {}

/// Stable integrity identifier for a verified semantic artifact. It deliberately detects
/// accidental corruption and version drift; it is not a cryptographic signature.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SemanticArtifactHash(u64);

impl SemanticArtifactHash {
    pub const fn value(self) -> u64 {
        self.0
    }

    pub fn from_hex(value: &str) -> Option<Self> {
        (value.len() == 16
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte.is_ascii_uppercase()))
        .then(|| u64::from_str_radix(value, 16).ok().map(Self))
        .flatten()
    }
}

impl fmt::Display for SemanticArtifactHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:016X}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SemanticArtifactError {
    Forge(ForgeError),
    KernelVersion(u16),
    SemanticVersion(u16),
    CertificateMismatch,
    GuardsMismatch,
    SealMismatch,
}

impl fmt::Display for SemanticArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forge(error) => write!(formatter, "cannot derive semantics: {error}"),
            Self::KernelVersion(version) => write!(
                formatter,
                "semantic artifact requires kernel version {version}, expected {KERNEL_SEMANTICS_VERSION}"
            ),
            Self::SemanticVersion(version) => write!(
                formatter,
                "semantic artifact uses version {version}, expected {SEMANTIC_ARTIFACT_VERSION}"
            ),
            Self::CertificateMismatch => {
                formatter.write_str("semantic certificate does not match capability")
            }
            Self::GuardsMismatch => formatter.write_str("runtime guards do not match capability"),
            Self::SealMismatch => {
                formatter.write_str("semantic artifact seal does not match content")
            }
        }
    }
}

impl std::error::Error for SemanticArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Forge(error) => Some(error),
            Self::KernelVersion(_)
            | Self::SemanticVersion(_)
            | Self::CertificateMismatch
            | Self::GuardsMismatch
            | Self::SealMismatch => None,
        }
    }
}

/// Immutable proof-carrying capability. Verification replays a fixed algebraic certificate and
/// validates this content seal; it never scans a caller's dataset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SemanticArtifact {
    capability: FoldSpec,
    certificate: DeltaCertificate,
    guards: RuntimeGuards,
    kernel_version: u16,
    semantic_version: u16,
    hash: SemanticArtifactHash,
}

impl SemanticArtifact {
    pub fn synthesize(capability: FoldSpec) -> Result<Self, SemanticArtifactError> {
        let certificate =
            DeltaForge::certificate_for(capability).map_err(SemanticArtifactError::Forge)?;
        let guards = RuntimeGuards::for_capability(capability);
        let artifact = Self {
            capability,
            certificate,
            guards,
            kernel_version: KERNEL_SEMANTICS_VERSION,
            semantic_version: SEMANTIC_ARTIFACT_VERSION,
            hash: SemanticArtifactHash::from_content(
                capability,
                certificate,
                guards,
                KERNEL_SEMANTICS_VERSION,
                SEMANTIC_ARTIFACT_VERSION,
            ),
        };
        artifact.verify()?;
        Ok(artifact)
    }

    pub fn from_record(
        capability: FoldSpec,
        certificate: DeltaCertificate,
        guards: RuntimeGuards,
        kernel_version: u16,
        semantic_version: u16,
        hash: SemanticArtifactHash,
    ) -> Result<Self, SemanticArtifactError> {
        let artifact = Self {
            capability,
            certificate,
            guards,
            kernel_version,
            semantic_version,
            hash,
        };
        artifact.verify()?;
        Ok(artifact)
    }

    pub fn verify(self) -> Result<(), SemanticArtifactError> {
        if self.kernel_version != KERNEL_SEMANTICS_VERSION {
            return Err(SemanticArtifactError::KernelVersion(self.kernel_version));
        }
        if self.semantic_version != SEMANTIC_ARTIFACT_VERSION {
            return Err(SemanticArtifactError::SemanticVersion(
                self.semantic_version,
            ));
        }
        if self.certificate
            != DeltaForge::certificate_for(self.capability).map_err(SemanticArtifactError::Forge)?
        {
            return Err(SemanticArtifactError::CertificateMismatch);
        }
        if self.guards != RuntimeGuards::for_capability(self.capability) {
            return Err(SemanticArtifactError::GuardsMismatch);
        }
        (self.hash
            == SemanticArtifactHash::from_content(
                self.capability,
                self.certificate,
                self.guards,
                self.kernel_version,
                self.semantic_version,
            ))
        .then_some(())
        .ok_or(SemanticArtifactError::SealMismatch)
    }

    pub const fn capability(self) -> FoldSpec {
        self.capability
    }

    pub const fn certificate(self) -> DeltaCertificate {
        self.certificate
    }

    pub const fn guards(self) -> RuntimeGuards {
        self.guards
    }

    pub const fn kernel_version(self) -> u16 {
        self.kernel_version
    }

    pub const fn semantic_version(self) -> u16 {
        self.semantic_version
    }

    pub const fn hash(self) -> SemanticArtifactHash {
        self.hash
    }
}

impl SemanticArtifactHash {
    const fn from_content(
        capability: FoldSpec,
        certificate: DeltaCertificate,
        guards: RuntimeGuards,
        kernel_version: u16,
        semantic_version: u16,
    ) -> Self {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        hash = fnv1a(hash, capability.as_str().as_bytes());
        hash = fnv1a(hash, certificate.identifier().as_bytes());
        hash = fnv1a(hash, guards.identifier().as_bytes());
        hash = fnv1a(hash, &kernel_version.to_le_bytes());
        hash = fnv1a(hash, &semantic_version.to_le_bytes());
        Self(hash)
    }
}

const fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        index += 1;
    }
    hash
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CertificateStatus {
    Verified,
    Cached,
}

impl CertificateStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Cached => "cached",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalBackend {
    Interpreter,
}

impl PhysicalBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PhysicalEvidence {
    Unmeasured,
}

impl PhysicalEvidence {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unmeasured => "unmeasured",
        }
    }
}

/// Replaceable execution choice. It references semantic content but never determines truth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhysicalRealization {
    semantic_hash: SemanticArtifactHash,
    backend: PhysicalBackend,
    evidence: PhysicalEvidence,
}

impl PhysicalRealization {
    pub const fn interpreter(semantic_hash: SemanticArtifactHash) -> Self {
        Self {
            semantic_hash,
            backend: PhysicalBackend::Interpreter,
            evidence: PhysicalEvidence::Unmeasured,
        }
    }

    pub const fn semantic_hash(self) -> SemanticArtifactHash {
        self.semantic_hash
    }

    pub const fn backend(self) -> PhysicalBackend {
        self.backend
    }

    pub const fn evidence(self) -> PhysicalEvidence {
        self.evidence
    }
}

/// A derived program for `fold(Add mod u64)`. It owns no mutable state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedSumPlan {
    certificate: DeltaCertificate,
    fold: SumFold,
}

impl DerivedSumPlan {
    pub const fn certificate(&self) -> DeltaCertificate {
        self.certificate
    }

    pub fn full(&self, input: &Vec<u64>) -> (u64, u64) {
        self.fold.full(input)
    }

    pub fn delta(&self, change: &ReplaceDelta, cache: &u64) -> Result<(u64, u64), ForgeError> {
        self.fold
            .delta(change, cache)
            .map_err(ForgeError::InvalidChange)
    }

    pub fn apply_output_delta(&self, total: u64, delta: u64) -> u64 {
        total.wrapping_add(delta)
    }

    /// Checks the concrete proof obligation before a derived plan can be promoted.
    pub fn check(&self, before: &Vec<u64>, change: &ReplaceDelta) -> Result<(), ForgeError> {
        let after = VectorU64
            .apply(before, change)
            .map_err(ForgeError::InvalidChange)?;
        let (old_total, cache) = self.full(before);
        let (output_delta, next_cache) = self.delta(change, &cache)?;
        let (expected_total, expected_cache) = self.full(&after);
        (self.apply_output_delta(old_total, output_delta) == expected_total
            && next_cache == expected_cache)
            .then_some(())
            .ok_or(ForgeError::CertificateViolation)
    }
}

/// An average represented as an exact fraction; formatting and rounding are deliberately a
/// caller concern, so the artifact never loses information during maintenance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactAverage {
    numerator: u128,
    denominator: usize,
}

impl ExactAverage {
    pub fn new(numerator: u128, denominator: usize) -> Result<Self, ForgeError> {
        (denominator > 0)
            .then_some(Self {
                numerator,
                denominator,
            })
            .ok_or(ForgeError::EmptyInput)
    }

    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    pub const fn denominator(self) -> usize {
        self.denominator
    }
}

/// The auxiliary state is derived by the declared average capability, never supplied by a
/// caller. It makes the exact replacement rule explicit and independently checkable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AverageCache {
    total: u128,
    count: usize,
}

impl AverageCache {
    pub const fn total(self) -> u128 {
        self.total
    }

    pub const fn count(self) -> usize {
        self.count
    }
}

/// A derived program for `average(u64)` with an exact `(sum, count)` cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedAveragePlan {
    certificate: DeltaCertificate,
}

impl DerivedAveragePlan {
    pub const fn certificate(&self) -> DeltaCertificate {
        self.certificate
    }

    pub fn full(&self, input: &[u64]) -> Result<(ExactAverage, AverageCache), ForgeError> {
        let count = input.len();
        let total = input.iter().try_fold(0_u128, |total, value| {
            total
                .checked_add(u128::from(*value))
                .ok_or(ForgeError::ArithmeticOverflow)
        })?;
        let average = ExactAverage::new(total, count)?;
        Ok((average, AverageCache { total, count }))
    }

    pub fn delta(
        &self,
        change: &ReplaceDelta,
        cache: &AverageCache,
    ) -> Result<(ExactAverage, AverageCache), ForgeError> {
        let total = change
            .changes()
            .iter()
            .try_fold(cache.total, |total, change| {
                total
                    .checked_sub(u128::from(change.old()))
                    .ok_or(ForgeError::InvalidCache)?
                    .checked_add(u128::from(change.new_value()))
                    .ok_or(ForgeError::ArithmeticOverflow)
            })?;
        let next_cache = AverageCache {
            total,
            count: cache.count,
        };
        Ok((ExactAverage::new(total, cache.count)?, next_cache))
    }

    /// Checks the concrete proof obligation before an average artifact can be promoted.
    pub fn check(&self, before: &[u64], change: &ReplaceDelta) -> Result<(), ForgeError> {
        let before = before.to_vec();
        let after = VectorU64
            .apply(&before, change)
            .map_err(ForgeError::InvalidChange)?;
        let (old_average, cache) = self.full(&before)?;
        let (next_average, next_cache) = self.delta(change, &cache)?;
        let (expected_average, expected_cache) = self.full(&after)?;
        (old_average.denominator() == next_average.denominator()
            && next_average == expected_average
            && next_cache == expected_cache)
            .then_some(())
            .ok_or(ForgeError::CertificateViolation)
    }
}

/// Exact population variance represented as
/// `(count * sum_of_squares - sum * sum) / (count * count)`.
/// The fraction is intentionally not rounded or converted to floating point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactPopulationVariance {
    numerator: u128,
    denominator: u128,
}

impl ExactPopulationVariance {
    pub fn new(numerator: u128, denominator: u128) -> Result<Self, ForgeError> {
        (denominator > 0)
            .then_some(Self {
                numerator,
                denominator,
            })
            .ok_or(ForgeError::EmptyInput)
    }

    pub const fn numerator(self) -> u128 {
        self.numerator
    }

    pub const fn denominator(self) -> u128 {
        self.denominator
    }

    fn from_moments(total: u128, squared_total: u128, count: usize) -> Result<Self, ForgeError> {
        if count == 0 {
            return Err(ForgeError::EmptyInput);
        }
        let count = count as u128;
        let scaled_squared_total = squared_total
            .checked_mul(count)
            .ok_or(ForgeError::ArithmeticOverflow)?;
        let squared_total_value = total
            .checked_mul(total)
            .ok_or(ForgeError::ArithmeticOverflow)?;
        let numerator = scaled_squared_total
            .checked_sub(squared_total_value)
            .ok_or(ForgeError::InvalidCache)?;
        let denominator = count
            .checked_mul(count)
            .ok_or(ForgeError::ArithmeticOverflow)?;
        Self::new(numerator, denominator)
    }
}

/// Three sufficient statistics for exact population variance under shape-preserving
/// replacement. Checked `u128` arithmetic makes the representable domain explicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VarianceCache {
    total: u128,
    squared_total: u128,
    count: usize,
}

impl VarianceCache {
    pub const fn total(self) -> u128 {
        self.total
    }

    pub const fn squared_total(self) -> u128 {
        self.squared_total
    }

    pub const fn count(self) -> usize {
        self.count
    }
}

/// A derived exact population-variance plan. It uses raw moments only because integer
/// arithmetic is checked and exact; it never applies the numerically unstable floating-point
/// form silently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedVariancePlan {
    certificate: DeltaCertificate,
}

impl DerivedVariancePlan {
    pub const fn certificate(&self) -> DeltaCertificate {
        self.certificate
    }

    pub fn full(
        &self,
        input: &[u64],
    ) -> Result<(ExactPopulationVariance, VarianceCache), ForgeError> {
        if input.is_empty() {
            return Err(ForgeError::EmptyInput);
        }
        let (total, squared_total) =
            input
                .iter()
                .try_fold((0_u128, 0_u128), |(total, squared_total), value| {
                    let value = u128::from(*value);
                    let squared = value
                        .checked_mul(value)
                        .ok_or(ForgeError::ArithmeticOverflow)?;
                    Ok::<_, ForgeError>((
                        total
                            .checked_add(value)
                            .ok_or(ForgeError::ArithmeticOverflow)?,
                        squared_total
                            .checked_add(squared)
                            .ok_or(ForgeError::ArithmeticOverflow)?,
                    ))
                })?;
        let cache = VarianceCache {
            total,
            squared_total,
            count: input.len(),
        };
        Ok((
            ExactPopulationVariance::from_moments(total, squared_total, input.len())?,
            cache,
        ))
    }

    pub fn delta(
        &self,
        change: &ReplaceDelta,
        cache: &VarianceCache,
    ) -> Result<(ExactPopulationVariance, VarianceCache), ForgeError> {
        if cache.count == 0 {
            return Err(ForgeError::InvalidCache);
        }
        let mut total = cache.total;
        let mut squared_total = cache.squared_total;
        for replacement in change.changes() {
            if replacement.index() >= cache.count {
                return Err(ForgeError::InvalidChange(ChangeError::IndexOutOfBounds(
                    replacement.index(),
                )));
            }
            let old = u128::from(replacement.old());
            let new = u128::from(replacement.new_value());
            let old_squared = old.checked_mul(old).ok_or(ForgeError::ArithmeticOverflow)?;
            let new_squared = new.checked_mul(new).ok_or(ForgeError::ArithmeticOverflow)?;
            total = total
                .checked_sub(old)
                .ok_or(ForgeError::InvalidCache)?
                .checked_add(new)
                .ok_or(ForgeError::ArithmeticOverflow)?;
            squared_total = squared_total
                .checked_sub(old_squared)
                .ok_or(ForgeError::InvalidCache)?
                .checked_add(new_squared)
                .ok_or(ForgeError::ArithmeticOverflow)?;
        }
        let next_cache = VarianceCache {
            total,
            squared_total,
            count: cache.count,
        };
        Ok((
            ExactPopulationVariance::from_moments(total, squared_total, cache.count)?,
            next_cache,
        ))
    }

    /// Checks the concrete replacement obligation against an independent full fold.
    pub fn check(&self, before: &[u64], change: &ReplaceDelta) -> Result<(), ForgeError> {
        let before = before.to_vec();
        let after = VectorU64
            .apply(&before, change)
            .map_err(ForgeError::InvalidChange)?;
        let (_, cache) = self.full(&before)?;
        let (next_variance, next_cache) = self.delta(change, &cache)?;
        let expected = self.full(&after)?;
        (next_variance == expected.0 && next_cache == expected.1)
            .then_some(())
            .ok_or(ForgeError::CertificateViolation)
    }
}

/// Constant-size evidence for a current minimum. `length` lets the plan prove when a
/// replacement batch covers the complete source; it does not retain the source itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MinimumCache {
    minimum: u64,
    multiplicity: usize,
    length: usize,
}

impl MinimumCache {
    pub const fn minimum(self) -> u64 {
        self.minimum
    }

    pub const fn multiplicity(self) -> usize {
        self.multiplicity
    }

    pub const fn length(self) -> usize {
        self.length
    }
}

/// A compact witness is sufficient until the final known occurrence is removed. At that
/// boundary the plan says that information is missing instead of guessing a new minimum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MinimumUpdate {
    Incremental { minimum: u64, cache: MinimumCache },
    RebuildRequired,
}

impl MinimumUpdate {
    pub const fn minimum(self) -> Option<u64> {
        match self {
            Self::Incremental { minimum, .. } => Some(minimum),
            Self::RebuildRequired => None,
        }
    }

    pub const fn cache(self) -> Option<MinimumCache> {
        match self {
            Self::Incremental { cache, .. } => Some(cache),
            Self::RebuildRequired => None,
        }
    }
}

/// A derived exact `MIN` plan with a three-word witness and explicit full fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DerivedMinimumPlan {
    certificate: DeltaCertificate,
}

impl DerivedMinimumPlan {
    pub const fn certificate(&self) -> DeltaCertificate {
        self.certificate
    }

    pub fn full(&self, input: &[u64]) -> Result<(u64, MinimumCache), ForgeError> {
        let mut values = input.iter().copied();
        let first = values.next().ok_or(ForgeError::EmptyInput)?;
        let (minimum, multiplicity) = values.fold((first, 1_usize), |state, value| {
            if value < state.0 {
                (value, 1)
            } else if value == state.0 {
                (state.0, state.1 + 1)
            } else {
                state
            }
        });
        Ok((
            minimum,
            MinimumCache {
                minimum,
                multiplicity,
                length: input.len(),
            },
        ))
    }

    pub fn delta(
        &self,
        change: &ReplaceDelta,
        cache: &MinimumCache,
    ) -> Result<MinimumUpdate, ForgeError> {
        if cache.length == 0 || cache.multiplicity == 0 || cache.multiplicity > cache.length {
            return Err(ForgeError::InvalidCache);
        }
        for replacement in change.changes() {
            if replacement.index() >= cache.length {
                return Err(ForgeError::InvalidChange(ChangeError::IndexOutOfBounds(
                    replacement.index(),
                )));
            }
        }

        let removed_witnesses = change
            .changes()
            .iter()
            .filter(|replacement| replacement.old() == cache.minimum)
            .count();
        let remaining_witnesses = cache
            .multiplicity
            .checked_sub(removed_witnesses)
            .ok_or(ForgeError::InvalidCache)?;
        let new_minimum = change
            .changes()
            .iter()
            .map(|replacement| replacement.new_value())
            .min();
        let new_minimum_count = new_minimum.map_or(0, |minimum| {
            change
                .changes()
                .iter()
                .filter(|replacement| replacement.new_value() == minimum)
                .count()
        });

        let known = match (remaining_witnesses, new_minimum) {
            (remaining, Some(new)) if remaining > 0 && new < cache.minimum => {
                Some((new, new_minimum_count))
            }
            (remaining, Some(new)) if remaining > 0 && new == cache.minimum => {
                Some((cache.minimum, remaining + new_minimum_count))
            }
            (remaining, _) if remaining > 0 => Some((cache.minimum, remaining)),
            (0, Some(new)) if new <= cache.minimum => Some((new, new_minimum_count)),
            (0, Some(new)) if change.changes().len() == cache.length => {
                Some((new, new_minimum_count))
            }
            _ => None,
        };

        Ok(
            known.map_or(MinimumUpdate::RebuildRequired, |(minimum, multiplicity)| {
                MinimumUpdate::Incremental {
                    minimum,
                    cache: MinimumCache {
                        minimum,
                        multiplicity,
                        length: cache.length,
                    },
                }
            }),
        )
    }

    /// Replays the concrete obligation and accepts either a proved local update or an exact
    /// rebuild request. A rebuild is part of the certificate, never a guessed delta.
    pub fn check(&self, before: &[u64], change: &ReplaceDelta) -> Result<(), ForgeError> {
        let before = before.to_vec();
        let after = VectorU64
            .apply(&before, change)
            .map_err(ForgeError::InvalidChange)?;
        let (_, cache) = self.full(&before)?;
        let expected = self.full(&after)?;
        match self.delta(change, &cache)? {
            MinimumUpdate::Incremental {
                minimum,
                cache: next_cache,
            } => (minimum == expected.0 && next_cache == expected.1)
                .then_some(())
                .ok_or(ForgeError::CertificateViolation),
            MinimumUpdate::RebuildRequired => Ok(()),
        }
    }
}

/// A capability-derived artifact. Every variant carries the program and certificate needed to
/// execute and validate that capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedArtifact {
    Sum(DerivedSumPlan),
    Average(DerivedAveragePlan),
    Minimum(DerivedMinimumPlan),
    PopulationVariance(DerivedVariancePlan),
}

pub struct DeltaForge;

impl DeltaForge {
    /// Evaluates every state template in the bounded grammar. The trace is available even when
    /// no candidate fits, so a refusal remains inspectable.
    pub fn analyze(spec: FoldSpec, budget: SynthesisBudget) -> SynthesisTrace {
        let templates = [
            (MaintenanceState::ModularTotal, 1, RebuildPolicy::Never),
            (MaintenanceState::ExactSumAndCount, 2, RebuildPolicy::Never),
            (
                MaintenanceState::MinimumAndMultiplicity,
                3,
                RebuildPolicy::WhenWitnessExhausted,
            ),
            (
                MaintenanceState::ExactFirstAndSecondMoments,
                3,
                RebuildPolicy::Never,
            ),
        ];
        let mut selected = None;
        let attempts = templates.map(|(state, auxiliary_words, rebuild)| {
            let semantic_match = matches!(
                (spec, state),
                (FoldSpec::AddModU64, MaintenanceState::ModularTotal)
                    | (
                        FoldSpec::AverageExactU64,
                        MaintenanceState::ExactSumAndCount
                    )
                    | (FoldSpec::MinU64, MaintenanceState::MinimumAndMultiplicity)
                    | (
                        FoldSpec::PopulationVarianceExactU64,
                        MaintenanceState::ExactFirstAndSecondMoments
                    )
            );
            let decision = if !semantic_match {
                CandidateDecision::SemanticMismatch
            } else if auxiliary_words > budget.max_auxiliary_words {
                CandidateDecision::StateBudgetExceeded
            } else if rebuild != RebuildPolicy::Never && !budget.allow_exact_rebuild {
                CandidateDecision::ExactRebuildDisallowed
            } else {
                selected = Some(state);
                CandidateDecision::Selected
            };
            CandidateAttempt {
                state,
                auxiliary_words,
                rebuild,
                decision,
            }
        });
        SynthesisTrace {
            spec,
            attempts,
            selected,
        }
    }

    pub fn synthesize(spec: FoldSpec) -> Result<DerivedSumPlan, ForgeError> {
        match spec {
            FoldSpec::AddModU64 => Ok(DerivedSumPlan {
                certificate: Self::certificate_for(FoldSpec::AddModU64)?,
                fold: SumFold,
            }),
            FoldSpec::AverageExactU64 | FoldSpec::MinU64 | FoldSpec::PopulationVarianceExactU64 => {
                Err(ForgeError::UnsupportedFold(spec))
            }
        }
    }

    /// Derives the complete artifact, including required auxiliary state, from a capability
    /// declaration. This is intentionally separate from the legacy SUM-only convenience API.
    pub fn synthesize_capability(spec: FoldSpec) -> Result<DerivedArtifact, ForgeError> {
        Self::synthesize_with_budget(spec, SynthesisBudget::default()).map(|(artifact, _)| artifact)
    }

    pub fn synthesize_with_budget(
        spec: FoldSpec,
        budget: SynthesisBudget,
    ) -> Result<(DerivedArtifact, SynthesisTrace), ForgeError> {
        let (certificate, trace) = Self::derive_certificate(spec, budget)?;
        match spec {
            FoldSpec::AddModU64 => Ok((
                DerivedArtifact::Sum(DerivedSumPlan {
                    certificate,
                    fold: SumFold,
                }),
                trace,
            )),
            FoldSpec::AverageExactU64 => Ok((
                DerivedArtifact::Average(DerivedAveragePlan { certificate }),
                trace,
            )),
            FoldSpec::MinU64 => Ok((
                DerivedArtifact::Minimum(DerivedMinimumPlan { certificate }),
                trace,
            )),
            FoldSpec::PopulationVarianceExactU64 => Ok((
                DerivedArtifact::PopulationVariance(DerivedVariancePlan { certificate }),
                trace,
            )),
        }
    }

    fn certificate_for(spec: FoldSpec) -> Result<DeltaCertificate, ForgeError> {
        Self::derive_certificate(spec, SynthesisBudget::default())
            .map(|(certificate, _)| certificate)
    }

    fn derive_certificate(
        spec: FoldSpec,
        budget: SynthesisBudget,
    ) -> Result<(DeltaCertificate, SynthesisTrace), ForgeError> {
        let trace = Self::analyze(spec, budget);
        let selected = match trace.selected {
            Some(selected) => selected,
            None => {
                let matching = trace
                    .attempts
                    .iter()
                    .find(|attempt| attempt.decision != CandidateDecision::SemanticMismatch)
                    .ok_or(ForgeError::UnsupportedFold(spec))?;
                return match matching.decision {
                    CandidateDecision::StateBudgetExceeded => {
                        Err(ForgeError::StateBudgetExceeded {
                            required_words: matching.auxiliary_words,
                            available_words: budget.max_auxiliary_words,
                        })
                    }
                    CandidateDecision::ExactRebuildDisallowed => {
                        Err(ForgeError::ExactRebuildDisallowed(spec))
                    }
                    CandidateDecision::SemanticMismatch | CandidateDecision::Selected => {
                        Err(ForgeError::UnsupportedFold(spec))
                    }
                };
            }
        };
        let certificate = match (spec, selected) {
            (FoldSpec::AddModU64, MaintenanceState::ModularTotal) => DeltaCertificate {
                fold: FoldSpec::AddModU64,
                algebra: AlgebraicClass::CommutativeGroup,
                maintenance_state: MaintenanceState::ModularTotal,
                update_rule: UpdateRule::SubtractOldThenAddNew,
            },
            (FoldSpec::AverageExactU64, MaintenanceState::ExactSumAndCount) => DeltaCertificate {
                fold: FoldSpec::AverageExactU64,
                algebra: AlgebraicClass::CommutativeGroupWithInvariantCardinality,
                maintenance_state: MaintenanceState::ExactSumAndCount,
                update_rule: UpdateRule::SubtractOldThenAddNewPreserveCount,
            },
            (FoldSpec::MinU64, MaintenanceState::MinimumAndMultiplicity) => DeltaCertificate {
                fold: FoldSpec::MinU64,
                algebra: AlgebraicClass::IdempotentCommutativeMonoid,
                maintenance_state: MaintenanceState::MinimumAndMultiplicity,
                update_rule: UpdateRule::AdjustWitnessOrRebuild,
            },
            (
                FoldSpec::PopulationVarianceExactU64,
                MaintenanceState::ExactFirstAndSecondMoments,
            ) => DeltaCertificate {
                fold: FoldSpec::PopulationVarianceExactU64,
                algebra: AlgebraicClass::CommutativeGroupWithInvariantCardinality,
                maintenance_state: MaintenanceState::ExactFirstAndSecondMoments,
                update_rule: UpdateRule::ReplaceExactMomentsPreserveCount,
            },
            _ => return Err(ForgeError::CertificateViolation),
        };
        Ok((certificate, trace))
    }
}
