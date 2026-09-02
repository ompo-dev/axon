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
}

impl FoldSpec {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AddModU64 => "AddModU64",
            Self::AverageExactU64 => "AverageExactU64",
            Self::MinU64 => "MinU64",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "AddModU64" => Some(Self::AddModU64),
            "AverageExactU64" => Some(Self::AverageExactU64),
            "MinU64" => Some(Self::MinU64),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgebraicClass {
    CommutativeGroup,
    CommutativeGroupWithInvariantCardinality,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MaintenanceState {
    ModularTotal,
    ExactSumAndCount,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UpdateRule {
    SubtractOldThenAddNew,
    SubtractOldThenAddNewPreserveCount,
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
            FoldSpec::MinU64 => "unsupported",
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
            Self::InvalidChange(error) => write!(formatter, "invalid change: {error:?}"),
            Self::EmptyInput => formatter.write_str("average requires at least one value"),
            Self::ArithmeticOverflow => formatter.write_str("artifact arithmetic overflowed"),
            Self::InvalidCache => {
                formatter.write_str("artifact cache is inconsistent with the change")
            }
            Self::CertificateViolation => formatter.write_str("artifact certificate was violated"),
        }
    }
}

impl std::error::Error for ForgeError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticSemantics {
    ModularU64,
    ExactU128,
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
                arithmetic: ArithmeticSemantics::ExactU128,
                change: ChangeSemantics::ReplaceFinalState,
                requires_nonempty_input: false,
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
            (ArithmeticSemantics::ExactU128, true) => "exact_u128_replace_nonempty_final_v1",
            (ArithmeticSemantics::ExactU128, false) => "exact_u128_replace_final_v1",
            (ArithmeticSemantics::ModularU64, true) => "unsupported",
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
            Self::EmptyInput => formatter.write_str("exact average requires at least one value"),
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

/// A capability-derived artifact. Every variant carries the program and certificate needed to
/// execute and validate that capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DerivedArtifact {
    Sum(DerivedSumPlan),
    Average(DerivedAveragePlan),
}

pub struct DeltaForge;

impl DeltaForge {
    pub fn synthesize(spec: FoldSpec) -> Result<DerivedSumPlan, ForgeError> {
        match spec {
            FoldSpec::AddModU64 => Ok(DerivedSumPlan {
                certificate: Self::certificate_for(FoldSpec::AddModU64)?,
                fold: SumFold,
            }),
            FoldSpec::AverageExactU64 | FoldSpec::MinU64 => Err(ForgeError::UnsupportedFold(spec)),
        }
    }

    /// Derives the complete artifact, including required auxiliary state, from a capability
    /// declaration. This is intentionally separate from the legacy SUM-only convenience API.
    pub fn synthesize_capability(spec: FoldSpec) -> Result<DerivedArtifact, ForgeError> {
        match spec {
            FoldSpec::AddModU64 => match Self::synthesize(FoldSpec::AddModU64) {
                Ok(plan) => Ok(DerivedArtifact::Sum(plan)),
                Err(error) => Err(error),
            },
            FoldSpec::AverageExactU64 => Ok(DerivedArtifact::Average(DerivedAveragePlan {
                certificate: Self::certificate_for(FoldSpec::AverageExactU64)?,
            })),
            FoldSpec::MinU64 => Err(ForgeError::UnsupportedFold(FoldSpec::MinU64)),
        }
    }

    fn certificate_for(spec: FoldSpec) -> Result<DeltaCertificate, ForgeError> {
        match spec {
            FoldSpec::AddModU64 => Ok(DeltaCertificate {
                fold: FoldSpec::AddModU64,
                algebra: AlgebraicClass::CommutativeGroup,
                maintenance_state: MaintenanceState::ModularTotal,
                update_rule: UpdateRule::SubtractOldThenAddNew,
            }),
            FoldSpec::AverageExactU64 => Ok(DeltaCertificate {
                fold: FoldSpec::AverageExactU64,
                algebra: AlgebraicClass::CommutativeGroupWithInvariantCardinality,
                maintenance_state: MaintenanceState::ExactSumAndCount,
                update_rule: UpdateRule::SubtractOldThenAddNewPreserveCount,
            }),
            FoldSpec::MinU64 => Err(ForgeError::UnsupportedFold(FoldSpec::MinU64)),
        }
    }
}
