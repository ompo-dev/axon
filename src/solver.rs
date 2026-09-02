use std::{fmt, path::Path};

use crate::{
    ArtifactStatus, ArtifactStore, ArtifactStoreError, AxonTask, CertificateStatus,
    DerivedArtifact, ExactAverage, ForgeError, PhysicalRealization, RuntimeGuardError,
    RuntimeGuards, SemanticArtifactHash,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactOutcome {
    ModularSum(u64),
    ExactAverage(ExactAverage),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SolveReceipt {
    task_name: String,
    artifact_status: ArtifactStatus,
    artifact_path: std::path::PathBuf,
    semantic_hash: SemanticArtifactHash,
    certificate_status: CertificateStatus,
    physical: PhysicalRealization,
    outcome: ArtifactOutcome,
}

#[derive(Debug)]
pub enum SolveError {
    Store(ArtifactStoreError),
    Guard(RuntimeGuardError),
    Forge(ForgeError),
}

impl fmt::Display for SolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => error.fmt(formatter),
            Self::Guard(error) => write!(formatter, "runtime guard rejected task: {error}"),
            Self::Forge(error) => write!(formatter, "cannot execute artifact: {error:?}"),
        }
    }
}

impl std::error::Error for SolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Guard(error) => Some(error),
            Self::Forge(error) => Some(error),
        }
    }
}

pub fn solve_task(store: &ArtifactStore, task: &AxonTask) -> Result<SolveReceipt, SolveError> {
    RuntimeGuards::for_capability(task.goal())
        .validate_values(task.values())
        .map_err(SolveError::Guard)?;
    let installed = store.install(task).map_err(SolveError::Store)?;
    let outcome = match installed.artifact() {
        DerivedArtifact::Sum(plan) => {
            ArtifactOutcome::ModularSum(plan.full(&task.values().to_vec()).0)
        }
        DerivedArtifact::Average(plan) => {
            ArtifactOutcome::ExactAverage(plan.full(task.values()).map_err(SolveError::Forge)?.0)
        }
    };
    Ok(SolveReceipt {
        task_name: task.name().to_owned(),
        artifact_status: installed.status(),
        artifact_path: installed.path().to_path_buf(),
        semantic_hash: installed.semantic().hash(),
        certificate_status: installed.certificate_status(),
        physical: installed.physical(),
        outcome,
    })
}

impl SolveReceipt {
    pub fn task_name(&self) -> &str {
        &self.task_name
    }

    pub const fn artifact_status(&self) -> ArtifactStatus {
        self.artifact_status
    }

    pub fn artifact_path(&self) -> &Path {
        &self.artifact_path
    }

    pub const fn semantic_hash(&self) -> SemanticArtifactHash {
        self.semantic_hash
    }

    pub const fn certificate_status(&self) -> CertificateStatus {
        self.certificate_status
    }

    pub const fn physical(&self) -> PhysicalRealization {
        self.physical
    }

    pub const fn outcome(&self) -> &ArtifactOutcome {
        &self.outcome
    }
}
