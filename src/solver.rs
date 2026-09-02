use std::{fmt, path::Path};

use crate::{
    ArtifactStatus, ArtifactStore, ArtifactStoreError, AxonTask, DerivedArtifact, ExactAverage,
    ForgeError,
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
    outcome: ArtifactOutcome,
}

#[derive(Debug)]
pub enum SolveError {
    Store(ArtifactStoreError),
    Forge(ForgeError),
}

impl fmt::Display for SolveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => error.fmt(formatter),
            Self::Forge(error) => write!(formatter, "cannot execute artifact: {error:?}"),
        }
    }
}

impl std::error::Error for SolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Forge(error) => Some(error),
        }
    }
}

pub fn solve_task(store: &ArtifactStore, task: &AxonTask) -> Result<SolveReceipt, SolveError> {
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

    pub const fn outcome(&self) -> &ArtifactOutcome {
        &self.outcome
    }
}
