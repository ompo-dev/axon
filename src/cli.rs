use std::{fmt, fs, path::PathBuf};

use crate::{ArtifactOutcome, ArtifactStore, AxonTask, SolveError, TaskError, solve_task};

#[derive(Debug)]
pub enum CliError {
    Usage,
    ReadTask {
        path: PathBuf,
        source: std::io::Error,
    },
    Task(TaskError),
    Solve(SolveError),
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Usage => {
                formatter.write_str("usage: axon solve <task.axon> [--artifact-dir <directory>]")
            }
            Self::ReadTask { path, .. } => {
                write!(formatter, "cannot read task `{}`", path.display())
            }
            Self::Task(error) => error.fmt(formatter),
            Self::Solve(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReadTask { source, .. } => Some(source),
            Self::Task(error) => Some(error),
            Self::Solve(error) => Some(error),
            Self::Usage => None,
        }
    }
}

pub fn solve_command(arguments: &[String]) -> Result<String, CliError> {
    let (task_path, artifact_root) = parse_arguments(arguments)?;
    let source = fs::read_to_string(&task_path).map_err(|source| CliError::ReadTask {
        path: task_path.clone(),
        source,
    })?;
    let task = AxonTask::parse(&source).map_err(CliError::Task)?;
    let receipt =
        solve_task(&ArtifactStore::open(artifact_root), &task).map_err(CliError::Solve)?;
    let outcome = match receipt.outcome() {
        ArtifactOutcome::ModularSum(total) => format!("modular_sum={total}"),
        ArtifactOutcome::ExactAverage(average) => {
            format!(
                "exact_average={}/{}",
                average.numerator(),
                average.denominator()
            )
        }
    };
    Ok(format!(
        "task={}\nartifact={}\nartifact_path={}\nresult={outcome}",
        receipt.task_name(),
        receipt.artifact_status().as_str(),
        receipt.artifact_path().display()
    ))
}

fn parse_arguments(arguments: &[String]) -> Result<(PathBuf, PathBuf), CliError> {
    if arguments.len() != 2 && arguments.len() != 4
        || arguments.first().map(String::as_str) != Some("solve")
    {
        return Err(CliError::Usage);
    }
    let task_path = PathBuf::from(&arguments[1]);
    let default_root = task_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(".axon-artifacts");
    if arguments.len() == 2 {
        return Ok((task_path, default_root));
    }
    if arguments[2] != "--artifact-dir" {
        return Err(CliError::Usage);
    }
    Ok((task_path, PathBuf::from(&arguments[3])))
}
