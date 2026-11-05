use std::{
    fmt,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    ArtifactStore, AxonTask, EpisodeData, EpisodeError, EpisodeReceipt, EpisodeStamp, EpisodeStore,
    SolveError, SolveReceipt, solve_task,
};

#[derive(Debug)]
pub enum ExperiencedSolveError {
    Memory {
        source: EpisodeError,
        executed: Option<Box<Result<SolveReceipt, SolveError>>>,
    },
    Solve {
        source: Box<SolveError>,
        episode: EpisodeReceipt,
    },
}

impl fmt::Display for ExperiencedSolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory {
                source,
                executed: None,
            } => write!(
                f,
                "experience preparation failed; task not executed: {source}"
            ),
            Self::Memory {
                source,
                executed: Some(result),
            } => {
                write!(
                    f,
                    "task execution finished ({result:?}), but experience was not acknowledged: {source}; reopen journal before retrying"
                )
            }
            Self::Solve { source, episode } => {
                write!(f, "{source}; failure recorded as episode {}", episode.id)
            }
        }
    }
}
impl std::error::Error for ExperiencedSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Memory { source, .. } => Some(source),
            Self::Solve { source, .. } => Some(source.as_ref()),
        }
    }
}

/// Legacy solver journal: records measured attempts, but does not infer a causal model or policy.
/// Task/artifact effects and journal commits are separate; a recording error retains the outcome.
/// The plastic observe/validate/commit path lives in `CognitiveCycle`.
pub fn solve_task_with_experience(
    artifacts: &ArtifactStore,
    task: &AxonTask,
    episodes: &mut EpisodeStore,
    context: &str,
) -> Result<(SolveReceipt, EpisodeReceipt), ExperiencedSolveError> {
    let prepare = |source| ExperiencedSolveError::Memory {
        source,
        executed: None,
    };
    let at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| prepare(EpisodeError::Invalid("clock before Unix epoch")))?
        .as_secs();
    let stamp = EpisodeStamp::new("axon.solve/v1", format!("solve-{}", episodes.next_id()), at);
    // Reserve for the longest outcome detail and status before creating any artifact.
    episodes
        .check_record(
            &stamp,
            &EpisodeData::Outcome {
                context: context.into(),
                action: format!("{}/rejected", task.goal().as_str()),
                success: false,
                cost_ns: 0,
                detail: "x".repeat(1_024),
            },
        )
        .map_err(prepare)?;
    let start = Instant::now();
    let execution = solve_task(artifacts, task);
    let elapsed = start.elapsed().as_nanos();
    let status = execution
        .as_ref()
        .map_or("rejected", |receipt| receipt.artifact_status().as_str());
    let result = match &execution {
        Ok(receipt) => format!(
            "semantic_hash={};outcome={:?}",
            receipt.semantic_hash(),
            receipt.outcome()
        ),
        Err(error) => format!("error={error:?}"),
    };
    let mut detail = format!(
        "task={};items={};input_fnv64={:016x};{result}",
        task.name(),
        task.values().len(),
        input_hash(task)
    );
    if detail.len() > 1_024 {
        let mut end = 1_024;
        while !detail.is_char_boundary(end) {
            end -= 1;
        }
        detail.truncate(end);
    }
    let cost_ns = match u64::try_from(elapsed) {
        Ok(value) => value,
        Err(_) => {
            return Err(ExperiencedSolveError::Memory {
                source: EpisodeError::Limit("elapsed nanoseconds"),
                executed: Some(Box::new(execution)),
            });
        }
    };
    let receipt = match episodes.record(
        stamp,
        EpisodeData::Outcome {
            context: context.into(),
            action: format!("{}/{status}", task.goal().as_str()),
            success: execution.is_ok(),
            cost_ns,
            detail,
        },
    ) {
        Ok(receipt) => receipt,
        Err(source) => {
            return Err(ExperiencedSolveError::Memory {
                source,
                executed: Some(Box::new(execution)),
            });
        }
    };
    match execution {
        Ok(result) => Ok((result, receipt)),
        Err(source) => Err(ExperiencedSolveError::Solve {
            source: Box::new(source),
            episode: receipt,
        }),
    }
}

fn input_hash(task: &AxonTask) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in task
        .goal()
        .as_str()
        .bytes()
        .chain(task.values().iter().flat_map(|n| n.to_le_bytes()))
    {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
