use std::fmt;

use crate::FoldSpec;

/// A small typed, declarative AXON task. The grammar deliberately names the data shape and
/// requested capability; it never accepts executable updater code.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxonTask {
    name: String,
    values: Vec<u64>,
    goal: FoldSpec,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TaskError {
    InvalidSyntax(&'static str),
    InvalidName,
    InvalidValue(String),
    UnknownGoal(String),
}

impl AxonTask {
    pub fn parse(source: &str) -> Result<Self, TaskError> {
        let lines: Vec<_> = source
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        if lines.len() != 4 || lines[3] != "}" {
            return Err(TaskError::InvalidSyntax(
                "expected task, data, goal, and closing brace",
            ));
        }

        let name = lines[0]
            .strip_prefix("task ")
            .and_then(|line| line.strip_suffix(" {"))
            .ok_or(TaskError::InvalidSyntax("expected `task name {`"))?;
        if !valid_name(name) {
            return Err(TaskError::InvalidName);
        }

        let values = parse_values(lines[1])?;
        let goal_name = lines[2]
            .strip_prefix("goal derive IncrementalArtifact<")
            .and_then(|line| line.strip_suffix('>'))
            .ok_or(TaskError::InvalidSyntax(
                "expected `goal derive IncrementalArtifact<Capability>`",
            ))?;
        let goal = FoldSpec::from_name(goal_name)
            .ok_or_else(|| TaskError::UnknownGoal(goal_name.to_owned()))?;

        Ok(Self {
            name: name.to_owned(),
            values,
            goal,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn values(&self) -> &[u64] {
        &self.values
    }

    pub const fn goal(&self) -> FoldSpec {
        self.goal
    }
}

impl fmt::Display for TaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSyntax(message) => write!(formatter, "invalid AXON task: {message}"),
            Self::InvalidName => formatter.write_str("invalid AXON task name"),
            Self::InvalidValue(value) => write!(formatter, "invalid u64 value `{value}`"),
            Self::UnknownGoal(goal) => write!(formatter, "unknown AXON capability `{goal}`"),
        }
    }
}

impl std::error::Error for TaskError {}

fn valid_name(name: &str) -> bool {
    let mut characters = name.chars();
    matches!(characters.next(), Some(character) if character.is_ascii_alphabetic() || character == '_')
        && characters.all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
}

fn parse_values(line: &str) -> Result<Vec<u64>, TaskError> {
    let values = line
        .strip_prefix("data numbers: Vec<u64> = [")
        .and_then(|line| line.strip_suffix(']'))
        .ok_or(TaskError::InvalidSyntax(
            "expected `data numbers: Vec<u64> = [values]`",
        ))?;
    if values.trim().is_empty() {
        return Ok(Vec::new());
    }
    values
        .split(',')
        .map(|value| {
            let value = value.trim();
            value
                .parse::<u64>()
                .map_err(|_| TaskError::InvalidValue(value.to_owned()))
        })
        .collect()
}
