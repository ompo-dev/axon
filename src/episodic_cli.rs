use std::{collections::BTreeMap, path::PathBuf};

use crate::{EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue, FactKey, cli::CliError};

pub(crate) const USAGE: &str = "\n  axon observe <subject> <predicate> <value> --context <context> --source <source> --at <time> --event <id> [--type text|int|bool] [--episodes <file.axep>]\n  axon revise <episode-id> <value> --source <source> --event <id> [--type text|int|bool] [--episodes <file.axep>]\n  axon retract <episode-id> --source <source> --event <id> [--episodes <file.axep>]\n  axon belief <subject> <predicate> --context <context> [--at <time>] [--known-at <episode-id>] [--episodes <file.axep>]\n  axon episode-history <subject> <predicate> --context <context> [--episodes <file.axep>]\n  axon episodes-status [--episodes <file.axep>]\n  axon remember-outcome <context> <action> --success true|false --cost-ns <u64> --source <source> --at <time> --event <id> --detail <text> [--episodes <file.axep>]\n  axon experience <context> [--episodes <file.axep>]";

struct Options<'a>(BTreeMap<&'a str, &'a str>);
impl<'a> Options<'a> {
    fn parse(
        arguments: &'a [String],
        positional: usize,
        allowed: &[&str],
    ) -> Result<Self, CliError> {
        if arguments.len() < 1 + positional
            || arguments[1..1 + positional]
                .iter()
                .any(|s| s.starts_with("--"))
        {
            return Err(CliError::Usage);
        }
        let mut values = BTreeMap::new();
        let (pairs, remainder) = arguments[1 + positional..].as_chunks::<2>();
        for pair in pairs {
            if !allowed.contains(&pair[0].as_str())
                || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err(CliError::Usage);
            }
        }
        if !remainder.is_empty() {
            return Err(CliError::Usage);
        }
        Ok(Self(values))
    }
    fn get(&self, key: &str) -> Option<&'a str> {
        self.0.get(key).copied()
    }
    fn required(&self, key: &str) -> Result<&'a str, CliError> {
        self.get(key).ok_or(CliError::Usage)
    }
    fn number(&self, key: &'static str) -> Result<Option<u64>, CliError> {
        self.get(key).map(|s| number(s, key)).transpose()
    }
    fn stamp(&self) -> Result<EpisodeStamp, CliError> {
        Ok(EpisodeStamp::new(
            self.required("--source")?,
            self.required("--event")?,
            number(self.required("--at")?, "--at")?,
        ))
    }
    fn path(&self) -> PathBuf {
        self.get("--episodes")
            .map(PathBuf::from)
            .unwrap_or_else(EpisodeStore::default_path)
    }
    fn value(&self, value: &str) -> Result<EpisodeValue, CliError> {
        match self.get("--type").unwrap_or("text") {
            "text" => Ok(EpisodeValue::Text(value.to_owned())),
            "int" => value
                .parse::<i64>()
                .map(EpisodeValue::Integer)
                .map_err(|_| invalid("--type int", value)),
            "bool" => boolean(value).map(EpisodeValue::Boolean),
            _ => Err(CliError::Usage),
        }
    }
}

fn invalid(option: &'static str, value: &str) -> CliError {
    CliError::InvalidNumber {
        option,
        value: value.into(),
    }
}
fn number(value: &str, option: &'static str) -> Result<u64, CliError> {
    value.parse().map_err(|_| invalid(option, value))
}
fn boolean(value: &str) -> Result<bool, CliError> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(invalid("boolean", value)),
    }
}

pub(crate) fn command(arguments: &[String]) -> Result<String, CliError> {
    let verb = arguments
        .first()
        .map(String::as_str)
        .ok_or(CliError::Usage)?;
    let (positional, allowed): (usize, &[&str]) = match verb {
        "observe" => (
            3,
            &[
                "--context",
                "--source",
                "--at",
                "--event",
                "--type",
                "--episodes",
            ],
        ),
        "revise" => (2, &["--source", "--event", "--type", "--episodes"]),
        "retract" => (1, &["--source", "--event", "--episodes"]),
        "belief" => (2, &["--context", "--at", "--known-at", "--episodes"]),
        "episode-history" => (2, &["--context", "--episodes"]),
        "episodes-status" => (0, &["--episodes"]),
        "remember-outcome" => (
            2,
            &[
                "--source",
                "--at",
                "--event",
                "--success",
                "--cost-ns",
                "--detail",
                "--episodes",
            ],
        ),
        "experience" => (1, &["--episodes"]),
        _ => return Err(CliError::Usage),
    };
    let options = Options::parse(arguments, positional, allowed)?;
    // Parse all required fields before opening (and possibly creating) the journal.
    let operation = match verb {
        "observe" => Operation::Record(
            options.stamp()?,
            EpisodeData::Fact {
                key: FactKey::new(&arguments[1], &arguments[2], options.required("--context")?),
                value: options.value(&arguments[3])?,
                replaces: None,
            },
        ),
        "revise" => Operation::Revise(
            number(&arguments[1], "episode-id")?,
            options.required("--source")?.into(),
            options.required("--event")?.into(),
            options.value(&arguments[2])?,
        ),
        "retract" => Operation::Retract(
            number(&arguments[1], "episode-id")?,
            options.required("--source")?.into(),
            options.required("--event")?.into(),
        ),
        "belief" => Operation::Belief(
            FactKey::new(&arguments[1], &arguments[2], options.required("--context")?),
            options.number("--at")?,
            options.number("--known-at")?,
        ),
        "episode-history" => Operation::History(FactKey::new(
            &arguments[1],
            &arguments[2],
            options.required("--context")?,
        )),
        "remember-outcome" => Operation::Record(
            options.stamp()?,
            EpisodeData::Outcome {
                context: arguments[1].clone(),
                action: arguments[2].clone(),
                success: boolean(options.required("--success")?)?,
                cost_ns: number(options.required("--cost-ns")?, "--cost-ns")?,
                detail: options.required("--detail")?.into(),
            },
        ),
        "experience" => Operation::Experience(&arguments[1]),
        _ => Operation::Status,
    };
    let (mut store, recovery) = EpisodeStore::open(options.path())?;
    let mut output = format!(
        "episodes={}\nreplayed_episodes={}\ndiscarded_tail_bytes={}",
        store.path().display(),
        recovery.replayed_episodes,
        recovery.discarded_tail_bytes
    );
    match operation {
        Operation::Record(stamp, data) => append_receipt(&mut output, store.record(stamp, data)?),
        Operation::Revise(id, source, event, value) => {
            append_receipt(&mut output, store.revise(id, source, event, value)?)
        }
        Operation::Retract(id, source, event) => {
            append_receipt(&mut output, store.retract(id, source, event)?)
        }
        Operation::Belief(key, at, known) => {
            let result = store.belief(&key, at, known)?;
            output.push_str(&format!(
                "\nstatus={}\nevidence_count={}\nexamined_episodes={}",
                result.status.as_str(),
                result.evidence.len(),
                result.examined_episodes
            ));
            for evidence in result.evidence {
                output.push_str(&format!(
                    "\nepisode={} source={:?} effective_at={} value={:?}",
                    evidence.episode_id, evidence.source, evidence.effective_at, evidence.value
                ));
            }
        }
        Operation::History(key) => {
            for episode in store.history(&key)? {
                output.push_str(&format!(
                    "\nepisode={} source={:?} event={:?} effective_at={} data={:?}",
                    episode.id(),
                    episode.stamp().source,
                    episode.stamp().event,
                    episode.stamp().effective_at,
                    episode.data()
                ));
            }
        }
        Operation::Experience(context) => {
            let summaries = store.experience(context)?;
            output.push_str(&format!(
                "\ncontext={context:?}\nactions={}",
                summaries.len()
            ));
            for (action, summary) in summaries {
                let (n, d) = summary.success_fraction();
                output.push_str(&format!("\naction={action:?} observations={} successes={} failures={} total_cost_ns={} mean_cost_ns={}/{} smoothed_success={n}/{d}", summary.observations, summary.successes, summary.observations - summary.successes, summary.total_cost_ns, summary.total_cost_ns, summary.observations));
            }
        }
        Operation::Status => output.push_str(&format!(
            "\nepisode_count={}\njournal_bytes={}",
            store.len(),
            store.file_bytes()?
        )),
    }
    Ok(output)
}

enum Operation<'a> {
    Record(EpisodeStamp, EpisodeData),
    Revise(u64, String, String, EpisodeValue),
    Retract(u64, String, String),
    Belief(FactKey, Option<u64>, Option<u64>),
    History(FactKey),
    Experience(&'a str),
    Status,
}

fn append_receipt(output: &mut String, receipt: crate::EpisodeReceipt) {
    output.push_str(&format!(
        "\nepisode_id={}\nduplicate={}",
        receipt.id, receipt.duplicate
    ));
}
