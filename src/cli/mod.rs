use std::ffi::OsString;
use std::path::PathBuf;

use crate::error::AxonError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RunMode {
    Deterministic,
    Stochastic,
}

#[derive(Debug)]
pub struct TuiArgs {
    pub brain: PathBuf,
    pub create_if_missing: bool,
    pub dict: Option<PathBuf>,
    pub mode: RunMode,
}

#[derive(Debug)]
pub struct IngestArgs {
    pub brain: PathBuf,
    pub dict: PathBuf,
}

#[derive(Debug)]
pub struct InspectArgs {
    pub brain: PathBuf,
}

#[derive(Debug)]
pub struct DumpHeaderArgs {
    pub brain: PathBuf,
}

#[derive(Debug)]
pub struct DumpRegionArgs {
    pub brain: PathBuf,
    pub region: String,
}

#[derive(Debug)]
pub struct VerifyArgs {
    pub brain: PathBuf,
}

#[derive(Debug)]
pub struct CompactArgs {
    pub brain: PathBuf,
}

#[derive(Debug)]
pub enum Command {
    Tui(TuiArgs),
    Ingest(IngestArgs),
    Inspect(InspectArgs),
    DumpHeader(DumpHeaderArgs),
    DumpRegion(DumpRegionArgs),
    Verify(VerifyArgs),
    Compact(CompactArgs),
}

#[derive(Debug)]
pub struct Cli {
    pub command: Command,
}

impl Cli {
    pub fn parse<I>(args: I) -> Result<Self, AxonError>
    where
        I: IntoIterator,
        I::Item: Into<OsString>,
    {
        let values: Vec<String> = args
            .into_iter()
            .map(|value| value.into().to_string_lossy().into_owned())
            .collect();
        if values.len() < 2 {
            return Err(AxonError::Parse(usage()));
        }
        let cmd = values[1].as_str();
        let rest = &values[2..];
        let command = match cmd {
            "tui" => Command::Tui(parse_tui(rest)?),
            "ingest" => Command::Ingest(parse_ingest(rest)?),
            "inspect" => Command::Inspect(InspectArgs {
                brain: parse_required_path(rest, "--brain")?,
            }),
            "dump-header" => Command::DumpHeader(DumpHeaderArgs {
                brain: parse_required_path(rest, "--brain")?,
            }),
            "dump-region" => Command::DumpRegion(DumpRegionArgs {
                brain: parse_required_path(rest, "--brain")?,
                region: parse_required_value(rest, "--region")?,
            }),
            "verify" => Command::Verify(VerifyArgs {
                brain: parse_required_path(rest, "--brain")?,
            }),
            "compact" => Command::Compact(CompactArgs {
                brain: parse_required_path(rest, "--brain")?,
            }),
            "--help" | "-h" | "help" => return Err(AxonError::Parse(usage())),
            _ => return Err(AxonError::Parse(format!("unknown command '{cmd}'\n{}", usage()))),
        };
        Ok(Self { command })
    }
}

fn parse_tui(args: &[String]) -> Result<TuiArgs, AxonError> {
    let brain = parse_required_path(args, "--brain")?;
    let dict = parse_optional_path(args, "--dict");
    let create_if_missing = has_flag(args, "--create-if-missing");
    let mode = match parse_optional_value(args, "--mode").as_deref() {
        Some("deterministic") => RunMode::Deterministic,
        Some("stochastic") | None => RunMode::Stochastic,
        Some(other) => {
            return Err(AxonError::Parse(format!(
                "invalid --mode '{other}', expected deterministic|stochastic"
            )))
        }
    };
    Ok(TuiArgs {
        brain,
        create_if_missing,
        dict,
        mode,
    })
}

fn parse_ingest(args: &[String]) -> Result<IngestArgs, AxonError> {
    Ok(IngestArgs {
        brain: parse_required_path(args, "--brain")?,
        dict: parse_required_path(args, "--dict")?,
    })
}

fn parse_required_path(args: &[String], key: &str) -> Result<PathBuf, AxonError> {
    parse_optional_path(args, key).ok_or_else(|| {
        AxonError::Parse(format!(
            "missing required argument {key}\n{}",
            usage_for_command(args)
        ))
    })
}

fn parse_optional_path(args: &[String], key: &str) -> Option<PathBuf> {
    parse_optional_value(args, key).map(PathBuf::from)
}

fn parse_required_value(args: &[String], key: &str) -> Result<String, AxonError> {
    parse_optional_value(args, key).ok_or_else(|| {
        AxonError::Parse(format!(
            "missing required argument {key}\n{}",
            usage_for_command(args)
        ))
    })
}

fn parse_optional_value(args: &[String], key: &str) -> Option<String> {
    let mut idx = 0usize;
    while idx < args.len() {
        if args[idx] == key {
            return args.get(idx + 1).cloned();
        }
        idx += 1;
    }
    None
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|arg| arg == key)
}

fn usage_for_command(args: &[String]) -> String {
    let _ = args;
    usage()
}

fn usage() -> String {
    [
        "Usage:",
        "  axon tui --brain <path.axon> [--create-if-missing] [--dict <dict.txt>] [--mode deterministic|stochastic]",
        "  axon ingest --brain <path.axon> --dict <dict.txt>",
        "  axon inspect --brain <path.axon>",
        "  axon dump-header --brain <path.axon>",
        "  axon dump-region --brain <path.axon> --region <semantic|memory|cortex|journal|obs>",
        "  axon verify --brain <path.axon>",
        "  axon compact --brain <path.axon>",
    ]
    .join("\n")
}
