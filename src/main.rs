mod axon_format;
mod cli;
mod config;
mod cortex;
mod error;
mod gpu;
mod inspect;
mod memory;
mod platform;
mod runtime;
mod semantic;
mod storage;
mod tui;

use std::process;

use cli::{Cli, Command};
use error::AxonError;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), AxonError> {
    let cli = Cli::parse(std::env::args())?;
    match cli.command {
        Command::Tui(args) => runtime::run_tui(args),
        Command::Ingest(args) => runtime::run_ingest(args),
        Command::Inspect(args) => inspect::inspect_brain(args),
        Command::DumpHeader(args) => inspect::dump_header(args),
        Command::DumpRegion(args) => inspect::dump_region(args),
        Command::Verify(args) => inspect::verify_brain(args),
        Command::Compact(args) => runtime::run_compact(args),
    }
}
