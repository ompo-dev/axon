//! Run an explicit AXON developmental campaign from JSON.

use axon_uic::{DevelopmentPlan, run_plan};
use std::{error::Error, fs, path::PathBuf};

const MAX_PLAN_BYTES: usize = 4 * 1024 * 1024;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: axon-uic-development-lab <plan.json> <journal.axep> [report.json]".into(),
        );
    }
    let plan_path = PathBuf::from(&args[0]);
    let journal = PathBuf::from(&args[1]);
    let bytes = fs::read(&plan_path)?;
    if bytes.is_empty() || bytes.len() > MAX_PLAN_BYTES {
        return Err("development plan byte budget".into());
    }
    let plan: DevelopmentPlan = serde_json::from_slice(&bytes)?;
    let run = run_plan(&plan, &journal)?;
    let report = serde_json::json!({
        "schema": "axon-development-report/v1",
        "scope": plan.scope,
        "context": plan.context,
        "training_tasks": plan.training.len(),
        "reserve_tasks": plan.reserve.len(),
        "report": run.report(),
    });
    if let Some(path) = args.get(2) {
        let output = PathBuf::from(path);
        if output.exists() {
            return Err("report output already exists".into());
        }
        fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    let r = run.report();
    println!(
        "development {}: reserve {}/{}; passes={}; steps={}; mse {:.6}->{:.6}",
        match r.stop {
            axon_uic::DevelopmentStop::Converged => "converged",
            axon_uic::DevelopmentStop::PassLimit => "pass-limit",
        },
        r.final_reserve_correct,
        r.reserve_tasks,
        r.passes,
        r.training_steps,
        r.initial_reserve_mse,
        r.final_reserve_mse
    );
    Ok(())
}
