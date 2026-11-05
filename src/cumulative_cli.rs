use crate::{
    Authority, Effect, EpisodeStamp, casework::plan_files, cli::CliError, cumulative::StudySeed,
    demand::genome::*,
};
use std::path::Path;

pub const USAGE: &str = "\n  axon study-inspect <study.axstudy>\n  axon case-plan <seed.axnum> <new-birth-directory> <data-directory> <sentence> <event-id>";
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    match args.first().map(String::as_str) {
        Some("study-inspect") if args.len() == 2 => {
            bounded_path(Path::new(&args[1]))?;
            let seed = StudySeed::open(Path::new(&args[1]), &auth)?;
            Ok(format!(
                "context={}\nrevision={}\nparent={:016X}\nskills={}\nframes={}\nquestion_policy={:?}\nquestion_limit={}\nsearch_work_limit={}\nencoded_bytes={}",
                seed.context(),
                seed.revision(),
                seed.parent(),
                seed.skills().len(),
                seed.frames().len(),
                seed.policy().question_policy,
                seed.policy().questions,
                seed.policy().total_work,
                seed.to_bytes()?.len()
            ))
        }
        Some("case-plan") if args.len() == 6 => {
            let genome = Genome::open_number(Path::new(&args[1]), &auth)?;
            let mut birth = genome.birth(Path::new(&args[2]), BirthLimits::default(), &auth)?;
            let plan = plan_files(
                birth.controller.runtime_mut(),
                Path::new(&args[3]),
                &args[4],
                EpisodeStamp::new("case-cli", &args[5], 0),
                &auth,
            )?;
            Ok(format!(
                "status=verified-dry-run\nitem={}\nneeded={}\nunits={}\ncost_cents={}\nbudget_cents={}\nchoices={}\nstock={:016X}\noffers={:016X}\nmodel={:016X}\nepisode={}\nduplicate={}\noptimizer_nodes={}\noptimizer_iterations={}",
                plan.item,
                plan.needed,
                plan.units,
                plan.cost_cents,
                plan.budget_cents,
                plan.choices
                    .iter()
                    .map(|c| format!("{}:{}", c.supplier, c.packs))
                    .collect::<Vec<_>>()
                    .join(","),
                plan.source_stock,
                plan.source_offers,
                plan.model,
                plan.episode,
                plan.duplicate,
                plan.nodes,
                plan.iterations
            ))
        }
        _ => Err(CliError::Usage),
    }
}
