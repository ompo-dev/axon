use crate::{
    Authority, Effect,
    behavior::BehaviorError,
    cli::CliError,
    factor::*,
    factor_lab::{self, Environment, Kind},
    text_skill::{publish_new, read_bounded},
};
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
pub(crate) const USAGE: &str = "\n  axon factor-demo <new-directory> [--inputs <markdown-directory>]\n  axon factor-publish <factors.axseed> <new-directory> [--inputs <markdown-directory>]";
impl From<FactorError> for CliError {
    fn from(e: FactorError) -> Self {
        Self::Factor(e)
    }
}
fn inputs(path: Option<&String>) -> Result<Vec<(PathBuf, Vec<u8>)>, FactorError> {
    let files = if let Some(path) = path {
        let mut paths = Vec::new();
        for entry in fs::read_dir(path)? {
            let p = entry?.path();
            if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("md")) {
                if paths.len() == 5 {
                    return Err(FactorError::Invalid("expected 2..5 Markdown files"));
                }
                paths.push(p);
            }
        }
        paths.sort();
        if !(2..=5).contains(&paths.len()) {
            return Err(FactorError::Invalid("expected 2..5 Markdown files"));
        }
        paths
            .into_iter()
            .map(|p| {
                let b = read_bounded(&p, 65536).map_err(BehaviorError::from)?;
                Ok((p, b))
            })
            .collect::<Result<Vec<_>, FactorError>>()?
    } else {
        [
            (
                "architecture.md",
                include_bytes!("../examples/factor-documents/architecture.md").as_slice(),
            ),
            (
                "deployment.md",
                include_bytes!("../examples/factor-documents/deployment.md").as_slice(),
            ),
            (
                "requirements.md",
                include_bytes!("../examples/factor-documents/requirements.md").as_slice(),
            ),
            (
                "validation.md",
                include_bytes!("../examples/factor-documents/validation.md").as_slice(),
            ),
        ]
        .into_iter()
        .map(|(name, bytes)| (PathBuf::from(name), bytes.to_vec()))
        .collect()
    };
    if files.iter().any(|(_, b)| {
        b.len() < 20 || !b.starts_with(b"# ") || b.contains(&0) || std::str::from_utf8(b).is_err()
    }) {
        return Err(FactorError::Invalid("titled UTF-8 Markdown required"));
    }
    Ok(files)
}
fn context(dimensions: usize) -> String {
    format!("demo/factor/workflow-v1/{dimensions}")
}
fn environment(dimensions: usize) -> Result<Environment, FactorError> {
    Ok(Environment::random(Kind::Workflow, dimensions, 719319)?.zero_initial())
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    match args.first().map(String::as_str) {
        Some("factor-demo") if args.len() == 2 || (args.len() == 4 && args[2] == "--inputs") => {
            let files = inputs(args.get(3))?;
            demo(Path::new(&args[1]), files).map_err(Into::into)
        }
        Some("factor-publish") if args.len() == 3 || (args.len() == 5 && args[3] == "--inputs") => {
            let files = inputs(args.get(4))?;
            let auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
            let started = Instant::now();
            let model =
                FactorModel::load_bundle(Path::new(&args[1]), &context(files.len()), &auth)?;
            let report = publish(&model, Path::new(&args[2]), &files, &auth)?;
            Ok(format!(
                "reused_model=true\nnew_learning_queries=0\nload_plan_publication_ms={:.3}\n{report}",
                started.elapsed().as_secs_f64() * 1000.0
            ))
        }
        _ => Err(CliError::Usage),
    }
}
fn publish(
    model: &FactorModel,
    directory: &Path,
    files: &[(PathBuf, Vec<u8>)],
    auth: &Authority,
) -> Result<String, FactorError> {
    let env = environment(files.len())?;
    let goal = model.goal(&env.goal_tokens())?;
    let planned = model
        .plan(&[], &goal, PlanBudget::default())?
        .ok_or(FactorError::Invalid("publication goal unreachable"))?;
    fs::create_dir(directory)?;
    let targets = files
        .iter()
        .enumerate()
        .map(|(i, (_, bytes))| (bytes.clone(), directory.join(format!("document-{i}.md"))))
        .collect::<Vec<_>>();
    let mut physical = env.publishing(targets.clone())?;
    let executed = model
        .execute_goal(&[], &goal, &mut physical, auth, PlanBudget::default())?
        .ok_or(FactorError::Invalid("publication plan disappeared"))?;
    if planned != executed {
        return Err(FactorError::Invalid("publication plan mismatch"));
    }
    let mut report = format!(
        "documents={}\nexecution_actions={}\nplan={executed:?}\n",
        files.len(),
        executed.len()
    );
    for (i, (bytes, path)) in targets.iter().enumerate() {
        let saved = read_bounded(path, 65536).map_err(BehaviorError::from)?;
        if saved != *bytes {
            return Err(FactorError::Invalid("published content mismatch"));
        }
        report.push_str(&format!(
            "document_{i}_source={}\ndocument_{i}_bytes={}\n",
            files[i].0.display(),
            saved.len()
        ));
    }
    report.push_str("scope=learned finite workflow controller; supplied sensor schema and goal tokens; no natural-language understanding or AGI\n");
    publish_new(&directory.join("publication.txt"), report.as_bytes())
        .map_err(BehaviorError::from)?;
    Ok(report)
}
fn demo(directory: &Path, files: Vec<(PathBuf, Vec<u8>)>) -> Result<String, FactorError> {
    if directory.exists() {
        return Err(FactorError::Invalid("demo destination must be new"));
    }
    let env = environment(files.len())?;
    let mut oracle = FrameCache::new(
        env.clone(),
        env.actions(),
        env.dimensions(),
        FrameBudget::default(),
        Authority::from([Effect::Simulate]),
    )?;
    let started = Instant::now();
    let model = learn(
        env.actions(),
        env.dimensions(),
        &context(files.len()),
        &mut oracle,
    )?;
    let fit_ms = started.elapsed().as_secs_f64() * 1000.0;
    let audit = factor_lab::audit(&model, &env, 65536)?;
    if audit.counterexample.is_some() {
        return Err(FactorError::Invalid("whole-product audit failed"));
    }
    fs::create_dir(directory)?;
    let authority = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    model.save_bundle_new(&directory.join("models"), &authority)?;
    let loaded = FactorModel::load_bundle(
        &directory.join("models/factors.axseed"),
        model.context(),
        &authority,
    )?;
    if loaded != model {
        return Err(FactorError::Invalid("factor reload mismatch"));
    }
    let publication = publish(&loaded, &directory.join("published"), &files, &authority)?;
    let report = format!(
        "fit_ms={fit_ms:.3}\nlearning_queries={}\nlearning_steps={}\ncache_nodes={}\nfactors={}\nfactor_states={}\nproduct_state_upper_bound={}\naudited_reachable_states={}\naudited_edges={}\nmodel_bytes={}\nreload_matches=true\n{publication}",
        oracle.queries(),
        oracle.steps(),
        oracle.cache_nodes(),
        model.dimensions(),
        model.factor_states(),
        model.product_states(),
        audit.reachable_pairs,
        audit.checked_edges,
        model.encoded_model_bytes()
    );
    publish_new(&directory.join("report.txt"), report.as_bytes()).map_err(BehaviorError::from)?;
    Ok(report)
}
