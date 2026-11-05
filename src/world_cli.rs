use crate::{Authority, Effect, EpisodeStore, cli::CliError, world::*};
use std::{
    collections::BTreeMap,
    fmt::Write,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
pub(crate) const USAGE: &str = "\n  axon world-demo <new-directory>\n  axon world-inspect <model.axworld>\n  axon world-plan <model.axworld> --context <context> --start <state> --on <mask> --off <mask>\n  axon world-rebuild <events.axep> --bits <1..6> --actions <1..8> --context <context> --model <new.axworld>";
impl From<WorldError> for CliError {
    fn from(e: WorldError) -> Self {
        Self::World(e)
    }
}
fn options(args: &[String], names: &[&str]) -> Result<BTreeMap<String, String>, CliError> {
    if args.len() != 2 + names.len() * 2 {
        return Err(CliError::Usage);
    }
    let mut values = BTreeMap::new();
    for pair in args[2..].as_chunks::<2>().0 {
        if !names.contains(&pair[0].as_str())
            || values.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(CliError::Usage);
        }
    }
    Ok(values)
}
fn number(values: &BTreeMap<String, String>, key: &'static str) -> Result<u8, CliError> {
    let value = values.get(key).ok_or(CliError::Usage)?;
    value.parse().map_err(|_| CliError::InvalidNumber {
        option: key,
        value: value.clone(),
    })
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let path = args.get(1).ok_or(CliError::Usage)?;
    match args[0].as_str() {
        "world-demo" if args.len() == 2 => demo(Path::new(path)),
        "world-inspect" if args.len() == 2 => {
            let model = WorldModel::load(path)?;
            let mut out = format!(
                "context={} bits={} actions={} interventions={} bytes={} fingerprint={:016x}\n",
                model.context(),
                model.shape().bits(),
                model.shape().actions(),
                model.interventions(),
                model.encoded_bytes(),
                model.fingerprint()
            );
            for (action, rule) in model.rules().iter().enumerate() {
                writeln!(out, "action={action} {rule:?}").expect("String");
            }
            Ok(out)
        }
        "world-plan" => {
            let values = options(args, &["--context", "--start", "--on", "--off"])?;
            let model = WorldModel::load(path)?;
            if model.context() != values["--context"] {
                return Err(WorldError::Invalid("context mismatch").into());
            }
            let start = number(&values, "--start")?;
            let goal = Goal {
                on: number(&values, "--on")?,
                off: number(&values, "--off")?,
            };
            Ok(format!(
                "{:?}\n",
                model.plan(start, goal, model.shape().all_states())?
            ))
        }
        "world-rebuild" => {
            let values = options(args, &["--bits", "--actions", "--context", "--model"])?;
            if !Path::new(path).is_file() {
                return Err(WorldError::Invalid("missing journal").into());
            }
            let store = EpisodeStore::open(path)?.0;
            let shape = WorldShape::new(number(&values, "--bits")?, number(&values, "--actions")?)?;
            let learner = WorldLearner::restore(&store, shape, &values["--context"])?;
            let model = learner.compile()?;
            model.save_new(&values["--model"])?;
            Ok(format!(
                "rebuilt={} records={} interventions={} fingerprint={:016x}\n",
                values["--model"],
                learner.records(),
                model.interventions(),
                model.fingerprint()
            ))
        }
        _ => Err(CliError::Usage),
    }
}
// Hidden from the learner. The controlled adapter can reset any starting state.
fn workcell(state: u8, action: u8) -> u8 {
    match action {
        0 if state & 1 != 0 => state | 2,
        1 if state & 1 != 0 && state & 4 == 0 => state ^ 8,
        2 => state ^ 1,
        3 if state & 8 == 0 => state & !16,
        4 if state & 2 != 0 => state ^ 4,
        _ => state,
    }
}
fn demo(directory: &Path) -> Result<String, CliError> {
    fs::create_dir(directory).map_err(|source| CliError::ReadTask {
        path: directory.into(),
        source,
    })?;
    let journal = directory.join("workcell.axep");
    let artifact = directory.join("workcell.axworld");
    let shape = WorldShape::new(5, 5)?;
    let context = "demo/workcell-v1";
    let authority = Authority::from([Effect::Simulate, Effect::WriteData]);
    let mut scope = ProbeScope::unrestricted(shape, 512);
    let started = Instant::now();
    let model;
    let mut out = String::new();
    {
        let mut store = EpisodeStore::open(&journal)?.0;
        let mut session = WorldSession::restore(&mut store, shape, context)?;
        let initial = session.learner().candidate_counts();
        // A deliberately confounded passive observation must not become a causal rule.
        session.record_passive(0, 0, 31)?;
        assert_eq!(session.learner().candidate_counts(), initial);
        writeln!(
            out,
            "initial_hypotheses={initial:?} passive_observation_pruned=false"
        )
        .expect("String");
        while let ProbeChoice::Experiment(probe) = session.learner().choose_probe(&scope)? {
            let after = session.intervene(probe, &mut scope, &authority, workcell)?;
            writeln!(
                out,
                "trial={} state={} action={} observed={} remaining={:?}",
                session.learner().interventions(),
                probe.state,
                probe.action,
                after,
                session.learner().candidate_counts()
            )
            .expect("String");
        }
        model = session.learner().compile()?;
        model.save_new(&artifact)?;
    }
    let fit_ms = started.elapsed().as_secs_f64() * 1000.0;
    let loaded = WorldModel::load(&artifact)?;
    let mut store = EpisodeStore::open(&journal)?.0;
    let mut session = WorldSession::restore(&mut store, shape, context)?;
    if session.learner().compile()? != loaded {
        return Err(WorldError::Invalid("replay mismatch").into());
    }
    let mut exact = 0;
    for state in 0..32 {
        for action in 0..5 {
            if loaded.predict(state, action)? != workcell(state, action) {
                return Err(WorldError::Invalid("audit mismatch").into());
            }
            exact += 1;
        }
    }
    let PlanResult::Found(plan) = loaded.plan(20, Goal { on: 11, off: 20 }, shape.all_states())?
    else {
        return Err(WorldError::Invalid("demo goal unreachable").into());
    };
    let final_state = session.execute(&loaded, &plan, &mut scope, &authority, workcell)?;
    writeln!(out, "learned_interventions={} fit_with_sync_ms={fit_ms:.3} model_bytes={} replay_exact=true audit={exact}/{exact}", model.interventions(), model.encoded_bytes()).expect("String");
    writeln!(
        out,
        "novel_goal_plan={:?} final_state={final_state} reached={}",
        plan.steps,
        plan.goal.matches(final_state)
    )
    .expect("String");
    writeln!(
        out,
        "artifact={} journal={}",
        artifact.display(),
        journal.display()
    )
    .expect("String");
    let report: PathBuf = directory.join("demo.txt");
    crate::text_skill::publish_new(&report, out.as_bytes()).map_err(WorldError::from)?;
    Ok(out)
}
