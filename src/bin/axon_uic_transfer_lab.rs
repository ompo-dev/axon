use axon_uic::{
    behavior::*,
    behavior_lab::{CachedOracle, Rng, audit_machine},
    transfer::*,
    transfer_lab::*,
};
use std::{collections::BTreeSet, error::Error, fs, path::Path, time::Instant};

fn random_adapt(
    models: &[&BehaviorModel],
    context: &str,
    oracle: &mut impl SequenceOracle,
    seed: u64,
) -> Result<Adaptation, TransferError> {
    let mut learner = TransferLearner::new(models, 4, context, TransferBudget::default())?;
    let mut rng = Rng::new(seed);
    let mut failure = "random prior exhausted".to_owned();
    for _ in 0..128 {
        if let Some(result) = learner.compile()? {
            if let Some(witness) = conformance_probe(&result.model, oracle, 65_536)? {
                failure = format!("prior refuted: {witness:?}");
                break;
            }
            return Ok(Adaptation {
                model: result.model,
                mode: AdaptationMode::StructuralTransfer,
                transfer_stats: result.stats,
                transfer_failure: None,
                source_fingerprints: result.source_fingerprints,
                evidence_digest: result.evidence_digest,
            });
        }
        let word = (0..1 + rng.pick(8))
            .map(|_| rng.pick(4) as u8)
            .collect::<Vec<_>>();
        if learner.preflight(&word).is_err() {
            continue;
        }
        let reply = oracle.query(&word)?;
        match learner.observe(&word, &reply) {
            Ok(()) => {}
            Err(TransferError::Refuted) => {
                failure = "random prior refuted".into();
                break;
            }
            Err(e) => return Err(e),
        }
    }
    let model = discover(4, context, oracle)?;
    Ok(Adaptation {
        model,
        mode: AdaptationMode::Discovery,
        transfer_stats: learner.stats(),
        transfer_failure: Some(failure),
        source_fingerprints: Vec::new(),
        evidence_digest: 0,
    })
}
fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: axon-uic-transfer-lab <seed> <cases:4..256> <new.csv>".into());
    }
    let seed = args[0].parse::<u64>()?;
    let cases = args[1].parse::<usize>()?;
    if !(4..=256).contains(&cases) {
        return Err("cases must be 4..256".into());
    }
    let path = Path::new(&args[2]);
    let models_dir = path.with_extension("models");
    if path.exists() || models_dir.exists() {
        return Err("outputs must be new".into());
    }
    fs::create_dir_all(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    fs::create_dir(&models_dir)?;
    let training = Instant::now();
    let mut sources = Vec::new();
    let mut source_queries = 0;
    let mut source_steps = 0;
    for family in [Family::Grid, Family::Protocol, Family::Pipeline] {
        let extent = if family == Family::Protocol { 3 } else { 2 };
        let env = Environment::new(family, extent, 0);
        let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
        let model = discover(4, &format!("lab/source/{}", family.name()), &mut oracle)?;
        if audit_machine(&model, 4, 0, |s, a| env.transition(s, a))?
            .counterexample
            .is_some()
        {
            return Err("source audit mismatch".into());
        }
        source_queries += oracle.raw.queries;
        source_steps += oracle.raw.steps;
        model.save_new(models_dir.join(format!("source-{}.axbehavior", family.name())))?;
        sources.push(model);
    }
    let source_ns = training.elapsed().as_nanos();
    let source_bytes: usize = sources.iter().map(BehaviorModel::encoded_bytes).sum();
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "seed",
        "case",
        "family",
        "policy",
        "mode",
        "fit_ns",
        "queries",
        "steps",
        "learning_probes",
        "hypotheses_initial",
        "hypotheses_remaining",
        "search_work",
        "hypothesis_bytes",
        "model_bytes",
        "states",
        "audit_edges",
        "exact",
        "goals",
        "goal_correct",
        "transfer_failure",
        "failure",
    ])?;
    let mut rng = Rng::new(seed);
    let mut exacts = [0_usize; 3];
    let mut goal_checks = 0;
    let mut goal_correct = 0;
    let mut total_queries = [0_usize; 3];
    let mut total_steps = [0_usize; 3];
    let mut total_ns = [0_u128; 3];
    let mut transfers = [0_usize; 2];
    let mut novel_discoveries = 0;
    for case in 0..cases {
        let (family, extent, label) = match case % 4 {
            0 => (Family::Grid, 2, "discrete-control"),
            1 => (Family::Protocol, 3, "symbolic-protocol"),
            2 => (Family::Pipeline, 2, "document-workflow"),
            _ => (Family::Grid, 3, "novel-control"),
        };
        let template = Environment::new(family, extent, 0);
        let initial = rng.pick(template.states());
        let target = Rewired::random(Environment::new(family, extent, initial), &mut rng);
        let context = format!("lab/target/{seed}/{case}");
        let order = if case % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
        for policy in order {
            let mut oracle = CachedOracle::new(target.clone(), BehaviorBudget::default());
            let start = Instant::now();
            let result = match policy {
                0 => adapt(
                    &sources.iter().collect::<Vec<_>>(),
                    4,
                    &context,
                    TransferBudget::default(),
                    &mut oracle,
                ),
                1 => random_adapt(
                    &sources.iter().collect::<Vec<_>>(),
                    &context,
                    &mut oracle,
                    seed ^ (case as u64).wrapping_mul(713_919),
                ),
                _ => discover(4, &context, &mut oracle).map(|model| Adaptation {
                    model,
                    mode: AdaptationMode::Discovery,
                    transfer_stats: TransferStats::default(),
                    transfer_failure: None,
                    source_fingerprints: Vec::new(),
                    evidence_digest: 0,
                }),
            };
            let fit_ns = start.elapsed().as_nanos();
            total_queries[policy] += oracle.raw.queries;
            total_steps[policy] += oracle.raw.steps;
            total_ns[policy] += fit_ns;
            let policy_name = ["active-transfer", "random-transfer", "from-scratch"][policy];
            let (adapted, error) = match result {
                Ok(a) => (Some(a), String::new()),
                Err(e) => (None, e.to_string()),
            };
            let mut audit_edges = 0;
            let mut exact = false;
            let mut goals = 0;
            let mut correct = 0;
            if let Some(a) = &adapted {
                let audit = audit_machine(&a.model, 4, initial, |s, action| {
                    target.transition(s, action)
                })?;
                audit_edges = audit.checked_edges;
                exact = audit.counterexample.is_none();
                if exact {
                    exacts[policy] += 1;
                }
                if policy < 2 && a.mode == AdaptationMode::StructuralTransfer {
                    transfers[policy] += 1;
                }
                if policy == 0 && label == "novel-control" && a.mode == AdaptationMode::Discovery {
                    novel_discoveries += 1;
                }
                let mut used = BTreeSet::new();
                for source in target.environment.symbols() {
                    let goal = target.output_for(source);
                    used.insert(goal);
                    let plan = a.model.plan_output(&[], goal, &BTreeSet::new())?;
                    goals += 1;
                    if let Some(word) = plan {
                        let mut state = initial;
                        let mut last = None;
                        for action in word {
                            let (next, out) = target.transition(state, action)?;
                            state = next;
                            last = Some(out);
                        }
                        if last == Some(goal) {
                            correct += 1;
                        }
                    }
                }
                let absent = (0..=u16::MAX)
                    .find(|n| !used.contains(n))
                    .expect("unused output");
                goals += 1;
                if a.model
                    .plan_output(&[], absent, &BTreeSet::new())?
                    .is_none()
                {
                    correct += 1;
                }
                a.model
                    .save_new(models_dir.join(format!("case-{case}-{policy}.axbehavior")))?;
                let loaded = BehaviorModel::load(
                    models_dir.join(format!("case-{case}-{policy}.axbehavior")),
                )?;
                if loaded != a.model {
                    return Err("reload mismatch".into());
                }
            }
            goal_checks += goals;
            goal_correct += correct;
            let stats = adapted
                .as_ref()
                .map_or(TransferStats::default(), |a| a.transfer_stats);
            writer.write_record([
                seed.to_string(),
                case.to_string(),
                label.into(),
                policy_name.into(),
                adapted
                    .as_ref()
                    .map_or("error".into(), |a| format!("{:?}", a.mode)),
                fit_ns.to_string(),
                oracle.raw.queries.to_string(),
                oracle.raw.steps.to_string(),
                stats.queries.to_string(),
                stats.initial_hypotheses.to_string(),
                stats.surviving_hypotheses.to_string(),
                stats.work.to_string(),
                stats.peak_hypothesis_bytes.to_string(),
                adapted
                    .as_ref()
                    .map_or(0, |a| a.model.encoded_bytes())
                    .to_string(),
                adapted.as_ref().map_or(0, |a| a.model.states()).to_string(),
                audit_edges.to_string(),
                exact.to_string(),
                goals.to_string(),
                correct.to_string(),
                adapted
                    .as_ref()
                    .and_then(|a| a.transfer_failure.clone())
                    .unwrap_or_default(),
                error,
            ])?;
        }
    }
    // A new representation becomes a reusable fourth prototype, not a replacement for old ones.
    let larger = Rewired::random(Environment::new(Family::Grid, 3, 0), &mut rng);
    let mut oracle = CachedOracle::new(larger.clone(), BehaviorBudget::default());
    let new_model = adapt(
        &sources.iter().collect::<Vec<_>>(),
        4,
        "lab/new-representation",
        TransferBudget::default(),
        &mut oracle,
    )?;
    let grew = new_model.mode == AdaptationMode::Discovery && new_model.model.states() == 9;
    let new_exact = audit_machine(&new_model.model, 4, 0, |s, a| larger.transition(s, a))?
        .counterexample
        .is_none();
    let growth_queries = oracle.raw.queries;
    let growth_steps = oracle.raw.steps;
    let new_path = models_dir.join("new-representation.axbehavior");
    new_model.model.save_new(&new_path)?;
    sources.push(BehaviorModel::load(&new_path)?);
    let new_target = Rewired::random(Environment::new(Family::Grid, 3, 8), &mut rng);
    let mut oracle = CachedOracle::new(new_target.clone(), BehaviorBudget::default());
    let reuse = adapt(
        &sources.iter().collect::<Vec<_>>(),
        4,
        "lab/reuse-new-representation",
        TransferBudget::default(),
        &mut oracle,
    )?;
    let reuse_exact = audit_machine(&reuse.model, 4, 8, |s, a| new_target.transition(s, a))?
        .counterexample
        .is_none();
    reuse
        .model
        .save_new(models_dir.join("reused-new-representation.axbehavior"))?;
    writer.flush()?;
    let data = writer.into_inner()?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    use std::io::Write;
    file.write_all(&data)?;
    file.sync_all()?;
    println!(
        "cases={cases}\nsource_learning_ns={source_ns}\nsource_queries={source_queries}\nsource_steps={source_steps}\nsource_model_bytes={source_bytes}\nexact_by_policy={exacts:?}\nstructural_transfers_active_random={transfers:?}\nnovel_models_learned={novel_discoveries}\nqueries_by_policy={total_queries:?}\nsteps_by_policy={total_steps:?}\nfit_ns_by_policy={total_ns:?}\ngoals_correct={goal_correct}/{goal_checks}\nnew_representation_9_states={grew}\nnew_representation_exact={new_exact}\nnew_representation_queries={growth_queries}\nnew_representation_steps={growth_steps}\nnew_representation_reused={:?}\nnew_representation_reuse_exact={reuse_exact}\nnew_representation_reuse_queries={}\nnew_representation_reuse_steps={}\n",
        reuse.mode, oracle.raw.queries, oracle.raw.steps
    );
    if exacts.iter().any(|&n| n != cases)
        || goal_correct != goal_checks
        || !grew
        || !new_exact
        || !reuse_exact
        || reuse.mode != AdaptationMode::StructuralTransfer
    {
        return Err("some cases failed; raw results preserved".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
