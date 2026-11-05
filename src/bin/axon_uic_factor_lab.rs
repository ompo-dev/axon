use axon_uic::{
    Authority, Effect,
    behavior::*,
    behavior_lab::{CachedOracle, Rng, audit_machine},
    factor::*,
    factor_lab::{self, Environment, Flatten, Kind},
    transfer::*,
    transfer_lab,
};
use std::{
    collections::{BTreeSet, VecDeque},
    error::Error,
    fs,
    io::Write,
    path::Path,
    time::Instant,
};

fn flat_audit<O: FrameOracle>(
    model: &BehaviorModel,
    truth: &Environment,
    codes: &Flatten<'_, O>,
) -> Result<(bool, usize, usize), Box<dyn Error>> {
    let root = (truth.initial_state(), 0_u16);
    let mut seen = BTreeSet::from([root]);
    let mut queue = VecDeque::from([root]);
    let mut edges = 0;
    while let Some((state, learned)) = queue.pop_front() {
        for action in 0..truth.actions() {
            let (next, actual) = truth.transition(state, action)?;
            let edge = model.edge(learned, action)?;
            edges += 1;
            if codes.frame(edge.output) != Some(actual) {
                return Ok((false, seen.len(), edges));
            }
            let pair = (next, edge.next);
            if seen.insert(pair) {
                if seen.len() > 65536 {
                    return Err("flat audit budget".into());
                }
                queue.push_back(pair);
            }
        }
    }
    Ok((true, seen.len(), edges))
}
fn distance(
    env: &Environment,
    history: &[u8],
    target: &[(usize, u16)],
) -> Result<Option<usize>, BehaviorError> {
    let matches = |f: Frame| target.iter().all(|&(i, t)| f.values()[i] == t);
    let mut root = env.initial_state();
    let mut last = None;
    for &a in history {
        let (next, frame) = env.transition(root, a)?;
        root = next;
        last = Some(frame);
    }
    if last.is_some_and(matches) {
        return Ok(Some(0));
    }
    let mut visited = BTreeSet::from([root]);
    let mut queue = VecDeque::from([(root, 0)]);
    while let Some((state, steps)) = queue.pop_front() {
        for a in 0..env.actions() {
            let (next, frame) = env.transition(state, a)?;
            if matches(frame) {
                return Ok(Some(steps + 1));
            }
            if visited.insert(next) {
                queue.push_back((next, steps + 1));
            }
        }
    }
    Ok(None)
}
fn goals(
    model: &FactorModel,
    env: &Environment,
    seed: u64,
) -> Result<(usize, usize), Box<dyn Error>> {
    let mut rng = Rng::new(seed);
    let mut correct = 0;
    let mut steps = 0;
    for i in 0..6 {
        let history = (0..rng.pick(6))
            .map(|_| rng.pick(env.actions() as usize) as u8)
            .collect::<Vec<_>>();
        let mut state = env.initial_state();
        for &a in &history {
            state = env.transition(state, a)?.0;
        }
        for _ in 0..1 + rng.pick(8) {
            state = env
                .transition(state, rng.pick(env.actions() as usize) as u8)?
                .0;
        }
        let mut target = env.target_tokens(state);
        if i % 2 == 0 {
            target.truncate(1);
        }
        if i == 5 {
            let factor = &model.factors()[0];
            let used = (0..factor.states())
                .flat_map(|state| {
                    (0..factor.actions())
                        .map(move |action| factor.edge(state as u16, action).expect("valid").output)
                })
                .collect::<BTreeSet<_>>();
            let absent = (0..=u16::MAX)
                .find(|t| !used.contains(t))
                .expect("unused token");
            target = vec![(0, absent)];
        }
        let goal = model.goal(&target)?;
        let expected = distance(env, &history, &target)?;
        let plan = model.plan(&history, &goal, PlanBudget::default())?;
        if plan.as_ref().map(Vec::len) != expected {
            continue;
        }
        if let Some(plan) = plan {
            let mut current = env.initial_state();
            let mut last = None;
            for &a in history.iter().chain(&plan) {
                let (next, frame) = env.transition(current, a)?;
                current = next;
                last = Some(frame);
            }
            if last.is_none_or(|f| target.iter().any(|&(i, t)| f.values()[i] != t)) {
                continue;
            }
            steps += plan.len();
        }
        correct += 1;
    }
    Ok((correct, steps))
}
fn write_new(path: &Path, data: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(data)?;
    file.sync_all()?;
    Ok(())
}
fn selection(seed: u64, cases: usize, path: &Path, models: &Path) -> Result<bool, Box<dyn Error>> {
    let mut sources = Vec::new();
    let mut source_queries = 0;
    let mut source_steps = 0;
    let started = Instant::now();
    for (i, family) in [
        transfer_lab::Family::Grid,
        transfer_lab::Family::Protocol,
        transfer_lab::Family::Pipeline,
    ]
    .into_iter()
    .enumerate()
    {
        let extent = if family == transfer_lab::Family::Protocol {
            3
        } else {
            2
        };
        let env = transfer_lab::Environment::new(family, extent, 0);
        let mut oracle = CachedOracle::new(env.clone(), BehaviorBudget::default());
        let model = discover(4, &format!("selection/source/{i}"), &mut oracle)?;
        if audit_machine(&model, 4, 0, |s, a| env.transition(s, a))?
            .counterexample
            .is_some()
        {
            return Err("source audit".into());
        }
        source_queries += oracle.raw.queries;
        source_steps += oracle.raw.steps;
        model.save_new(models.join(format!("selection-source-{i}.axbehavior")))?;
        sources.push(model);
    }
    let source_ns = started.elapsed().as_nanos();
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "seed",
        "case",
        "round",
        "policy",
        "fit_ns",
        "queries",
        "steps",
        "learning_probes",
        "exact",
        "same_model_and_experiments",
        "failure",
    ])?;
    let mut rng = Rng::new(seed);
    let mut passed = true;
    let mut total = [0_u128; 3];
    let mut exacts = [0; 3];
    for case in 0..cases {
        let (family, extent) = match case % 4 {
            0 => (transfer_lab::Family::Grid, 2),
            1 => (transfer_lab::Family::Protocol, 3),
            2 => (transfer_lab::Family::Pipeline, 2),
            _ => (transfer_lab::Family::Grid, 3),
        };
        let env = transfer_lab::Environment::new(family, extent, 0);
        let initial = rng.pick(env.states());
        let target = transfer_lab::Rewired::random(
            transfer_lab::Environment::new(family, extent, initial),
            &mut rng,
        );
        for round in 0..5 {
            let context = format!("selection/target/{seed}/{case}/{round}");
            let mut records = Vec::new();
            let order = if (case + round) % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            };
            for policy in order {
                let mut oracle = CachedOracle::new(target.clone(), BehaviorBudget::default());
                let start = Instant::now();
                let adapted = if policy < 2 {
                    adapt_with_selection(
                        &sources.iter().collect::<Vec<_>>(),
                        4,
                        &context,
                        TransferBudget::default(),
                        &mut oracle,
                        if policy == 0 {
                            ProbeSelection::Compact
                        } else {
                            ProbeSelection::Reference
                        },
                    )
                    .map(|a| (a.model, a.transfer_stats.queries))
                } else {
                    discover(4, &context, &mut oracle).map(|m| (m, 0))
                };
                let ns = start.elapsed().as_nanos();
                total[policy] += ns;
                let exact = if let Ok((model, _)) = &adapted {
                    audit_machine(model, 4, initial, |s, a| target.transition(s, a))?
                        .counterexample
                        .is_none()
                } else {
                    false
                };
                if exact {
                    exacts[policy] += 1;
                } else {
                    passed = false;
                }
                records.push((
                    policy,
                    ns,
                    oracle.raw.queries,
                    oracle.raw.steps,
                    exact,
                    adapted,
                ));
            }
            records.sort_by_key(|r| r.0);
            let parity = match (&records[0].5, &records[1].5) {
                (Ok((a, pa)), Ok((b, pb))) => {
                    a == b
                        && pa == pb
                        && records[0].2 == records[1].2
                        && records[0].3 == records[1].3
                }
                _ => false,
            };
            passed &= parity;
            for (policy, ns, queries, steps, exact, result) in records {
                let (probes, failure) = match result {
                    Ok((_, p)) => (p, String::new()),
                    Err(e) => (0, e.to_string()),
                };
                writer.write_record([
                    seed.to_string(),
                    case.to_string(),
                    round.to_string(),
                    ["compact", "reference", "from-scratch"][policy].into(),
                    ns.to_string(),
                    queries.to_string(),
                    steps.to_string(),
                    probes.to_string(),
                    exact.to_string(),
                    parity.to_string(),
                    failure,
                ])?;
            }
        }
    }
    writer.flush()?;
    write_new(path, &writer.into_inner()?)?;
    println!(
        "selection_source_ns={source_ns}\nselection_source_queries={source_queries}\nselection_source_steps={source_steps}\nselection_fit_ns={total:?}\nselection_exact={exacts:?}\nselection_parity={passed}"
    );
    Ok(passed)
}
fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: axon-uic-factor-lab <seed> <cases:5..64> <new.csv>".into());
    }
    let seed = args[0].parse::<u64>()?;
    let cases = args[1].parse::<usize>()?;
    if !(5..=64).contains(&cases) {
        return Err("cases 5..64".into());
    }
    let path = Path::new(&args[2]);
    let models = path.with_extension("models");
    let selection_path = path.with_extension("selection.csv");
    if path.exists() || models.exists() || selection_path.exists() {
        return Err("outputs must be new".into());
    }
    fs::create_dir_all(
        path.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    fs::create_dir(&models)?;
    let mut writer = csv::Writer::from_writer(Vec::new());
    writer.write_record([
        "seed",
        "case",
        "regime",
        "policy",
        "fit_ns",
        "queries",
        "steps",
        "states",
        "product_upper",
        "model_bytes",
        "audit_states",
        "audit_edges",
        "exact",
        "goals",
        "goal_correct",
        "plan_steps",
        "failure",
    ])?;
    let mut factor_exact = 0;
    let mut flat_exact = 0;
    let mut flat_refused = 0;
    let mut wrong_flat = 0;
    let mut goal_correct = 0;
    let mut passed = true;
    let authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut rng = Rng::new(seed);
    for case in 0..cases {
        let (kind, n, label) = match case % 5 {
            0 => (Kind::Cycles, 3, "small-parallel"),
            1 => (Kind::Cycles, 6, "large-parallel"),
            2 => (Kind::Workflow, 2, "small-hidden-workflow"),
            3 => (Kind::Workflow, 4, "large-hidden-workflow"),
            _ => (Kind::Coupled, 2, "coupled-negative"),
        };
        let mut env = Environment::random(kind, n, rng.next_u64())?;
        if kind == Kind::Workflow {
            env = env.zero_initial();
        }
        let context = format!("factor/reserve/{seed}/{case}");
        let order = if case % 2 == 0 { [0, 1] } else { [1, 0] };
        for policy in order {
            let mut oracle = FrameCache::new(
                env.clone(),
                env.actions(),
                env.dimensions(),
                FrameBudget::default(),
                Authority::from([Effect::Simulate]),
            )?;
            if policy == 0 {
                let start = Instant::now();
                let result = learn(env.actions(), env.dimensions(), &context, &mut oracle);
                let ns = start.elapsed().as_nanos();
                let mut states = 0;
                let mut product = 0;
                let mut bytes = 0;
                let mut audit_states = 0;
                let mut edges = 0;
                let mut exact = false;
                let mut correct = 0;
                let mut plan_steps = 0;
                let mut failure = String::new();
                match result {
                    Ok(model) => {
                        states = model.factor_states();
                        product = model.product_states();
                        bytes = model.encoded_model_bytes();
                        let audit = factor_lab::audit(&model, &env, 65536)?;
                        audit_states = audit.reachable_pairs;
                        edges = audit.checked_edges;
                        exact = audit.counterexample.is_none();
                        model.save_bundle_new(&models.join(format!("case-{case}")), &authority)?;
                        let loaded = FactorModel::load_bundle(
                            &models.join(format!("case-{case}/factors.axseed")),
                            &context,
                            &authority,
                        )?;
                        if loaded != model {
                            return Err("reload mismatch".into());
                        }
                        if exact {
                            factor_exact += 1;
                            (correct, plan_steps) =
                                goals(&loaded, &env, seed ^ (case as u64 + 19713))?;
                        }
                    }
                    Err(e) => failure = e.to_string(),
                }
                passed &= exact && correct == 6;
                goal_correct += correct;
                writer.write_record([
                    seed.to_string(),
                    case.to_string(),
                    label.into(),
                    "projected".into(),
                    ns.to_string(),
                    oracle.queries().to_string(),
                    oracle.steps().to_string(),
                    states.to_string(),
                    product.to_string(),
                    bytes.to_string(),
                    audit_states.to_string(),
                    edges.to_string(),
                    exact.to_string(),
                    "6".into(),
                    correct.to_string(),
                    plan_steps.to_string(),
                    failure,
                ])?;
            } else {
                let mut flattened = Flatten::new(&mut oracle);
                let start = Instant::now();
                let result = discover(env.actions(), &format!("{context}/flat"), &mut flattened);
                let ns = start.elapsed().as_nanos();
                let mut states = 0;
                let mut bytes = 0;
                let mut audit_states = 0;
                let mut edges = 0;
                let mut exact = false;
                let mut failure = String::new();
                match result {
                    Ok(model) => {
                        states = model.states();
                        bytes = model.encoded_bytes();
                        (exact, audit_states, edges) = flat_audit(&model, &env, &flattened)?;
                        model.save_new(models.join(format!("flat-{case}.axbehavior")))?;
                        if exact {
                            flat_exact += 1;
                        } else {
                            wrong_flat += 1;
                            passed = false;
                        }
                    }
                    Err(e) => {
                        flat_refused += 1;
                        failure = e.to_string();
                    }
                }
                writer.write_record([
                    seed.to_string(),
                    case.to_string(),
                    label.into(),
                    "monolithic".into(),
                    ns.to_string(),
                    oracle.queries().to_string(),
                    oracle.steps().to_string(),
                    states.to_string(),
                    "0".into(),
                    bytes.to_string(),
                    audit_states.to_string(),
                    edges.to_string(),
                    exact.to_string(),
                    "0".into(),
                    "0".into(),
                    "0".into(),
                    failure,
                ])?;
            }
        }
    }
    writer.flush()?;
    write_new(path, &writer.into_inner()?)?;
    println!(
        "factor_cases={cases}\nfactor_exact={factor_exact}\nmonolithic_exact={flat_exact}\nmonolithic_refused={flat_refused}\nmonolithic_wrong={wrong_flat}\nfactor_goals={goal_correct}/{}",
        6 * cases
    );
    passed &= selection(seed ^ 0xC1317, cases, &selection_path, &models)?;
    if !passed {
        return Err("some audits/parity/goals failed; raw data preserved".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
