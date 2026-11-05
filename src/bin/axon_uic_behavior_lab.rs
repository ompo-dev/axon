use axon_uic::{behavior::*, behavior_lab::*};
use std::{
    collections::BTreeSet,
    error::Error,
    fmt::Write,
    fs::{self, OpenOptions},
    io::Write as IoWrite,
    path::PathBuf,
    time::Instant,
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: axon-uic-behavior-lab <seed> <worlds:1..128> <new.csv>".into());
    }
    let seed: u64 = args[1].parse()?;
    let count: usize = args[2].parse()?;
    if !(1..=128).contains(&count) {
        return Err("world count".into());
    }
    let models = PathBuf::from(format!("{}.models", args[3]));
    fs::create_dir(&models)?;
    let mut output = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args[3])?;
    let mut csv = String::from(
        "world,actions,hidden_base_states,minimal_truth_states,learned_states,rounds,suffixes,queries,steps,trie_nodes,fit_us,load_us,model_bytes,audit_edges,exact,goals,goal_correct,multistep_goals,max_goal_steps,failure\n",
    );
    let mut rng = Rng::new(seed);
    let mut exact = 0;
    let mut goals_exact = 0;
    let mut edges_total = 0;
    let mut failed = 0;
    for index in 0..count {
        let hidden = HiddenMachine::generate(&mut rng);
        let actions = hidden.actions;
        let minimal = hidden.minimal_states();
        let mut goal_rng = Rng::new(seed.wrapping_add(0x91732).wrapping_add(index as u64));
        let goals: Vec<(Word, u16)> = (0..24)
            .map(|i| {
                let n = goal_rng.pick(7);
                let history = (0..n)
                    .map(|_| goal_rng.pick(actions as usize) as u8)
                    .collect();
                let target = if i == 23 {
                    u16::MAX
                } else {
                    hidden
                        .transition(
                            goal_rng.pick(hidden.states()),
                            goal_rng.pick(actions as usize) as u8,
                        )
                        .1
                };
                (history, target)
            })
            .collect();
        let mut oracle = CachedOracle::new(hidden.clone(), BehaviorBudget::default());
        let clock = Instant::now();
        let discovered = discover(
            actions,
            "reserved/behavior-v1",
            BehaviorBudget::default(),
            &mut oracle,
        );
        let fit = clock.elapsed().as_micros();
        match discovered {
            Ok((model, revisions, suffixes)) => {
                let path = models.join(format!("{index}.axbehavior"));
                model.save_new(&path)?;
                let loading = Instant::now();
                let reloaded = BehaviorModel::load(path)?;
                let load = loading.elapsed().as_micros();
                if reloaded != model {
                    return Err("model reload mismatch".into());
                }
                let audit = audit_machine(&model, actions, 0, |s, a| Ok(hidden.transition(s, a)))?;
                let valid = audit.counterexample.is_none() && model.states() == minimal;
                let (mut goal_correct, mut multistep, mut max_steps) = (0, 0, 0);
                if valid {
                    exact += 1;
                    edges_total += audit.checked_edges;
                    for (history, goal) in &goals {
                        let plan = model.plan_output(history, *goal, &BTreeSet::new())?;
                        if *goal == u16::MAX {
                            if plan.is_none() {
                                goal_correct += 1;
                            }
                            continue;
                        }
                        if let Some(plan) = plan {
                            multistep += usize::from(plan.len() > 1);
                            max_steps = max_steps.max(plan.len());
                            let mut word = history.clone();
                            word.extend(plan);
                            let actual = RawOracle::new(hidden.clone()).query(&word)?;
                            if actual == model.predict(&word)? && actual.last() == Some(goal) {
                                goal_correct += 1;
                            }
                        }
                    }
                    goals_exact += goal_correct;
                } else {
                    failed += 1;
                }
                writeln!(
                    csv,
                    "{index},{actions},{},{minimal},{},{},{suffixes},{},{},{},{fit},{load},{},{},{valid},24,{goal_correct},{multistep},{max_steps},{}",
                    hidden.states(),
                    model.states(),
                    revisions.len(),
                    oracle.raw.queries,
                    oracle.raw.steps,
                    oracle.trie_nodes(),
                    model.encoded_bytes(),
                    audit.checked_edges,
                    usize::from(!valid)
                )?;
            }
            Err(error) => {
                failed += 1;
                writeln!(
                    csv,
                    "{index},{actions},{},{minimal},0,0,0,{},{},{},{fit},0,0,0,false,24,0,0,0,1",
                    hidden.states(),
                    oracle.raw.queries,
                    oracle.raw.steps,
                    oracle.trie_nodes()
                )?;
                eprintln!("world={index} failure={error}");
            }
        }
    }
    output.write_all(csv.as_bytes())?;
    output.sync_all()?;
    println!(
        "seed={seed} worlds={count} exact_minimal_models={exact} failed={failed} audited_edges={edges_total} goals_correct={goals_exact}/{} report={}",
        count * 24,
        args[3]
    );
    Ok(())
}
