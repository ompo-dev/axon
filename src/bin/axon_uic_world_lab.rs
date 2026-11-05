mod world_support;
use axon_uic::world::*;
use std::{error::Error, fmt::Write, fs::OpenOptions, io::Write as IoWrite, time::Instant};
use world_support::{HiddenWorld, Rng};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let audit_goals = args.len() == 5 && args[4] == "--audit-goals";
    if args.len() != 4 && !audit_goals {
        return Err(
            "usage: axon-uic-world-lab <seed> <worlds:1..256> <new.csv> [--audit-goals]".into(),
        );
    }
    let seed: u64 = args[1].parse()?;
    let worlds: usize = args[2].parse()?;
    if !(1..=256).contains(&worlds) {
        return Err("world count out of range".into());
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    let mut rng = Rng::new(seed);
    let mut csv = String::from(
        "world,bits,actions,policy,identified,interventions,learning_us,selection_us,transition_checks,transition_correct,goal_checks,goal_correct,reachable_goals,model_bytes\n",
    );
    let mut active_trials = 0;
    let mut random_trials = 0;
    let mut scan_trials = 0;
    let mut total_transitions = 0;
    let mut total_goals = 0;
    let mut random_failures = 0;
    if audit_goals {
        csv = "world,start,target,shortest_steps\n".into();
    }
    let (mut nontrivial, mut multistep, mut unreachable, mut longest) = (0, 0, 0, 0);
    for index in 0..worlds {
        let world = HiddenWorld::generate(&mut rng);
        let mut goals = Vec::new();
        for n in 0..40 {
            let start = rng.pick(world.shape.states());
            let target = if n < 32 {
                let mut s = start;
                for _ in 0..(1 + rng.pick(10)) {
                    s = world.step(s, rng.pick(world.shape.actions()));
                }
                s
            } else {
                rng.pick(world.shape.states())
            };
            goals.push((start, target));
        }
        if audit_goals {
            // Inspect the frozen goals with the oracle only; no fitting or model updates.
            for &(start, target) in &goals {
                let depth = world.shortest(start, target);
                match depth {
                    Some(n) => {
                        nontrivial += usize::from(n > 0);
                        multistep += usize::from(n > 1);
                        longest = longest.max(n);
                    }
                    None => unreachable += 1,
                }
                writeln!(
                    csv,
                    "{index},{start},{target},{}",
                    depth.map_or(-1, |n| n as i32)
                )?;
            }
            continue;
        }
        for policy in ["active", "random", "scan"] {
            let started = Instant::now();
            let mut learner = WorldLearner::new(world.shape, "lab/hidden-v1")?;
            let mut random = Rng::new(seed.wrapping_add(index as u64).wrapping_add(0x8f719abc));
            let scope = ProbeScope::unrestricted(world.shape, 512);
            let mut selection_us = 0;
            for trial in 0..512 {
                if learner.identified() {
                    break;
                }
                let selecting = Instant::now();
                let (state, action) = match policy {
                    "active" => match learner.choose_probe(&scope)? {
                        ProbeChoice::Experiment(p) => (p.state, p.action),
                        _ => return Err("active policy unexpectedly stalled".into()),
                    },
                    "random" => (
                        random.pick(world.shape.states()),
                        random.pick(world.shape.actions()),
                    ),
                    _ => (
                        (trial / usize::from(world.shape.actions())) as u8,
                        (trial % usize::from(world.shape.actions())) as u8,
                    ),
                };
                selection_us += selecting.elapsed().as_micros();
                learner.observe(Transition {
                    before: state,
                    action,
                    after: world.step(state, action),
                    kind: ObservationKind::Intervention,
                })?;
            }
            let learning_us = started.elapsed().as_micros();
            let trials = usize::from(learner.interventions());
            match policy {
                "active" => active_trials += trials,
                "random" => random_trials += trials,
                _ => scan_trials += trials,
            }
            let (mut checks, mut correct, mut goal_correct, mut reachable, mut model_bytes) =
                (0, 0, 0, 0, 0);
            if learner.identified() {
                let model = learner.compile()?;
                model_bytes = model.encoded_bytes();
                for state in 0..world.shape.states() {
                    for action in 0..world.shape.actions() {
                        checks += 1;
                        if model.predict(state, action)? == world.step(state, action) {
                            correct += 1;
                        }
                    }
                }
                for &(start, target) in &goals {
                    let expected = world.shortest(start, target);
                    if expected.is_some() {
                        reachable += 1;
                    }
                    let result = model.plan(
                        start,
                        Goal {
                            on: target,
                            off: world.shape.mask() ^ target,
                        },
                        world.shape.all_states(),
                    )?;
                    match result {
                        PlanResult::Found(plan) => {
                            let mut actual = start;
                            let valid = plan.steps.iter().all(|step| {
                                if actual != step.before {
                                    return false;
                                }
                                actual = world.step(actual, step.action);
                                actual == step.after
                            });
                            if valid && actual == target && expected == Some(plan.steps.len()) {
                                goal_correct += 1;
                            }
                        }
                        PlanResult::Unreachable if expected.is_none() => goal_correct += 1,
                        _ => (),
                    }
                }
                if correct != checks || goal_correct != goals.len() {
                    return Err("independent oracle mismatch".into());
                }
                if policy == "active" {
                    total_transitions += checks;
                    total_goals += goals.len();
                }
            } else if policy == "random" {
                random_failures += 1;
            } else {
                return Err("policy exhausted before identification".into());
            }
            let goal_checks = if learner.identified() { goals.len() } else { 0 };
            writeln!(
                csv,
                "{index},{},{},{policy},{},{trials},{learning_us},{selection_us},{checks},{correct},{goal_checks},{goal_correct},{reachable},{model_bytes}",
                world.shape.bits(),
                world.shape.actions(),
                learner.identified()
            )?;
        }
    }
    output.write_all(csv.as_bytes())?;
    output.sync_all()?;
    if audit_goals {
        println!(
            "oracle_only=true seed={seed} worlds={worlds} nontrivial={nontrivial} multistep={multistep} unreachable={unreachable} longest={longest}"
        );
        return Ok(());
    }
    println!(
        "seed={seed} worlds={worlds} active_trials={active_trials} random_trials={random_trials} scan_trials={scan_trials} random_unidentified={random_failures} active_exact_transitions={total_transitions} active_exact_goals={total_goals} report={}",
        args[3]
    );
    Ok(())
}
