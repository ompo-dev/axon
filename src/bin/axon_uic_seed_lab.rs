use axon_uic::{
    behavior_lab::Rng,
    numeric::{evolution::*, seed::*, *},
    telemetry::{LabTelemetry, Value as V},
};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

const SCOPE: &str = "seed-lab/development-v1";
const POPULATION: usize = 8;
const ROUNDS: usize = 3;
const TASKS: usize = 6;
const WORK: u64 = 8_000_000;

fn create(path: &Path, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(())
}
fn archive(root: &Path, seed: &DevelopmentSeed) -> Result<(), Box<dyn std::error::Error>> {
    let path = root.join(format!("revision-{:016x}.axdna", seed.revision_id()));
    if path.exists() {
        let old = DevelopmentSeed::open(&path, SCOPE)?;
        if old.fingerprint() != seed.fingerprint()
            || old.parents() != seed.parents()
            || old.generation() != seed.generation()
        {
            return Err("revision identity conflict".into());
        }
        return Ok(());
    }
    seed.save_new(&path)?;
    Ok(())
}
// The host alone knows these oracles. The learner receives independent numerical examples.
fn truth(case: usize, x: &[f64]) -> f64 {
    match case {
        0 => (x[0] - x[2]).hypot(x[1] - x[3]),
        1 => (x[0] + x[2]).hypot(x[1] + x[3]),
        2 => (x[0] * x[2]).hypot(x[1] * x[3]),
        3 => (x[0] + x[1]) * (x[2] - x[3]),
        4 => (x[0] - x[2]) * (x[1] - x[3]),
        _ => x[0].sin(),
    }
}
fn anchor(case: usize, x: &[f64]) -> f64 {
    match case {
        0 => x[0] * x[0],
        1 => x[0] * x[1],
        _ => x[0].hypot(x[1]),
    }
}
fn rows(case: usize, random: u64, count: usize, scale: f64, anchors: bool) -> Vec<NumericExample> {
    let arity = if anchors {
        if case == 0 { 1 } else { 2 }
    } else if case == 5 {
        1
    } else {
        4
    };
    let mut rng = Rng::new(random);
    (0..count)
        .map(|i| {
            let mut input = (0..arity)
                .map(|_| (rng.pick(4001) as f64 - 2000.0) * scale / 251.0)
                .collect::<Vec<_>>();
            if scale == 1.0 && i <= arity {
                input.fill(0.0);
                if i > 0 {
                    input[i - 1] = 1.0;
                }
            } else if i.is_multiple_of(17) {
                input[i % arity] = 0.0;
            }
            if arity == 4 && i > arity && i.is_multiple_of(31) {
                input[2] = input[0];
                input[3] = input[1];
            }
            let output = if anchors {
                anchor(case, &input)
            } else {
                truth(case, &input)
            };
            NumericExample::new(input, output)
        })
        .collect()
}
fn agrees(skill: &EvolvedSkill, examples: &[NumericExample]) -> bool {
    examples.iter().all(|e| {
        skill
            .predict(skill.context(), &e.input)
            .is_ok_and(|v| NumericTolerance::default().agrees(v, e.output))
    })
}
fn replay(
    seed: &DevelopmentSeed,
    random: u64,
    count: usize,
) -> Result<(bool, u64, u64, String), Box<dyn std::error::Error>> {
    let mut good = true;
    let mut correct = 0;
    let mut work = 0;
    let mut witnesses = String::new();
    for skill in seed.skills() {
        let (anchors, case) = if let Some(id) = skill.context().strip_prefix("seed-lab/anchor-") {
            (true, id.parse::<usize>()?)
        } else {
            (
                false,
                skill
                    .context()
                    .strip_prefix("seed-lab/task-")
                    .ok_or("unknown benchmark skill scope")?
                    .parse::<usize>()?,
            )
        };
        let examples = rows(
            case,
            random.wrapping_add(case as u64 * 193 + u64::from(anchors) * 8731),
            count,
            41.0,
            anchors,
        );
        let matched = examples
            .iter()
            .filter(|e| {
                skill
                    .predict(skill.context(), &e.input)
                    .is_ok_and(|v| NumericTolerance::default().agrees(v, e.output))
            })
            .count();
        correct += matched as u64;
        work += (count * skill.program().instructions()) as u64;
        good &= matched == count;
        if matched != count
            && let Some(row) = examples.iter().find(|e| {
                !skill
                    .predict(skill.context(), &e.input)
                    .is_ok_and(|v| NumericTolerance::default().agrees(v, e.output))
            })
        {
            witnesses.push_str(&format!(
                "module={} input={:?} expected={} actual={:?} expression={}\n",
                skill.context(),
                row.input,
                row.output,
                skill.predict(skill.context(), &row.input).ok(),
                skill.program().expression()
            ));
        }
    }
    Ok((good, correct, work, witnesses))
}
#[derive(Clone)]
struct Candidate {
    seed: DevelopmentSeed,
    score: SeedScore,
    id: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 4 {
        return Err(
            "usage: axon-uic-seed-lab <new-dir> [seed] [heldout-per-module 100..10000] [development|reserved]".into(),
        );
    }
    let root = Path::new(&args[0]);
    let random = args
        .get(1)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(70143863);
    let count = args
        .get(2)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1000);
    if !(100..=10000).contains(&count) {
        return Err("verification quota".into());
    }
    let validation = args.get(3).map(String::as_str).unwrap_or("development");
    if !["development", "reserved"].contains(&validation) {
        return Err("unknown seed validation protocol".into());
    }
    if let Some(parent) = root.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(root)?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment("seed");
    telemetry.emit(
        "lab_start",
        &[
            ("seed", V::Count(random)),
            ("population", V::Count(POPULATION as u64)),
            ("rounds", V::Count(ROUNDS as u64)),
            ("tasks", V::Count(TASKS as u64)),
            ("validation", V::Text(validation)),
            (
                "telemetry",
                V::Text("seed-selection-and-search-checkpoints"),
            ),
        ],
    )?;
    let empty = DevelopmentSeed::new(SCOPE, EvolutionBudget::balanced(), 65536)?;
    archive(root, &empty)?;
    let mut inherited = empty.clone();
    let mut bootstrap_work = 0;
    for case in 0..3 {
        let training = rows(case, random.wrapping_add(case as u64 * 91), 16, 1.0, true);
        let admission = rows(
            case,
            random.wrapping_add(381973 + case as u64 * 971),
            64,
            1.7,
            true,
        )
        .into_iter()
        .filter(|a| !training.iter().any(|t| t.input == a.input))
        .take(8)
        .collect::<Vec<_>>();
        let tools = inherited.skills().iter().collect::<Vec<_>>();
        let outcome = EvolvedSkill::evolve(
            "anchor",
            &format!("seed-lab/anchor-{case}"),
            training,
            admission,
            &tools,
            random.wrapping_add(case as u64 * 119),
            EvolutionBudget::balanced(),
            NumericTolerance::default(),
        )?;
        bootstrap_work += outcome.statistics.work;
        let skill = outcome
            .skill
            .ok_or("bootstrap did not acquire all three anchors; outputs preserved")?;
        inherited = inherited.extend(SCOPE, skill)?;
        archive(root, &inherited)?;
        telemetry.emit(
            "seed_bootstrap",
            &[
                ("task", V::Count(case as u64)),
                ("work", V::Count(outcome.statistics.work)),
                ("modules", V::Count(inherited.skills().len() as u64)),
            ],
        )?;
    }
    inherited.save_new(&root.join("founder.axdna"))?;
    empty.save_new(&root.join("amnesia-control.axdna"))?;
    let mut genotypes = vec![inherited.clone(), empty.clone()];
    let mut seen = genotypes
        .iter()
        .map(DevelopmentSeed::fingerprint)
        .collect::<std::collections::BTreeSet<_>>();
    for attempt in 0..256 {
        if genotypes.len() == POPULATION {
            break;
        }
        let child = inherited.mutate(SCOPE, random.wrapping_add(attempt * 479 + 7))?;
        if seen.insert(child.fingerprint()) {
            genotypes.push(child);
        }
    }
    if genotypes.len() != POPULATION {
        return Err("initial diversity quota".into());
    }
    let mut report = String::from(
        "candidate,round,genotype,revision,parent_a,parent_b,capabilities,retention_ok,violations,work,bytes,instructions,modules,frontier\n",
    );
    let mut development = String::from(
        "candidate,round,task,outcome,reused,search_work,development_correct,development_count,expression\n",
    );
    let mut final_front = Vec::new();
    let mut next_id = 0;
    let mut total_search = 0;
    for round in 0..ROUNDS {
        let mut candidates = Vec::new();
        for mut seed in genotypes {
            archive(root, &seed)?;
            let id = next_id;
            next_id += 1;
            let origin = seed.revision_id();
            let parents = seed.parents();
            let p = seed.policy();
            telemetry.emit(
                "seed_candidate",
                &[
                    ("candidate", V::Count(id as u64)),
                    ("generation", V::Count(round as u64)),
                    ("genotype", V::Text(&format!("{:016x}", seed.fingerprint()))),
                    ("revision", V::Text(&format!("{origin:016x}"))),
                    ("origin_revision", V::Text(&format!("{origin:016x}"))),
                    ("parent_a", V::Text(&format!("{:016x}", parents[0]))),
                    ("parent_b", V::Text(&format!("{:016x}", parents[1]))),
                    ("modules", V::Count(seed.skills().len() as u64)),
                    ("population", V::Count(p.population as u64)),
                    ("genes", V::Count(p.genes as u64)),
                    ("mutations", V::Count(p.mutations as u64)),
                    ("refinement", V::Flag(p.local_refinement)),
                    ("bindings", V::Flag(p.refine_inherited_calls)),
                ],
            )?;
            let mut mask = 0;
            let mut work = 0;
            let mut violations = 0;
            for task in 0..TASKS {
                let context = format!("seed-lab/task-{task}");
                let training = rows(
                    task,
                    random.wrapping_add(task as u64 * 913 + 701),
                    16,
                    1.0,
                    false,
                );
                let admission = rows(
                    task,
                    random.wrapping_add(task as u64 * 781 + 187931),
                    64,
                    1.9,
                    false,
                )
                .into_iter()
                .filter(|a| !training.iter().any(|t| t.input == a.input))
                .take(8)
                .collect::<Vec<_>>();
                let cached = seed
                    .skills()
                    .iter()
                    .find(|s| s.context() == context)
                    .filter(|s| agrees(s, &training))
                    .cloned();
                let reused = cached.is_some();
                let (skill, stop, search_work) = if let Some(skill) = cached {
                    let charge = (24 * skill.program().instructions()) as u64;
                    if agrees(&skill, &admission) {
                        (Some(skill), EvolutionStop::Admitted, charge)
                    } else {
                        (None, EvolutionStop::Refuted, charge)
                    }
                } else {
                    let tools = seed.skills().iter().collect::<Vec<_>>();
                    let mut io_error = None;
                    let outcome = EvolvedSkill::evolve_observed(
                        "opaque",
                        &context,
                        training,
                        admission,
                        &tools,
                        random.wrapping_add(id as u64 * 1709 + task as u64 * 179),
                        seed.budget(WORK)?,
                        NumericTolerance::default(),
                        |progress| {
                            if telemetry.enabled() && io_error.is_none() {
                                io_error = telemetry
                                    .emit(
                                        "seed_progress",
                                        &[
                                            ("candidate", V::Count(id as u64)),
                                            ("task", V::Count(task as u64)),
                                            ("generation", V::Count(round as u64)),
                                            ("work", V::Count(progress.statistics.work)),
                                            (
                                                "evaluations",
                                                V::Count(progress.statistics.evaluations),
                                            ),
                                            ("loss", V::Number(progress.statistics.best_loss)),
                                        ],
                                    )
                                    .err();
                            }
                        },
                    )?;
                    if let Some(error) = io_error {
                        return Err(error.into());
                    }
                    (outcome.skill, outcome.stop, outcome.statistics.work)
                };
                total_search += search_work;
                work += search_work;
                let selection = rows(
                    task,
                    random.wrapping_add(task as u64 * 137 + 923731),
                    128,
                    11.0,
                    false,
                );
                let correct = skill.as_ref().map_or(0, |s| {
                    selection
                        .iter()
                        .filter(|e| {
                            s.predict(s.context(), &e.input)
                                .is_ok_and(|v| NumericTolerance::default().agrees(v, e.output))
                        })
                        .count()
                });
                if let Some(ref skill) = skill {
                    work += (128 * skill.program().instructions()) as u64;
                }
                let accepted = correct == 128;
                if accepted {
                    mask |= 1 << task;
                    if !reused {
                        match seed.extend(SCOPE, skill.as_ref().expect("scored model").clone()) {
                            Ok(child) => {
                                archive(root, &child)?;
                                seed = child;
                            }
                            Err(_) => violations += 1,
                        }
                    }
                }
                let expression = skill
                    .as_ref()
                    .map_or(String::new(), |s| s.program().expression());
                let outcome = if accepted {
                    "DevelopmentPassed".into()
                } else {
                    format!("{stop:?}")
                };
                development.push_str(&format!("{id},{round},{task},{outcome},{reused},{search_work},{correct},128,\"{expression}\"\n"));
                telemetry.emit(
                    "seed_task",
                    &[
                        ("candidate", V::Count(id as u64)),
                        ("task", V::Count(task as u64)),
                        ("generation", V::Count(round as u64)),
                        ("outcome", V::Text(&outcome)),
                        ("reused", V::Flag(reused)),
                        ("work", V::Count(search_work)),
                        ("correct", V::Count(correct as u64)),
                        ("scored", V::Count(128)),
                    ],
                )?;
            }
            let (retention_ok, _, retention_work, witnesses) =
                replay(&seed, random.wrapping_add(638591 + round as u64 * 831), 64)?;
            if !witnesses.is_empty() {
                create(
                    &root.join(format!("candidate-{id}.retention-failures.txt")),
                    &witnesses,
                )?;
            }
            work += retention_work;
            let score = SeedScore {
                capabilities: mask,
                tasks: TASKS as u8,
                retention_ok,
                violations,
                work,
                bytes: seed.to_bytes().len() as u64,
                instructions: seed
                    .skills()
                    .iter()
                    .map(|s| s.program().instructions() as u64)
                    .sum(),
            };
            seed.save_new(&root.join(format!("candidate-{id}.axdna")))?;
            telemetry.emit(
                "seed_evaluation",
                &[
                    ("candidate", V::Count(id as u64)),
                    ("generation", V::Count(round as u64)),
                    ("genotype", V::Text(&format!("{:016x}", seed.fingerprint()))),
                    ("revision", V::Text(&format!("{:016x}", seed.revision_id()))),
                    ("parent_a", V::Text(&format!("{:016x}", seed.parents()[0]))),
                    ("parent_b", V::Text(&format!("{:016x}", seed.parents()[1]))),
                    ("capabilities", V::Count(mask)),
                    ("solved", V::Count(mask.count_ones() as u64)),
                    ("tasks", V::Count(TASKS as u64)),
                    ("retention_ok", V::Flag(retention_ok)),
                    ("violations", V::Count(violations as u64)),
                    ("work", V::Count(work)),
                    ("capsule_bytes", V::Count(score.bytes)),
                    ("instructions", V::Count(score.instructions)),
                    ("modules", V::Count(seed.skills().len() as u64)),
                ],
            )?;
            candidates.push(Candidate { seed, score, id });
        }
        let front = pareto_front(&candidates.iter().map(|c| c.score).collect::<Vec<_>>())?;
        if front.is_empty() {
            return Err("no feasible developmental survivor; inspect artifacts".into());
        }
        for (index, c) in candidates.iter().enumerate() {
            let parents = c.seed.parents();
            report.push_str(&format!(
                "{},{round},{:016x},{:016x},{:016x},{:016x},{},{},{},{},{},{},{},{}\n",
                c.id,
                c.seed.fingerprint(),
                c.seed.revision_id(),
                parents[0],
                parents[1],
                c.score.capabilities,
                c.score.retention_ok,
                c.score.violations,
                c.score.work,
                c.score.bytes,
                c.score.instructions,
                c.seed.skills().len(),
                front.contains(&index)
            ));
        }
        let selected = front
            .iter()
            .map(|i| candidates[*i].id as u64)
            .collect::<Vec<_>>();
        telemetry.emit(
            "seed_selection",
            &[
                ("generation", V::Count(round as u64)),
                ("selected", V::Counts(&selected)),
            ],
        )?;
        create(&root.join(format!("round-{round}.population.csv")), &report)?;
        create(
            &root.join(format!("round-{round}.development.csv")),
            &development,
        )?;
        final_front = front.into_iter().map(|i| candidates[i].clone()).collect();
        if round + 1 == ROUNDS {
            break;
        }
        genotypes = vec![empty.clone(), final_front[0].seed.clone()];
        let mut seen = genotypes
            .iter()
            .map(DevelopmentSeed::fingerprint)
            .collect::<std::collections::BTreeSet<_>>();
        for attempt in 0..512 {
            if genotypes.len() == POPULATION {
                break;
            }
            let a = &final_front[attempt as usize % final_front.len()].seed;
            let b = &final_front[(attempt as usize + 1) % final_front.len()].seed;
            let random = random.wrapping_add(round as u64 * 931 + attempt * 791 + 1);
            let crossed = if attempt % 2 == 0 {
                a.crossover(SCOPE, b, random).unwrap_or_else(|_| a.clone())
            } else {
                a.clone()
            };
            archive(root, &crossed)?;
            let child = crossed.mutate(SCOPE, random)?;
            if seen.insert(child.fingerprint()) {
                genotypes.push(child);
            }
        }
        if genotypes.len() != POPULATION {
            return Err("offspring diversity quota".into());
        }
    }
    // No further search or offspring selection after this durable boundary.
    let frozen = final_front
        .iter()
        .map(|c| {
            format!(
                "{},{:016x},{:016x}\n",
                c.id,
                c.seed.fingerprint(),
                c.seed.revision_id()
            )
        })
        .collect::<String>();
    create(
        &root.join("frozen.csv"),
        &format!("candidate,genotype,revision\n{frozen}"),
    )?;
    telemetry.emit(
        "curriculum_frozen",
        &[("survivors", V::Count(final_front.len() as u64))],
    )?;
    let mut verification = String::from("candidate,reserve_seed,correct,heldout,retention_ok\n");
    let mut total_correct = 0;
    let mut total_heldout = 0;
    let mut failed = 0;
    let mut index = 0;
    for candidate in &final_front {
        let file = root.join(format!("candidate-{}.axdna", candidate.id));
        let reopened = DevelopmentSeed::open(&file, SCOPE)?;
        let reserves = if validation == "reserved" {
            [90271317u64, 45108629, 13794283]
        } else {
            [7071379u64, 19478319, 81397183]
        };
        for reserve in reserves {
            let (okay, correct, _, witnesses) = replay(&reopened, reserve, count)?;
            if !witnesses.is_empty() {
                create(
                    &root.join(format!(
                        "candidate-{}.verification-{reserve}.failures.txt",
                        candidate.id
                    )),
                    &witnesses,
                )?;
            }
            let heldout = (count * reopened.skills().len()) as u64;
            total_correct += correct;
            total_heldout += heldout;
            failed += usize::from(!okay);
            verification.push_str(&format!(
                "{},{reserve},{correct},{heldout},{okay}\n",
                candidate.id
            ));
            telemetry.emit(
                "seed_reserved",
                &[
                    ("candidate", V::Count(candidate.id as u64)),
                    ("reserve_seed", V::Count(reserve)),
                    ("correct", V::Count(correct)),
                    ("heldout", V::Count(heldout)),
                    ("retention_ok", V::Flag(okay)),
                ],
            )?;
        }
        for skill in reopened.skills() {
            let (arity, output, wiring) = skill.wiring();
            telemetry.emit(
                "genome",
                &[
                    ("case", V::Count(index)),
                    ("candidate", V::Count(candidate.id as u64)),
                    ("name", V::Text(skill.context())),
                    ("arity", V::Count(arity as u64)),
                    ("output_signal", V::Count(output as u64)),
                    ("wiring", V::Counts(&wiring)),
                    ("expression", V::Text(&skill.program().expression())),
                    (
                        "instructions",
                        V::Count(skill.program().instructions() as u64),
                    ),
                    ("capsule_bytes", V::Count(skill.to_bytes().len() as u64)),
                    ("active_genes", V::Count(skill.active_genes() as u64)),
                ],
            )?;
            index += 1;
        }
    }
    create(&root.join("results.csv"), &verification)?;
    create(&root.join("population.csv"), &report)?;
    create(&root.join("development.csv"), &development)?;
    let best = final_front
        .iter()
        .map(|c| c.score.capabilities.count_ones())
        .max()
        .unwrap_or(0);
    let summary = format!(
        "AXON DEVELOPMENT SEED LAB\nseed={random}\nrounds={ROUNDS}\ncandidates={next_id}\nsurvivors={}\nbest_development_capabilities={best}/{TASKS}\nbootstrap_work={bootstrap_work}\ntotal_search_work={total_search}\nreserved_correct={total_correct}\nreserved_evaluated={total_heldout}\nreserve_failures={failed}\nwhole_ns={}\nPolicy and typed learned programs are inherited. Pareto scores use development only. Reuse in later rounds is not novel-task learning. Bootstrap cost is reported; cold control discards that acquired knowledge. Not a general brain, native self-modification, watts or AGI.\n",
        final_front.len(),
        started.elapsed().as_nanos()
    );
    create(&root.join("summary.txt"), &summary)?;
    print!("{summary}");
    telemetry.emit(
        "lab_result",
        &[
            ("passed", V::Flag(failed == 0)),
            ("candidates", V::Count(next_id as u64)),
            ("survivors", V::Count(final_front.len() as u64)),
            ("solved", V::Count(best as u64)),
            ("tasks", V::Count(TASKS as u64)),
            ("correct", V::Count(total_correct)),
            ("heldout", V::Count(total_heldout)),
            ("bootstrap_work", V::Count(bootstrap_work)),
            ("work", V::Count(total_search)),
        ],
    )?;
    if failed > 0 {
        return Err("reserved retention failed; descendants were not retuned".into());
    }
    Ok(())
}
