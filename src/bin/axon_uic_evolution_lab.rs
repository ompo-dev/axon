use axon_uic::{
    behavior_lab::Rng,
    numeric::{evolution::*, *},
    telemetry::{LabTelemetry, Value as V},
};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    time::Instant,
};

fn rows(case: usize, seed: u64, count: usize, scale: f64) -> Vec<NumericExample> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|i| {
            let mut value = || (rng.pick(2001) as f64 - 1000.0) / 137.0 * scale;
            match case {
                0 => {
                    let x = if i % 101 == 0 && scale != 1.3 {
                        0.0
                    } else {
                        value()
                    };
                    NumericExample::new(vec![x], x * x)
                }
                1 => {
                    let (x, y) = match i % 101 {
                        0 => (0.0, value()),
                        1 => (value(), 0.0),
                        _ => (value(), value()),
                    };
                    NumericExample::new(vec![x, y], 4.0 * x * y)
                }
                2 => {
                    let (x, y) = (value(), value());
                    NumericExample::new(vec![x, y], x.hypot(y))
                }
                3..=5 => {
                    let (lo, hi, magnitude) = if i % 101 < 4 {
                        let (lo, hi) = [(-3.0, 0.0), (0.0, 3.0), (2.0, 2.0), (0.0, 0.0)][i % 101];
                        let exponent = if scale == 1.0 {
                            -2
                        } else if scale < 2.0 {
                            2
                        } else {
                            6
                        };
                        (lo, hi, 2.0_f64.powi(exponent + (value().abs() as i32) % 4))
                    } else {
                        let lo = value();
                        let hi = lo + value().abs() + scale;
                        let a = 0.2 + value().abs() / scale;
                        (lo, hi, a)
                    };
                    let a = if i % 2 == 0 { magnitude } else { -magnitude };
                    let input = vec![a, -a * (lo + hi), a * lo * hi];
                    // Labels come from the generating roots, not the quadratic formula.
                    let output = match case {
                        3 => a.abs() * (hi - lo),
                        4 => {
                            if a > 0.0 {
                                hi
                            } else {
                                lo
                            }
                        }
                        _ => {
                            if a > 0.0 {
                                lo
                            } else {
                                hi
                            }
                        }
                    };
                    NumericExample::new(input, output)
                }
                6 => {
                    let (x, y, z) = (value(), value(), value());
                    NumericExample::new(vec![x, y, z], x.hypot(y).hypot(z))
                }
                _ => {
                    let (x, y) = (value(), value());
                    let (u, v) = if i % 101 == 0 {
                        (x, y)
                    } else {
                        (value(), value())
                    };
                    NumericExample::new(vec![x, y, u, v], (x - u).hypot(y - v))
                }
            }
        })
        .collect()
}
fn create(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(text.as_bytes())?;
    f.sync_all()?;
    Ok(())
}
fn data(rows: &[NumericExample]) -> String {
    let mut s = (0..rows[0].input.len())
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join("\t");
    s.push_str("\ty\n");
    for row in rows {
        for x in &row.input {
            s.push_str(&format!("{x:.17}\t"));
        }
        s.push_str(&format!("{:.17}\n", row.output));
    }
    s
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=5).contains(&args.len()) {
        return Err(
            "axon-uic-evolution-lab <new-directory> [seed] [heldout-per-task 100..100000] [hybrid|mutation-only|no-inheritance|balanced] [core|transfer]".into(),
        );
    }
    let root = Path::new(&args[0]);
    let seed = args
        .get(1)
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(82641973);
    let count = args
        .get(2)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(10000);
    if !(100..=100000).contains(&count) {
        return Err("heldout quota".into());
    }
    let mode = args.get(3).map(String::as_str).unwrap_or("hybrid");
    if !["hybrid", "mutation-only", "no-inheritance", "balanced"].contains(&mode) {
        return Err("evolution mode".into());
    }
    let suite = args.get(4).map(String::as_str).unwrap_or("core");
    if !["core", "transfer"].contains(&suite) {
        return Err("evolution suite".into());
    }
    let tasks = if suite == "transfer" { 8 } else { 7 };
    let mut telemetry = LabTelemetry::from_environment("evolution");
    telemetry.emit(
        "lab_start",
        &[
            ("seed", V::Count(seed)),
            ("strategy", V::Text(mode)),
            ("suite", V::Text(suite)),
            ("tasks", V::Count(tasks)),
            ("telemetry", V::Text("search-checkpoints")),
        ],
    )?;
    if let Some(parent) = root.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(root)?;
    let mut learned = Vec::<Option<EvolvedSkill>>::new();
    let timer = Instant::now();
    let mut report = String::from(
        "case,status,generation,evaluations,work,mutations,crossovers,best_loss,active_genes,instructions,dependencies,capsule_bytes,correct,heldout,training_ns,local_proposals,residual_errors,expression\n",
    );
    let mut acquisition = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(root.join("acquisition.csv"))?;
    writeln!(
        acquisition,
        "case,status,generation,evaluations,work,mutations,crossovers,best_loss,elapsed_ns"
    )?;
    let mut outcomes = Vec::new();
    for case in 0..tasks as usize {
        let context = format!("evolution/case-{case}");
        let training = rows(case, seed.wrapping_add(971 * case as u64), 16, 1.0);
        let admission: Vec<_> = rows(case, seed.wrapping_add(376183 + 543 * case as u64), 32, 1.3)
            .into_iter()
            .filter(|a| !training.iter().any(|t| t.input == a.input))
            .take(8)
            .collect();
        if admission.len() != 8 {
            return Err("not enough novel admission inputs".into());
        }
        create(
            &root.join(format!("case-{case}.training.tsv")),
            &data(&training),
        )?;
        create(
            &root.join(format!("case-{case}.admission.tsv")),
            &data(&admission),
        )?;
        let tools: Vec<_> = if mode == "no-inheritance" {
            Vec::new()
        } else {
            learned.iter().filter_map(Option::as_ref).collect()
        };
        telemetry.emit(
            "acquisition_start",
            &[
                ("case", V::Count(case as u64)),
                ("inherited", V::Count(tools.len() as u64)),
                ("training", V::Count(training.len() as u64)),
                ("admission", V::Count(admission.len() as u64)),
            ],
        )?;
        let mut telemetry_error = None;
        let run = EvolvedSkill::evolve_observed(
            "acquired",
            &context,
            training,
            admission,
            &tools,
            seed.wrapping_add(7013 * case as u64),
            EvolutionBudget {
                local_refinement: mode != "mutation-only",
                ..if mode == "balanced" {
                    EvolutionBudget::balanced()
                } else {
                    EvolutionBudget::default()
                }
            },
            NumericTolerance::default(),
            |p| {
                if telemetry.enabled() && telemetry_error.is_none() {
                    let phase = format!("{:?}", p.phase);
                    telemetry_error = telemetry
                        .emit(
                            "evolution_progress",
                            &[
                                ("case", V::Count(case as u64)),
                                ("phase", V::Text(&phase)),
                                ("generation", V::Count(p.statistics.generation as u64)),
                                ("evaluations", V::Count(p.statistics.evaluations)),
                                ("work", V::Count(p.statistics.work)),
                                ("loss", V::Number(p.statistics.best_loss)),
                                ("local_work", V::Count(p.allocation.refinement_work)),
                                ("truncated", V::Count(p.allocation.truncated_refinements)),
                                ("correct", V::Count(p.correct as u64)),
                                ("active_genes", V::Count(p.active_genes as u64)),
                                ("diversity", V::Count(p.distinct_genomes as u64)),
                            ],
                        )
                        .err();
                }
            },
        )?;
        if let Some(error) = telemetry_error {
            return Err(error.into());
        }
        let status = format!("{:?}", run.stop);
        telemetry.emit(
            "acquisition_result",
            &[
                ("case", V::Count(case as u64)),
                ("status", V::Text(&status)),
                ("work", V::Count(run.statistics.work)),
                ("refinement_work", V::Count(run.allocation.refinement_work)),
                ("truncated", V::Count(run.allocation.truncated_refinements)),
            ],
        )?;
        let s = run.statistics;
        let mut trace = String::from(
            "generation,evaluations,work,loss,correct,active_genes,distinct_genomes,genome,parent_a,parent_b\n",
        );
        for g in run.generations {
            trace.push_str(&format!(
                "{},{},{},{:.17},{},{},{},{},{},{}\n",
                g.generation,
                g.evaluations,
                g.work,
                g.loss,
                g.correct,
                g.active_genes,
                g.distinct_genomes,
                g.genome,
                g.parents[0],
                g.parents[1]
            ));
        }
        create(&root.join(format!("case-{case}.generations.csv")), &trace)?;
        writeln!(
            acquisition,
            "{case},{:?},{},{},{},{},{},{:.17},{}",
            run.stop,
            s.generation,
            s.evaluations,
            s.work,
            s.mutations,
            s.crossovers,
            s.best_loss,
            s.elapsed_ns
        )?;
        acquisition.flush()?;
        if let Some(r) = run.refutation {
            create(
                &root.join(format!("case-{case}.refutation.txt")),
                &format!(
                    "input={:?}\nexpected={}\nactual={:?}\nexpression={}\n",
                    r.input, r.expected, r.actual, r.expression
                ),
            )?;
        }
        if let Some(skill) = run.skill {
            let path = root.join(format!("case-{case}.axevo"));
            skill.save_new(&path)?;
            let skill = EvolvedSkill::open(&path, &context)?;
            if telemetry.enabled() {
                let (arity, output, wiring) = skill.wiring();
                telemetry.emit(
                    "genome",
                    &[
                        ("case", V::Count(case as u64)),
                        ("arity", V::Count(arity as u64)),
                        ("output_signal", V::Count(output as u64)),
                        ("wiring", V::Counts(&wiring)),
                    ],
                )?;
            }
            telemetry.emit(
                "skill",
                &[
                    ("case", V::Count(case as u64)),
                    ("expression", V::Text(&skill.program().expression())),
                    (
                        "instructions",
                        V::Count(skill.program().instructions() as u64),
                    ),
                    ("dependencies", V::Count(skill.dependencies().len() as u64)),
                    ("capsule_bytes", V::Count(skill.to_bytes().len() as u64)),
                    (
                        "identity",
                        V::Text(&format!("{:016x}", skill.fingerprint())),
                    ),
                    ("active_genes", V::Count(skill.active_genes() as u64)),
                ],
            )?;
            println!(
                "case={case} status={:?} generation={} expression={}",
                run.stop,
                s.generation,
                skill.program().expression()
            );
            learned.push(Some(skill));
        } else {
            println!(
                "case={case} status={:?} generation={} loss={} work={}",
                run.stop, s.generation, s.best_loss, s.work
            );
            learned.push(None);
        }
        outcomes.push((run.stop, s));
    }
    acquisition.sync_all()?;
    telemetry.emit("curriculum_frozen", &[("tasks", V::Count(tasks))])?;
    // Freeze the entire curriculum before examining the final reserve.
    let mut passed = 0;
    let mut correct_total = 0;
    let mut refuted = 0;
    for (case, skill) in learned.iter().enumerate() {
        let (stop, s) = outcomes[case];
        if let Some(skill) = skill {
            let heldout = rows(
                case,
                seed.wrapping_add(8163947 + case as u64 * 7151),
                count,
                25.0,
            );
            let correct = heldout
                .iter()
                .filter(|e| {
                    skill
                        .predict(skill.context(), &e.input)
                        .is_ok_and(|v| NumericTolerance::default().agrees(v, e.output))
                })
                .count();
            let residual_errors = if (4..=5).contains(&case) {
                heldout
                    .iter()
                    .filter(|e| {
                        let Ok(x) = skill.predict(skill.context(), &e.input) else {
                            return true;
                        };
                        let ax = e.input[0] * x * x;
                        let bx = e.input[1] * x;
                        let c = e.input[2];
                        let residual = ax + bx + c;
                        !residual.is_finite()
                            || residual.abs() > 1e-9 * (1.0 + ax.abs() + bx.abs() + c.abs())
                    })
                    .count()
            } else {
                0
            };
            correct_total += correct;
            let complete = correct == count && residual_errors == 0;
            passed += usize::from(complete);
            refuted += usize::from(!complete);
            let status = if complete {
                format!("{stop:?}")
            } else {
                "HoldoutRefuted".into()
            };
            telemetry.emit(
                "verification",
                &[
                    ("case", V::Count(case as u64)),
                    ("correct", V::Count(correct as u64)),
                    ("heldout", V::Count(count as u64)),
                    ("residual_errors", V::Count(residual_errors as u64)),
                    ("status", V::Text(&status)),
                ],
            )?;
            report.push_str(&format!(
                "{case},{status},{},{},{},{},{},{:.17},{},{},{},{},{correct},{count},{},{},{residual_errors},\"{}\"\n",
                s.generation,
                s.evaluations,
                s.work,
                s.mutations,
                s.crossovers,
                s.best_loss,
                skill.active_genes(),
                skill.program().instructions(),
                skill.dependencies().len(),
                skill.to_bytes().len(),
                s.elapsed_ns,
                s.local_proposals,
                skill.program().expression()
            ));
        } else {
            telemetry.emit(
                "verification",
                &[
                    ("case", V::Count(case as u64)),
                    ("correct", V::Count(0)),
                    ("heldout", V::Count(count as u64)),
                    ("status", V::Text("Abstained")),
                ],
            )?;
            report.push_str(&format!(
                "{case},{stop:?},{},{},{},{},{},{:.17},,,,,0,{count},{},{},,\"\"\n",
                s.generation,
                s.evaluations,
                s.work,
                s.mutations,
                s.crossovers,
                s.best_loss,
                s.elapsed_ns,
                s.local_proposals
            ));
        }
    }
    create(&root.join("results.csv"), &report)?;
    let summary = format!(
        "AXON EVOLUTION LAB\nseed={seed}\nmode={mode}\nsuite={suite}\ntasks={tasks}\npassed={passed}\nholdout_refuted={refuted}\ncorrect={correct_total}\nheldout={}\nwhole_ns={}\nSupplied: arithmetic/root operators, graph grammar, readout/local search, selection/mutation/crossover, labeled curriculum and budgets. Learned: wiring, active program, inherited calls. Curriculum frozen before heldout; zero/repeated roots and both signs of a included. Not general seed evolution or open intelligence.\n",
        tasks as usize * count,
        timer.elapsed().as_nanos()
    );
    create(&root.join("summary.txt"), &summary)?;
    print!("{summary}");
    telemetry.emit(
        "lab_result",
        &[
            ("passed", V::Count(passed as u64)),
            ("tasks", V::Count(tasks)),
            ("correct", V::Count(correct_total as u64)),
            ("heldout", V::Count(tasks * count as u64)),
            ("refuted", V::Count(refuted as u64)),
            (
                "elapsedNs",
                V::Count(timer.elapsed().as_nanos().min(u64::MAX as u128) as u64),
            ),
        ],
    )?;
    if refuted > 0 {
        return Err("heldout refuted admitted programs; evidence preserved".into());
    }
    Ok(())
}
