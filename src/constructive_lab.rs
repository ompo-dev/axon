use crate::{
    Authority, Effect,
    behavior_lab::Rng,
    demand::genome::bounded_path,
    demand::{
        Catalog, DemandError, DemandRuntime, Module, ModuleKind, ModuleSpec, Seed,
        controller::CompositionBudget,
        genome::{BirthLimits, Genome, MAX_BYTES},
    },
    live_store::checksum,
    numeric::{
        NumericExample, NumericTolerance,
        evolution::{EvolvedSkill, constructive::*},
    },
    telemetry::{LabTelemetry, Value as V},
    text_skill::publish_new,
};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

// Only the host judge sees these functions; search receives input/output demonstrations.
fn truth(case: usize, x: &[f64]) -> f64 {
    match case {
        0 => x[0] * x[0],
        1 => x[0].hypot(x[1]),
        2 => (x[0] - x[2]).hypot(x[1] - x[3]),
        3 => x[0].hypot(x[1]) / x[2].hypot(x[3]),
        4 => (x[0] + x[1]) * (x[2] - x[3]),
        6 => x[0] + x[1],
        7 => x[0] - x[1],
        8 => x[0] * x[1],
        9 => x[0] / x[1],
        _ => x[0].sin(),
    }
}
fn rows(case: usize, seed: u64, n: usize, scale: f64) -> Vec<NumericExample> {
    let arity = match case {
        0 | 5 => 1,
        1 | 6..=9 => 2,
        _ => 4,
    };
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let mut x: Vec<_> = (0..arity)
                .map(|_| (rng.pick(4001) as f64 - 2000.0) / 317.0 * scale)
                .collect();
            if scale == 1.0 && i < arity + 1 {
                x.fill(0.0);
                if i > 0 {
                    x[i - 1] = scale;
                }
            }
            if case == 3 && x[2] == 0.0 && x[3] == 0.0 {
                x[3] = scale;
            }
            if case == 9 && x[1] == 0.0 {
                x[1] = scale;
            }
            let output = truth(case, &x);
            NumericExample::new(x, output)
        })
        .collect()
}
fn table(path: &Path, data: &[NumericExample]) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(Vec::new());
    let arity = data[0].input.len();
    writer.write_record((0..arity).map(|i| format!("x{i}")).chain(["output".into()]))?;
    for row in data {
        writer.write_record(
            row.input
                .iter()
                .chain([&row.output])
                .map(ToString::to_string),
        )?;
    }
    publish_new(path, &writer.into_inner()?)?;
    Ok(())
}
pub fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=5).contains(&args.len()) {
        return Err(
            "usage: axon-uic-constructive-lab <new-output> [rng] [new-graphs-per-task 100..10000] [elitist|diverse|reinforced] [legacy|arithmetic]"
                .into(),
        );
    }
    let output = PathBuf::from(&args[0]);
    bounded_path(&output)?;
    fs::create_dir(&output)?;
    let seed = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(75218341u64);
    let limit = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(5000usize);
    if !(100..=10000).contains(&limit) {
        return Err("constructive graph limit 100..10000".into());
    }
    let policy = match args.get(3).map(String::as_str).unwrap_or("elitist") {
        "elitist" => ConstructivePolicy::Elitist,
        "diverse" => ConstructivePolicy::Diverse,
        "reinforced" => ConstructivePolicy::Reinforced,
        _ => return Err("unknown constructive policy".into()),
    };
    let arithmetic = match args.get(4).map(String::as_str).unwrap_or("legacy") {
        "legacy" => false,
        "arithmetic" => true,
        _ => return Err("unknown constructive curriculum".into()),
    };
    let version = if policy == ConstructivePolicy::Elitist && !arithmetic {
        "constructive-v3"
    } else {
        "constructive-v6"
    };
    let families: Vec<usize> = if arithmetic {
        vec![6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
    } else {
        (0..6).collect()
    };
    let goals = families.len() - 1;
    let registry = std::env::var_os("AXON_CONSTRUCTIVE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".axon-memory/constructive"));
    bounded_path(&registry)?;
    fs::create_dir_all(&registry)?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment(version);
    telemetry.emit(
        "lab_start",
        &[
            ("method", V::Text("ConstructiveSearch::open")),
            ("rows", V::Count(limit as u64)),
        ],
    )?;
    // Freeze every train/admission/final table before the first selection. No final-label feedback.
    let mut suites = Vec::new();
    let mut protocol = String::from(
        "protocol=constructive-v3\nrepresentation=acyclic arithmetic graphs; +,-,*,/,sqrt and -1,0,1,2 supplied\nnovelty=exact normalized graph within exact demonstrations/ancestral programs/budget; not semantic equivalence\nselection=training only; cached elites; mutations and crossover; no formula names; exportable programs only for promotion\ngrowth=one operation slot on 96 consecutive evaluations without capability improvement; promotion and plateau are independent decisions\nlimits=20000 unique graphs; 100000000 fitness scalar work per task including certification ceiling; 2..16 genes; one worker\nbirth=AXCG0001 decimal circuit without demonstrations, fitness, RNG or personal episodes; AXGN0001 bundle retains module proof examples\nfinal=1024 independent inputs per task; no selection or retry from final results; v3 fresh final instances, known task families\nnegative=sine, outside the supplied finite arithmetic grammar; failure retained\nenergy_and_RSS=unavailable\n",
    );
    protocol = protocol.replacen(
        "protocol=constructive-v3",
        &format!("protocol={version}"),
        1,
    );
    protocol.push_str(&format!("executed_protocol={version}\npolicy={}\ncurriculum={}\ngoal_tasks={goals}\nsearch_feedback=UCB1 over fresh,mutation,crossover,composition; training-case improvement only; search-level, not general lifetime RL\nportfolio=per-case champions plus recent proposals; inherited approved programs\n", policy.name(), if arithmetic { "arithmetic" } else { "legacy" }));
    protocol.push_str(&format!("initial_genes={}\ncomposition=generic two-branch sweep; no formula dispatch; half of proposals plus adaptive choices\n", if policy == ConstructivePolicy::Elitist { 2 } else { 4 }));
    let mut admission_passed = 0;
    let mut goal_final_correct = 0;
    for (case, family) in families.iter().copied().enumerate() {
        let train = rows(family, seed ^ (case as u64 * 97 + 11), 16, 1.0);
        let admission = rows(family, 0x61ab937 + case as u64, 16, 1.73);
        let final_rows = rows(
            family,
            (if version == "constructive-v3" {
                0xf531927
            } else {
                0x7169ab3
            }) + case as u64,
            1024,
            7.19,
        );
        protocol.push_str(&format!("task-{case}-family={family}\n"));
        for (name, data) in [
            ("training", &train),
            ("admission", &admission),
            ("final", &final_rows),
        ] {
            let path = output.join(format!("task-{case}-{name}.tsv"));
            table(&path, data)?;
            protocol.push_str(&format!(
                "task-{case}-{name}-fnv64={:016X}\n",
                checksum(&fs::read(path)?)
            ));
        }
        suites.push((train, admission, final_rows));
    }
    publish_new(&output.join("protocol.txt"), protocol.as_bytes())?;
    let mut acquired = Vec::<EvolvedSkill>::new();
    let mut birth_probes = Vec::new();
    let mut results = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(Vec::new());
    results.write_record([
        "case",
        "status",
        "new_graphs",
        "total_graphs",
        "genes",
        "growths",
        "work",
        "training_correct",
        "training_total",
        "final_correct",
        "final_wrong",
        "final_refused",
        "dna_bytes",
        "dna_digits",
        "novel_vs_parents",
        "elapsed_ns",
        "expression",
    ])?;
    let mut total_new = 0;
    let mut admitted = 0;
    for (case, (train, admission, final_rows)) in suites.into_iter().enumerate() {
        let clock = Instant::now();
        let ancestors: Vec<_> = acquired.iter().rev().take(8).rev().collect();
        let mut scope = Vec::new();
        scope.extend(fs::read(output.join(format!("task-{case}-training.tsv")))?);
        for parent in &ancestors {
            scope.extend(
                CircuitDna::from_skill(parent)
                    .program()?
                    .expression()
                    .bytes(),
            );
        }
        let dir = registry.join(if version == "constructive-v3" {
            format!("task-{case}-v3-{:016X}", checksum(&scope))
        } else {
            format!(
                "task-{case}-v6-{}-{}-{:016X}",
                policy.name(),
                if arithmetic { "arithmetic" } else { "legacy" },
                checksum(&scope)
            )
        });
        bounded_path(&dir)?;
        fs::create_dir_all(&dir)?;
        let mut search = ConstructiveSearch::open_with_policy(
            &dir.join("search.axep"),
            train,
            &ancestors,
            if policy == ConstructivePolicy::Elitist {
                ConstructiveBudget::default()
            } else {
                ConstructiveBudget {
                    initial_genes: 4,
                    ..ConstructiveBudget::default()
                }
            },
            NumericTolerance::default(),
            seed ^ (case as u64 * 173),
            policy,
        )?;
        let previous = search.progress().completed;
        telemetry.emit(
            "stage",
            &[
                ("case", V::Count(case as u64)),
                ("method", V::Text("ConstructiveSearch::resume")),
                (
                    "detail",
                    V::Text(&format!("Tarefa {case}: {previous} grafos anteriores")),
                ),
            ],
        )?;
        let mut telemetry_error = None;
        let new = search.run(limit, |p| {
            if telemetry_error.is_none() {
                telemetry_error = telemetry
                    .emit(
                        "evolution_progress",
                        &[
                            ("method", V::Text("ConstructiveSearch::run")),
                            ("case", V::Count(case as u64)),
                            ("evaluations", V::Count(p.completed as u64)),
                            ("unique", V::Count(p.seen as u64)),
                            ("genes", V::Count(p.genes as u64)),
                            ("growths", V::Count(p.growths as u64)),
                            ("correct", V::Count(p.correct as u64)),
                            ("loss", V::Number(p.loss)),
                            ("work", V::Count(p.work)),
                            (
                                "duplicate_proposals",
                                V::Count(p.duplicate_proposals as u64),
                            ),
                        ],
                    )
                    .err();
            }
        })?;
        if let Some(e) = telemetry_error {
            return Err(e.into());
        }
        total_new += new;
        let progress = search.progress();
        if let Some(feedback) = search.feedback() {
            let detail = format!(
                "operators=fresh,mutation,crossover,composition; trials={:?}; rewards={:?}; policy={}",
                feedback.trials,
                feedback.rewards,
                policy.name()
            );
            telemetry.emit(
                "stage",
                &[
                    ("method", V::Text("ConstructiveSearch::feedback")),
                    ("case", V::Count(case as u64)),
                    ("operator_trials", V::Counts(&feedback.trials)),
                    ("detail", V::Text(&detail)),
                ],
            )?;
            publish_new(
                &output.join(format!("task-{case}-feedback.txt")),
                detail.as_bytes(),
            )?;
        }
        let candidate = search.best_dna().expect("positive batch budget");
        let program = candidate.program()?;
        let expression = program.expression();
        let novel = !ancestors.iter().any(|p| {
            p.program().expression() == expression && p.program().arity() == program.arity()
        });
        // Exactly one post-selection admission. A failed candidate is not replaced after looking at final labels.
        let certified = search.certify(&format!("task-{case}"), "constructive/v1", admission);
        let status = if search.winner().is_none() {
            "not-acquired"
        } else if certified.is_err() {
            "refuted"
        } else {
            "admitted"
        };
        let born = CircuitDna::from_number(&candidate.number())?.program()?;
        let (mut good, mut wrong, mut refused) = (0, 0, 0);
        let mut predictions = csv::WriterBuilder::new()
            .delimiter(b'\t')
            .from_writer(Vec::new());
        predictions.write_record(["row", "expected", "actual", "verdict"])?;
        for (i, row) in final_rows.iter().enumerate() {
            let actual = born.evaluate(&row.input).ok();
            let verdict = match actual {
                Some(v) if NumericTolerance::default().agrees(v, row.output) => {
                    good += 1;
                    "correct"
                }
                Some(_) => {
                    wrong += 1;
                    "wrong"
                }
                None => {
                    refused += 1;
                    "refused"
                }
            };
            predictions.write_record([
                i.to_string(),
                row.output.to_string(),
                actual.map_or_else(String::new, |v| v.to_string()),
                verdict.into(),
            ])?;
        }
        publish_new(
            &output.join(format!("task-{case}-predictions.tsv")),
            &predictions.into_inner()?,
        )?;
        if case < goals {
            goal_final_correct += good;
        }
        publish_new(
            &output.join(format!("task-{case}-circuit.txt")),
            candidate.number().as_bytes(),
        )?;
        search.snapshot_new(&output.join(format!("task-{case}-journal.axep")))?;
        if let Ok(skill) = certified {
            skill.save_new(&output.join(format!("task-{case}.axevo")))?;
            let (arity, output_signal, wiring) = skill.wiring();
            telemetry.emit(
                "genome",
                &[
                    ("method", V::Text("EvolvedSkill::wiring")),
                    ("case", V::Count(case as u64)),
                    ("arity", V::Count(arity as u64)),
                    ("output_signal", V::Count(output_signal as u64)),
                    ("wiring", V::Counts(&wiring)),
                    ("expression", V::Text(&expression)),
                ],
            )?;
            acquired.push(skill);
            birth_probes.push(if version == "constructive-v3" {
                final_rows[..16].to_vec()
            } else {
                final_rows.clone()
            });
            admitted += 1;
            if case < goals {
                admission_passed += 1;
            }
        }
        telemetry.emit(
            "validation",
            &[
                ("method", V::Text("CircuitDna::program")),
                ("case", V::Count(case as u64)),
                (
                    "phase",
                    V::Text(if previous > 0 {
                        "development-recheck"
                    } else if version == "constructive-v3" {
                        "prospective-final"
                    } else {
                        "post-selection"
                    }),
                ),
                ("correct", V::Count(good)),
                ("wrong", V::Count(wrong)),
                ("refused", V::Count(refused)),
                ("checked", V::Count(final_rows.len() as u64)),
                ("admitted", V::Flag(status == "admitted")),
                ("expression", V::Text(&expression)),
            ],
        )?;
        results.write_record([
            case.to_string(),
            status.into(),
            new.to_string(),
            progress.completed.to_string(),
            candidate.genes().to_string(),
            progress.growths.to_string(),
            progress.work.to_string(),
            progress.correct.to_string(),
            "16".into(),
            good.to_string(),
            wrong.to_string(),
            refused.to_string(),
            candidate.to_bytes().len().to_string(),
            candidate.number().len().to_string(),
            novel.to_string(),
            clock.elapsed().as_nanos().to_string(),
            expression,
        ])?;
    }
    publish_new(&output.join("results.tsv"), &results.into_inner()?)?;
    if !acquired.is_empty() {
        let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
        let mut catalog = Catalog::new(&output);
        for skill in &acquired {
            catalog.register(
                ModuleSpec {
                    id: skill.name(),
                    kind: ModuleKind::Evolved,
                    file: &format!("{}.axevo", skill.name()),
                    input: "numbers",
                    output: "number",
                    cost: skill.program().instructions() as u32,
                },
                &auth,
            )?;
        }
        let mut runtime = DemandRuntime::new(
            catalog,
            Seed {
                active_modules: 1,
                active_bytes: 262144,
                ..Seed::default()
            },
        )?;
        let genome = Genome::capture(
            "constructive/v1",
            &mut runtime,
            CompositionBudget::default(),
            &[],
            MAX_BYTES,
            &auth,
        )?;
        genome.save_number_new(&output.join("seed.axnum"), &auth)?;
        let restored = Genome::from_number(&genome.number())?;
        let mut birth = restored.birth(&output.join("birth"), BirthLimits::default(), &auth)?;
        let initial_loads = birth.controller.runtime().stats().loads;
        let mut retention = 0u64;
        for (skill, probes) in acquired.iter().zip(&birth_probes) {
            birth.controller.runtime_mut().with_module(
                skill.name(),
                skill.context(),
                &auth,
                |module| {
                    let Module::Evolved(child) = module else {
                        return Err(DemandError::Invalid("constructive birth module"));
                    };
                    for row in probes {
                        let value = child
                            .predict(skill.context(), &row.input)
                            .map_err(|e| DemandError::Module(e.to_string()))?;
                        if !NumericTolerance::default().agrees(value, row.output) {
                            return Err(DemandError::Invalid("constructive birth retention"));
                        }
                        retention += 1;
                    }
                    Ok(())
                },
            )?;
        }
        let stats = birth.controller.runtime().stats();
        let birth_result = format!(
            "format=AXGN0001\nmodules={}\nencoded_bytes={}\ndecimal_digits={}\ninitial_module_loads={initial_loads}\nfinal_module_loads={}\nactive_modules={}\nretention_checks={retention}\npersonal_episodes=0\nproof_examples=retained in evolved capsules; not personal episodes\ncompact_circuits=no examples; separate AXCG0001 files\n",
            acquired.len(),
            genome.encoded_bytes(),
            genome.number().len(),
            stats.loads,
            stats.active_modules
        );
        publish_new(&output.join("birth-result.txt"), birth_result.as_bytes())?;
        telemetry.emit(
            "birth",
            &[
                ("method", V::Text("Genome::birth")),
                ("modules", V::Count(acquired.len() as u64)),
                ("encoded_bytes", V::Count(genome.encoded_bytes() as u64)),
                ("decimal_digits", V::Count(genome.number().len() as u64)),
                ("loads", V::Count(stats.loads)),
                ("checked", V::Count(retention)),
            ],
        )?;
    } else {
        publish_new(
            &output.join("birth-result.txt"),
            b"status=unavailable\nreason=no admitted modules\n",
        )?;
    }
    let result = format!(
        "protocol={version}\npolicy={}\nnew_graphs={total_new}\nadmitted_tasks={admitted}\ntasks={}\ngoal_tasks={goals}\ngoal_admitted={admission_passed}\nall_goals_admitted={}\ngoal_final_correct={goal_final_correct}\nall_goals_passed={}\nfinal_attempts={}\nprocess_elapsed_ns={}\npersonal_episodes=0\nRSS=unavailable\nenergy=unavailable\n",
        policy.name(),
        families.len(),
        admission_passed == goals,
        admission_passed == goals && goal_final_correct == (goals * 1024) as u64,
        families.len() * 1024,
        started.elapsed().as_nanos()
    );
    publish_new(&output.join("result.txt"), result.as_bytes())?;
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("ConstructiveSearch::certify")),
            ("admitted", V::Count(admitted)),
            ("tasks", V::Count(families.len() as u64)),
            ("goal_tasks", V::Count(goals as u64)),
            ("goal_admitted", V::Count(admission_passed as u64)),
            ("all_goals_admitted", V::Flag(admission_passed == goals)),
            ("goal_final_correct", V::Count(goal_final_correct)),
            (
                "all_goals_passed",
                V::Flag(admission_passed == goals && goal_final_correct == (goals * 1024) as u64),
            ),
            ("new_candidates", V::Count(total_new as u64)),
            ("elapsed_ns", V::Count(started.elapsed().as_nanos() as u64)),
        ],
    )?;
    Ok(())
}
