use axon_uic::{
    behavior_lab::Rng,
    numeric::{library::*, *},
};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    hint::black_box,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy)]
enum Task {
    Magnitude4,
    PointDistance,
    ScaledSignal,
    Inclination,
    Residual,
    ReciprocalMagnitude,
    SquaredResidual,
}
impl Task {
    fn name(self) -> &'static str {
        match self {
            Self::Magnitude4 => "magnitude4",
            Self::PointDistance => "point_distance",
            Self::ScaledSignal => "scaled_signal",
            Self::Inclination => "inclination",
            Self::Residual => "residual",
            Self::ReciprocalMagnitude => "reciprocal_magnitude",
            Self::SquaredResidual => "squared_residual",
        }
    }
    fn arity(self) -> usize {
        match self {
            Self::Magnitude4 | Self::PointDistance | Self::ScaledSignal => 4,
            Self::Inclination | Self::Residual => 3,
            Self::ReciprocalMagnitude | Self::SquaredResidual => 2,
        }
    }
    fn cost(self) -> usize {
        if matches!(
            self,
            Self::Inclination | Self::Residual | Self::SquaredResidual
        ) {
            5
        } else {
            7
        }
    }
    fn truth(self, x: &[f64]) -> f64 {
        match self {
            Self::Magnitude4 => x[0].hypot(x[1]).hypot(x[2]).hypot(x[3]),
            Self::PointDistance => (x[0] - x[2]).hypot(x[1] - x[3]),
            Self::ScaledSignal => (x[0] * x[1]).hypot(x[2] * x[3]),
            Self::Inclination => x[0].hypot(x[1]) / x[2],
            Self::Residual => (x[0] - x[1]).hypot(x[2]),
            Self::ReciprocalMagnitude => x[0].recip().hypot(x[1].recip()),
            Self::SquaredResidual => (x[0] - x[1]).powi(2),
        }
    }
    fn samples(self, seed: u64, count: usize, training: bool) -> Vec<NumericExample> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|i| {
                let mut x: Vec<_> = (0..self.arity())
                    .map(|_| {
                        if training {
                            (rng.pick(169) as f64 - 84.0) / 7.0
                        } else {
                            (rng.pick(200001) as f64 - 100000.0) / 37.0
                        }
                    })
                    .collect();
                if i % 17 == 0 {
                    x[0] = 0.0;
                }
                if matches!(self, Self::Inclination) && x[2] == 0.0 {
                    x[2] = 0.25;
                }
                if matches!(self, Self::ReciprocalMagnitude) {
                    for value in &mut x {
                        if *value == 0.0 {
                            *value = 0.25;
                        }
                    }
                }
                let y = self.truth(&x);
                NumericExample::new(x, y)
            })
            .collect()
    }
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
fn tsv(rows: &[NumericExample]) -> String {
    let mut s = (0..rows[0].input.len())
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join("\t");
    s.push_str("\ty\n");
    for row in rows {
        for n in &row.input {
            s.push_str(&format!("{n:.17}\t"));
        }
        s.push_str(&format!("{:.17}\n", row.output));
    }
    s
}
fn csv(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
fn load_sources(directory: &Path) -> Result<Vec<NumericSkill>, Box<dyn Error>> {
    let mut paths = fs::read_dir(directory)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    let mut skills = Vec::new();
    for path in paths {
        if path.extension().and_then(|s| s.to_str()) != Some("axnum") {
            continue;
        }
        if skills.len() >= 64 {
            return Err("source count limit".into());
        }
        let mut bytes = Vec::new();
        File::open(&path)?.take(32769).read_to_end(&mut bytes)?;
        skills.push(NumericSkill::from_bytes(&bytes)?);
    }
    Ok(skills)
}
fn acquire_extra_sources(skills: &mut Vec<NumericSkill>, seed: u64) -> Result<(), NumericError> {
    // These are demonstrated tasks, not tool definitions. Mining never sees the oracle or its label.
    for case in 0..6 {
        let mut rng = Rng::new(seed.wrapping_add(case * 3571));
        let training = (0..12)
            .map(|_| {
                let x = (rng.pick(169) as f64 - 84.0) / 7.0;
                let square = x * x;
                let y = match case {
                    0 => square,
                    1 => square + 1.0,
                    2 => square - 1.0,
                    3 => square * 2.0,
                    4 => square / 2.0,
                    _ => square + 2.0,
                };
                NumericExample::new(vec![x], y)
            })
            .collect();
        let skill = NumericSkill::learn(
            format!("opaque_extra_{case}"),
            format!("abstraction-lab/past/{case}"),
            training,
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )?;
        let mut reserved = Rng::new(seed.wrapping_add(0x735ffe19).wrapping_add(case * 983));
        for _ in 0..1024 {
            let x = (reserved.pick(200001) as f64 - 100000.0) / 37.0;
            let square = x * x;
            let y = match case {
                0 => square,
                1 => square + 1.0,
                2 => square - 1.0,
                3 => square * 2.0,
                4 => square / 2.0,
                _ => square + 2.0,
            };
            if !skill.tolerance().agrees(skill.program().evaluate(&[x])?, y) {
                return Err(NumericError::Invalid(
                    "new source failed reserved validation",
                ));
            }
        }
        skills.push(skill);
    }
    Ok(())
}
fn median_times(
    program: &NumericProgram,
    execution: &NumericExecution,
    rows: &[NumericExample],
) -> (u64, u64) {
    let mut stack = Vec::new();
    let mut shared = Vec::new();
    for round in 0..7 {
        for mode in 0..2 {
            let dag = (mode + round) % 2 == 1;
            let start = Instant::now();
            for row in rows {
                let value = if dag {
                    execution.evaluate(black_box(&row.input))
                } else {
                    program.evaluate(black_box(&row.input))
                };
                black_box(value).expect("validated heldout inputs");
            }
            let elapsed = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            if dag {
                shared.push(elapsed);
            } else {
                stack.push(elapsed);
            }
        }
    }
    stack.sort_unstable();
    shared.sort_unstable();
    (stack[3], shared[3])
}
fn search_fields(error: &NumericError) -> String {
    match error {
        NumericError::Budget(s) | NumericError::NoProgram(s) => format!(
            "{},{},{},{}",
            s.candidates, s.scalar_work, s.retained_terms, s.elapsed_ns
        ),
        _ => ",,,".to_owned(),
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=4).contains(&args.len()) {
        return Err("usage: axon-uic-abstraction-lab <source-capsule-directory> <new-output-directory> [seed] [rows-per-task 100..100000]".into());
    }
    let seed = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(30719581_u64);
    let rows = args
        .get(3)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(50000_usize);
    if !(100..=100000).contains(&rows) {
        return Err("heldout row limit".into());
    }
    let output = PathBuf::from(&args[1]);
    let start = Instant::now();
    let mut skills = load_sources(Path::new(&args[0]))?;
    let prior_sources = skills.len();
    if prior_sources == 0 || prior_sources > 58 {
        return Err("requires 1..58 prior capsules; six source tasks will be acquired".into());
    }
    let acquisition_started = Instant::now();
    acquire_extra_sources(&mut skills, seed.wrapping_add(0x33ee7149))?;
    let acquisition_ns = acquisition_started.elapsed().as_nanos();
    let library = NumericLibrary::mine(
        "abstraction-lab/experience/v1",
        &skills.iter().collect::<Vec<_>>(),
        MiningBudget::default(),
    )?;
    fs::create_dir(&output)?;
    let mut manifest = String::from("skill\tcontext\n");
    for (i, skill) in skills.iter().enumerate() {
        let name = format!("source-{i:03}.axnum");
        skill.save_new(&output.join(&name))?;
        manifest.push_str(&format!("{name}\t{}\n", skill.context()));
    }
    write_new(&output.join("sources.tsv"), manifest.as_bytes())?;
    let reload_started = Instant::now();
    library.save_new(&output.join("mined.axlib"))?;
    let library = NumericLibrary::open(&output.join("mined.axlib"), library.context())?;
    let library_save_reload_ns = reload_started.elapsed().as_nanos();
    let mut modules =
        String::from("identity,arity,body_instructions,support,net_saved_tree_nodes,expression\n");
    for m in library.modules() {
        modules.push_str(&format!(
            "{:016x},{},{},{},{},{}\n",
            m.fingerprint(),
            m.program().arity(),
            m.program().instructions(),
            m.sources().len(),
            m.net_saved_nodes(),
            csv(&m.program().expression())
        ));
    }
    write_new(&output.join("modules.csv"), modules.as_bytes())?;
    let mut report = String::from(
        "task,mode,status,examples,max_cost,candidates,scalar_work,search_terms,search_ns,expanded_instructions,shared_instructions,model_bytes,correct,total,max_absolute_error,expression_or_error\n",
    );
    let mut performance = String::from(
        "task,rows,stack_median_ns,shared_median_ns,stack_scratch_bytes,shared_scratch_bytes,compiled_retained_bytes,compile_ns,seven_alternating_rounds\n",
    );
    let tasks = [
        Task::Magnitude4,
        Task::PointDistance,
        Task::ScaledSignal,
        Task::Inclination,
        Task::Residual,
        Task::ReciprocalMagnitude,
        Task::SquaredResidual,
    ];
    let mut development_report = String::from(
        "task,attempt,proposed_tool,compatible,candidates,scalar_work,search_ns,total_development_work,total_development_ns,proposal_patterns,proposal_visits\n",
    );
    let mut baseline_solved = 0;
    let mut compression_solved = 0;
    let mut mined_solved = 0;
    let mut correct_total = 0;
    let mut bits_total = 0;
    let mut new_models = Vec::new();
    for (i, task) in tasks.into_iter().enumerate() {
        let context = format!("abstraction-lab/{}/v1", task.name());
        let training = task.samples(seed.wrapping_add(i as u64 * 7919), 12, true);
        write_new(
            &output.join(format!("{}.training.tsv", task.name())),
            tsv(&training).as_bytes(),
        )?;
        let budget = NumericBudget {
            max_cost: task.cost(),
            max_terms: 200000,
            max_work: 100000000,
        };
        let baseline = NumericSkill::learn(
            "opaque",
            &context,
            training.clone(),
            &[],
            budget,
            NumericTolerance::default(),
        );
        match baseline {
            Ok(skill) => {
                baseline_solved += 1;
                let s = skill.search();
                report.push_str(&format!(
                    "{},primitive,fit_only,12,{},{},{},{},{},{},,,,,,{}\n",
                    task.name(),
                    task.cost(),
                    s.candidates,
                    s.scalar_work,
                    s.retained_terms,
                    s.elapsed_ns,
                    skill.program().instructions(),
                    csv(&skill.program().expression())
                ));
            }
            Err(e) => report.push_str(&format!(
                "{},primitive,refused,12,{},{},,,,,,,{}\n",
                task.name(),
                task.cost(),
                search_fields(&e),
                csv(&e.to_string())
            )),
        }
        let learned = NumericSkill::learn_from_library(
            "opaque",
            &context,
            training.clone(),
            &library,
            budget,
            NumericTolerance::default(),
        );
        match learned {
            Ok(skill) => {
                compression_solved += 1;
                let s = skill.search();
                report.push_str(&format!(
                    "{},compression,fit_only,12,{},{},{},{},{},{},,,,,,{}\n",
                    task.name(),
                    task.cost(),
                    s.candidates,
                    s.scalar_work,
                    s.retained_terms,
                    s.elapsed_ns,
                    skill.program().instructions(),
                    csv(&skill.program().expression())
                ));
            }
            Err(e) => {
                report.push_str(&format!(
                    "{},compression,refused,12,{},{},,,,,,,{}\n",
                    task.name(),
                    task.cost(),
                    search_fields(&e),
                    csv(&e.to_string())
                ));
            }
        }
        let (skill, trace) = match library.develop(
            "opaque",
            &context,
            training,
            budget,
            DevelopmentBudget::default(),
            NumericTolerance::default(),
        ) {
            Ok(result) => result,
            Err(e) => {
                report.push_str(&format!(
                    "{},developing,refused,12,{},{},,,,,,,{}\n",
                    task.name(),
                    task.cost(),
                    search_fields(&e),
                    csv(&e.to_string())
                ));
                continue;
            }
        };
        for (attempt, entry) in trace.attempts.iter().enumerate() {
            development_report.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                task.name(),
                attempt,
                entry
                    .proposed_tool
                    .map(|id| format!("{id:016x}"))
                    .unwrap_or_default(),
                entry.compatible,
                entry.search.candidates,
                entry.search.scalar_work,
                entry.search.elapsed_ns,
                trace.scalar_work,
                trace.elapsed_ns,
                trace.proposal_patterns,
                trace.proposal_visits
            ));
        }
        let model = output.join(format!("{}.axnum", task.name()));
        skill.save_new(&model)?;
        let skill = NumericSkill::open(&model, &context)?;
        let compile_started = Instant::now();
        let execution = skill.program().compile_execution();
        let compile_ns = compile_started.elapsed().as_nanos();
        // Test inputs and labels are generated only after the learned artifacts are frozen.
        let heldout = task.samples(
            seed.wrapping_add(0x517cc1b727220a95)
                .wrapping_add(i as u64 * 19391),
            rows,
            false,
        );
        write_new(
            &output.join(format!("{}.heldout.tsv", task.name())),
            tsv(&heldout).as_bytes(),
        )?;
        let mut correct = 0;
        let mut exact = 0;
        let mut error = 0.0_f64;
        for e in &heldout {
            let stack = skill.predict(&context, &e.input)?;
            let shared = execution.evaluate(&e.input)?;
            if stack.to_bits() == shared.to_bits() {
                exact += 1;
            }
            if skill.tolerance().agrees(shared, e.output) {
                correct += 1;
            }
            error = error.max((shared - e.output).abs());
        }
        let s = skill.search();
        report.push_str(&format!(
            "{},developing,{},12,{},{},{},{},{},{},{},{},{},{},{},{}\n",
            task.name(),
            if correct == rows && exact == rows {
                "passed"
            } else {
                "refuted"
            },
            task.cost(),
            s.candidates,
            s.scalar_work,
            s.retained_terms,
            s.elapsed_ns,
            skill.program().instructions(),
            execution.instructions(),
            skill.to_bytes().len(),
            correct,
            rows,
            error,
            csv(&skill.program().expression())
        ));
        if correct == rows && exact == rows {
            mined_solved += 1;
            let (stack, shared) = median_times(skill.program(), &execution, &heldout);
            performance.push_str(&format!(
                "{},{},{},{},{},{},{},{},true\n",
                task.name(),
                rows,
                stack,
                shared,
                skill.program().scratch_bytes(),
                execution.scratch_bytes(),
                execution.retained_bytes(),
                compile_ns
            ));
        }
        correct_total += correct;
        bits_total += exact;
        new_models.push(skill);
    }
    write_new(&output.join("results.csv"), report.as_bytes())?;
    write_new(&output.join("execution.csv"), performance.as_bytes())?;
    write_new(
        &output.join("development.csv"),
        development_report.as_bytes(),
    )?;
    let duplicate = NumericLibrary::mine(
        "abstraction-lab/duplicates",
        &vec![&skills[0]; 32],
        MiningBudget::default(),
    )?;
    let mut controls = format!(
        "32 duplicate sources: unique={}, modules={}\n",
        duplicate.statistics().unique_sources,
        duplicate.modules().len()
    );
    if !duplicate.modules().is_empty() {
        return Err("duplicate-source control failed".into());
    }
    let sine = [0.2_f64, 0.7, 1.3, 2.1, 3.2]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x.sin()))
        .collect();
    let unsupported = NumericSkill::learn_from_library(
        "unsupported",
        "negative/sine",
        sine,
        &library,
        NumericBudget {
            max_cost: 5,
            max_terms: 4000,
            max_work: 100000,
        },
        NumericTolerance::default(),
    );
    if unsupported.is_ok() {
        return Err("unsupported-task control unexpectedly fit".into());
    }
    controls.push_str(&format!(
        "unsupported sine with mined library: {:?}\n",
        unsupported.err()
    ));
    let mut all = skills.iter().collect::<Vec<_>>();
    all.extend(new_models.iter());
    let next = NumericLibrary::mine(
        "abstraction-lab/experience/v2",
        &all,
        MiningBudget::default(),
    )?;
    next.save_new(&output.join("next-epoch.axlib"))?;
    controls.push_str(&format!(
        "new experience consolidated: unique={}, modules={}, net_saved_nodes={}\n",
        next.statistics().unique_sources,
        next.modules().len(),
        next.statistics().net_saved_nodes
    ));
    write_new(&output.join("controls.txt"), controls.as_bytes())?;
    let s = library.statistics();
    let summary = format!(
        "AXON AUTOMATIC ABSTRACTION LAB / EMPIRICAL NUMERICAL TRANSFER\nseed={seed}\nrows_per_task={rows}\nsource_directory={}\nprior_sources={prior_sources}\nnew_source_acquisition_ns={acquisition_ns}\nnew_source_reserved_checks=6144\nsupplied_sources={}\nunique_sources={}\nmodules={}\nmining_ns={}\ntree_visits={}\npatterns={}\noriginal_tree_nodes={}\nrewritten_tree_nodes={}\ndefinition_nodes={}\nnet_saved_tree_nodes={}\nlibrary_bytes_including_source_audit={}\nretained_library_bytes={}\nlibrary_save_reload_ns={library_save_reload_ns}\nprimitive_fit_only={baseline_solved}/7\ncompression_fit_only={compression_solved}/7\ndemand_verified={mined_solved}/7\ncorrect={correct_total}\nheldout_total={}\nshared_execution_bit_identical={bits_total}\nwhole_lab_ns={}\nCosts exclude the prior acquisition of input capsules but include six new source tasks. Tree-node savings are not encoded-byte savings. Demand adds bounded attempts, not free search. No general proof, watts, language fluency or AGI claim.\n",
        args[0],
        s.supplied_sources,
        s.unique_sources,
        library.modules().len(),
        s.elapsed_ns,
        s.tree_visits,
        s.patterns,
        s.original_nodes,
        s.rewritten_nodes,
        s.definition_nodes,
        s.net_saved_nodes,
        library.to_bytes().len(),
        library.retained_bytes(),
        7 * rows,
        start.elapsed().as_nanos()
    );
    write_new(&output.join("summary.txt"), summary.as_bytes())?;
    print!("{summary}");
    if mined_solved != 7 || correct_total != 7 * rows || bits_total != 7 * rows {
        return Err("one or more bounded transfer tasks failed; inspect results.csv".into());
    }
    Ok(())
}
