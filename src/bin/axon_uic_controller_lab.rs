mod controller_support;
use axon_uic::{
    Authority, Effect,
    behavior::Resettable,
    behavior_lab::Rng,
    demand::{controller::*, *},
    telemetry::{LabTelemetry, Value as V},
    text_skill::read_bounded,
};
use controller_support::*;
use std::{
    collections::BTreeSet,
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    process::{Command, Stdio},
};

const WIRE: [&str; 16] = [
    "event",
    "id",
    "goal",
    "predicted",
    "actual",
    "baseline",
    "actions",
    "phase",
    "correct",
    "scored",
    "physical_correct",
    "baseline_correct",
    "drifts",
    "queries",
    "peak_bytes",
    "peak_modules",
];
fn create(path: &Path, value: &str) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(value.as_bytes())?;
    f.sync_all()?;
    Ok(())
}
fn emit<W: Write>(
    writer: &mut csv::Writer<W>,
    event: &str,
    position: (u64, usize),
    values: (Option<i64>, Option<i64>, Option<i64>),
    actions: usize,
    phase: &str,
    totals: [u64; 8],
) -> Result<(), Box<dyn Error>> {
    let (id, goal) = position;
    let (predicted, actual, baseline) = values;
    let mut row = vec![
        event.into(),
        id.to_string(),
        goal.to_string(),
        predicted.map_or(String::new(), |v| v.to_string()),
        actual.map_or(String::new(), |v| v.to_string()),
        baseline.map_or(String::new(), |v| v.to_string()),
        actions.to_string(),
        phase.into(),
    ];
    row.extend(totals.map(|v| v.to_string()));
    writer.write_record(row)?;
    writer.flush()?;
    Ok(())
}
fn replay(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 6 {
        return Err(
            "replay <root> <stream.tsv> <results.csv> <device-seed> <repair-version>".into(),
        );
    }
    let root = Path::new(&args[1]);
    axon_uic::demand::file_name(&args[2])?;
    axon_uic::demand::file_name(&args[3])?;
    let device_seed = args[4].parse::<u64>()?;
    let repair = args[5].parse::<usize>()?;
    if repair == 0 {
        return Err("repair identity must be nonzero".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    let mut runtime =
        controller_support::runtime(Catalog::load(root.join("modules.axseed"), &auth)?, 5)?;
    let mut procedures = Vec::new();
    for (i, context) in GOALS.iter().enumerate() {
        procedures.push(Procedure::open(
            &root.join(format!("goal-{i}.axgoal")),
            context,
            &mut runtime,
            &auth,
        )?);
    }
    let baseline_procedure = procedures[0].clone();
    let mut baseline_runtime =
        controller_support::runtime(Catalog::load(root.join("modules.axseed"), &auth)?, 5)?;
    let mut baseline_cursor = baseline_procedure.cursor();
    let mut controller = GoalController::new(runtime);
    for p in procedures {
        controller.install(p, &auth)?;
    }
    let mut cursor = controller.runtime().behavior_cursor("device", DEVICE)?;
    let mut device = Device::new(device_seed);
    let bytes = read_bounded(&root.join(&args[2]), 16_777_216)?;
    let mut inputs = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(bytes.as_slice());
    if inputs.headers()?.iter().collect::<Vec<_>>()
        != ["id", "goal", "input", "feedback", "x0", "x1"]
    {
        return Err("invalid stream schema".into());
    }
    let rows = inputs.records().collect::<Result<Vec<_>, _>>()?;
    if rows.is_empty() || rows.len() > 20000 {
        return Err("stream quota".into());
    }
    let result = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(&args[3]))?;
    let mut out = csv::Writer::from_writer(result);
    out.write_record([
        "id",
        "goal",
        "predicted",
        "actual",
        "baseline",
        "policy_correct",
        "physical_correct",
        "phase",
        "actions",
        "device_output",
    ])?;
    let mut progress = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(std::io::stdout());
    progress.write_record(WIRE)?;
    let (mut correct, mut physical, mut baseline_correct, mut drifts, mut queries) =
        (0, 0, 0, 0, 0);
    let (mut old_loads, mut peak_bytes, mut peak_modules) = (0, 0, 0);
    for (index, row) in rows.iter().enumerate() {
        let id = row[0].parse::<u64>()?;
        let goal = row[1].parse::<usize>()?;
        if id != index as u64 + 1 || goal >= GOALS.len() {
            return Err("stream ID/goal".into());
        }
        if index == rows.len() / 2 {
            device.change();
        }
        let baseline = baseline_procedure
            .predict(
                GOALS[0],
                &mut baseline_runtime,
                &mut baseline_cursor,
                &row[2],
                &auth,
            )
            .ok();
        let decision = controller.propose(GOALS[goal], id, &row[2], &auth);
        let predicted = decision.as_ref().ok().map(|d| d.value);
        let totals = [
            correct,
            index as u64,
            physical,
            baseline_correct,
            drifts,
            queries,
            peak_bytes,
            peak_modules,
        ];
        if index < 8 {
            emit(
                &mut progress,
                "prediction",
                (id, goal),
                (predicted, None, baseline),
                0,
                "PredictionBeforeFeedback",
                totals,
            )?;
        }
        let mut actions = 0;
        let mut reached = false;
        let mut observed = None;
        let mut phase = "Refused".to_string();
        if let Ok(decision) = decision {
            let receipt = controller.dispatch(
                id,
                ("device", DEVICE),
                &mut cursor,
                &mut device,
                &BTreeSet::new(),
                &auth,
            )?;
            actions = receipt.observations.len();
            observed = receipt.observations.last().map(|(_, output)| *output);
            debug_assert!(observed.is_none_or(|v| v == device.state()));
            reached = receipt.status == DispatchStatus::Reached;
            phase = format!("{:?}", receipt.status);
            if receipt.status == DispatchStatus::Uncertain {
                return Err("uncertain effect; no retry or fabricated success".into());
            }
            if receipt.status == DispatchStatus::Refuted {
                drifts += 1;
                emit(
                    &mut progress,
                    "drift",
                    (id, goal),
                    (predicted, None, baseline),
                    actions,
                    "DeviceSuspended",
                    [
                        correct,
                        index as u64,
                        physical,
                        baseline_correct,
                        drifts,
                        queries,
                        peak_bytes,
                        peak_modules,
                    ],
                )?;
                let stats = controller.runtime().stats();
                old_loads += stats.loads;
                peak_bytes = peak_bytes.max(stats.peak_active_bytes as u64);
                peak_modules = peak_modules.max(stats.peak_active_modules as u64);
                let (q, _) = learn_device(root, repair, &device, &auth)?;
                queries += q as u64;
                device.reset()?;
                let c = catalog(root, repair, &auth)?;
                c.save_new(root.join(format!("modules-{repair}.axseed")), &auth)?;
                controller.rebind(controller_support::runtime(c, 5)?, &auth)?;
                cursor = controller.runtime().behavior_cursor("device", DEVICE)?;
                emit(
                    &mut progress,
                    "repair",
                    (id, goal),
                    (predicted, None, baseline),
                    actions,
                    "DeviceRelearnedNoRetry",
                    [
                        correct,
                        index as u64,
                        physical,
                        baseline_correct,
                        drifts,
                        queries,
                        peak_bytes,
                        peak_modules,
                    ],
                )?;
            }
            // Feedback becomes visible to the controller only after prediction and physical dispatch.
            let actual = row[3].parse::<i64>()?;
            controller.feedback(GOALS[goal], id, actual)?;
            let _ = decision;
        }
        let actual = row[3].parse::<i64>()?;
        let policy_ok = predicted == Some(actual);
        let physical_ok = reached && observed == u16::try_from(actual).ok();
        correct += u64::from(policy_ok);
        physical += u64::from(physical_ok);
        baseline_correct += u64::from(baseline == Some(actual));
        let stats = controller.runtime().stats();
        peak_bytes = peak_bytes.max(stats.peak_active_bytes as u64);
        peak_modules = peak_modules.max(stats.peak_active_modules as u64);
        out.write_record([
            id.to_string(),
            goal.to_string(),
            predicted.map_or(String::new(), |v| v.to_string()),
            actual.to_string(),
            baseline.map_or(String::new(), |v| v.to_string()),
            policy_ok.to_string(),
            physical_ok.to_string(),
            phase.clone(),
            actions.to_string(),
            observed.map_or(String::new(), |v| v.to_string()),
        ])?;
        if index < 8 || (index + 1) % 64 == 0 || index + 1 == rows.len() {
            emit(
                &mut progress,
                "feedback",
                (id, goal),
                (predicted, Some(actual), baseline),
                actions,
                &phase,
                [
                    correct,
                    index as u64 + 1,
                    physical,
                    baseline_correct,
                    drifts,
                    queries,
                    peak_bytes,
                    peak_modules,
                ],
            )?;
        }
    }
    out.flush()?;
    out.get_ref().sync_all()?;
    let loads = old_loads + controller.runtime().stats().loads;
    emit(
        &mut progress,
        "result",
        (rows.len() as u64, 0),
        (None, None, None),
        0,
        &format!("{loads}ModuleLoads/{}ObservedActions", device.effects()),
        [
            correct,
            rows.len() as u64,
            physical,
            baseline_correct,
            drifts,
            queries,
            peak_bytes,
            peak_modules,
        ],
    )?;
    Ok(())
}
struct Worker(std::process::Child);
impl Drop for Worker {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}
fn stream(root: &Path, seed: u64, count: usize, scale: usize) -> Result<String, Box<dyn Error>> {
    let name = format!("stream-{seed}.tsv");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(&name))?;
    let mut writer = csv::WriterBuilder::new().delimiter(b'\t').from_writer(file);
    writer.write_record(["id", "goal", "input", "feedback", "x0", "x1"])?;
    let mut previous = 0;
    let mut id = 0u64;
    let mut rng = Rng::new(seed);
    for start in (0..count).step_by(64) {
        for goal in 0..2 {
            for _ in start..(start + 64).min(count) {
                id += 1;
                let values = if rng.pick(2) == 0 {
                    [rng.pick(8) as i64, rng.pick(8) as i64]
                } else {
                    [12 + rng.pick(scale) as i64, 12 + rng.pick(scale) as i64]
                };
                let actual = expectation(goal, values, &mut previous);
                writer.write_record([
                    id.to_string(),
                    goal.to_string(),
                    wire(seed.wrapping_add(id), values),
                    actual.to_string(),
                    values[0].to_string(),
                    values[1].to_string(),
                ])?;
            }
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(name)
}
fn run(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.is_empty() || args.len() > 4 {
        return Err("axon-uic-controller-lab <new-dir> [seed] [count-per-goal 100..10000] [development|reserved]".into());
    }
    let root = Path::new(&args[0]);
    let random = args
        .get(1)
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(DEV_SEED);
    let count = args
        .get(2)
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1000);
    let validation = args.get(3).map(String::as_str).unwrap_or("development");
    if !(100..=10000).contains(&count) || !["development", "reserved"].contains(&validation) {
        return Err("controller protocol bounds".into());
    }
    if let Some(parent) = root.parent() {
        fs::create_dir_all(parent)?;
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    let mut telemetry = LabTelemetry::from_environment("controller");
    telemetry.emit(
        "lab_start",
        &[
            ("validation", V::Text(validation)),
            (
                "telemetry",
                V::Text("typed-composition-and-64-frame-checkpoints"),
            ),
            ("seed", V::Count(random)),
        ],
    )?;
    let bootstrap = bootstrap(root, random, &auth)?;
    telemetry.emit(
        "controller_bootstrap",
        &[
            ("work", V::Count(bootstrap.numeric_work)),
            ("reactive_work", V::Count(bootstrap.reactive_work)),
            ("queries", V::Count(bootstrap.device_queries as u64)),
            ("actions", V::Count(bootstrap.device_steps as u64)),
        ],
    )?;
    let mut runtime =
        controller_support::runtime(Catalog::load(root.join("modules.axseed"), &auth)?, 5)?;
    let mut frozen = String::from("goal,identity,steps\n");
    for (i, context) in GOALS.iter().enumerate() {
        let outcome = Procedure::discover(
            context,
            traces(i, 7139161, 3),
            traces(i, 37918713, 2),
            &mut runtime,
            &auth,
            CompositionBudget::default(),
        )?;
        let p = outcome
            .procedure
            .ok_or("composition refused; inspect bootstrap and keep artifacts")?;
        p.save_new(&root.join(format!("goal-{i}.axgoal")), &auth)?;
        let steps = p
            .steps()
            .iter()
            .map(|step| match step {
                ProcedureStep::Module { id, .. } => format!(
                    "{} / {:?}",
                    id,
                    runtime.catalog().get(id).expect("validated").kind
                ),
                ProcedureStep::DecodeScalars => "decode_scalars".into(),
                ProcedureStep::ExactInteger => "exact_integer".into(),
            })
            .collect::<Vec<String>>();
        let labels = steps.iter().map(String::as_str).collect::<Vec<_>>();
        frozen.push_str(&format!(
            "{i},{:016x},{}\n",
            p.fingerprint(),
            p.steps().len()
        ));
        telemetry.emit(
            "composition",
            &[
                ("case", V::Count(i as u64)),
                ("name", V::Text(context)),
                ("phase", V::Text("Admitted")),
                ("steps", V::Texts(&labels)),
                ("work", V::Count(outcome.statistics.work)),
                (
                    "evaluations",
                    V::Count(outcome.statistics.candidates as u64),
                ),
                ("expression", V::Text(&format!("{:?}", p.steps()))),
                ("identity", V::Text(&format!("{:016x}", p.fingerprint()))),
                ("capsule_bytes", V::Count(p.to_bytes().len() as u64)),
            ],
        )?;
    }
    create(&root.join("frozen.csv"), &frozen)?;
    telemetry.emit("curriculum_frozen", &[("tasks", V::Count(2))])?;
    let seeds = if validation == "reserved" {
        RESERVES.to_vec()
    } else {
        vec![DEV_SEED]
    };
    let (mut total, mut correct, mut physical, mut baseline, mut drifts, mut queries) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut combined = String::from(
        "seed,correct,scored,physical_correct,baseline_correct,drifts,queries,peak_bytes,peak_modules\n",
    );
    for seed in seeds {
        let input = stream(
            root,
            seed,
            count,
            if validation == "reserved" { 101 } else { 31 },
        )?;
        let output = format!("results-{seed}.csv");
        let mut child = Worker(
            Command::new(std::env::current_exe()?)
                .args([
                    "replay",
                    root.to_str().ok_or("path UTF-8")?,
                    &input,
                    &output,
                    &random.to_string(),
                    &seed.to_string(),
                ])
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()?,
        );
        let stdout = child.0.stdout.take().expect("pipe");
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(stdout);
        if reader.headers()?.iter().collect::<Vec<_>>() != WIRE {
            return Err("child telemetry schema".into());
        }
        let mut result = None;
        for record in reader.records() {
            let r = record?;
            if r.len() != WIRE.len() {
                return Err("child telemetry width".into());
            }
            let totals = r
                .iter()
                .skip(8)
                .map(str::parse::<u64>)
                .collect::<Result<Vec<_>, _>>()?;
            let id = r[1].parse::<u64>()?;
            let goal = r[2].parse::<u64>()?;
            let number = |s: &str| -> Result<V<'_>, Box<dyn Error>> {
                Ok(if s.is_empty() {
                    V::Null
                } else {
                    V::Number(s.parse::<i64>()? as f64)
                })
            };
            telemetry.emit(
                &format!("controller_{}", &r[0]),
                &[
                    (
                        "method",
                        V::Text(match &r[0] {
                            "prediction" => "GoalController::propose",
                            "feedback" => "GoalController::feedback",
                            "drift" => "DemandRuntime::observe_behavior",
                            "repair" => "GoalController::rebind",
                            "result" => "ControllerReplay::summary",
                            _ => return Err("unknown child checkpoint".into()),
                        }),
                    ),
                    ("seed", V::Count(seed)),
                    ("channel", V::Count(goal)),
                    ("observation", V::Count(id)),
                    ("predicted", number(&r[3])?),
                    ("actual", number(&r[4])?),
                    ("phase", V::Text(&r[7])),
                    ("actions", V::Count(r[6].parse()?)),
                    ("correct", V::Count(totals[0])),
                    ("scored", V::Count(totals[1])),
                    ("physical_correct", V::Count(totals[2])),
                    ("baseline_correct", V::Count(totals[3])),
                    ("drifts", V::Count(totals[4])),
                    ("queries", V::Count(totals[5])),
                    ("peak_bytes", V::Count(totals[6])),
                    ("peak_modules", V::Count(totals[7])),
                ],
            )?;
            if &r[0] == "result" {
                result = Some(totals);
            }
        }
        if !child.0.wait()?.success() {
            return Err("child replay failed; keep all partial evidence".into());
        }
        let r = result.ok_or("child omitted final checkpoint")?;
        // Recompute every policy label independently from the input table after child inference.
        let mut inputs = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(root.join(&input))?;
        let mut outputs = csv::Reader::from_path(root.join(&output))?;
        let mut previous = 0;
        let mut verified = 0;
        let mut checked = 0;
        let mut physical_verified = 0;
        let mut baseline_verified = 0;
        let input_rows = inputs.records().collect::<Result<Vec<_>, _>>()?;
        let output_rows = outputs.records().collect::<Result<Vec<_>, _>>()?;
        if input_rows.len() != output_rows.len() || input_rows.len() != 2 * count {
            return Err("replay output cardinality".into());
        }
        for (a, b) in input_rows.iter().zip(&output_rows) {
            let goal = a[1].parse::<usize>()?;
            let expected = expectation(goal, [a[4].parse()?, a[5].parse()?], &mut previous);
            if a[0] != b[0] || a[1] != b[1] || expected != b[3].parse::<i64>()? {
                return Err("replay provenance/feedback mismatch".into());
            }
            verified += u64::from(b[2].parse::<i64>().ok() == Some(expected));
            baseline_verified += u64::from(b[4].parse::<i64>().ok() == Some(expected));
            let reached = &b[7] == "Reached" && b[9].parse::<i64>().ok() == Some(expected);
            if reached != b[6].parse::<bool>()? {
                return Err("device receipt differs from physical score".into());
            }
            physical_verified += u64::from(reached);
            checked += 1;
        }
        if [verified, checked, physical_verified, baseline_verified] != r[..4] {
            return Err("checkpoint totals differ from full replay output".into());
        }
        correct += verified;
        total += checked;
        physical += physical_verified;
        baseline += baseline_verified;
        drifts += r[4];
        queries += r[5];
        combined.push_str(&format!(
            "{seed},{verified},{checked},{physical_verified},{baseline_verified},{},{},{},{}\n",
            r[4], r[5], r[6], r[7]
        ));
    }
    create(&root.join("results.csv"), &combined)?;
    let summary = format!(
        "AXON COMPOSITIONAL GOAL CONTROLLER\nvalidation={validation}\npolicy_correct={correct}\nevents={total}\nphysical_correct={physical}\nfixed_route_correct={baseline}\ndetected_device_changes={drifts}\nrelearning_queries={queries}\nNumerical bootstrap work={} reactive work={} device queries={} device steps={}\nKnown-family synthetic streams and resettable simulated device, not AGI or physical robotics. Every drift miss is retained; no decision retry.\n",
        bootstrap.numeric_work,
        bootstrap.reactive_work,
        bootstrap.device_queries,
        bootstrap.device_steps
    );
    create(&root.join("summary.txt"), &summary)?;
    let passed = correct == total
        && physical + drifts == total
        && drifts == if validation == "reserved" { 3 } else { 1 };
    telemetry.emit(
        "lab_result",
        &[
            ("passed", V::Flag(passed)),
            ("correct", V::Count(correct)),
            ("heldout", V::Count(total)),
            ("physical_correct", V::Count(physical)),
            ("baseline_correct", V::Count(baseline)),
            ("drifts", V::Count(drifts)),
            ("queries", V::Count(queries)),
        ],
    )?;
    println!("{summary}");
    if !passed {
        return Err("controller protocol failed; preserve all results, no replacement seed".into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|a| a == "replay") {
        replay(&args)
    } else {
        run(&args)
    }
}
