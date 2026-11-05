#[allow(dead_code)]
mod controller_support;
use axon_uic::{
    Authority, Effect,
    behavior_lab::Rng,
    demand::{controller::*, genome::*, *},
    numeric::seed::{SeedScore, pareto_front},
    telemetry::{LabTelemetry, Value as V},
    text_skill::read_bounded,
};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

const GOALS: [&str; 8] = [
    "study/a", "study/b", "study/c", "study/d", "study/e", "study/f", "study/g", "study/h",
];
const RESERVES: [u64; 3] = [80127491, 25609387, 61432871];
const DEV: u64 = 84371921;
type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn authority() -> Authority {
    Authority::from([Effect::ReadData, Effect::WriteData])
}
fn file(path: &Path, text: &str) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(text.as_bytes())?;
    f.sync_all()?;
    Ok(())
}
fn output(goal: usize, x: i64, y: i64, prior: &mut i64) -> i64 {
    let current = match goal {
        0 | 2 => x + y,
        1 | 3 => x * y,
        4 | 6 => i64::from(x + y < 20),
        5 | 7 => i64::from(x * y < 20),
        _ => unreachable!(),
    };
    if [2, 3, 6, 7].contains(&goal) {
        let old = *prior;
        *prior = current;
        old
    } else {
        current
    }
}
fn examples(goal: usize, seed: u64, traces: usize, rows: usize, wide: bool) -> Vec<GoalTrace> {
    let mut rng = Rng::new(seed);
    (0..traces)
        .map(|trace| {
            let mut prior = 0;
            (0..rows)
                .map(|i| {
                    let (x, y) = match i % 16 {
                        0 => (0, 0),
                        1 => (1, 1),
                        2 => (4, 5),
                        3 => (9, 0),
                        4 => (20, 1),
                        5 => (-3, 4),
                        _ if rng.pick(2) == 0 => (rng.pick(12) as i64 - 4, rng.pick(12) as i64 - 4),
                        _ => {
                            let n = if wide { 385 } else { 41 };
                            (
                                rng.pick(n) as i64 - n as i64 / 3,
                                rng.pick(n) as i64 - n as i64 / 3,
                            )
                        }
                    };
                    GoalExample {
                        input: format!(
                            "invoice-{}|{x},{y}",
                            seed.wrapping_add((trace * rows + i) as u64)
                        ),
                        output: output(goal, x, y, &mut prior),
                    }
                })
                .collect()
        })
        .collect()
}
fn lessons(path: &Path, targets: &[usize], training: u64, admission: u64) -> Result<()> {
    let mut w = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    );
    w.write_record(["goal", "phase", "trace", "input", "output"])?;
    for &goal in targets {
        for (phase, seed, n) in [("train", training, 3), ("admit", admission, 2)] {
            for (t, rows) in examples(goal, seed.wrapping_add(goal as u64 * 191), n, 16, false)
                .iter()
                .enumerate()
            {
                for row in rows {
                    w.write_record([
                        GOALS[goal],
                        phase,
                        &t.to_string(),
                        &row.input,
                        &row.output.to_string(),
                    ])?;
                }
            }
        }
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
}
fn score_input(path: &Path, targets: &[usize], seed: u64, count: usize) -> Result<()> {
    let mut w = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    );
    w.write_record(["goal", "input", "output"])?;
    for &goal in targets {
        for row in &examples(goal, seed.wrapping_add(goal as u64 * 197), 1, count, true)[0] {
            w.write_record([GOALS[goal], &row.input, &row.output.to_string()])?;
        }
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(())
}
fn rows(path: &Path, header: &[&str]) -> Result<Vec<csv::StringRecord>> {
    let b = read_bounded(path, 16_777_216)?;
    let mut r = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(b.as_slice());
    if r.headers()?.iter().collect::<Vec<_>>() != header {
        return Err("invalid genome lab schema".into());
    }
    let rows = r.records().collect::<std::result::Result<Vec<_>, _>>()?;
    if rows.len() > 80000 {
        return Err("genome lab row quota".into());
    }
    Ok(rows)
}
fn worker(args: &[String]) -> Result<()> {
    if args.len() != 6 {
        return Err("worker <number> <new-root> <lessons.tsv> <score.tsv> <new-number>".into());
    }
    let auth = authority();
    let g = Genome::open_number(Path::new(&args[1]), &auth)?;
    let mut birth = g.birth(Path::new(&args[2]), BirthLimits::default(), &auth)?;
    let lesson_rows = rows(
        Path::new(&args[3]),
        &["goal", "phase", "trace", "input", "output"],
    )?;
    let mut targets: BTreeMap<String, [BTreeMap<usize, GoalTrace>; 2]> = BTreeMap::new();
    for r in lesson_rows {
        let phase = match &r[1] {
            "train" => 0,
            "admit" => 1,
            _ => return Err("lesson phase".into()),
        };
        let t = r[2].parse::<usize>()?;
        if targets.len() > 8 || t >= 8 {
            return Err("lesson quota".into());
        }
        targets.entry(r[0].into()).or_default()[phase]
            .entry(t)
            .or_default()
            .push(GoalExample {
                input: r[3].into(),
                output: r[4].parse()?,
            });
    }
    if targets.is_empty() || targets.len() > 8 {
        return Err("lesson goals".into());
    }
    let mut learned = Vec::new();
    let mut work = 0;
    let mut attempts = 0;
    let mut acquired = 0;
    for (context, traces) in targets {
        let [train, admit] = traces;
        let run = Procedure::discover(
            &context,
            train.into_values().collect(),
            admit.into_values().collect(),
            birth.controller.runtime_mut(),
            &auth,
            g.composition(),
        )?;
        work += run.statistics.work;
        attempts += 1;
        if let Some(p) = run.procedure {
            acquired += 1;
            learned.push(p);
        }
    }
    let child = g.consolidate(&learned, birth.controller.runtime_mut(), &auth)?;
    child.save_number_new(Path::new(&args[5]), &auth)?;
    let mut runtime = DemandRuntime::new(
        Catalog::load(Path::new(&args[2]).join("birth.axseed"), &auth)?,
        g.residency(),
    )?;
    let procedures = child.procedures()?;
    let mut states = procedures.iter().map(Procedure::cursor).collect::<Vec<_>>();
    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut out = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(Path::new(&args[2]).join("predictions.tsv"))?,
    );
    out.write_record(["goal", "input", "prediction", "actual", "correct"])?;
    let mut peak_bytes = 0;
    let mut peak_modules = 0;
    for row in rows(Path::new(&args[4]), &["goal", "input", "output"])? {
        let i = procedures.iter().position(|p| p.context() == &row[0]);
        // Predict before parsing the external score. Labels never choose the procedure.
        let predicted = i.and_then(|i| {
            procedures[i]
                .predict(&row[0], &mut runtime, &mut states[i], &row[1], &auth)
                .ok()
        });
        let actual = row[2].parse::<i64>()?;
        let correct = predicted == Some(actual);
        let count = counts.entry(row[0].into()).or_default();
        count.0 += usize::from(correct);
        count.1 += 1;
        out.write_record([
            &row[0],
            &row[1],
            &predicted.map_or(String::new(), |v| v.to_string()),
            &row[2],
            if correct { "1" } else { "0" },
        ])?;
        peak_bytes = peak_bytes.max(runtime.stats().active_bytes);
        peak_modules = peak_modules.max(runtime.active_ids().len());
    }
    out.flush()?;
    out.get_ref().sync_all()?;
    let mut mask = 0u64;
    let mut correct = 0;
    let mut scored = 0;
    for (context, (n, total)) in counts {
        correct += n;
        scored += total;
        if n == total
            && total > 0
            && let Some(i) = GOALS.iter().position(|g| *g == context)
        {
            mask |= 1 << i;
        }
    }
    let stages: usize = procedures.iter().map(|p| p.steps().len()).sum();
    println!(
        "{mask}\t{work}\t{}\t{stages}\t{attempts}\t{acquired}\t{correct}\t{scored}\t{peak_bytes}\t{peak_modules}",
        child.encoded_bytes()
    );
    Ok(())
}
#[derive(Clone)]
struct Candidate {
    id: usize,
    seed: PathBuf,
    score: SeedScore,
    depth: usize,
}
fn invoke(
    number: &Path,
    root: &Path,
    lessons: &Path,
    scores: &Path,
    child: &Path,
) -> Result<[u64; 10]> {
    let result = Command::new(std::env::current_exe()?)
        .arg("worker")
        .arg(number)
        .arg(root)
        .arg(lessons)
        .arg(scores)
        .arg(child)
        .output()?;
    if !result.status.success() {
        return Err(format!("worker failed: {}", String::from_utf8_lossy(&result.stderr)).into());
    }
    let text = String::from_utf8(result.stdout)?;
    let values = text
        .trim()
        .split('\t')
        .map(str::parse::<u64>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    values
        .try_into()
        .map_err(|_| "invalid worker summary".into())
}
fn audit(scores: &Path, predictions: &Path) -> Result<[u64; 3]> {
    let expected = rows(scores, &["goal", "input", "output"])?;
    let observed = rows(
        predictions,
        &["goal", "input", "prediction", "actual", "correct"],
    )?;
    if expected.len() != observed.len() {
        return Err("worker result length".into());
    }
    let mut counts = BTreeMap::<usize, (u64, u64)>::new();
    for (a, b) in expected.iter().zip(&observed) {
        let actual = a[2].parse::<i64>()?;
        let correct = b[2].parse::<i64>().ok() == Some(actual);
        if a[0] != b[0] || a[1] != b[1] || a[2] != b[3] || &b[4] != if correct { "1" } else { "0" }
        {
            return Err("worker result audit".into());
        }
        let goal = GOALS
            .iter()
            .position(|g| *g == &a[0])
            .ok_or("unknown scored goal")?;
        let c = counts.entry(goal).or_default();
        c.0 += u64::from(correct);
        c.1 += 1;
    }
    let mut mask = 0;
    let mut correct = 0;
    let mut scored = 0;
    for (goal, (n, total)) in counts {
        correct += n;
        scored += total;
        if n == total && total > 0 {
            mask |= 1 << goal;
        }
    }
    Ok([mask, correct, scored])
}
fn parent(args: &[String]) -> Result<()> {
    if !(4..=6).contains(&args.len()) {
        return Err("<new-root> <rng> <count> <development|reserved> [members] [rounds]".into());
    }
    let root = Path::new(&args[0]);
    let random = args[1].parse::<u64>()?;
    let count = args[2].parse::<usize>()?;
    let reserved = match args[3].as_str() {
        "development" => false,
        "reserved" => true,
        _ => return Err("protocol".into()),
    };
    let members = args.get(4).map_or(Ok(32), |s| s.parse::<usize>())?;
    let rounds = args.get(5).map_or(Ok(8), |s| s.parse::<usize>())?;
    if !(100..=10000).contains(&count) || !(2..=32).contains(&members) || !(1..=8).contains(&rounds)
    {
        return Err("cohort quota".into());
    }
    let mut telemetry = LabTelemetry::from_environment("common-genome-v1");
    let auth = authority();
    fs::create_dir(root)?;
    file(
        &root.join("protocol.txt"),
        include_str!("../../evals/protocols/common-genome-v3.txt"),
    )?;
    let boot_auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    let bootstrap = controller_support::bootstrap(&root.join("bootstrap"), random, &boot_auth)?;
    let mut runtime = controller_support::runtime(
        controller_support::catalog(&root.join("bootstrap"), 0, &auth)?,
        4,
    )?;
    let original = Genome::capture(
        "axon/common-genome-v1",
        &mut runtime,
        CompositionBudget {
            max_steps: 4,
            ..CompositionBudget::default()
        },
        &[],
        65536,
        &auth,
    )?;
    original.save_number_new(&root.join("founder.axnum"), &auth)?;
    telemetry.emit(
        "stage",
        &[
            ("method", V::Text("Genome::capture")),
            ("protocol", V::Text("common-genome-v3")),
            ("modules", V::Count(original.module_count() as u64)),
            ("capsule_bytes", V::Count(original.encoded_bytes() as u64)),
            (
                "work",
                V::Count(bootstrap.numeric_work + bootstrap.reactive_work),
            ),
            ("queries", V::Count(bootstrap.device_queries as u64)),
        ],
    )?;
    let teaching = root.join("development-lessons.tsv");
    let scoring = root.join("development-score.tsv");
    lessons(
        &teaching,
        &[0, 1, 2, 3, 4, 5],
        random.wrapping_add(191),
        random.wrapping_add(7919),
    )?;
    score_input(
        &scoring,
        &[0, 1, 2, 3, 4, 5],
        random.wrapping_add(71279),
        32,
    )?;
    let mut rng = Rng::new(random);
    let mut parents = Vec::<Candidate>::new();
    let mut all = Vec::new();
    let mut attempts = 0;
    let mut acquisitions = 0;
    let mut total_work = 0;
    let mut ledger = csv::Writer::from_writer(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("population.csv"))?,
    );
    ledger.write_record([
        "candidate",
        "round",
        "parent",
        "genome",
        "depth",
        "resident",
        "mask",
        "work",
        "bytes",
        "stages",
        "attempts",
        "acquired",
        "correct",
        "scored",
    ])?;
    for round in 0..rounds {
        let mut cohort = Vec::new();
        for slot in 0..members {
            let id = round * members + slot;
            let ancestor = if parents.is_empty() {
                original.clone()
            } else {
                Genome::open_number(&parents[rng.pick(parents.len())].seed, &auth)?
            };
            let g = ancestor.offspring(rng.next_u64())?;
            let initial = root.join(format!("initial-{id}.axnum"));
            g.save_number_new(&initial, &auth)?;
            telemetry.emit(
                "seed_candidate",
                &[
                    ("method", V::Text("Genome::offspring")),
                    ("candidate", V::Count(id as u64)),
                    ("generation", V::Count(round as u64)),
                    ("genotype", V::Text(&format!("{:016X}", g.fingerprint()))),
                    ("parent_a", V::Text(&format!("{:016X}", g.parent()))),
                ],
            )?;
            let seed = root.join(format!("candidate-{id}.axnum"));
            let dir = root.join(format!("birth-{id}"));
            let [
                mask,
                work,
                bytes,
                stages,
                trials,
                acquired,
                correct,
                scored,
                peak_bytes,
                peak_modules,
            ] = invoke(&initial, &dir, &teaching, &scoring, &seed)?;
            attempts += trials;
            acquisitions += acquired;
            total_work += work;
            // Audit complete worker results against the parent's labelled score, not only totals.
            if audit(&scoring, &dir.join("predictions.tsv"))? != [mask, correct, scored] {
                return Err("worker result audit".into());
            }
            let child = Genome::open_number(&seed, &auth)?;
            ledger.write_record([
                id.to_string(),
                round.to_string(),
                format!("{:016X}", g.parent()),
                format!("{:016X}", child.fingerprint()),
                g.composition().max_steps.to_string(),
                g.residency().active_modules.to_string(),
                mask.to_string(),
                work.to_string(),
                bytes.to_string(),
                stages.to_string(),
                trials.to_string(),
                acquired.to_string(),
                correct.to_string(),
                scored.to_string(),
            ])?;
            ledger.flush()?;
            let inherited = ancestor
                .procedures()?
                .iter()
                .filter_map(|p| GOALS.iter().position(|g| *g == p.context()))
                .fold(0u64, |mask, i| mask | (1 << i));
            let score = SeedScore {
                capabilities: mask,
                tasks: 6,
                retention_ok: mask & inherited == inherited,
                violations: 0,
                work,
                bytes,
                instructions: stages,
            };
            telemetry.emit(
                "seed_evaluation",
                &[
                    ("method", V::Text("Genome::birth")),
                    ("candidate", V::Count(id as u64)),
                    ("generation", V::Count(round as u64)),
                    ("parent_a", V::Text(&format!("{:016X}", g.parent()))),
                    ("artifact", V::Text(&format!("candidate-{id}.axnum"))),
                    (
                        "genotype",
                        V::Text(&format!("{:016X}", child.fingerprint())),
                    ),
                    ("solved", V::Count(mask.count_ones() as u64)),
                    ("tasks", V::Count(6)),
                    ("modules", V::Count(child.procedure_count() as u64)),
                    ("capsule_bytes", V::Count(bytes)),
                    ("work", V::Count(work)),
                    ("retention_ok", V::Flag(score.retention_ok)),
                    ("peak_bytes", V::Count(peak_bytes)),
                    ("peak_modules", V::Count(peak_modules)),
                ],
            )?;
            let c = Candidate {
                id,
                seed,
                score,
                depth: g.composition().max_steps,
            };
            cohort.push(c.clone());
            all.push(c);
        }
        let front = pareto_front(&cohort.iter().map(|c| c.score).collect::<Vec<_>>())?;
        if front.is_empty() {
            return Err("no viable genomes".into());
        }
        let mut keep = Vec::new();
        // Preserve policy niches using development-only scores, without reserve feedback.
        for depth in 4..=6 {
            if let Some((i, _)) = cohort
                .iter()
                .enumerate()
                .filter(|(_, c)| c.depth == depth && c.score.feasible())
                .max_by_key(|(_, c)| {
                    (
                        c.score.capabilities.count_ones(),
                        std::cmp::Reverse(c.score.work),
                    )
                })
                && !keep.contains(&i)
            {
                keep.push(i);
            }
        }
        let mut frontier = front;
        frontier.sort_by_key(|&i| {
            (
                std::cmp::Reverse(cohort[i].score.capabilities.count_ones()),
                cohort[i].score.work,
                cohort[i].score.bytes,
                i,
            )
        });
        for i in frontier {
            if keep.len() == 8 {
                break;
            }
            if !keep.contains(&i) {
                keep.push(i);
            }
        }
        parents = keep.iter().map(|&i| cohort[i].clone()).collect();
        telemetry.emit(
            "seed_selection",
            &[
                ("method", V::Text("pareto_front")),
                ("generation", V::Count(round as u64)),
                (
                    "selected",
                    V::Counts(&parents.iter().map(|c| c.id as u64).collect::<Vec<_>>()),
                ),
            ],
        )?;
    }
    ledger.flush()?;
    ledger.get_ref().sync_all()?;
    parents.sort_by_key(|c| {
        (
            std::cmp::Reverse(c.score.capabilities.count_ones()),
            c.score.work,
            c.score.bytes,
            c.id,
        )
    });
    let mut frozen_parents = Vec::<Candidate>::new();
    for depth in 4..=6 {
        if let Some(c) = parents.iter().find(|c| c.depth == depth) {
            frozen_parents.push(c.clone());
        }
    }
    for c in &parents {
        if frozen_parents.len() == 4 {
            break;
        }
        if !frozen_parents.iter().any(|kept| kept.id == c.id) {
            frozen_parents.push(c.clone());
        }
    }
    parents = frozen_parents;
    let mut frozen = String::from("candidate,genome\n");
    for c in &parents {
        frozen.push_str(&format!(
            "{},{:016X}\n",
            c.id,
            Genome::open_number(&c.seed, &auth)?.fingerprint()
        ));
    }
    file(&root.join("frozen.csv"), &frozen)?;
    let transfer = root.join("transfer-lessons.tsv");
    lessons(&transfer, &[6, 7], 17094361, 49721833)?;
    let seeds = if reserved {
        RESERVES.to_vec()
    } else {
        vec![DEV.wrapping_add(919)]
    };
    let mut correct = 0;
    let mut scored = 0;
    let mut coverage = 0;
    let mut transfer_attempts = 0;
    let mut result = csv::Writer::from_writer(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("results.csv"))?,
    );
    result.write_record([
        "candidate",
        "input_seed",
        "coverage_mask",
        "correct",
        "scored",
        "work",
        "peak_bytes",
        "peak_modules",
    ])?;
    for c in &parents {
        for &seed in &seeds {
            let scores = root.join(format!("reserve-{seed}.tsv"));
            if !scores.exists() {
                score_input(&scores, &[0, 1, 2, 3, 4, 5, 6, 7], seed, count)?;
            }
            let dir = root.join(format!("transfer-{}-{seed}", c.id));
            let number = root.join(format!("transfer-{}-{seed}.axnum", c.id));
            let [
                mask,
                work,
                _,
                _,
                trials,
                _,
                n,
                total,
                peak_bytes,
                peak_modules,
            ] = invoke(&c.seed, &dir, &transfer, &scores, &number)?;
            if audit(&scores, &dir.join("predictions.tsv"))? != [mask, n, total] {
                return Err("reserve summary audit".into());
            }
            if seed == seeds[0] {
                for p in Genome::open_number(&number, &auth)?.procedures()? {
                    let goal = GOALS
                        .iter()
                        .position(|g| *g == p.context())
                        .ok_or("procedure goal")?;
                    let names = p
                        .steps()
                        .iter()
                        .map(|step| match step {
                            ProcedureStep::Module { id, .. } => id.clone(),
                            ProcedureStep::DecodeScalars => "decode_scalars".into(),
                            ProcedureStep::ExactInteger => "exact_integer".into(),
                        })
                        .collect::<Vec<_>>();
                    let refs = names.iter().map(String::as_str).collect::<Vec<_>>();
                    telemetry.emit(
                        "skill",
                        &[
                            ("method", V::Text("Genome::consolidate")),
                            ("case", V::Count((c.id * 8 + goal) as u64)),
                            ("name", V::Text(&format!("#{} / {}", c.id, p.context()))),
                            ("steps", V::Texts(&refs)),
                            ("capsule_bytes", V::Count(p.to_bytes().len() as u64)),
                            ("status", V::Text("Reopened")),
                        ],
                    )?;
                }
            }
            correct += n;
            scored += total;
            coverage += u64::from((mask & 192) == 192);
            transfer_attempts += trials;
            result.write_record([
                c.id.to_string(),
                seed.to_string(),
                mask.to_string(),
                n.to_string(),
                total.to_string(),
                work.to_string(),
                peak_bytes.to_string(),
                peak_modules.to_string(),
            ])?;
            telemetry.emit(
                "validation",
                &[
                    ("method", V::Text("GenomeTransfer::audit")),
                    ("case", V::Count(c.id as u64)),
                    ("input_seed", V::Count(seed)),
                    ("correct", V::Count(n)),
                    ("scored", V::Count(total)),
                    (
                        "status",
                        V::Text(if mask == 255 { "Passed" } else { "Partial" }),
                    ),
                    ("coverage_mask", V::Count(mask)),
                    ("peak_bytes", V::Count(peak_bytes)),
                    ("peak_modules", V::Count(peak_modules)),
                ],
            )?;
        }
    }
    result.flush()?;
    result.get_ref().sync_all()?;
    let summary = format!(
        "AXON COMMON GENOME\nvalidation={}\ncandidates={}\ncomposition_attempts={attempts}\nadmitted_compositions={acquisitions}\nsearch_work={total_work}\nfrozen_survivors={}\ntransfer_attempts={transfer_attempts}\nfull_transfer_workers={coverage}\ncorrect={correct}\nscored={scored}\nNo source module paths were passed to newborn workers. Known synthetic families, not AGI.\n",
        args[3],
        all.len(),
        parents.len()
    );
    file(&root.join("summary.txt"), &summary)?;
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("GenomeLab::summary")),
            (
                "status",
                V::Text(if correct == scored {
                    "Passed"
                } else {
                    "Partial"
                }),
            ),
            ("correct", V::Count(correct)),
            ("scored", V::Count(scored)),
            ("attempts", V::Count(attempts)),
            ("acquired", V::Count(acquisitions)),
            ("work", V::Count(total_work)),
            ("survivors", V::Count(parents.len() as u64)),
            ("candidates", V::Count(all.len() as u64)),
            (
                "solved",
                V::Count(
                    all.iter()
                        .map(|c| c.score.capabilities.count_ones())
                        .max()
                        .unwrap_or(0) as u64,
                ),
            ),
            ("tasks", V::Count(6)),
        ],
    )?;
    print!("{summary}");
    Ok(())
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let result = if args.first().is_some_and(|s| s == "worker") {
        worker(&args)
    } else {
        parent(&args)
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
