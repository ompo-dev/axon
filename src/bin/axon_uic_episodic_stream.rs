mod episodic_support;

use axon_uic::{
    ArtifactOutcome, ArtifactStore, AxonTask, BeliefStatus, EpisodeData, EpisodeStore,
    EpisodeValue, ExactAverage, ExactPopulationVariance, ExperiencedSolveError,
    solve_task_with_experience,
};
use episodic_support::{Rng, generate, reference};
use std::{
    fs::{self, OpenOptions},
    hint::black_box,
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

fn main() {
    if let Err(error) = run() {
        eprintln!("episodic stream: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut seed = 20261012_u64;
    let mut count = 1536_usize;
    let mut queries = 20_000_usize;
    let mut output = None::<PathBuf>;
    let mut training = None::<PathBuf>;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "axon-uic-episodic-stream [--seed u64] [--episodes 64..4096] [--queries 100..100000] --output new.json [--train-journal new.axep]"
            );
            return Ok(());
        }
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--seed" => seed = value.parse()?,
            "--episodes" => count = value.parse()?,
            "--queries" => queries = value.parse()?,
            "--output" if output.is_none() => output = Some(value.into()),
            "--train-journal" if training.is_none() => training = Some(value.into()),
            _ => return Err(format!("unknown or duplicate option {flag}").into()),
        }
    }
    if !(64..=4096).contains(&count) || !(100..=100_000).contains(&queries) {
        return Err("workload outside bounded domain".into());
    }
    let output = output.ok_or("--output is required")?;
    if training.as_ref().is_some_and(|path| path.exists()) {
        return Err("training journal must be new; existing experiences are preserved".into());
    }
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    let root = std::env::temp_dir().join(format!(
        "axon-episodic-benchmark-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir(&root)?;
    let result = evaluate(&root, seed, count, queries, training);
    // All files in this directory belong to this run; the persistent journal is outside it.
    fs::remove_dir_all(&root)?;
    let json = result?;
    report.write_all(json.as_bytes())?;
    report.sync_all()?;
    println!("{json}");
    Ok(())
}

fn evaluate(
    root: &std::path::Path,
    seed: u64,
    count: usize,
    queries: usize,
    training: Option<PathBuf>,
) -> Result<String, Box<dyn std::error::Error>> {
    let started_at = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let path = root.join("stream.axep");
    let mut store = EpisodeStore::open(&path)?.0;
    let mut rng = Rng(seed);
    let start = Instant::now();
    let events = generate(&mut store, count, &mut rng);
    let append_ns = start.elapsed().as_nanos();
    let journal_bytes = store.file_bytes()?;
    let revisions = events
        .iter()
        .filter(|event| event.replaces.is_some() && event.value.is_some())
        .count();
    let retractions = events.iter().filter(|event| event.value.is_none()).count();
    drop(store);
    let start = Instant::now();
    let (store, recovery) = EpisodeStore::open(&path)?;
    let replay_ns = start.elapsed().as_nanos();
    assert_eq!(recovery.replayed_episodes, count);
    assert_eq!(recovery.discarded_tail_bytes, 0);
    episodic_support::compare(&store, &events, &rng.key(), u64::MAX, count as u64);
    let mut indexed_ns = 0_u128;
    let mut scan_ns = 0_u128;
    let mut examined = 0_u128;
    let mut statuses = [0_usize; 3];
    for i in 0..queries {
        let key = rng.key();
        let at = rng.next() % 100;
        let known = rng.next() % (count as u64 + 1);
        let (actual, expected);
        // Paired measurements alternate execution order to reduce systematic warm-cache bias.
        if i % 2 == 0 {
            let start = Instant::now();
            actual = black_box(store.belief(black_box(&key), Some(at), Some(known))?);
            indexed_ns += start.elapsed().as_nanos();
            let start = Instant::now();
            expected = black_box(reference(black_box(&events), &key, at, known));
            scan_ns += start.elapsed().as_nanos();
        } else {
            let start = Instant::now();
            expected = black_box(reference(black_box(&events), &key, at, known));
            scan_ns += start.elapsed().as_nanos();
            let start = Instant::now();
            actual = black_box(store.belief(black_box(&key), Some(at), Some(known))?);
            indexed_ns += start.elapsed().as_nanos();
        }
        examined += actual.examined_episodes as u128;
        statuses[match actual.status {
            BeliefStatus::Missing => 0,
            BeliefStatus::Consistent => 1,
            BeliefStatus::Conflict => 2,
        }] += 1;
        let mut evidence = actual
            .evidence
            .into_iter()
            .map(|e| {
                let EpisodeValue::Integer(value) = e.value else {
                    panic!("integer fixture");
                };
                (e.episode_id, e.source, e.effective_at, value)
            })
            .collect::<Vec<_>>();
        evidence.sort();
        assert_eq!(
            (actual.status, evidence),
            expected,
            "query {i}, seed {seed}"
        );
    }
    drop(store);
    let artifact_root = if let Some(path) = &training {
        path.with_extension("artifacts").join(".axon-artifacts")
    } else {
        root.join("artifacts")
    };
    if artifact_root.exists() {
        return Err("training artifact directory must be new".into());
    }
    let training_path = training.unwrap_or_else(|| root.join("solver.axep"));
    let mut experience = EpisodeStore::open(&training_path)?.0;
    let artifacts = ArtifactStore::open(&artifact_root);
    let hardware =
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| std::env::consts::ARCH.into());
    let build = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let context = format!("solver-5-items/{build}/{hardware}");
    let goals = [
        "AddModU64",
        "AverageExactU64",
        "MinU64",
        "PopulationVarianceExactU64",
    ];
    let mut verified = 0;
    for round in 0..25_u64 {
        for (operator, goal) in goals.iter().enumerate() {
            let values = [round, round + 2, round + 5, round + 7, round + 11];
            let task = AxonTask::parse(&format!(
                "task measured_{operator} {{\n data numbers: Vec<u64> = [{}, {}, {}, {}, {}]\n goal derive IncrementalArtifact<{goal}>\n}}",
                values[0], values[1], values[2], values[3], values[4]
            ))?;
            let expected = match operator {
                0 => ArtifactOutcome::ModularSum(5 * round + 25),
                1 => {
                    ArtifactOutcome::ExactAverage(ExactAverage::new(u128::from(5 * round + 25), 5)?)
                }
                2 => ArtifactOutcome::ExactMinimum(round),
                _ => {
                    ArtifactOutcome::ExactPopulationVariance(ExactPopulationVariance::new(370, 25)?)
                }
            };
            let receipt =
                solve_task_with_experience(&artifacts, &task, &mut experience, &context)?.0;
            assert_eq!(*receipt.outcome(), expected);
            verified += 1;
        }
    }
    let rejected_context = format!("solver-invalid-input/{build}/{hardware}");
    let mut rejections = 0;
    for (i, goal) in goals.iter().enumerate().skip(1) {
        let task = AxonTask::parse(&format!(
            "task rejected_{i} {{\n data numbers: Vec<u64> = []\n goal derive IncrementalArtifact<{goal}>\n}}"
        ))?;
        match solve_task_with_experience(&artifacts, &task, &mut experience, &rejected_context) {
            Err(ExperiencedSolveError::Solve { .. }) => rejections += 1,
            _ => return Err("empty input did not yield a logged guard failure".into()),
        }
    }
    let mut actions_json = Vec::new();
    for ctx in [&context, &rejected_context] {
        for (action, summary) in experience.experience(ctx)? {
            // These internally generated fields contain no control characters; Debug quotes strings.
            actions_json.push(format!("{{\"context\":{ctx:?},\"action\":{action:?},\"observations\":{},\"successes\":{},\"total_cost_ns\":{}}}", summary.observations, summary.successes, summary.total_cost_ns));
        }
    }
    let trained_episodes = experience.len();
    let training_bytes = experience.file_bytes()?;
    drop(experience);
    let start = Instant::now();
    let (experience, recovery) = EpisodeStore::open(&training_path)?;
    let training_replay_ns = start.elapsed().as_nanos();
    assert_eq!(recovery.replayed_episodes, verified + rejections);
    assert_eq!(
        experience
            .experience(&context)?
            .iter()
            .map(|(_, s)| s.observations)
            .sum::<u64>(),
        verified as u64
    );
    for id in 1..=trained_episodes as u64 {
        assert!(matches!(
            experience.episode(id)?.data(),
            EpisodeData::Outcome { .. }
        ));
    }
    Ok(format!(
        "{{\"schema\":\"axon-episodic-stream-v1\",\"started_at_unix\":{started_at},\"seed\":{seed},\"build\":{build:?},\"hardware\":{hardware:?},\"episodes\":{count},\"revisions\":{revisions},\"retractions\":{retractions},\"queries\":{queries},\"exact_matches\":{queries},\"missing_consistent_conflict\":{statuses:?},\"journal_bytes\":{journal_bytes},\"append_ns\":{append_ns},\"replay_ns\":{replay_ns},\"indexed_query_ns\":{indexed_ns},\"full_scan_query_ns\":{scan_ns},\"indexed_examined_total\":{examined},\"full_scan_examined_total\":{},\"solver_verified\":{verified},\"solver_rejected\":{rejections},\"solver_episodes\":{trained_episodes},\"solver_journal_bytes\":{training_bytes},\"solver_replay_ns\":{training_replay_ns},\"solver_actions\":[{}]}}",
        count as u128 * queries as u128,
        actions_json.join(",")
    ))
}
