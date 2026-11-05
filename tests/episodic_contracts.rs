#[path = "../src/bin/episodic_support.rs"]
mod support;

use axon_uic::{
    ArtifactStore, AxonTask, BeliefStatus, EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue,
    ExperiencedSolveError, FactKey, cli, solve_task_with_experience,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use support::{Rng, compare, generate};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "axon-episode-contract-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
    fn journal(&self) -> PathBuf {
        self.0.join("experience.axep")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn command(args: &[&str], path: &std::path::Path) -> Result<String, cli::CliError> {
    let mut args = args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    args.extend(["--episodes".into(), path.to_str().unwrap().into()]);
    cli::command(&args)
}

#[test]
fn generated_temporal_histories_match_an_independent_full_scan_before_and_after_replay() {
    let temp = Temp::new();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let mut rng = Rng(0xabc_2026_1012);
    let events = generate(&mut store, 512, &mut rng);
    for _ in 0..2_000 {
        let key = rng.key();
        compare(&store, &events, &key, rng.next() % 100, rng.next() % 513);
    }
    drop(store);
    let (store, report) = EpisodeStore::open(temp.journal()).unwrap();
    assert_eq!(report.replayed_episodes, 512);
    for _ in 0..2_000 {
        let key = rng.key();
        compare(&store, &events, &key, rng.next() % 100, rng.next() % 513);
    }
}

#[test]
fn unrelated_contexts_and_facts_do_not_expand_the_examined_slice() {
    let temp = Temp::new();
    let mut store = EpisodeStore::open(temp.journal()).unwrap().0;
    let key = FactKey::new("tank", "liters", "north");
    store
        .observe(
            EpisodeStamp::new("meter", "1", 1),
            key.clone(),
            EpisodeValue::Integer(10),
        )
        .unwrap();
    for i in 0..256 {
        store
            .observe(
                EpisodeStamp::new("meter", format!("other-{i}"), 2),
                FactKey::new(format!("tank-{i}"), "liters", "south"),
                EpisodeValue::Integer(99),
            )
            .unwrap();
    }
    let result = store.belief(&key, None, None).unwrap();
    assert_eq!(result.status, BeliefStatus::Consistent);
    assert_eq!(result.examined_episodes, 1);
    assert_eq!(result.evidence[0].value, EpisodeValue::Integer(10));
    assert_eq!(
        store
            .belief(&FactKey::new("tank", "liters", "south"), None, None)
            .unwrap()
            .status,
        BeliefStatus::Missing
    );
}

#[test]
fn cli_preserves_conflicts_and_historical_corrections_with_typed_values() {
    let temp = Temp::new();
    let path = temp.journal();
    for (source, n) in [("a", "10"), ("b", "11")] {
        command(
            &[
                "observe",
                "tank",
                "liters",
                n,
                "--type",
                "int",
                "--context",
                "lab",
                "--source",
                source,
                "--event",
                "reading",
                "--at",
                "20",
            ],
            &path,
        )
        .unwrap();
    }
    let query = ["belief", "tank", "liters", "--context", "lab"];
    assert!(command(&query, &path).unwrap().contains("status=conflict"));
    command(
        &[
            "revise",
            "2",
            "10",
            "--type",
            "int",
            "--source",
            "b",
            "--event",
            "correction",
        ],
        &path,
    )
    .unwrap();
    assert!(
        command(&query, &path)
            .unwrap()
            .contains("status=consistent")
    );
    assert!(
        command(
            &[
                "belief",
                "tank",
                "liters",
                "--context",
                "lab",
                "--known-at",
                "2"
            ],
            &path
        )
        .unwrap()
        .contains("status=conflict")
    );
    assert!(
        command(
            &["episode-history", "tank", "liters", "--context", "lab"],
            &path
        )
        .unwrap()
        .contains("replaces: Some(2)")
    );
    assert!(
        command(
            &[
                "observe",
                "relay",
                "enabled",
                "yes",
                "--type",
                "bool",
                "--context",
                "lab",
                "--source",
                "a",
                "--event",
                "bool",
                "--at",
                "20"
            ],
            &path
        )
        .is_err()
    );
    assert!(
        command(&["episodes-status"], &path)
            .unwrap()
            .contains("episode_count=3")
    );
}

#[test]
fn cli_rejects_ambiguous_options_and_missing_fields_before_creating_journal() {
    let temp = Temp::new();
    for args in [
        vec!["experience", "lab", "--episodes", "other"],
        vec!["belief", "x", "p"],
        vec!["episodes-status", "--unknown", "x"],
        vec!["retract", "zero", "--source", "a", "--event", "x"],
        vec![
            "observe",
            "x",
            "p",
            "v",
            "--at",
            "bad",
            "--context",
            "c",
            "--source",
            "s",
            "--event",
            "e",
        ],
    ] {
        assert!(command(&args, &temp.journal()).is_err());
        assert!(!temp.journal().exists());
    }
}

#[test]
fn actual_solver_experiences_record_created_reused_and_rejected_attempts() {
    let temp = Temp::new();
    let artifacts = ArtifactStore::open(temp.0.join("artifacts"));
    let mut episodes = EpisodeStore::open(temp.journal()).unwrap().0;
    let task = AxonTask::parse("task measured {\n data numbers: Vec<u64> = [2, 4, 9]\n goal derive IncrementalArtifact<AverageExactU64>\n}\n").unwrap();
    assert_eq!(
        solve_task_with_experience(&artifacts, &task, &mut episodes, "host-fixed/3-items")
            .unwrap()
            .1
            .id,
        1
    );
    solve_task_with_experience(&artifacts, &task, &mut episodes, "host-fixed/3-items").unwrap();
    let bad = AxonTask::parse("task empty {\n data numbers: Vec<u64> = []\n goal derive IncrementalArtifact<AverageExactU64>\n}\n").unwrap();
    assert!(
        matches!(solve_task_with_experience(&artifacts, &bad, &mut episodes, "host-fixed/0-items"), Err(ExperiencedSolveError::Solve { episode, .. }) if episode.id == 3)
    );
    assert_eq!(episodes.experience("host-fixed/3-items").unwrap().len(), 2);
    let EpisodeData::Outcome {
        success,
        detail,
        cost_ns,
        ..
    } = episodes.episode(1).unwrap().data()
    else {
        panic!()
    };
    assert!(*success && *cost_ns > 0);
    assert!(detail.contains("items=3;input_fnv64="));
    assert!(detail.contains("ExactAverage"));
    let rejection = episodes.experience("host-fixed/0-items").unwrap()[0].1;
    assert_eq!(rejection.observations, 1);
    assert_eq!(rejection.successes, 0);
    drop(episodes);
    assert_eq!(EpisodeStore::open(temp.journal()).unwrap().0.len(), 3);
}

#[test]
fn invalid_experience_metadata_blocks_solver_effects() {
    let temp = Temp::new();
    let root = temp.0.join("artifacts");
    let artifacts = ArtifactStore::open(&root);
    let mut episodes = EpisodeStore::open(temp.journal()).unwrap().0;
    let task = AxonTask::parse("task no_effect {\n data numbers: Vec<u64> = [1]\n goal derive IncrementalArtifact<AddModU64>\n}\n").unwrap();
    assert!(matches!(
        solve_task_with_experience(&artifacts, &task, &mut episodes, "invalid\ncontext"),
        Err(ExperiencedSolveError::Memory { executed: None, .. })
    ));
    assert!(!root.exists());
    assert!(episodes.is_empty());
}

#[test]
fn solve_cli_opt_in_records_success_and_guard_failure_without_changing_default() {
    let temp = Temp::new();
    let task = temp.0.join("task.axon");
    fs::write(&task, "task logged {\n data numbers: Vec<u64> = [1, 2]\n goal derive IncrementalArtifact<AverageExactU64>\n}\n").unwrap();
    let args = vec!["solve".to_owned(), task.to_str().unwrap().into()];
    assert!(!cli::command(&args).unwrap().contains("episode_id="));
    assert!(!temp.journal().exists());
    let mut logged = args.clone();
    logged.extend([
        "--episodes".into(),
        temp.journal().to_str().unwrap().into(),
        "--experience-context".into(),
        "fixed-host/2-items".into(),
    ]);
    assert!(cli::command(&logged).unwrap().contains("episode_id=1"));
    fs::write(&task, "task invalid {\n data numbers: Vec<u64> = []\n goal derive IncrementalArtifact<AverageExactU64>\n}\n").unwrap();
    assert!(
        cli::command(&logged)
            .unwrap_err()
            .to_string()
            .contains("failure recorded as episode 2")
    );
    let mut incomplete = args;
    incomplete.extend(["--episodes".into(), temp.journal().to_str().unwrap().into()]);
    assert!(cli::command(&incomplete).is_err());
}
