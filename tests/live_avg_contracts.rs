use axon_uic::{
    AverageStrategy, ChangeError, FoldSpec, LiveAverage, LiveAverageStore, LiveError, Replace,
    ReplaceDelta, SemanticArtifact,
};
use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

fn state(values: Vec<u64>) -> LiveAverage {
    LiveAverage::materialize(
        7,
        values,
        SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap(),
    )
    .unwrap()
}

#[test]
fn live_production_cost_excludes_audit_and_checks_overflow() {
    use std::time::Duration;
    let contract = axon_uic::LiveBenchContract {
        setup: Duration::from_nanos(1),
        steady: Duration::from_nanos(2),
        durability: Duration::from_nanos(3),
        recovery: Duration::from_nanos(4),
        audit: Duration::MAX,
    };
    assert_eq!(contract.production(), Ok(Duration::from_nanos(10)));
    assert!(
        axon_uic::LiveBenchContract {
            setup: Duration::MAX,
            ..contract
        }
        .production()
        .is_err()
    );
}

fn path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "axon-live-{}-{}.bin",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn fractions_extremes_and_ten_thousand_batches_remain_exact() {
    let mut live = state(vec![1, 2]);
    assert_eq!(live.query().numerator(), 3);
    assert_eq!(live.query().denominator(), 2);
    for event in 1..=10_000 {
        let delta =
            ReplaceDelta::try_new(vec![Replace::new(0, live.values()[0], u64::MAX - event)])
                .unwrap();
        live.apply(7, event - 1, event, &delta).unwrap();
        assert_eq!(live.query().numerator(), (u64::MAX - event) as u128 + 2);
    }
    assert_eq!(live.last_event(), 10_000);
    assert_eq!(live.version(), 10_000);
}

#[test]
fn invalid_batches_leave_the_entire_state_unchanged() {
    let mut live = state(vec![1, 2, 3]);
    let before = live.clone();
    let stale = ReplaceDelta::try_new(vec![Replace::new(0, 1, 5), Replace::new(2, 99, 4)]).unwrap();
    assert!(matches!(
        live.apply(7, 0, 1, &stale),
        Err(LiveError::Change(ChangeError::StaleOldValue(2)))
    ));
    assert_eq!(live, before);
    let valid = ReplaceDelta::try_new(vec![Replace::new(0, 1, 5)]).unwrap();
    assert!(live.apply(8, 0, 1, &valid).is_err());
    assert!(live.apply(7, 1, 1, &valid).is_err());
    assert!(live.apply(7, 0, 2, &valid).is_err());
    let out = ReplaceDelta::try_new(vec![Replace::new(0, 1, 8), Replace::new(9, 3, 7)]).unwrap();
    assert!(live.apply(7, 0, 1, &out).is_err());
    assert_eq!(live, before);
    live.apply(7, 0, 1, &valid).unwrap();
    let after = live.clone();
    assert!(live.apply(7, 0, 1, &valid).is_err());
    assert_eq!(live, after);
}

#[test]
fn strategy_switching_preserves_full_source_and_cache() {
    let mut live = state(vec![u64::MAX; 32]);
    for event in 1..=64 {
        let i = event as usize % 32;
        let delta = ReplaceDelta::try_new(vec![Replace::new(i, live.values()[i], event)]).unwrap();
        let strategy = if event % 2 == 0 {
            AverageStrategy::Full
        } else {
            AverageStrategy::Incremental
        };
        live.apply_with_strategy(event, &delta, strategy).unwrap();
        assert_eq!(
            live.query().numerator(),
            live.values().iter().map(|v| *v as u128).sum::<u128>()
        );
    }
}

#[test]
fn semantic_and_empty_guards_are_enforced() {
    let avg = SemanticArtifact::synthesize(FoldSpec::AverageExactU64).unwrap();
    assert!(LiveAverage::materialize(7, vec![], avg).is_err());
    let sum = SemanticArtifact::synthesize(FoldSpec::AddModU64).unwrap();
    assert!(LiveAverage::materialize(7, vec![1], sum).is_err());
    let mut live = state(vec![1]);
    live.apply(7, 0, 1, &ReplaceDelta::try_new(vec![]).unwrap())
        .unwrap();
    assert_eq!(live.query().numerator(), 1);
}

#[test]
fn journal_reopens_replays_once_and_rejects_another_writer() {
    let path = path();
    let mut store = LiveAverageStore::create(&path, state(vec![1, 2])).unwrap();
    assert!(LiveAverageStore::open(&path).is_err());
    assert!(LiveAverageStore::create(&path, state(vec![9])).is_err());
    let delta = ReplaceDelta::try_new(vec![Replace::new(0, 1, u64::MAX)]).unwrap();
    store.apply(7, 0, 1, &delta).unwrap();
    let expected = store.state().unwrap().clone();
    drop(store);
    for _ in 0..3 {
        let (store, report) = LiveAverageStore::open(&path).unwrap();
        assert_eq!(report.replayed_events, 1);
        assert_eq!(store.state().unwrap(), &expected);
    }
    fs::remove_file(path).unwrap();
}

#[test]
fn every_partial_final_frame_recovers_last_complete_commit_and_can_continue() {
    let original = path();
    let mut store = LiveAverageStore::create(&original, state(vec![1, 2])).unwrap();
    let initial_len = fs::metadata(&original).unwrap().len() as usize;
    let delta = ReplaceDelta::try_new(vec![Replace::new(0, 1, 9)]).unwrap();
    store.apply(7, 0, 1, &delta).unwrap();
    drop(store);
    let bytes = fs::read(&original).unwrap();
    for cut in initial_len..bytes.len() {
        let partial = path();
        fs::write(&partial, &bytes[..cut]).unwrap();
        let (mut recovered, report) = LiveAverageStore::open(&partial).unwrap();
        assert_eq!(recovered.state().unwrap(), &state(vec![1, 2]));
        assert_eq!(report.discarded_tail_bytes as usize, cut - initial_len);
        recovered.apply(7, 0, 1, &delta).unwrap();
        drop(recovered);
        let (recovered, _) = LiveAverageStore::open(&partial).unwrap();
        assert_eq!(recovered.state().unwrap().values(), &[9, 2]);
        drop(recovered);
        fs::remove_file(partial).unwrap();
    }
    fs::remove_file(original).unwrap();
}

#[test]
fn complete_corruption_is_rejected_without_truncating_the_file() {
    let path = path();
    let mut store = LiveAverageStore::create(&path, state(vec![1, 2])).unwrap();
    store
        .apply(
            7,
            0,
            1,
            &ReplaceDelta::try_new(vec![Replace::new(0, 1, 9)]).unwrap(),
        )
        .unwrap();
    drop(store);
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::End(-1)).unwrap();
    file.write_all(&[123]).unwrap();
    drop(file);
    let before = fs::read(&path).unwrap();
    assert!(LiveAverageStore::open(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    fs::remove_file(path).unwrap();
}

#[test]
fn crash_child() {
    let Ok(path) = std::env::var("AXON_TEST_CRASH_PATH") else {
        return;
    };
    let events: u64 = std::env::var("AXON_TEST_CRASH_EVENTS")
        .unwrap()
        .parse()
        .unwrap();
    let mut store = LiveAverageStore::create(&path, state(vec![1, 2])).unwrap();
    for event in 1..=events {
        let old = store.state().unwrap().values()[0];
        store
            .apply(
                7,
                event - 1,
                event,
                &ReplaceDelta::try_new(vec![Replace::new(0, old, event + 10)]).unwrap(),
            )
            .unwrap();
    }
    std::fs::write(format!("{path}.ready"), b"synced").unwrap();
    loop {
        std::thread::park();
    }
}

#[test]
fn killed_process_recovers_all_acknowledged_events_and_continues() {
    for events in [1, 13, 47] {
        let path = path();
        let ready = PathBuf::from(format!("{}.ready", path.display()));
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "crash_child", "--nocapture"])
            .env("AXON_TEST_CRASH_PATH", &path)
            .env("AXON_TEST_CRASH_EVENTS", events.to_string())
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        while !ready.exists() && start.elapsed().as_secs() < 10 {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        child.kill().unwrap();
        child.wait().unwrap();
        assert!(ready.exists(), "child did not reach durable commit");
        let (mut store, report) = LiveAverageStore::open(&path).unwrap();
        assert_eq!(report.replayed_events, events);
        assert_eq!(store.state().unwrap().values(), &[events + 10, 2]);
        store
            .apply(
                7,
                events,
                events + 1,
                &ReplaceDelta::try_new(vec![Replace::new(1, 2, 99)]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            store.state().unwrap().query().numerator(),
            events as u128 + 109
        );
        drop(store);
        fs::remove_file(path).unwrap();
        fs::remove_file(ready).unwrap();
    }
}

#[test]
fn compaction_preserves_nonzero_version_and_bounds_the_journal() {
    let path = path();
    let mut store = LiveAverageStore::create(&path, state(vec![1, 2])).unwrap();
    let initial_bytes = fs::metadata(&path).unwrap().len();
    for event in 1..=128 {
        let old = store.state().unwrap().values()[0];
        store
            .apply(
                7,
                event - 1,
                event,
                &ReplaceDelta::try_new(vec![Replace::new(0, old, event * 3)]).unwrap(),
            )
            .unwrap();
        if event % 16 == 0 {
            let expected = store.state().unwrap().clone();
            let compact = store.compact().unwrap();
            assert_eq!(compact.bytes_after, initial_bytes);
            assert!(compact.bytes_before > compact.bytes_after);
            assert_eq!(compact.checkpoint_event, event);
            assert!(LiveAverageStore::open(&path).is_err());
            assert_eq!(store.state().unwrap(), &expected);
            drop(store);
            let (reopened, report) = LiveAverageStore::open(&path).unwrap();
            assert_eq!(report.replayed_events, 0);
            assert_eq!(reopened.state().unwrap(), &expected);
            store = reopened;
        }
    }
    let no_change = store.compact().unwrap();
    assert_eq!(no_change.bytes_before, no_change.bytes_after);
    drop(store);
    fs::remove_file(&path).unwrap();
    fs::remove_file(format!("{}.lock", path.display())).unwrap();
}

#[test]
fn invalid_event_does_not_grow_the_journal() {
    let path = path();
    let mut store = LiveAverageStore::create(&path, state(vec![1, 2])).unwrap();
    let initial_bytes = fs::metadata(&path).unwrap().len();
    let delta = ReplaceDelta::try_new(vec![Replace::new(0, 9, 3)]).unwrap();
    assert!(store.apply(7, 0, 1, &delta).is_err());
    assert_eq!(fs::metadata(&path).unwrap().len(), initial_bytes);
    assert_eq!(store.state().unwrap().values(), &[1, 2]);
    drop(store);
    fs::remove_file(&path).unwrap();
    fs::remove_file(format!("{}.lock", path.display())).unwrap();
}
