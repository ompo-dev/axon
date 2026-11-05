use axon_uic::nursery::*;
use std::{
    collections::BTreeSet,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-nursery-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let p = self.0.canonicalize().unwrap();
        assert!(p.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        std::fs::remove_dir_all(p).unwrap();
    }
}
fn config() -> SearchConfig {
    SearchConfig {
        maximum: 99,
        population: 4,
        patience: 1,
        max_candidates: 100,
        ..SearchConfig::default()
    }
}
fn assessment(correct: u64) -> Assessment {
    Assessment {
        checked: 10,
        correct,
        wrong: 0,
        encoded_bytes: 100,
        elapsed_ns: 100,
    }
}

#[test]
fn grows_one_digit_and_finds_a_real_winner_without_repetition() {
    let t = Temp::new();
    let mut n = Nursery::open(&t.0.join("registry.axep"), "fixture/v1", config(), 91).unwrap();
    let mut seen = BTreeSet::new();
    while let Some(p) = n.propose().unwrap() {
        assert!(seen.insert(p.dna));
        assert_eq!(p.digits, if n.growths() == 0 { 1 } else { 2 });
        n.commit(
            p,
            assessment(if p.dna == 99 {
                10
            } else {
                p.dna.count_ones() as u64
            }),
        )
        .unwrap();
    }
    assert_eq!(n.stop(), Some(SearchStop::Winner));
    assert_eq!(n.best().unwrap().dna, 99);
    assert_eq!(n.digits(), 2);
    assert_eq!(n.growths(), 1);
}
#[test]
fn resume_never_reexecutes_evaluated_seeds_or_changes_their_scores() {
    let t = Temp::new();
    let path = t.0.join("registry.axep");
    let first;
    {
        let mut n = Nursery::open(&path, "fixture/v1", config(), 91).unwrap();
        for _ in 0..13 {
            let p = n.propose().unwrap().unwrap();
            n.commit(p, assessment(3)).unwrap();
        }
        first = n.entries().copied().collect::<Vec<_>>();
    }
    let mut n = Nursery::open(&path, "fixture/v1", config(), 91).unwrap();
    assert_eq!(n.entries().copied().collect::<Vec<_>>(), first);
    let old: BTreeSet<_> = first.iter().map(|i| i.dna).collect();
    while let Some(p) = n.propose().unwrap() {
        assert!(!old.contains(&p.dna));
        n.commit(p, assessment(3)).unwrap();
    }
    assert_eq!(n.completed(), 100);
    assert_eq!(n.stop(), Some(SearchStop::SpaceExhausted));
}
#[test]
fn interrupted_reservations_are_quarantined_not_evaluated_twice() {
    let t = Temp::new();
    let path = t.0.join("registry.axep");
    let lost;
    {
        let mut n = Nursery::open(&path, "fixture/v1", config(), 91).unwrap();
        lost = n.propose().unwrap().unwrap().dna;
    }
    let mut n = Nursery::open(&path, "fixture/v1", config(), 91).unwrap();
    assert_eq!(n.unfinished(), 1);
    while let Some(p) = n.propose().unwrap() {
        assert_ne!(p.dna, lost);
        n.commit(p, assessment(0)).unwrap();
    }
    assert_eq!(n.seen(), 100);
    assert_eq!(n.completed(), 99);
    assert_eq!(n.stop(), Some(SearchStop::UnconfirmedExhaustion));
}
#[test]
fn journal_lock_scope_and_frozen_limits_prevent_cross_campaign_score_reuse() {
    let t = Temp::new();
    let path = t.0.join("registry.axep");
    let n = Nursery::open(&path, "suite-a", config(), 1).unwrap();
    assert!(Nursery::open(&path, "suite-a", config(), 1).is_err());
    drop(n);
    assert!(Nursery::open(&path, "suite-b", config(), 1).is_err());
    let mut changed = config();
    changed.max_encoded_bytes += 1;
    assert!(Nursery::open(&path, "suite-a", changed, 1).is_err());
}
#[test]
fn correctness_does_not_override_measured_resource_limits() {
    let t = Temp::new();
    let mut n = Nursery::open(&t.0.join("registry.axep"), "fixture/v1", config(), 1).unwrap();
    let p = n.propose().unwrap().unwrap();
    let mut a = assessment(10);
    a.encoded_bytes = 200000;
    n.commit(p, a).unwrap();
    assert_eq!(n.stop(), None);
    let p = n.propose().unwrap().unwrap();
    let mut a = assessment(10);
    a.elapsed_ns = 2_000_000_000;
    n.commit(p, a).unwrap();
    assert_eq!(n.stop(), None);
    let p = n.propose().unwrap().unwrap();
    n.commit(p, assessment(10)).unwrap();
    assert_eq!(n.stop(), Some(SearchStop::Winner));
}
#[test]
fn parent_front_preserves_memory_time_tradeoffs_without_rerunning_elites() {
    let t = Temp::new();
    let mut n = Nursery::open(&t.0.join("registry.axep"), "fixture/v1", config(), 1).unwrap();
    for (bytes, ns) in [(100, 200), (200, 100), (300, 300), (150, 150)] {
        let p = n.propose().unwrap().unwrap();
        let mut a = assessment(5);
        a.encoded_bytes = bytes;
        a.elapsed_ns = ns;
        n.commit(p, a).unwrap();
    }
    let parents = n.parents();
    assert_eq!(parents.len(), 3);
    assert!(parents.iter().any(|p| p.assessment.encoded_bytes == 100));
    assert!(parents.iter().any(|p| p.assessment.elapsed_ns == 100));
    assert_eq!(n.completed(), 4);
}
#[test]
fn malformed_scores_and_duplicate_commits_do_not_enter_the_archive() {
    let t = Temp::new();
    let mut n = Nursery::open(&t.0.join("registry.axep"), "fixture/v1", config(), 1).unwrap();
    let p = n.propose().unwrap().unwrap();
    assert!(n.propose().is_err());
    assert!(n.commit(p, assessment(11)).is_err());
    assert_eq!(n.completed(), 0);
    n.commit(p, assessment(5)).unwrap();
    assert!(n.commit(p, assessment(5)).is_err());
    let p = n.propose().unwrap().unwrap();
    let mut a = assessment(3);
    a.checked = 9;
    assert!(n.commit(p, a).is_err());
    assert_eq!(n.completed(), 1);
}
#[test]
fn candidate_quota_stops_explicitly_without_announcing_a_winner() {
    let t = Temp::new();
    let mut c = config();
    c.max_candidates = 7;
    let mut n = Nursery::open(&t.0.join("registry.axep"), "fixture/v1", c, 1).unwrap();
    while let Some(p) = n.propose().unwrap() {
        n.commit(p, assessment(0)).unwrap();
    }
    assert_eq!(n.stop(), Some(SearchStop::CandidateLimit));
    assert_eq!(n.seen(), 7);
}

#[test]
fn optimization_continues_after_a_winner_but_preserves_caps_and_reservations() {
    let t = Temp::new();
    let path = t.0.join("registry.axep");
    let mut c = config();
    c.max_candidates = 7;
    let mut n = Nursery::open(&path, "optimization/v1", c, 1).unwrap();
    let first = n.propose_optimization().unwrap().unwrap();
    n.commit(first, assessment(10)).unwrap();
    assert_eq!(n.stop(), Some(SearchStop::Winner));
    assert!(n.propose().unwrap().is_none());
    drop(n);
    let mut n = Nursery::open(&path, "optimization/v1", c, 1).unwrap();
    let mut seen = BTreeSet::from([first.dna]);
    while let Some(p) = n.propose_optimization().unwrap() {
        assert!(seen.insert(p.dna));
        n.commit(p, assessment(10)).unwrap();
    }
    assert_eq!(n.completed(), 7);
    assert_eq!(n.seen(), 7);
    assert_eq!(n.stop(), Some(SearchStop::Winner));
}

#[test]
fn imports_exclude_old_genotypes_without_inventing_fitness() {
    let t = Temp::new();
    let path = t.0.join("registry.axep");
    {
        let mut n = Nursery::open(&path, "fixture/v1", config(), 1).unwrap();
        assert!(n.import_seen(0, "old-run-a").unwrap());
        assert!(!n.import_seen(0, "old-run-b").unwrap());
        assert!(n.import_seen(99, "old-run-a").unwrap());
        assert_eq!(n.best(), None);
    }
    let mut n = Nursery::open(&path, "fixture/v1", config(), 1).unwrap();
    while let Some(p) = n.propose().unwrap() {
        assert_ne!(p.dna, 0);
        assert_ne!(p.dna, 99);
        n.commit(p, assessment(3)).unwrap();
    }
    assert_eq!(n.imported(), 2);
    assert_eq!(n.completed(), 98);
    assert_eq!(n.unfinished(), 0);
    assert_eq!(n.stop(), Some(SearchStop::HistoricalExhaustion));
}

#[test]
fn snapshots_keep_the_writer_lock_and_allow_further_commits() {
    let t = Temp::new();
    let path = t.0.join("registry.axep");
    let copy = t.0.join("snapshot.axep");
    let mut n = Nursery::open(&path, "fixture/v1", config(), 1).unwrap();
    let p = n.propose().unwrap().unwrap();
    n.commit(p, assessment(3)).unwrap();
    n.snapshot_new(&copy).unwrap();
    assert!(n.snapshot_new(&copy).is_err());
    let restored = Nursery::open(&copy, "fixture/v1", config(), 1).unwrap();
    assert_eq!(restored.completed(), 1);
    assert_eq!(restored.best(), n.best());
    drop(restored);
    assert!(Nursery::open(&path, "fixture/v1", config(), 1).is_err());
    let p = n.propose().unwrap().unwrap();
    n.commit(p, assessment(4)).unwrap();
    assert_eq!(n.completed(), 2);
    drop(n);
    assert_eq!(
        Nursery::open(&path, "fixture/v1", config(), 1)
            .unwrap()
            .completed(),
        2
    );
}
