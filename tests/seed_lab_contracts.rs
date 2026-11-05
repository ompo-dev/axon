use axon_uic::numeric::seed::DevelopmentSeed;
use std::{fs, process::Command};
#[test]
fn cohort_mutates_selects_preserves_and_reopens_real_developmental_genotypes() {
    let root = std::env::temp_dir().join(format!("axon-seed-lab-{}", std::process::id()));
    assert!(!root.exists());
    let result = Command::new(env!("CARGO_BIN_EXE_axon-uic-seed-lab"))
        .arg(&root)
        .args(["70143863", "100", "development"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{} / {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let summary = fs::read_to_string(root.join("summary.txt")).unwrap();
    assert!(summary.contains("candidates=24"));
    assert!(summary.contains("best_development_capabilities=5/6"));
    assert!(summary.contains("reserve_failures=0"));
    let population = fs::read_to_string(root.join("population.csv")).unwrap();
    assert_eq!(population.lines().count(), 25);
    assert!(population.contains("parent_a,parent_b"));
    let frozen = fs::read_to_string(root.join("frozen.csv")).unwrap();
    assert!(frozen.lines().count() > 1);
    for line in frozen.lines().skip(1) {
        let id = line.split(',').next().unwrap();
        let seed = DevelopmentSeed::open(
            &root.join(format!("candidate-{id}.axdna")),
            "seed-lab/development-v1",
        )
        .unwrap();
        assert!(!seed.skills().is_empty());
        assert!(seed.to_bytes().len() <= seed.byte_cap());
        for parent in seed.parents().into_iter().filter(|p| *p != 0) {
            assert!(root.join(format!("revision-{parent:016x}.axdna")).is_file());
        }
    }
    let absolute = root.canonicalize().unwrap();
    assert!(absolute.starts_with(std::env::temp_dir().canonicalize().unwrap()));
    fs::remove_dir_all(absolute).unwrap();
}
