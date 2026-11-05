#![cfg(feature = "text-workflows")]
use std::{fs, process::Command};
#[test]
fn independent_process_reopens_compositions_and_records_every_drift_miss() {
    let root = std::env::temp_dir().join(format!("axon-controller-lab-{}", std::process::id()));
    assert!(!root.exists());
    let output = Command::new(env!("CARGO_BIN_EXE_axon-uic-controller-lab"))
        .arg(&root)
        .args(["72918463", "100", "development"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{} / {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = fs::read_to_string(root.join("summary.txt")).unwrap();
    assert!(summary.contains("policy_correct=200\n"));
    assert!(summary.contains("physical_correct=199\n"));
    assert!(summary.contains("detected_device_changes=1\n"));
    assert!(root.join("modules-72918463.axseed").is_file());
    for i in 0..2 {
        assert_eq!(
            &fs::read(root.join(format!("goal-{i}.axgoal"))).unwrap()[..8],
            b"AXGP0001"
        );
    }
    let mut reader = csv::Reader::from_path(root.join("results-72918463.csv")).unwrap();
    let rows = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(rows.len(), 200);
    assert_eq!(rows.iter().filter(|r| &r[7] == "Refuted").count(), 1);
    assert!(rows.iter().all(|r| &r[5] == "true"));
    assert_eq!(rows.iter().filter(|r| &r[6] == "true").count(), 199);
    let absolute = root.canonicalize().unwrap();
    assert!(absolute.starts_with(std::env::temp_dir().canonicalize().unwrap()));
    fs::remove_dir_all(absolute).unwrap();
}
