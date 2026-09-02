use std::time::{SystemTime, UNIX_EPOCH};

use axon_uic::{
    ArtifactOutcome, ArtifactStatus, ArtifactStore, AxonTask, ExactAverage, FoldSpec, solve_task,
};

fn temporary_artifact_root(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("axon-uic-{label}-{}-{nonce}", std::process::id()))
}

#[test]
fn typed_task_declares_data_and_a_capability_goal() {
    let task = AxonTask::parse(
        "task average_stream {\n  data numbers: Vec<u64> = [2, 4, 9]\n  goal derive IncrementalArtifact<AverageExactU64>\n}\n",
    )
    .unwrap();

    assert_eq!(task.name(), "average_stream");
    assert_eq!(task.values(), &[2, 4, 9]);
    assert_eq!(task.goal(), FoldSpec::AverageExactU64);
}

#[test]
fn typed_task_rejects_a_name_that_cannot_become_an_artifact_path() {
    assert!(AxonTask::parse(
        "task 12_bad {\n  data numbers: Vec<u64> = [2]\n  goal derive IncrementalArtifact<AddModU64>\n}\n"
    )
    .is_err());
    assert!(AxonTask::parse(
        "task ../escape {\n  data numbers: Vec<u64> = [2]\n  goal derive IncrementalArtifact<AddModU64>\n}\n"
    )
    .is_err());
}

#[test]
fn stored_artifact_is_created_once_and_reused_for_the_same_typed_goal() {
    let root = temporary_artifact_root("reuse");
    let store = ArtifactStore::open(&root);
    let task = AxonTask::parse(
        "task sum_stream {\n  data numbers: Vec<u64> = [1, 2, 3]\n  goal derive IncrementalArtifact<AddModU64>\n}\n",
    )
    .unwrap();

    let created = store.install(&task).unwrap();
    let reused = store.install(&task).unwrap();

    assert_eq!(created.status(), ArtifactStatus::Created);
    assert_eq!(reused.status(), ArtifactStatus::Reused);
    assert_eq!(created.path(), reused.path());
    assert!(created.path().is_file());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn stored_artifact_refuses_a_changed_capability_or_corrupt_record() {
    let root = temporary_artifact_root("guard");
    let store = ArtifactStore::open(&root);
    let sum = AxonTask::parse(
        "task same_name {\n  data numbers: Vec<u64> = [1]\n  goal derive IncrementalArtifact<AddModU64>\n}\n",
    )
    .unwrap();
    let average = AxonTask::parse(
        "task same_name {\n  data numbers: Vec<u64> = [1]\n  goal derive IncrementalArtifact<AverageExactU64>\n}\n",
    )
    .unwrap();

    let artifact_path = store.install(&sum).unwrap().path().to_path_buf();
    assert!(store.install(&average).is_err());
    std::fs::write(&artifact_path, "not an artifact").unwrap();
    assert!(store.install(&sum).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn solver_executes_a_persisted_average_artifact_with_exact_output() {
    let root = temporary_artifact_root("solve");
    let store = ArtifactStore::open(&root);
    let task = AxonTask::parse(
        "task average_stream {\n  data numbers: Vec<u64> = [2, 4, 9]\n  goal derive IncrementalArtifact<AverageExactU64>\n}\n",
    )
    .unwrap();

    let first = solve_task(&store, &task).unwrap();
    let second = solve_task(&store, &task).unwrap();

    assert_eq!(first.artifact_status(), ArtifactStatus::Created);
    assert_eq!(second.artifact_status(), ArtifactStatus::Reused);
    assert_eq!(
        first.outcome(),
        &ArtifactOutcome::ExactAverage(ExactAverage::new(15, 3).unwrap())
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn axon_solve_cli_persists_then_reuses_a_typed_artifact() {
    let root = temporary_artifact_root("cli");
    std::fs::create_dir_all(&root).unwrap();
    let task_path = root.join("average.axon");
    let artifact_root = root.join("artifacts");
    std::fs::write(
        &task_path,
        "task average_cli {\n  data numbers: Vec<u64> = [2, 4, 9]\n  goal derive IncrementalArtifact<AverageExactU64>\n}\n",
    )
    .unwrap();

    let first = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .args(["solve", task_path.to_str().unwrap(), "--artifact-dir"])
        .arg(&artifact_root)
        .output()
        .unwrap();
    let second = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .args(["solve", task_path.to_str().unwrap(), "--artifact-dir"])
        .arg(&artifact_root)
        .output()
        .unwrap();

    assert!(first.status.success());
    assert!(second.status.success());
    assert!(
        String::from_utf8(first.stdout)
            .unwrap()
            .contains("artifact=created")
    );
    assert!(
        String::from_utf8(second.stdout)
            .unwrap()
            .contains("artifact=reused")
    );
    std::fs::remove_dir_all(root).unwrap();
}
