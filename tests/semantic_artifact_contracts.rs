use std::time::{SystemTime, UNIX_EPOCH};

use axon_uic::{
    ArtifactStatus, ArtifactStore, AxonTask, CertificateStatus, PhysicalBackend,
    SemanticArtifactHash, solve_task,
};

fn temporary_artifact_root(label: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "axon-semantic-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn average_task(values: &str) -> AxonTask {
    AxonTask::parse(&format!(
        "task average_stream {{\n  data numbers: Vec<u64> = [{values}]\n  goal derive IncrementalArtifact<AverageExactU64>\n}}\n"
    ))
    .unwrap()
}

#[test]
fn semantic_certificate_is_sealed_once_and_reused_by_content() {
    let root = temporary_artifact_root("seal");
    let store = ArtifactStore::open(&root);
    let task = average_task("2, 4, 9");

    let created = store.install(&task).unwrap();
    let reused = store.install(&task).unwrap();

    assert_eq!(created.status(), ArtifactStatus::Created);
    assert_eq!(created.certificate_status(), CertificateStatus::Verified);
    assert_eq!(reused.certificate_status(), CertificateStatus::Cached);
    assert_eq!(created.semantic().hash(), reused.semantic().hash());
    assert_eq!(reused.physical().semantic_hash(), reused.semantic().hash());
    assert_eq!(reused.physical().backend(), PhysicalBackend::Interpreter);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn semantic_seal_or_kernel_version_corruption_refuses_reuse() {
    let root = temporary_artifact_root("corrupt");
    let store = ArtifactStore::open(&root);
    let task = average_task("2, 4, 9");
    let path = store.install(&task).unwrap().path().to_path_buf();
    let content = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        content.replace("kernel_semantics_version=1", "kernel_semantics_version=0"),
    )
    .unwrap();

    assert!(store.install(&task).is_err());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn runtime_guards_reject_empty_exact_average_before_execution() {
    let root = temporary_artifact_root("guards");
    let store = ArtifactStore::open(&root);

    assert!(solve_task(&store, &average_task("")).is_err());
    let root_exists = root.exists();
    if root_exists {
        std::fs::remove_dir_all(&root).unwrap();
    }
    assert!(!root_exists);
}

#[test]
fn legacy_v1_cache_does_not_block_a_new_semantic_artifact() {
    let root = temporary_artifact_root("legacy");
    std::fs::create_dir_all(&root).unwrap();
    let legacy_path = root.join("average_stream.artifact");
    std::fs::write(
        &legacy_path,
        "axon-uic-artifact-v1\ncapability=AverageExactU64\n",
    )
    .unwrap();

    let installed = ArtifactStore::open(&root)
        .install(&average_task("2, 4, 9"))
        .unwrap();

    assert_eq!(installed.status(), ArtifactStatus::Created);
    assert_ne!(installed.path(), legacy_path);
    assert!(installed.path().exists());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn semantic_hash_requires_canonical_uppercase_16_digit_hex() {
    assert!(SemanticArtifactHash::from_hex("0000000000000001").is_some());
    assert!(SemanticArtifactHash::from_hex("1").is_none());
    assert!(SemanticArtifactHash::from_hex("000000000000000a").is_none());
}
