use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

fn temporary_root(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "axon-knowledge-cli-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn axon(arguments: &[&str], current_directory: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axon"))
        .args(arguments)
        .current_dir(current_directory)
        .output()
        .expect("AXON command starts")
}

fn stdout(output: Output) -> String {
    assert!(
        output.status.success(),
        "command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("AXON output is UTF-8")
}

#[test]
fn public_cli_learns_updates_recalls_abstains_and_forgets() {
    let root = temporary_root("workflow");
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("engine.md"),
        "The cobalt engine uses a ceramic rotor and magnetic bearings.\n",
    )
    .unwrap();
    fs::write(
        corpus.join("garden.md"),
        "The garden irrigation schedule starts before sunrise.\n",
    )
    .unwrap();

    let learned = stdout(axon(&["learn", "corpus"], &root));
    assert!(learned.contains("added_sources=2"));
    assert!(learned.contains("total_sources=2"));
    let unchanged = stdout(axon(&["learn", "corpus"], &root));
    assert!(unchanged.contains("unchanged_sources=2"));

    let recalled = stdout(axon(
        &["recall", "cermaic cobalt rotor", "--limit", "2"],
        &root,
    ));
    assert!(recalled.contains("status=found"));
    assert!(recalled.contains("engine.md"));
    assert!(recalled.contains("lines=1-1"));
    assert!(recalled.contains("indexed_documents=2"));
    assert!(recalled.contains("scored_documents=1"));

    let abstained = stdout(axon(
        &["recall", "photosynthesis orchid chlorophyll"],
        &root,
    ));
    assert!(abstained.contains("status=no_evidence"));
    assert!(abstained.contains("evidence_count=0"));

    fs::write(
        corpus.join("engine.md"),
        "The silver engine uses a carbon rotor and fluid bearings.\n",
    )
    .unwrap();
    let updated = stdout(axon(&["learn", "corpus/engine.md"], &root));
    assert!(updated.contains("updated_sources=1"));
    let stale = stdout(axon(&["recall", "cobalt ceramic magnetic"], &root));
    assert!(stale.contains("status=no_evidence"));

    let status = stdout(axon(&["memory-status"], &root));
    assert!(status.contains("sources=2"));
    assert!(status.contains("chunks=2"));
    let forgotten = stdout(axon(&["forget", "corpus"], &root));
    assert!(forgotten.contains("removed_sources=2"));
    assert!(forgotten.contains("total_sources=0"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_cli_evaluates_ranked_retrieval_against_frozen_queries() {
    let root = temporary_root("evaluation");
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("mars.md"),
        "Mars has two small moons named Phobos and Deimos.\n",
    )
    .unwrap();
    fs::write(
        corpus.join("ocean.md"),
        "The Pacific is the largest ocean on Earth.\n",
    )
    .unwrap();
    fs::write(
        root.join("questions.eval"),
        "which moons orbit Mars\tcorpus/mars.md\nwhat is the largest ocean\tcorpus/ocean.md\norchid chlorophyll photosynthesis\t-\n",
    )
    .unwrap();
    stdout(axon(&["learn", "corpus"], &root));

    let evaluated = stdout(axon(
        &["memory-eval", "questions.eval", "--limit", "3"],
        &root,
    ));
    assert!(evaluated.contains("cases=3"));
    assert!(evaluated.contains("positive_cases=2"));
    assert!(evaluated.contains("negative_cases=1"));
    assert!(evaluated.contains("top1=2/2"));
    assert!(evaluated.contains("recall_at_3=2/2"));
    assert!(evaluated.contains("correct_abstentions=1/1"));
    assert!(evaluated.contains("overall_correct=3/3"));
    assert!(evaluated.contains("mrr=1.000000"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_cli_learns_from_relevance_feedback_across_processes() {
    let root = temporary_root("feedback");
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    let text = "Cobalt engine calibration follows the verified protocol.\n";
    fs::write(corpus.join("cobalt-engine.md"), text).unwrap();
    fs::write(corpus.join("manual.md"), text).unwrap();
    stdout(axon(&["learn", "corpus"], &root));

    let before = stdout(axon(&["recall", "cobalt engine calibration"], &root));
    assert!(before.contains("cobalt-engine.md"));
    assert!(before.contains("feedback_factor=1.000000"));

    let taught = stdout(axon(
        &[
            "teach",
            "cobalt engine calibration",
            "--relevant",
            "corpus/manual.md",
        ],
        &root,
    ));
    assert!(taught.contains("action=teach"));
    assert!(taught.contains("label=relevant"));
    assert!(taught.contains("relevant_observations=1"));

    let after = stdout(axon(&["recall", "cobalt engine calibration"], &root));
    let preferred_position = after.find("manual.md").unwrap();
    let old_position = after.find("cobalt-engine.md").unwrap();
    assert!(preferred_position < old_position);
    assert!(after.contains("feedback_factor=2.000000"));
    let status = stdout(axon(&["memory-status"], &root));
    assert!(status.contains("feedback_records=1"));
    assert!(status.contains("feedback_observations=1"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_cli_composes_cited_evidence_and_evaluates_the_reasoning_path() {
    let root = temporary_root("composed-reasoning");
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("durability.md"),
        "# Integrity\n\nAxiom validates stored content.\n\n# Exclusion\n\nMutex prevents a second writer.\n\n# Recovery\n\nPriorstate survives an interrupted publication.\n",
    )
    .unwrap();
    fs::write(
        root.join("composition.eval"),
        "axiom mutex priorstate\tcorpus/durability.md\naxiom wallet crypton\t-\n",
    )
    .unwrap();
    stdout(axon(&["learn", "corpus"], &root));

    let recalled = stdout(axon(&["recall", "axiom mutex priorstate"], &root));
    assert!(recalled.contains("status=no_evidence"));
    let reasoned = stdout(axon(&["ask", "axiom mutex priorstate"], &root));
    assert!(reasoned.contains("status=found"));
    assert!(reasoned.contains("mode=extractive_composition"));
    assert!(reasoned.contains("covered_terms=axiom,mutex,priorstate"));
    assert!(reasoned.contains("evidence_count=3"));
    assert!(reasoned.contains("source="));
    assert!(reasoned.contains("lines="));

    let evaluated = stdout(axon(
        &["memory-reason-eval", "composition.eval", "--limit", "3"],
        &root,
    ));
    assert!(evaluated.contains("mode=composed"));
    assert!(evaluated.contains("top1=1/1"));
    assert!(evaluated.contains("correct_abstentions=1/1"));
    assert!(evaluated.contains("overall_correct=2/2"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_cli_builds_and_evaluates_connected_cross_source_evidence() {
    let root = temporary_root("evidence-graph");
    let corpus = root.join("corpus");
    fs::create_dir_all(&corpus).unwrap();
    fs::write(
        corpus.join("origin.md"),
        "The orion engine delegates through the cobalt relay.",
    )
    .unwrap();
    fs::write(
        corpus.join("destination.md"),
        "The cobalt relay opens the cedar vault.",
    )
    .unwrap();
    fs::write(
        corpus.join("isolated.md"),
        "A juniper archive stores an ivory chamber.",
    )
    .unwrap();
    fs::write(
        root.join("coherence.eval"),
        "orion engine cedar vault\tcorpus/origin.md&corpus/destination.md\norion engine ivory chamber\t-\n",
    )
    .unwrap();
    stdout(axon(&["learn", "corpus"], &root));

    let investigated = stdout(axon(
        &["investigate", "orion engine cedar vault", "--limit", "3"],
        &root,
    ));
    assert!(investigated.contains("status=found"));
    assert!(investigated.contains("mode=evidence_graph"));
    assert!(investigated.contains("evidence_count=2"));
    assert!(investigated.contains("link_count=1"));
    assert!(investigated.contains("link_1_kind=shared_terms"));
    assert!(investigated.contains("link_1_bridge_terms=cobalt,relay"));
    assert!(investigated.contains("origin.md"));
    assert!(investigated.contains("destination.md"));

    let disconnected = stdout(axon(&["investigate", "orion engine ivory chamber"], &root));
    assert!(disconnected.contains("status=no_evidence"));
    assert!(disconnected.contains("path_count=0"));

    let evaluated = stdout(axon(
        &["memory-coherence-eval", "coherence.eval", "--limit", "3"],
        &root,
    ));
    assert!(evaluated.contains("mode=evidence_graph"));
    assert!(evaluated.contains("top1=1/1"));
    assert!(evaluated.contains("correct_abstentions=1/1"));
    assert!(evaluated.contains("overall_correct=2/2"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn public_cli_refuses_a_corrupt_memory_file() {
    let root = temporary_root("corrupt");
    fs::create_dir_all(root.join(".axon-memory")).unwrap();
    fs::write(
        root.join(".axon-memory/knowledge.axm"),
        b"not an AXON memory",
    )
    .unwrap();

    let output = axon(&["memory-status"], &root);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("invalid AXON memory"),
        "unexpected stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::remove_dir_all(root).unwrap();
}
