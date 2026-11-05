#![cfg(feature = "curriculum")]

use axon_uic::structural::sequence::lexical_encoder::synaptic::{Genes, Model, Position, Rule};
use axon_uic::{CognitiveCycle, EpisodeStore};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(1);

fn episode_path(label: &str) -> PathBuf {
    let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "axon-cognitive-cycle-{label}-{}-{id}.axep",
        std::process::id()
    ))
}

fn model() -> Model {
    Model::new(Genes {
        seed: 47,
        dimensions: 8,
        neurons: 32,
        layers: 2,
        position: Position::None,
        rule: Rule::Delta,
        decay: 1.0,
        learning_rate: 0.5,
    })
    .unwrap()
}

fn vectors() -> (Vec<f32>, Vec<f32>) {
    (
        vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8],
        vec![0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1],
    )
}

fn cleanup(path: &PathBuf) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(path.with_extension("axep.lock"));
}

#[test]
fn validated_experience_updates_plastic_state_without_mutating_model() {
    let path = episode_path("validated");
    let (mut episodes, _) = EpisodeStore::open(&path).unwrap();
    let model = model();
    let model_before = model.to_bytes().unwrap();
    let mut cycle = CognitiveCycle::new(model, "scope", "cycle/validated").unwrap();
    let before = cycle.state_snapshot().unwrap();
    let (key, value) = vectors();

    let receipt = cycle
        .observe(
            &mut episodes,
            "test",
            "step-1",
            1,
            &key,
            &value,
            true,
            17,
            200_000,
            262_144,
        )
        .unwrap();

    assert!(receipt.validated);
    assert!(receipt.learned);
    assert!(receipt.synapses_written > 0);
    assert!(receipt.prediction_error > 0.0);
    assert_ne!(before, cycle.state_snapshot().unwrap());
    assert_eq!(model_before, cycle.model().to_bytes().unwrap());
    assert_eq!(episodes.len(), 1);
    drop(episodes);
    cleanup(&path);
}

#[test]
fn rejected_experience_is_recorded_without_plastic_learning() {
    let path = episode_path("rejected");
    let (mut episodes, _) = EpisodeStore::open(&path).unwrap();
    let mut cycle = CognitiveCycle::new(model(), "scope", "cycle/rejected").unwrap();
    let before = cycle.state_snapshot().unwrap();
    let (key, value) = vectors();

    let receipt = cycle
        .observe(
            &mut episodes,
            "test",
            "step-1",
            1,
            &key,
            &value,
            false,
            19,
            200_000,
            262_144,
        )
        .unwrap();

    assert!(!receipt.validated);
    assert!(!receipt.learned);
    assert_eq!(receipt.synapses_written, 0);
    assert_eq!(before, cycle.state_snapshot().unwrap());
    assert_eq!(episodes.len(), 1);
    drop(episodes);
    cleanup(&path);
}

#[test]
fn journal_collision_cannot_commit_a_candidate_state() {
    let path = episode_path("collision");
    let (mut episodes, _) = EpisodeStore::open(&path).unwrap();
    let mut cycle = CognitiveCycle::new(model(), "scope", "cycle/collision").unwrap();
    let (key, value) = vectors();
    cycle
        .observe(
            &mut episodes,
            "test",
            "same-event",
            1,
            &key,
            &value,
            false,
            0,
            200_000,
            262_144,
        )
        .unwrap();
    let before = cycle.state_snapshot().unwrap();
    let error = cycle.observe(
        &mut episodes,
        "test",
        "same-event",
        1,
        &key,
        &value,
        true,
        0,
        200_000,
        262_144,
    );

    assert!(error.is_err());
    assert_eq!(before, cycle.state_snapshot().unwrap());
    assert_eq!(episodes.len(), 1);
    drop(episodes);
    cleanup(&path);
}

#[test]
fn same_genome_and_experiences_replay_identically() {
    let path_a = episode_path("replay-a");
    let path_b = episode_path("replay-b");
    let (mut episodes_a, _) = EpisodeStore::open(&path_a).unwrap();
    let (mut episodes_b, _) = EpisodeStore::open(&path_b).unwrap();
    let mut a = CognitiveCycle::new(model(), "scope", "cycle/replay").unwrap();
    let mut b = CognitiveCycle::new(model(), "scope", "cycle/replay").unwrap();
    let (key, value) = vectors();
    let receipt_a = a
        .observe(
            &mut episodes_a,
            "test",
            "step-1",
            1,
            &key,
            &value,
            true,
            0,
            200_000,
            262_144,
        )
        .unwrap();
    let receipt_b = b
        .observe(
            &mut episodes_b,
            "test",
            "step-1",
            1,
            &key,
            &value,
            true,
            0,
            200_000,
            262_144,
        )
        .unwrap();

    assert_eq!(receipt_a.predicted, receipt_b.predicted);
    assert_eq!(receipt_a.prediction_error, receipt_b.prediction_error);
    assert_eq!(receipt_a.synapses_written, receipt_b.synapses_written);
    assert_eq!(a.state_snapshot().unwrap(), b.state_snapshot().unwrap());
    drop(episodes_a);
    drop(episodes_b);
    cleanup(&path_a);
    cleanup(&path_b);
}

#[test]
fn validated_online_association_reduces_error_against_frozen_control() {
    let path = episode_path("online");
    let (mut episodes, _) = EpisodeStore::open(&path).unwrap();
    let mut learner = CognitiveCycle::new(model(), "scope", "cycle/online").unwrap();
    let frozen = CognitiveCycle::new(model(), "scope", "cycle/frozen").unwrap();
    let (key, value) = vectors();
    let before = learner.predict(&key, 200_000, 262_144).unwrap();
    let frozen_before = frozen.predict(&key, 200_000, 262_144).unwrap();

    for step in 0..6 {
        learner
            .observe(
                &mut episodes,
                "test",
                &format!("online-{step}"),
                step,
                &key,
                &value,
                true,
                0,
                200_000,
                262_144,
            )
            .unwrap();
    }

    let after = learner.predict(&key, 200_000, 262_144).unwrap();
    let frozen_after = frozen.predict(&key, 200_000, 262_144).unwrap();
    let before_error = mse(&before, &value);
    let after_error = mse(&after, &value);
    let frozen_error = mse(&frozen_after, &value);

    assert!(
        after_error < before_error,
        "{before_error} -> {after_error}"
    );
    assert_eq!(frozen_before, frozen_after);
    assert_eq!(frozen_error, mse(&frozen_before, &value));
    assert_eq!(episodes.len(), 6);
    drop(episodes);
    cleanup(&path);
}

#[test]
fn validated_experience_replays_after_journal_reopen() {
    let path = episode_path("replay-persisted");
    let (mut episodes, _) = EpisodeStore::open(&path).unwrap();
    let mut original = CognitiveCycle::new(model(), "scope", "cycle/persisted").unwrap();
    let model_before = original.model().to_bytes().unwrap();
    let (key, value) = vectors();

    for step in 0..3 {
        original
            .observe(
                &mut episodes,
                "test",
                &format!("replay-{step}"),
                step,
                &key,
                &value,
                true,
                0,
                200_000,
                262_144,
            )
            .unwrap();
    }
    original
        .observe(
            &mut episodes,
            "test",
            "rejected",
            4,
            &key,
            &value,
            false,
            0,
            200_000,
            262_144,
        )
        .unwrap();
    let original_snapshot = original.state_snapshot().unwrap();
    drop(episodes);

    let (episodes, recovery) = EpisodeStore::open(&path).unwrap();
    assert_eq!(recovery.replayed_episodes, 4);
    let mut restored = CognitiveCycle::new(model(), "scope", "cycle/persisted").unwrap();
    let replay = restored.replay(&episodes, 200_000, 262_144).unwrap();

    assert_eq!(replay.replayed, 3);
    assert_eq!(replay.skipped, 1);
    assert_eq!(replay.position, 3);
    assert_eq!(original_snapshot, restored.state_snapshot().unwrap());
    assert_eq!(model_before, restored.model().to_bytes().unwrap());
    drop(episodes);
    cleanup(&path);
}

fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected)
        .map(|(predicted, expected)| (predicted - expected).powi(2))
        .sum::<f32>()
        / predicted.len() as f32
}
