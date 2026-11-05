#![cfg(feature = "curriculum")]

use axon_uic::structural::sequence::lexical_encoder::synaptic::{Genes, Model, Position, Rule};
use axon_uic::{
    DevelopmentBudget, DevelopmentError, DevelopmentStop, DevelopmentTask, EpisodeStore,
    run_development,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_PATH: AtomicU64 = AtomicU64::new(1);

fn path(label: &str) -> PathBuf {
    let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "axon-development-{label}-{}-{id}.axep",
        std::process::id()
    ))
}

fn model() -> Model {
    Model::new(Genes {
        seed: 907,
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

fn task(id: &str, scale: f32, target: f32) -> DevelopmentTask {
    DevelopmentTask::new(
        id,
        vec![
            0.1 * scale,
            -0.2 * scale,
            0.3 * scale,
            -0.4 * scale,
            0.5 * scale,
            -0.6 * scale,
            0.7 * scale,
            -0.8 * scale,
        ],
        vec![
            target,
            -target,
            target * 0.5,
            -target * 0.5,
            target * 0.25,
            -target * 0.25,
            0.0,
            0.0,
        ],
    )
    .unwrap()
}

fn cleanup(path: &PathBuf) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(path.with_extension("axep.lock"));
}

#[test]
fn active_development_converges_on_scale_invariant_reserve_without_leakage() {
    let file = path("converges");
    let (mut episodes, _) = EpisodeStore::open(&file).unwrap();
    let training = [task("family-a", 1.0, 0.8)];
    let reserve = [task("family-a-heldout", 3.0, 0.8)];
    let run = run_development(
        model(),
        "development-scope",
        "development/converges",
        &mut episodes,
        &training,
        &reserve,
        DevelopmentBudget {
            max_passes: 16,
            tolerance: 1e-4,
            ..DevelopmentBudget::default()
        },
    )
    .unwrap();

    assert_eq!(run.report().stop, DevelopmentStop::Converged);
    assert!(run.report().converged);
    assert_eq!(run.report().final_reserve_correct, 1);
    assert!(run.report().final_reserve_mse < run.report().initial_reserve_mse);
    assert_eq!(run.report().training_steps, run.report().passes);
    assert_eq!(episodes.len(), run.report().training_steps);
    cleanup(&file);
}

#[test]
fn reserve_is_never_trained_and_plateau_is_reported_without_false_success() {
    let file = path("plateau");
    let (mut episodes, _) = EpisodeStore::open(&file).unwrap();
    let training = [task("train", 1.0, 0.8)];
    let reserve = [task("unseen", 0.37, -0.8)];
    let run = run_development(
        model(),
        "development-scope",
        "development/plateau",
        &mut episodes,
        &training,
        &reserve,
        DevelopmentBudget {
            max_passes: 4,
            tolerance: 1e-4,
            ..DevelopmentBudget::default()
        },
    )
    .unwrap();

    assert_eq!(run.report().stop, DevelopmentStop::PassLimit);
    assert!(!run.report().converged);
    assert_eq!(run.report().reserve_checks, 5);
    assert_eq!(episodes.len(), 4);
    cleanup(&file);
}

#[test]
fn duplicate_keys_and_ids_are_rejected_before_training() {
    let file = path("rejects");
    let (mut episodes, _) = EpisodeStore::open(&file).unwrap();
    let same_key = task("same-a", 1.0, 0.8);
    let duplicate_id = task("same-a", 2.0, 0.8);
    let error = match run_development(
        model(),
        "development-scope",
        "development/rejects",
        &mut episodes,
        &[same_key.clone()],
        &[duplicate_id],
        DevelopmentBudget::default(),
    ) {
        Ok(_) => panic!("duplicate task id was accepted"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        DevelopmentError::Invalid("duplicate task id")
    ));
    assert_eq!(episodes.len(), 0);

    let duplicate_key = task("different-id", 1.0, 0.8);
    let error = match run_development(
        model(),
        "development-scope",
        "development/rejects-key",
        &mut episodes,
        &[same_key],
        &[duplicate_key],
        DevelopmentBudget::default(),
    ) {
        Ok(_) => panic!("training/reserve key overlap was accepted"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        DevelopmentError::Invalid("training/reserve key overlap")
    ));
    assert_eq!(episodes.len(), 0);
    cleanup(&file);
}
