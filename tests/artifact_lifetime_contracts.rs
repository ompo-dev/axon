use std::time::Duration;

use axon_uic::{ArtifactLifetime, ArtifactLifetimeError, BenchContract, BenchPhase, BreakEven};

fn contract(phase: BenchPhase, milliseconds: u64) -> BenchContract {
    BenchContract::empty().with_phase(phase, Duration::from_millis(milliseconds))
}

#[test]
fn artifact_lifetime_charges_creation_once_and_reuses_steady_state() {
    let artifact = ArtifactLifetime::try_new(
        contract(BenchPhase::Synthesis, 30),
        vec![
            contract(BenchPhase::Execution, 5),
            contract(BenchPhase::Execution, 5),
            contract(BenchPhase::Execution, 5),
        ],
        BenchContract::empty(),
    )
    .unwrap();

    assert_eq!(artifact.cold_create().unwrap(), Duration::from_millis(30));
    assert_eq!(artifact.hot(0).unwrap(), Duration::from_millis(5));
    assert_eq!(
        artifact.lifecycle_one(0).unwrap(),
        Duration::from_millis(35)
    );
    assert_eq!(artifact.steady_state(3).unwrap(), Duration::from_millis(15));
    assert_eq!(artifact.amortized(3).unwrap(), Duration::from_millis(45));
}

#[test]
fn break_even_requires_strictly_lower_total_cost() {
    let baseline = ArtifactLifetime::try_new(
        BenchContract::empty(),
        vec![
            contract(BenchPhase::Execution, 10),
            contract(BenchPhase::Execution, 10),
            contract(BenchPhase::Execution, 10),
            contract(BenchPhase::Execution, 10),
        ],
        BenchContract::empty(),
    )
    .unwrap();
    let candidate = ArtifactLifetime::try_new(
        contract(BenchPhase::Synthesis, 20),
        vec![
            contract(BenchPhase::Execution, 4),
            contract(BenchPhase::Execution, 4),
            contract(BenchPhase::Execution, 4),
            contract(BenchPhase::Execution, 4),
        ],
        BenchContract::empty(),
    )
    .unwrap();

    assert_eq!(
        candidate.first_break_even_against(&baseline, 4).unwrap(),
        BreakEven::AtUses(4)
    );
}

#[test]
fn artifact_lifetime_rejects_zero_or_unmeasured_reuse() {
    let artifact = ArtifactLifetime::try_new(
        BenchContract::empty(),
        vec![contract(BenchPhase::Execution, 1)],
        BenchContract::empty(),
    )
    .unwrap();

    assert_eq!(artifact.amortized(0), Err(ArtifactLifetimeError::ZeroUses));
    assert_eq!(
        artifact.steady_state(2),
        Err(ArtifactLifetimeError::UseUnavailable(2))
    );
    assert_eq!(
        artifact.lifecycle_one(usize::MAX),
        Err(ArtifactLifetimeError::UseUnavailable(usize::MAX))
    );
}
