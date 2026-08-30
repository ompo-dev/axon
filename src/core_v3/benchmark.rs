use super::abduction::{
    AbductiveEngine, CausalModel, Contradiction, Counterfactual, NegativeArchive, ReframeHypothesis,
};

/// Synthetic world where correlation is real but either direct causal direction is falsified.
#[derive(Clone, Debug, PartialEq)]
pub struct JumpBenchmarkResult {
    pub obvious_counterfactual_loss: u32,
    pub selected: ReframeHypothesis,
}

pub fn run_jump_benchmark() -> JumpBenchmarkResult {
    let obvious = CausalModel::with_direct_cause("a", "b", 256);
    let contradiction = Contradiction::between("a", "b")
        .with_shared_observations(8)
        .with_counterfactual(Counterfactual::expect("a", "b", false))
        .with_counterfactual(Counterfactual::expect("b", "a", false));
    let engine = AbductiveEngine::default();
    let islands = engine.reframe(&obvious, &contradiction, &NegativeArchive::default());
    let selected = engine
        .best(&islands)
        .expect("abductive engine must generate candidates")
        .proposal
        .clone();

    JumpBenchmarkResult {
        obvious_counterfactual_loss: obvious.counterfactual_loss(&contradiction),
        selected,
    }
}
