//! V5/Ω experimental: cognição multi-substrato orientada a valor por custo.
//!
//! O módulo é deliberadamente isolado do runtime estável. Os custos são
//! declarações de modelo até existirem contadores de hardware reais.

mod backend;
mod cost;
mod placement;
mod population;
mod profiler;
mod program;
mod scheduler;
mod state;
mod world;

pub use backend::{
    BackendPlan, BackendProfile, CognitiveOperation, PhysicalBackend, PhysicalCognitiveCompiler,
};
pub use cost::{CostError, CostOrigin, CostVector, CostWeights};
pub use placement::LocationPlasticity;
pub use population::{
    ActiveExperimentPlanner, ExperimentChoice, ExperimentPlan, Intervention, PopulationOfWorlds,
    WorldFamily, WorldFitness, WorldModel,
};
pub use profiler::{CognitiveMacro, CognitiveOpcode, MacroDispatch, ThoughtProfiler};
pub use program::{
    AbstractionCompiler, InductionResult, ProgramCell, ProgramInstruction, ProgramStatus,
};
pub use scheduler::{
    ExecutionAudit, ThermalCandidate, ThermalPlan, ThermodynamicBudget, ThermodynamicScheduler,
};
pub use state::{CognitiveEvent, CognitiveSubstrate, MultiSubstrateState};
pub use world::{
    ObjectWorld, Relation, ReversibleState, StateMutation, StructuralOperator, WorldMutation,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v5_contracts_cover_programs_worlds_reversibility_and_physical_routing() {
        let _ = ThermodynamicScheduler::default();
        let _ = ProgramCell::from_repeating_pair("alternation", "A", "B", 3);
        let _ = PopulationOfWorlds::default();
        let _ = ReversibleState::default();
        let _ = PhysicalCognitiveCompiler;
    }
}
