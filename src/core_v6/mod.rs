//! AXON V6/Ω6: runtime cognitivo experimental dirigido por Factors e mensagens.
//!
//! Esta primeira versão é determinística, single-threaded e isolada do runtime
//! estável. Ela testa os contratos de conhecimento versionado, trabalho ativo
//! esparso e programas compiláveis antes de qualquer hardware especializado.

mod factor;
mod ids;
mod learnability;
mod learning;
mod ledger;
mod message;
mod metrics;
mod program;
mod runtime;
mod validation;

pub use factor::{
    Factor, FactorEdge, FactorError, FactorGraph, FactorKind, LearningMeta, LocalityHint,
    RelationKind, RepresentationKind, StateRef, ValidityDomain,
};
pub use ids::{ClaimId, FactorId, PatchId, ProgramId, RevisionId};
pub use learnability::{
    Experiment, InformationRequest, LearnabilityDecision, LearnabilityEvidence, LearnabilityGate,
    ReframeRequest,
};
pub use learning::{EligibilityTrace, LocalLearner, TeachingSignal};
pub use ledger::{
    Claim, ClaimView, EpistemicLedger, EpistemicStatus, LedgerError, NegativeKnowledge, TruthValue,
};
pub use message::{
    CognitiveMessage, MessageDisposition, MessagePayload, MessageScheduler, SchedulerTier,
};
pub use metrics::CognitiveMetrics;
pub use program::{
    Guard, OpCode, Program, ProgramDispatch, ProgramLibrary, ProgramStats, ProgramStatus,
    ProgramVm, ReasoningTrace, TraceStep, Value, ValueType, VmOutcome,
};
pub use runtime::{CognitiveBudget, CognitiveRuntime, ProcessingOutcome};
pub use validation::{
    CognitivePatch, PatchDecision, PatchTarget, ValidationKernel, ValidationReport,
    VerificationLevel,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v6_contracts_cover_factors_ledger_programs_learning_and_validation() {
        let _ = FactorGraph::default();
        let _ = EpistemicLedger::default();
        let _ = ProgramVm;
        let _ = LearnabilityGate;
        let _ = ValidationKernel;
    }
}
