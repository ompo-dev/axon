//! Kernel confiável para propostas de autoaperfeiçoamento limitadas.

use std::collections::BTreeSet;

use super::ids::{PatchId, RevisionId};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PatchTarget {
    ProgramLibrary,
    RetrievalPolicy,
    Heuristic,
    TrustedKernel,
    Benchmark,
    Verifier,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CognitivePatch {
    pub id: PatchId,
    pub target: PatchTarget,
    pub purpose: String,
    pub invariants: BTreeSet<String>,
    pub tests: Vec<String>,
    pub rollback: RevisionId,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum VerificationLevel {
    Heuristic,
    Statistical,
    Independent,
    HeldOut,
    Deterministic,
    Formal,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationReport {
    pub correctness_delta_milliunits: i32,
    pub latency_delta_ns: i64,
    pub active_bytes_delta: i64,
    pub regressions: u32,
    pub verifier_level: VerificationLevel,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatchDecision {
    Candidate,
    Rejected(String),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ValidationKernel;

impl ValidationKernel {
    pub fn validate(&self, patch: &CognitivePatch, report: &ValidationReport) -> PatchDecision {
        if matches!(
            patch.target,
            PatchTarget::TrustedKernel | PatchTarget::Benchmark | PatchTarget::Verifier
        ) {
            return PatchDecision::Rejected(
                "trusted kernel assets are immutable to cognitive patches".to_string(),
            );
        }
        if patch.tests.is_empty() || !patch.invariants.contains("rollback-preserved") {
            return PatchDecision::Rejected(
                "patch requires tests and rollback-preserved invariant".to_string(),
            );
        }
        if !report.accepted
            || report.regressions > 0
            || report.verifier_level < VerificationLevel::HeldOut
        {
            return PatchDecision::Rejected(
                "independent held-out validation did not accept the patch".to_string(),
            );
        }
        PatchDecision::Candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch(target: PatchTarget) -> CognitivePatch {
        CognitivePatch {
            id: PatchId(1),
            target,
            purpose: "make retrieval cheaper".to_string(),
            invariants: BTreeSet::from(["rollback-preserved".to_string()]),
            tests: vec!["held-out retrieval suite".to_string()],
            rollback: RevisionId(1),
        }
    }

    fn accepted() -> ValidationReport {
        ValidationReport {
            correctness_delta_milliunits: 0,
            latency_delta_ns: -10,
            active_bytes_delta: 0,
            regressions: 0,
            verifier_level: VerificationLevel::HeldOut,
            accepted: true,
        }
    }

    #[test]
    fn patch_cannot_modify_kernel_benchmark_or_its_own_verifier() {
        let kernel = ValidationKernel;
        assert!(matches!(
            kernel.validate(&patch(PatchTarget::TrustedKernel), &accepted()),
            PatchDecision::Rejected(_)
        ));
        assert!(matches!(
            kernel.validate(&patch(PatchTarget::Benchmark), &accepted()),
            PatchDecision::Rejected(_)
        ));
        assert!(matches!(
            kernel.validate(&patch(PatchTarget::Verifier), &accepted()),
            PatchDecision::Rejected(_)
        ));
        assert_eq!(
            kernel.validate(&patch(PatchTarget::ProgramLibrary), &accepted()),
            PatchDecision::Candidate
        );
    }
}
