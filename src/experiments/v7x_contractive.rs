//! Laboratório V7-X: quatro hipóteses contrativas, determinísticas e refutáveis.

use crate::core_v7x::{
    CapitalDispatch, CognitiveCapitalRuntime, ContractedFactor, DecisionCertificate,
    FactorContract, FactorImplementation, ImplementationKind, MIB, SemanticResolution,
    SemanticVirtualMemory, VersionedWorld, WorldBase,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticCapabilityPoint {
    pub budget_mib: u64,
    pub active_mib: u64,
    pub archived_mib: u64,
    pub protected_recall: f64,
    pub exact_recall: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct V7XContractiveReport {
    pub certificate_preserved: bool,
    pub approximate_rejected_when_unstable: bool,
    pub selected_implementation: ImplementationKind,
    pub semantic_envelope: Vec<SemanticCapabilityPoint>,
    pub transactional_remorph_preserves_protected_facts: bool,
    pub worlds_tested: usize,
    pub cow_storage_ratio: f64,
    pub cow_branch_values_preserved: bool,
    pub capital_capability_accuracy: f64,
    pub capital_first_cost: u64,
    pub capital_familiar_cost: u64,
    pub verified_reusable_structures: u64,
}

impl V7XContractiveReport {
    pub fn to_markdown(&self) -> String {
        let semantic_rows = self
            .semantic_envelope
            .iter()
            .map(|point| {
                format!(
                    "| {} MiB | {} MiB | {} MiB | {:.1}% | {:.1}% |",
                    point.budget_mib,
                    point.active_mib,
                    point.archived_mib,
                    point.protected_recall * 100.0,
                    point.exact_recall * 100.0,
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "# AXON V7-X / Contractive Lab\n\
\n- REALIZE preservou certificado: {}; aproximado barato recusado: {}; realização escolhida: {:?}.\n\
- Remorph transacional preservou fatos protegidos: {}.\n\
- Mundos COW: {}; razão full-copy/shared: {:.1}x; valores por branch preservados: {}.\n\
- Capital cognitivo: capability exata {:.1}%; custo primeiro/familiar: {}/{} unidades; estruturas verificadas: {}.\n\
\n| Orçamento semântico lógico | Ativo | Arquivado | Recall protegido | Recall exato completo |\n\
|---:|---:|---:|---:|---:|\n\
{semantic_rows}\n\
\nA tabela mede a capacidade do corpus determinístico e o armazenamento lógico por resolução; não mede AGI nem comprime fisicamente 4 GiB de dados.\n",
            self.certificate_preserved,
            self.approximate_rejected_when_unstable,
            self.selected_implementation,
            self.transactional_remorph_preserves_protected_facts,
            self.worlds_tested,
            self.cow_storage_ratio,
            self.cow_branch_values_preserved,
            self.capital_capability_accuracy * 100.0,
            self.capital_first_cost,
            self.capital_familiar_cost,
            self.verified_reusable_structures,
        )
    }
}

pub(super) fn run() -> V7XContractiveReport {
    let required = contract(10, &["same-route"]);
    let factor = ContractedFactor {
        required_contract: required,
        certificate: DecisionCertificate {
            winner_lower_bound: 110,
            runner_up_upper_bound: 100,
        },
        implementations: vec![
            implementation(ImplementationKind::Exact, 0, 100, &["same-route"]),
            implementation(ImplementationKind::Approximate, 6, 1, &["same-route"]),
            implementation(
                ImplementationKind::Compiled,
                0,
                10,
                &["same-route", "compiled-guard"],
            ),
        ],
    };
    let realization = factor.realize().expect("compiled contract is eligible");

    let budgets_mib = [64, 128, 512, 4_096];
    let semantic_envelope = budgets_mib
        .into_iter()
        .map(|budget_mib| {
            let memory = SemanticVirtualMemory::synthetic(1_024, budget_mib * MIB)
                .expect("summary tier fits all V7-X budgets");
            SemanticCapabilityPoint {
                budget_mib,
                active_mib: memory.morphology().active_bytes / MIB,
                archived_mib: memory.morphology().archived_detail_bytes / MIB,
                protected_recall: memory.protected_recall_fraction(),
                exact_recall: memory.recall_fraction(SemanticResolution::Exact),
            }
        })
        .collect::<Vec<_>>();
    let low_memory = SemanticVirtualMemory::synthetic(1_024, 64 * MIB).unwrap();
    let remorphed = low_memory
        .commit(low_memory.stage_remorph(512 * MIB).unwrap())
        .unwrap();

    let world = VersionedWorld::from_base(WorldBase::new((0..1_024).collect()));
    let branches = (0..10_000)
        .map(|index| {
            world
                .fork(&[(index % 1_024, (index as u64).wrapping_mul(17))])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let shared_bytes = VersionedWorld::shared_footprint_bytes(&branches).max(1);
    let full_copy_bytes =
        VersionedWorld::full_copy_footprint_bytes(branches.len(), world.base_bytes());
    let cow_branch_values_preserved = branches.iter().enumerate().all(|(index, branch)| {
        branch.value(index % 1_024) == Some((index as u64).wrapping_mul(17))
    });

    let mut capital = CognitiveCapitalRuntime::new(3).unwrap();
    let mut correct = 0_u64;
    let mut first_cost = 0;
    let mut familiar_cost = 0;
    for call in 0..8 {
        let (next, outcome) = capital.solve("triangular", 256).unwrap();
        correct += u64::from(outcome.answer == 32_896 && outcome.verified);
        if call == 0 {
            first_cost = outcome.primitive_cost_units;
        }
        if outcome.dispatch == CapitalDispatch::Compiled {
            familiar_cost = outcome.primitive_cost_units;
        }
        capital = next;
    }

    V7XContractiveReport {
        certificate_preserved: realization.preserved_certificate,
        approximate_rejected_when_unstable: realization.kind != ImplementationKind::Approximate,
        selected_implementation: realization.kind,
        semantic_envelope,
        transactional_remorph_preserves_protected_facts: low_memory.protected_recall_fraction()
            == 1.0
            && remorphed.protected_recall_fraction() == 1.0,
        worlds_tested: branches.len(),
        cow_storage_ratio: full_copy_bytes as f64 / shared_bytes as f64,
        cow_branch_values_preserved,
        capital_capability_accuracy: correct as f64 / 8.0,
        capital_first_cost: first_cost,
        capital_familiar_cost: familiar_cost,
        verified_reusable_structures: capital.verified_capital(),
    }
}

fn contract(error: u64, claims: &[&str]) -> FactorContract {
    FactorContract::new(
        "factor:route",
        1,
        error,
        claims.iter().map(|claim| (*claim).to_string()).collect(),
        0xC0DE,
    )
}

fn implementation(
    kind: ImplementationKind,
    error: u64,
    cost: u64,
    claims: &[&str],
) -> FactorImplementation {
    FactorImplementation {
        kind,
        contract: contract(error, claims),
        lifetime_cost_units: cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v7x_report_separates_contracts_envelope_worlds_and_capital() {
        let report = run();

        assert!(report.certificate_preserved);
        assert!(report.approximate_rejected_when_unstable);
        assert_eq!(report.selected_implementation, ImplementationKind::Compiled);
        assert_eq!(report.semantic_envelope.len(), 4);
        assert!(
            report
                .semantic_envelope
                .iter()
                .all(|point| point.protected_recall == 1.0)
        );
        assert!(
            report.semantic_envelope.last().unwrap().exact_recall
                > report.semantic_envelope.first().unwrap().exact_recall
        );
        assert!(report.transactional_remorph_preserves_protected_facts);
        assert_eq!(report.worlds_tested, 10_000);
        assert!(report.cow_storage_ratio > 100.0);
        assert!(report.cow_branch_values_preserved);
        assert_eq!(report.capital_capability_accuracy, 1.0);
        assert!(report.capital_familiar_cost < report.capital_first_cost);
        assert_eq!(report.verified_reusable_structures, 1);
    }
}
