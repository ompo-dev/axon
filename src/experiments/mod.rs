//! Experimentos sintéticos determinísticos para as primitivas V3/V4.
//!
//! Esta camada mede hipóteses arquiteturais sem integrar o runtime estável ou
//! transformar estimativas de custo em alegações de hardware.

mod control;
mod factorized;
mod hypercell;
mod jump;
mod lambda_kernel;
mod v5_omega;
mod v6_omega;
mod v6x_physics;
mod v7_morphogenic;
mod v7x_contractive;

pub use control::{ControlReport, ResourceTotals};
pub use factorized::GeneralizationReport;
pub use hypercell::HypercellReport;
pub use jump::JumpReport;
pub use lambda_kernel::LambdaKernelReport;
pub use v5_omega::V5OmegaReport;
pub use v6_omega::V6OmegaReport;
pub use v6x_physics::V6XPhysicsReport;
pub use v7_morphogenic::V7MorphogenicReport;
pub use v7x_contractive::V7XContractiveReport;

/// Resultado completo de uma rodada do laboratório sintético.
#[derive(Clone, Debug, PartialEq)]
pub struct ScientificReport {
    pub hypercell: HypercellReport,
    pub generalization: GeneralizationReport,
    pub jump: JumpReport,
    pub control: ControlReport,
    pub v5_omega: V5OmegaReport,
    pub v6_omega: V6OmegaReport,
    pub v6_x: V6XPhysicsReport,
    pub v7_morphogenic: V7MorphogenicReport,
    pub v7_x: V7XContractiveReport,
    pub lambda: LambdaKernelReport,
}

/// Executa hipóteses pequenas, determinísticas e falsificáveis.
///
/// O relatório não é uma alegação de desempenho em hardware ou de descoberta
/// científica geral: cada resultado vale somente para o mundo sintético que o
/// produziu e para as primitivas explicitamente comparadas.
pub fn run_scientific_suite() -> ScientificReport {
    ScientificReport {
        hypercell: hypercell::run(),
        generalization: factorized::run(),
        jump: jump::run(),
        control: control::run(),
        v5_omega: v5_omega::run(),
        v6_omega: v6_omega::run(),
        v6_x: v6x_physics::run(),
        v7_morphogenic: v7_morphogenic::run(),
        v7_x: v7x_contractive::run(),
        lambda: lambda_kernel::run(),
    }
}

impl ScientificReport {
    /// Renderização estável para execução local e para o registro do experimento.
    pub fn to_markdown(&self) -> String {
        let markdown = format!(
            "# Axon Experimental Lab\n\
\n## 1. Álgebra de HyperCells\n\
- Ensaios: {hyper_trials}; dimensão: {hyper_dimension}.\n\
- Recuperação exata densa bipolar: {dense_exact:.1}%.\n\
- Recuperação por busca com 3% de ruído: {dense_noisy:.1}%.\n\
- Recuperação exata ternária esparsa: {sparse_exact:.1}%.\n\
- Retenção média de sinal ternário esparso: {sparse_retention:.1}%.\n\
- Conclusão: bind multiplicativo é reversível com chaves densas sem zero;\n\
  usar zero como estado do mesmo operador destrói informação e não pode ser\n\
  tratado como algebraicamente auto-inverso.\n\
\n## 2. Generalização por aprendizado local\n\
- Combinações não vistas: {holdouts}.\n\
- Lookup associativo: {lookup_accuracy:.1}%.\n\
- Regra local fatorizada: {factorized_accuracy:.1}%.\n\
- Conclusão: a vantagem aparece neste mundo somente porque a linguagem de\n\
  representação separa os dois fatores causais; não prova generalização geral.\n\
\n## 3. Jump causal por intervenção\n\
- Mundos: {jump_worlds} (direto, reverso e causa latente).\n\
- Adaptação que preserva A→B: {adapt_accuracy:.1}%.\n\
- REFRAME com contrafactuais: {reframe_accuracy:.1}%.\n\
- REFRAME sem intervenções: {observational_accuracy:.1}%.\n\
- Hipóteses avaliadas por mundo reframe: {candidates:.1}.\n\
- Conclusão: observação correlacional sozinha permanece ambígua; intervenções\n\
  são a evidência que permite identificar a mudança estrutural neste benchmark.\n\
\n## 4. Controle, custo declarado e reversibilidade\n\
- Seleções corretas do scheduler: {scheduler_correct}/{scheduler_tasks}.\n\
- Uso estimado do scheduler: {scheduled_events} eventos, {scheduled_bytes} bytes,\n\
  {scheduled_energy} µJ.\n\
- Baseline sempre deliberativo: {baseline_events} eventos, {baseline_bytes} bytes,\n\
  {baseline_energy} µJ.\n\
- Mutações protegidas bifurcadas: {protected_forks}/{protected_mutations};\n\
  rollbacks exatos: {rollbacks}/{protected_mutations}.\n\
- Conclusão: são estimativas declaradas de estratégia, não medição física; a\n\
  próxima etapa exige instrumentação de bytes, tempo e energia no runtime real.\n\
\n## 5. V5/Ω: programas, mundos e custo de informação
\
- ProgramCell induzido em holdout: {v5_program_accuracy:.1}%; compressão: {v5_compression:.1}x.
\
- Famílias estruturais preservadas: {v5_families}; intervenção ativa selecionada: {v5_intervention}.
\
- Rollback reversível exato: {v5_rollback}; conceitos ativos/dormentes: {v5_active}/{v5_dormant}.
\
- Custo lógico de rota antes/depois de coativação: {v5_route_before}/{v5_route_after}.
\
- Thought JIT compilou/deotimizou: {v5_jit_compiled}/{v5_jit_deoptimized}.
\
- Verificação exata em backend digital: {v5_exact}; similaridade em backend aproximado: {v5_approximate}.
\
- Conclusão: os custos e backends nesta etapa são declarados; a V5 valida
\
  contratos de seleção e reversibilidade, não eficiência física de chip.\n",
            hyper_trials = self.hypercell.trials,
            hyper_dimension = self.hypercell.dimension,
            dense_exact = self.hypercell.dense_exact_recovery * 100.0,
            dense_noisy = self.hypercell.dense_noisy_retrieval * 100.0,
            sparse_exact = self.hypercell.sparse_exact_recovery * 100.0,
            sparse_retention = self.hypercell.sparse_signal_retention * 100.0,
            holdouts = self.generalization.holdout_cases,
            lookup_accuracy = self.generalization.lookup_holdout_accuracy * 100.0,
            factorized_accuracy = self.generalization.factorized_holdout_accuracy * 100.0,
            jump_worlds = self.jump.worlds,
            adapt_accuracy = self.jump.direct_adaptation_intervention_accuracy * 100.0,
            reframe_accuracy = self.jump.reframe_intervention_accuracy * 100.0,
            observational_accuracy = self.jump.observational_only_identification_accuracy * 100.0,
            candidates = self.jump.mean_reframe_candidates,
            scheduler_correct = self.control.correct_selections,
            scheduler_tasks = self.control.tasks,
            scheduled_events = self.control.scheduled_estimated_use.events_processed,
            scheduled_bytes = self.control.scheduled_estimated_use.bytes_moved,
            scheduled_energy = self.control.scheduled_estimated_use.microjoules,
            baseline_events = self
                .control
                .always_deliberate_estimated_use
                .events_processed,
            baseline_bytes = self.control.always_deliberate_estimated_use.bytes_moved,
            baseline_energy = self.control.always_deliberate_estimated_use.microjoules,
            protected_forks = self.control.protected_forks,
            protected_mutations = self.control.protected_mutations,
            rollbacks = self.control.rollback_successes,
            v5_program_accuracy = self.v5_omega.program_holdout_accuracy * 100.0,
            v5_compression = self.v5_omega.compression_ratio,
            v5_families = self.v5_omega.population_families_retained,
            v5_intervention = self.v5_omega.active_intervention_selected,
            v5_rollback = self.v5_omega.reversible_rollback_exact,
            v5_active = self.v5_omega.active_concepts,
            v5_dormant = self.v5_omega.dormant_concepts,
            v5_route_before = self.v5_omega.route_cost_before,
            v5_route_after = self.v5_omega.route_cost_after,
            v5_jit_compiled = self.v5_omega.thought_macro_compiled,
            v5_jit_deoptimized = self.v5_omega.thought_macro_deoptimized,
            v5_exact = self.v5_omega.exact_backend_is_digital,
            v5_approximate = self.v5_omega.similarity_backend_is_approximate,
        );
        let markdown = format!(
            "{markdown}\n## 6. V6/Ω6: Factors, ledger e execução adaptativa\n- Fatos indexados/one-shot retido: {}/{}; working set local: {} Factor; ABR: {:.4}.\n- Supersessão nova/histórico: {}/{}; domínio barato/abrangente: {}/{}.\n- Mensagens suprimidas/processadas: {}/{}; Thought JIT compilou/deotimizou/equivalente: {}/{}/{}.\n- Learnability Gate completo/experimento ativo: {}/{}; conhecimento negativo/patch limitado: {}/{}.\n- Conclusão: a V6 valida contratos determinísticos para Factors, trabalho ativo, conhecimento versionado e programas verificáveis; não mede hardware ou escalabilidade industrial.\n",
            self.v6_omega.facts_stored,
            self.v6_omega.one_shot_fact_retained,
            self.v6_omega.active_factors_for_local_query,
            self.v6_omega.active_byte_ratio,
            self.v6_omega.supersession_current_is_new,
            self.v6_omega.supersession_history_len,
            self.v6_omega.cheap_domain_model_selected,
            self.v6_omega.broad_domain_model_selected,
            self.v6_omega.messages_suppressed,
            self.v6_omega.messages_processed,
            self.v6_omega.jit_compiled,
            self.v6_omega.jit_deoptimized,
            self.v6_omega.jit_equivalent_result,
            self.v6_omega.learnability_gate_complete,
            self.v6_omega.active_experiment_selected,
            self.v6_omega.negative_knowledge_retained,
            self.v6_omega.patch_is_safely_limited,
        );
        let markdown = format!(
            "{markdown}\n## 7. V6-X / Ψ-IR: realização física auditável\n- Prova exata digital: {}; Boundary Tax bloqueou p-bit falso-barato: {}; p-bit co-localizado para sampling: {}.\n- Proveniência mista recusada: {}; UNCOMPUTE válido/barato: {}; ERASE quando mais barato: {}; custos declarados: {}.\n- Conclusão: são planos declarados de realização, não medições de hardware físico.\n",
            self.v6_x.exact_verification_uses_digital,
            self.v6_x.boundary_tax_rejects_false_pbit_win,
            self.v6_x.pbit_sampling_selected_when_colocated,
            self.v6_x.mixed_cost_provenance_rejected,
            self.v6_x.uncompute_selected_when_valid_and_cheaper,
            self.v6_x.erase_selected_when_uncompute_is_costlier,
            self.v6_x.all_selected_costs_are_declared,
        );
        let markdown = format!(
            "{markdown}\n## 8. V7: Morphogenic Resource Runtime\n- Orcamentos testados: {}; determinismo: {}; qualidade monotona: {}.\n- 64 MiB arquiva mais do que executa: {}; 16 GiB expande ProgramCache/CandidateWorlds: {}/{}.\n- Q(64 MiB): {:.3}; Q(512 MiB): {:.3}; Q(16 GiB): {:.3}; seed minimo recusado: {}.\n- Conclusao: valida alocacao cognitiva sob memoria; corpo, idioma e ferramentas ainda sao compiladores futuros.\n",
            self.v7_morphogenic.budgets_tested,
            self.v7_morphogenic.same_seed_is_deterministic,
            self.v7_morphogenic.quality_is_monotonic,
            self.v7_morphogenic.low_memory_archives_more_than_it_runs,
            self.v7_morphogenic.high_memory_expands_programs,
            self.v7_morphogenic.high_memory_expands_candidate_worlds,
            self.v7_morphogenic.q64,
            self.v7_morphogenic.q512,
            self.v7_morphogenic.q16g,
            self.v7_morphogenic.minimum_seed_rejected,
        );
        let markdown = format!(
            "{markdown}\n## 9. V7-X: contratos, memória semântica, COW e capital\n- REALIZE preservou certificado/recusou aproximado instável: {}/{}; execução escolhida: {:?}.\n- Envelope semântico: {} pontos; recall protegido sob todos os budgets: {}.\n- Remorph transacional preservou fatos protegidos: {}; COW em {} mundos: {:.1}x de redução lógica; branches preservados: {}.\n- Capital: capability {:.1}%; custo primeiro/familiar: {}/{}; estruturas verificadas: {}.\n- Conclusão: são contratos e experimentos determinísticos; o sweep físico mede cópias, consultas e repetição no host, mas não prova inteligência geral.\n",
            self.v7_x.certificate_preserved,
            self.v7_x.approximate_rejected_when_unstable,
            self.v7_x.selected_implementation,
            self.v7_x.semantic_envelope.len(),
            self.v7_x
                .semantic_envelope
                .iter()
                .all(|point| point.protected_recall == 1.0),
            self.v7_x.transactional_remorph_preserves_protected_facts,
            self.v7_x.worlds_tested,
            self.v7_x.cow_storage_ratio,
            self.v7_x.cow_branch_values_preserved,
            self.v7_x.capital_capability_accuracy * 100.0,
            self.v7_x.capital_first_cost,
            self.v7_x.capital_familiar_cost,
            self.v7_x.verified_reusable_structures,
        );
        let markdown = format!(
            "{markdown}\n## 10. AXON-Λ: contratos, light cone e quotient\n- Refinamento/certificado: {}/{}; cone local B/F/A: {}/{}/{}; paridade e delta: {}/{}.\n- Cascata global A/paridade/full: {}/{}/{}; estado inteiro via overlay: {}.\n- LIFT exato/classes: {}/{}; journal de conformance disponível: {}.\n- Conclusão: a hipótese vale apenas para Factors afins em cadeia; o sweep físico mede 1M de nós materializados separadamente.\n",
            self.lambda.contract_refinement_preserved,
            self.lambda.unsafe_approximate_rejected,
            self.lambda.local_demanded_factors,
            self.lambda.local_changed_factors,
            self.lambda.local_active_factors,
            self.lambda.local_matches_full,
            self.lambda.local_delta_selected,
            self.lambda.global_active_factors,
            self.lambda.global_matches_full,
            self.lambda.global_full_selected,
            self.lambda.overlay_matches_full_state,
            self.lambda.lifted_sum_matches_individual_sum,
            self.lambda.lifted_classes,
            self.lambda.canonical_journal_available,
        );
        markdown.replace("\\\\n\\\\", "\\n")
    }
}

#[cfg(test)]
mod tests {
    use super::run_scientific_suite;

    #[test]
    fn suite_separates_algebra_generalization_jump_and_control_hypotheses() {
        let report = run_scientific_suite();

        assert!(report.hypercell.dense_exact_recovery > 0.99);
        assert!(report.hypercell.sparse_exact_recovery < 0.10);
        assert!(report.generalization.factorized_holdout_accuracy > 0.70);
        assert!(report.generalization.lookup_holdout_accuracy < 0.60);
        assert!(report.jump.reframe_intervention_accuracy > 0.99);
        assert!(report.jump.direct_adaptation_intervention_accuracy < 0.40);
        assert!(report.jump.observational_only_identification_accuracy < 0.40);
        assert!(
            report.control.scheduled_estimated_use.microjoules
                < report.control.always_deliberate_estimated_use.microjoules
        );
        assert_eq!(
            report.control.protected_forks,
            report.control.protected_mutations
        );
        assert_eq!(
            report.control.rollback_successes,
            report.control.protected_mutations
        );
        assert_eq!(report.v5_omega.program_holdout_accuracy, 1.0);
        assert!(report.v5_omega.compression_ratio > 1.0);
        assert_eq!(report.v5_omega.population_families_retained, 3);
        assert!(report.v5_omega.active_intervention_selected);
        assert!(report.v5_omega.reversible_rollback_exact);
        assert!(report.v5_omega.active_concepts < report.v5_omega.dormant_concepts);
        assert!(report.v5_omega.route_cost_after < report.v5_omega.route_cost_before);
        assert!(report.v5_omega.thought_macro_compiled);
        assert!(report.v5_omega.thought_macro_deoptimized);
        assert!(report.v5_omega.exact_backend_is_digital);
        assert!(report.v5_omega.similarity_backend_is_approximate);
        assert!(report.v6_omega.one_shot_fact_retained);
        assert!(report.v6_omega.active_byte_ratio < 0.01);
        assert!(report.v6_omega.supersession_current_is_new);
        assert!(report.v6_omega.cheap_domain_model_selected);
        assert!(report.v6_omega.broad_domain_model_selected);
        assert_eq!(report.v6_omega.messages_suppressed, 1);
        assert_eq!(report.v6_omega.messages_processed, 1);
        assert!(report.v6_omega.jit_compiled);
        assert!(report.v6_omega.jit_deoptimized);
        assert!(report.v6_omega.jit_equivalent_result);
        assert!(report.v6_omega.learnability_gate_complete);
        assert!(report.v6_omega.active_experiment_selected);
        assert!(report.v6_omega.negative_knowledge_retained);
        assert!(report.v6_omega.patch_is_safely_limited);
        assert!(report.v6_x.exact_verification_uses_digital);
        assert!(report.v6_x.boundary_tax_rejects_false_pbit_win);
        assert!(report.v6_x.pbit_sampling_selected_when_colocated);
        assert!(report.v6_x.mixed_cost_provenance_rejected);
        assert!(report.v6_x.uncompute_selected_when_valid_and_cheaper);
        assert!(report.v6_x.erase_selected_when_uncompute_is_costlier);
        assert!(report.v6_x.all_selected_costs_are_declared);
        assert!(report.v7_morphogenic.same_seed_is_deterministic);
        assert!(report.v7_morphogenic.quality_is_monotonic);
        assert!(report.v7_morphogenic.low_memory_archives_more_than_it_runs);
        assert!(report.v7_morphogenic.high_memory_expands_programs);
        assert!(report.v7_morphogenic.high_memory_expands_candidate_worlds);
        assert!(report.v7_morphogenic.minimum_seed_rejected);
        assert!(report.v7_x.certificate_preserved);
        assert!(report.v7_x.approximate_rejected_when_unstable);
        assert!(report.v7_x.transactional_remorph_preserves_protected_facts);
        assert!(report.v7_x.cow_branch_values_preserved);
        assert_eq!(report.v7_x.capital_capability_accuracy, 1.0);
        assert!(report.lambda.local_matches_full);
        assert!(report.lambda.global_full_selected);
        assert!(report.lambda.overlay_matches_full_state);
    }

    #[test]
    fn suite_is_reproducible_across_repeated_runs() {
        assert_eq!(run_scientific_suite(), run_scientific_suite());
    }
}
