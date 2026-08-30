//! Experimentos sintéticos determinísticos para as primitivas V3/V4.
//!
//! Esta camada mede hipóteses arquiteturais sem integrar o runtime estável ou
//! transformar estimativas de custo em alegações de hardware.

mod control;
mod factorized;
mod hypercell;
mod jump;

pub use control::{ControlReport, ResourceTotals};
pub use factorized::GeneralizationReport;
pub use hypercell::HypercellReport;
pub use jump::JumpReport;

/// Resultado completo de uma rodada do laboratório sintético.
#[derive(Clone, Debug, PartialEq)]
pub struct ScientificReport {
    pub hypercell: HypercellReport,
    pub generalization: GeneralizationReport,
    pub jump: JumpReport,
    pub control: ControlReport,
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
  próxima etapa exige instrumentação de bytes, tempo e energia no runtime real.\n",
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
    }

    #[test]
    fn suite_is_reproducible_across_repeated_runs() {
        assert_eq!(run_scientific_suite(), run_scientific_suite());
    }
}
