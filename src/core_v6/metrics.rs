//! Métricas do runtime V6; nenhuma unidade energética é inferida sem origem.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CognitiveMetrics {
    pub factor_visits: u64,
    pub messages_processed: u64,
    pub messages_suppressed: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub active_factors_peak: u64,
    pub programs_executed: u64,
    pub compiled_hits: u64,
    pub deoptimizations: u64,
    pub retrieval_hits: u64,
    pub reframe_attempts: u64,
    pub structural_mutations: u64,
    pub local_updates: u64,
    pub reused_verified_structures: u64,
    pub total_cognitive_operations: u64,
}

impl CognitiveMetrics {
    pub fn active_byte_ratio(&self, active_bytes: u64, total_knowledge_bytes: u64) -> f64 {
        if total_knowledge_bytes == 0 {
            0.0
        } else {
            active_bytes as f64 / total_knowledge_bytes as f64
        }
    }

    pub fn knowledge_scaling_coefficient(
        &self,
        inference_cost_delta: f64,
        knowledge_size_delta: u64,
    ) -> f64 {
        if knowledge_size_delta == 0 {
            0.0
        } else {
            inference_cost_delta / knowledge_size_delta as f64
        }
    }

    pub fn cognitive_reuse(&self) -> f64 {
        if self.total_cognitive_operations == 0 {
            0.0
        } else {
            self.reused_verified_structures as f64 / self.total_cognitive_operations as f64
        }
    }
}
