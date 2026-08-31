//! Laboratorio V7: morfogenese cognitiva sob orcamento de memoria.

use crate::core_v7::{
    CognitiveRegion, MorphogenicCompiler, RegionTier, ResourceBudget, WorkloadProfile,
};

const MIB: u64 = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct V7MorphogenicReport {
    pub budgets_tested: usize,
    pub quality_is_monotonic: bool,
    pub same_seed_is_deterministic: bool,
    pub low_memory_archives_more_than_it_runs: bool,
    pub high_memory_expands_programs: bool,
    pub high_memory_expands_candidate_worlds: bool,
    pub minimum_seed_rejected: bool,
    pub q64: f64,
    pub q512: f64,
    pub q16g: f64,
}

impl V7MorphogenicReport {
    pub fn to_markdown(&self) -> String {
        format!(
            "# Axon V7 / Morphogenic Runtime Lab\n\
\n- Orcamentos testados: {}.\n\
- Mesma seed deterministica: {}.\n\
- Qualidade monotona com memoria: {}.\n\
- 64 MiB arquiva mais do que executa: {}.\n\
- 16 GiB expande ProgramCache: {}.\n\
- 16 GiB expande CandidateWorlds: {}.\n\
- Orcamento abaixo do seed recusado: {}.\n\
- Q(64 MiB): {:.3}; Q(512 MiB): {:.3}; Q(16 GiB): {:.3}.\n\
- Resultado: valida uma economia cognitiva deterministica sob memoria; nao\n  prova linguagem, corpo fisico ou AGI geral.\n",
            self.budgets_tested,
            self.same_seed_is_deterministic,
            self.quality_is_monotonic,
            self.low_memory_archives_more_than_it_runs,
            self.high_memory_expands_programs,
            self.high_memory_expands_candidate_worlds,
            self.minimum_seed_rejected,
            self.q64,
            self.q512,
            self.q16g,
        )
    }
}

pub(super) fn run() -> V7MorphogenicReport {
    let compiler = MorphogenicCompiler::default();
    let workload = WorkloadProfile::research();
    let budgets = [64, 128, 256, 512, 1_024, 4_096, 16_384];
    let plans = budgets
        .into_iter()
        .map(|mib| {
            compiler
                .compile(ResourceBudget::memory_only(mib * MIB), workload)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let quality_is_monotonic = plans
        .windows(2)
        .all(|pair| pair[1].quality >= pair[0].quality);
    let same_seed_is_deterministic = compiler
        .compile(ResourceBudget::memory_only(512 * MIB), workload)
        .unwrap()
        == compiler
            .compile(ResourceBudget::memory_only(512 * MIB), workload)
            .unwrap();
    let low = &plans[0];
    let high = plans.last().expect("plans");
    let high_memory_expands_programs = high
        .allocation(CognitiveRegion::ProgramCache)
        .is_some_and(|allocation| allocation.tier == RegionTier::Expanded);
    let high_memory_expands_candidate_worlds = high
        .allocation(CognitiveRegion::CandidateWorlds)
        .is_some_and(|allocation| allocation.tier == RegionTier::Expanded);
    let minimum_seed_rejected = compiler
        .compile(ResourceBudget::memory_only(63 * MIB), workload)
        .is_err();

    V7MorphogenicReport {
        budgets_tested: plans.len(),
        quality_is_monotonic,
        same_seed_is_deterministic,
        low_memory_archives_more_than_it_runs: low.archived_bytes > low.active_bytes,
        high_memory_expands_programs,
        high_memory_expands_candidate_worlds,
        minimum_seed_rejected,
        q64: plans[0].quality,
        q512: plans[3].quality,
        q16g: high.quality,
    }
}
