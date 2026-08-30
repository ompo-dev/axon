use std::collections::{BTreeMap, BTreeSet};

use crate::core_v5::{
    AbstractionCompiler, ActiveExperimentPlanner, BackendProfile, CognitiveOpcode,
    CognitiveOperation, CostOrigin, CostVector, ExperimentPlan, Intervention, LocationPlasticity,
    MacroDispatch, PhysicalBackend, PhysicalCognitiveCompiler, PopulationOfWorlds, ReversibleState,
    ThermalCandidate, ThermodynamicBudget, ThermodynamicScheduler, ThoughtProfiler, WorldFamily,
    WorldFitness, WorldModel,
};

/// Resultados do protótipo V5/Ω. Custos permanecem declarados nesta rodada.
#[derive(Clone, Debug, PartialEq)]
pub struct V5OmegaReport {
    pub program_holdout_accuracy: f32,
    pub compression_ratio: f32,
    pub population_families_retained: u32,
    pub active_intervention_selected: bool,
    pub reversible_rollback_exact: bool,
    pub active_concepts: u32,
    pub dormant_concepts: u32,
    pub route_cost_before: u32,
    pub route_cost_after: u32,
    pub thought_macro_compiled: bool,
    pub thought_macro_deoptimized: bool,
    pub exact_backend_is_digital: bool,
    pub similarity_backend_is_approximate: bool,
    pub cost_origin: CostOrigin,
}

impl V5OmegaReport {
    pub fn to_markdown(&self) -> String {
        format!(
            "# Axon V5/Ω Lab\n\
\n- ProgramCell em holdout: {:.1}%; compressão: {:.1}x.\n\
- População estrutural: {} famílias; intervenção ativa: {}.\n\
- Rollback reversível: {}; conjunto ativo/dormente: {}/{}.\n\
- Localidade lógica: custo de rota {} -> {}.\n\
- Thought JIT compilou/deotimizou: {}/{}.\n\
- Backend exato digital: {}; similaridade aproximada: {}.\n\
- Custos desta execução: {:?} (não são telemetria física).\n",
            self.program_holdout_accuracy * 100.0,
            self.compression_ratio,
            self.population_families_retained,
            self.active_intervention_selected,
            self.reversible_rollback_exact,
            self.active_concepts,
            self.dormant_concepts,
            self.route_cost_before,
            self.route_cost_after,
            self.thought_macro_compiled,
            self.thought_macro_deoptimized,
            self.exact_backend_is_digital,
            self.similarity_backend_is_approximate,
            self.cost_origin,
        )
    }
}

pub(super) fn run() -> V5OmegaReport {
    let training = tokens(&["A", "B", "A", "B", "A", "B"]);
    let holdout = tokens(&["A", "B", "A", "B"]);
    let induction = AbstractionCompiler::default()
        .induce_repeating_pair("alternate", &training, &holdout)
        .expect("the built-in trace has a compressive repeating pair");
    let program_holdout_accuracy = f32::from(induction.cell.execute(2) == holdout);
    let compression_ratio =
        induction.raw_description_length as f32 / induction.program_description_length as f32;

    let population = PopulationOfWorlds::default()
        .admit(world("direct", WorldFamily::Direct, 0.8))
        .admit(world("reverse", WorldFamily::Reverse, 0.8))
        .admit(world("latent", WorldFamily::Latent, 0.8));
    let population_families_retained = population
        .select_diverse(3)
        .into_iter()
        .map(|world| world.family)
        .collect::<BTreeSet<_>>()
        .len() as u32;
    let active_intervention_selected = matches!(
        ActiveExperimentPlanner.choose(&[
            Intervention {
                id: "correlation".to_string(),
                predicted_outcomes: BTreeMap::from([
                    ("direct".to_string(), true),
                    ("reverse".to_string(), true),
                    ("latent".to_string(), true),
                ]),
                cost: CostVector::declared(1, 1, 0, 0, 1),
            },
            Intervention {
                id: "intervene-a".to_string(),
                predicted_outcomes: BTreeMap::from([
                    ("direct".to_string(), true),
                    ("reverse".to_string(), false),
                    ("latent".to_string(), false),
                ]),
                cost: CostVector::declared(1, 1, 0, 0, 1),
            },
        ]),
        ExperimentPlan::Run(choice) if choice.id == "intervene-a"
    );

    let original = ReversibleState::default().with_value("hypothesis", "A-causes-B");
    let reversible_rollback_exact = original
        .apply("hypothesis", "C-common-cause", "counterfactual-failure")
        .undo_last()
        .is_some_and(|restored| {
            restored.get("hypothesis") == original.get("hypothesis") && restored.journal_len() == 1
        });

    let placement = LocationPlasticity::new(2)
        .place("einstein", 1)
        .place("relativity", 9);
    let route_cost_before = placement
        .route_cost("einstein", "relativity")
        .expect("concepts are placed");
    let route_cost_after = placement
        .observe_joint_use("einstein", "relativity")
        .observe_joint_use("einstein", "relativity")
        .route_cost("einstein", "relativity")
        .expect("concepts remain placed");

    let scheduler = ThermodynamicScheduler::default();
    let budget = ThermodynamicBudget {
        max_weighted_cost: 100.0,
        origin: CostOrigin::Declared,
    };
    let plan = scheduler
        .schedule(
            budget,
            &[
                ThermalCandidate {
                    id: "wake-all".to_string(),
                    active_concepts: (0..128).map(|id| format!("c{id}")).collect(),
                    estimated_cost: CostVector::declared(8, 20, 0, 0, 1),
                    utility_milliunits: 20,
                },
                ThermalCandidate {
                    id: "local-query".to_string(),
                    active_concepts: BTreeSet::from([
                        "einstein".to_string(),
                        "relativity".to_string(),
                    ]),
                    estimated_cost: CostVector::declared(2, 2, 0, 0, 1),
                    utility_milliunits: 10,
                },
            ],
        )
        .expect("a declared plan fits the budget");
    let active_concepts = plan.active_concepts.len() as u32;
    let dormant_concepts = 128 - active_concepts;

    let guards = BTreeSet::from(["dense-key".to_string()]);
    let profiler = ThoughtProfiler::new(2).expect("a positive threshold is valid");
    let trace = [CognitiveOpcode::Bind, CognitiveOpcode::Compare];
    let (profiler, _) = profiler
        .record_verified_trace(&trace, &guards)
        .expect("nonempty trace is valid");
    let (profiler, macro_) = profiler
        .record_verified_trace(&trace, &guards)
        .expect("nonempty trace is valid");
    let macro_ = macro_.expect("two verified runs compile an instruction");
    let thought_macro_compiled = true;
    let thought_macro_deoptimized = matches!(
        profiler.dispatch(&macro_, &BTreeSet::new()),
        MacroDispatch::Deoptimized { .. }
    );

    let exact_backend = PhysicalCognitiveCompiler
        .select(
            CognitiveOperation::ExactVerification,
            &[
                profile(
                    PhysicalBackend::HdcApprox,
                    false,
                    &[CognitiveOperation::ExactVerification],
                    1,
                ),
                profile(
                    PhysicalBackend::CpuExact,
                    true,
                    &[CognitiveOperation::ExactVerification],
                    10,
                ),
            ],
        )
        .expect("exact CPU profile is available");
    let similarity_backend = PhysicalCognitiveCompiler
        .select(
            CognitiveOperation::SimilaritySearch,
            &[
                profile(
                    PhysicalBackend::CpuExact,
                    true,
                    &[CognitiveOperation::SimilaritySearch],
                    10,
                ),
                profile(
                    PhysicalBackend::HdcApprox,
                    false,
                    &[CognitiveOperation::SimilaritySearch],
                    1,
                ),
            ],
        )
        .expect("similarity profiles are available");

    V5OmegaReport {
        program_holdout_accuracy,
        compression_ratio,
        population_families_retained,
        active_intervention_selected,
        reversible_rollback_exact,
        active_concepts,
        dormant_concepts,
        route_cost_before,
        route_cost_after,
        thought_macro_compiled,
        thought_macro_deoptimized,
        exact_backend_is_digital: exact_backend.backend == PhysicalBackend::CpuExact,
        similarity_backend_is_approximate: similarity_backend.backend == PhysicalBackend::HdcApprox,
        cost_origin: exact_backend.estimated_cost.origin,
    }
}

fn tokens(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| (*value).to_string()).collect()
}

fn world(id: &str, family: WorldFamily, fitness: f32) -> WorldModel {
    WorldModel {
        id: id.to_string(),
        parent: None,
        family,
        assumptions: BTreeSet::new(),
        transformations: Vec::new(),
        fitness: WorldFitness {
            prediction: fitness,
            generalization: fitness,
            simplicity: fitness,
            novelty: fitness,
            falsifiability: fitness,
        },
    }
}

fn profile(
    backend: PhysicalBackend,
    supports_exact: bool,
    operations: &[CognitiveOperation],
    compute_ops: u64,
) -> BackendProfile {
    BackendProfile {
        backend,
        supports_exact,
        supported_operations: operations.to_vec(),
        estimated_cost: CostVector::declared(compute_ops, 0, 0, 0, 1),
    }
}
