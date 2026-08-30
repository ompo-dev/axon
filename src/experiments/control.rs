use std::collections::BTreeSet;

use crate::core_v4::{
    CandidateStrategy, Capability, CognitiveLevel, CognitiveMode, CognitiveRequest,
    CognitiveScheduler, CognitiveSignals, ComputeBudget, FirewallDecision, KnowledgeMutation,
    KnowledgeTier, MemoryFirewall, ReframeEvidence, ResourceUse, ReversibleJournal,
    StrategyOutcome,
};

/// Soma de recursos declarados por uma política de controle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ResourceTotals {
    pub events_processed: u32,
    pub bytes_moved: u64,
    pub microjoules: u64,
}

impl ResourceTotals {
    fn add(self, use_: ResourceUse) -> Self {
        Self {
            events_processed: self.events_processed + use_.events_processed,
            bytes_moved: self.bytes_moved + use_.bytes_moved,
            microjoules: self.microjoules + use_.microjoules,
        }
    }
}

/// Evidência do plano de controle, sem alegar que microjoules estimados são físicos.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlReport {
    pub tasks: u32,
    pub correct_selections: u32,
    pub scheduled_estimated_use: ResourceTotals,
    pub always_deliberate_estimated_use: ResourceTotals,
    pub protected_mutations: u32,
    pub protected_forks: u32,
    pub rollback_successes: u32,
}

pub(super) fn run() -> ControlReport {
    let budget = ComputeBudget::new(4, 512, 100).expect("built-in budget is valid");
    let mut scheduled = ResourceTotals::default();
    let mut correct = 0_u32;
    let mut tasks = 0_u32;

    let lookup = lookup_scenario(budget);
    correct += u32::from(lookup.mode == CognitiveMode::Retrieve);
    scheduled = scheduled.add(lookup.estimated_use);
    tasks += 1;

    let routine = routine_scenario(budget);
    correct += u32::from(routine.mode == CognitiveMode::Reflex);
    scheduled = scheduled.add(routine.estimated_use);
    tasks += 1;

    let deliberate = deliberate_scenario(budget);
    correct += u32::from(deliberate.mode == CognitiveMode::Think);
    scheduled = scheduled.add(deliberate.estimated_use);
    tasks += 1;

    let reframe = reframe_scenario(budget);
    correct += u32::from(reframe.mode == CognitiveMode::Reframe);
    scheduled = scheduled.add(reframe.estimated_use);
    tasks += 1;

    let always_deliberate = (0..tasks).fold(ResourceTotals::default(), |total, _| {
        total.add(ResourceUse::new(4, 480, 90))
    });

    let firewall = MemoryFirewall;
    let mut journal = ReversibleJournal::default();
    let mut protected_forks = 0_u32;
    let mut rollback_successes = 0_u32;
    const PROTECTED_MUTATIONS: u32 = 12;
    for id in 0..PROTECTED_MUTATIONS {
        let mutation = KnowledgeMutation::new(
            format!("protected-rule-{id}"),
            "verified",
            "unverified-change",
        );
        protected_forks += u32::from(matches!(
            firewall.propose(KnowledgeTier::Protected, mutation.clone()),
            FirewallDecision::ForkCandidate { .. }
        ));
        journal = journal.append(mutation.clone());
        rollback_successes += u32::from(journal.rollback(mutation.id) == Some(mutation));
    }

    ControlReport {
        tasks,
        correct_selections: correct,
        scheduled_estimated_use: scheduled,
        always_deliberate_estimated_use: always_deliberate,
        protected_mutations: PROTECTED_MUTATIONS,
        protected_forks,
        rollback_successes,
    }
}

fn lookup_scenario(budget: ComputeBudget) -> crate::core_v4::ScheduledPlan {
    let request = request(
        "lookup",
        signals(0.1, 0.8, 0.1, 0.1),
        [
            Capability::SemanticLookup,
            Capability::DeliberativeReasoning,
        ],
    );
    let candidates = [
        candidate(
            CognitiveMode::Retrieve,
            CognitiveLevel::L0,
            0.96,
            0.10,
            ResourceUse::new(1, 32, 3),
            [Capability::SemanticLookup],
        ),
        candidate(
            CognitiveMode::Think,
            CognitiveLevel::L2,
            0.99,
            0.40,
            ResourceUse::new(3, 384, 72),
            [Capability::DeliberativeReasoning],
        ),
    ];
    CognitiveScheduler::new(budget)
        .plan(&request, &candidates)
        .expect("lookup has a viable strategy")
}

fn routine_scenario(budget: ComputeBudget) -> crate::core_v4::ScheduledPlan {
    let request = request(
        "routine",
        signals(0.1, 0.8, 0.1, 0.1),
        [Capability::ProceduralCircuit, Capability::SemanticLookup],
    );
    let candidates = [
        candidate(
            CognitiveMode::Reflex,
            CognitiveLevel::L0,
            0.70,
            0.10,
            ResourceUse::new(1, 32, 3),
            [Capability::ProceduralCircuit],
        ),
        candidate(
            CognitiveMode::Retrieve,
            CognitiveLevel::L0,
            0.85,
            0.10,
            ResourceUse::new(1, 64, 5),
            [Capability::SemanticLookup],
        ),
    ];
    let scheduler = (0..3).fold(CognitiveScheduler::new(budget), |scheduler, _| {
        scheduler.record_outcome(
            "routine",
            CognitiveMode::Reflex,
            StrategyOutcome {
                verified: true,
                actual_use: ResourceUse::new(1, 32, 3),
            },
        )
    });
    scheduler
        .plan(&request, &candidates)
        .expect("routine has a viable strategy")
}

fn deliberate_scenario(budget: ComputeBudget) -> crate::core_v4::ScheduledPlan {
    let request = request(
        "reason",
        signals(0.7, 0.9, 0.7, 0.4),
        [
            Capability::SemanticLookup,
            Capability::DeliberativeReasoning,
        ],
    );
    let candidates = [
        candidate(
            CognitiveMode::Retrieve,
            CognitiveLevel::L0,
            0.60,
            0.10,
            ResourceUse::new(1, 32, 3),
            [Capability::SemanticLookup],
        ),
        candidate(
            CognitiveMode::Think,
            CognitiveLevel::L2,
            0.93,
            0.70,
            ResourceUse::new(3, 320, 60),
            [Capability::DeliberativeReasoning],
        ),
    ];
    CognitiveScheduler::new(budget)
        .plan(&request, &candidates)
        .expect("reasoning has a viable strategy")
}

fn reframe_scenario(budget: ComputeBudget) -> crate::core_v4::ScheduledPlan {
    let request = request(
        "anomaly",
        signals(1.0, 0.9, 0.9, 0.8),
        [Capability::Reframe],
    )
    .with_reframe_evidence(ReframeEvidence::try_new(0.9, 0.9, 0.9, 0.05).unwrap());
    let candidates = [
        candidate(
            CognitiveMode::Ask,
            CognitiveLevel::L0,
            0.99,
            0.10,
            ResourceUse::new(1, 8, 1),
            [],
        ),
        candidate(
            CognitiveMode::Reframe,
            CognitiveLevel::L5,
            0.85,
            0.80,
            ResourceUse::new(4, 480, 90),
            [Capability::Reframe],
        ),
    ];
    CognitiveScheduler::new(budget)
        .plan(&request, &candidates)
        .expect("reframe has a viable strategy")
}

fn request(
    task: &str,
    signals: CognitiveSignals,
    capabilities: impl IntoIterator<Item = Capability>,
) -> CognitiveRequest {
    CognitiveRequest::new(
        task,
        signals,
        capabilities.into_iter().collect::<BTreeSet<_>>(),
    )
}

fn signals(surprise: f32, goal: f32, uncertainty: f32, risk: f32) -> CognitiveSignals {
    CognitiveSignals::try_new(surprise, goal, uncertainty, risk, 0.1, 0.1)
        .expect("built-in signals are normalized")
}

fn candidate(
    mode: CognitiveMode,
    level: CognitiveLevel,
    confidence: f32,
    gain: f32,
    use_: ResourceUse,
    capabilities: impl IntoIterator<Item = Capability>,
) -> CandidateStrategy {
    CandidateStrategy::try_new(
        mode,
        level,
        confidence,
        gain,
        use_,
        capabilities.into_iter().collect::<BTreeSet<_>>(),
    )
    .expect("built-in candidate is valid")
}
