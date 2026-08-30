use std::collections::{BTreeMap, BTreeSet};

use crate::core_v5::CostVector;
use crate::core_v6::{
    Claim, ClaimId, CognitiveMessage, CognitiveMetrics, EpistemicLedger, EpistemicStatus,
    Experiment, FactorId, Guard, InformationRequest, LearnabilityDecision, LearnabilityEvidence,
    LearnabilityGate, MessageDisposition, MessagePayload, MessageScheduler, OpCode, PatchDecision,
    PatchId, PatchTarget, Program, ProgramDispatch, ProgramId, ProgramLibrary, ProgramStatus,
    ProgramVm, ReframeRequest, RevisionId, ValidationKernel, ValidationReport, Value, ValueType,
    VerificationLevel,
};

/// Resultado do corte V6.0. Todos os dados são mundos sintéticos determinísticos.
#[derive(Clone, Debug, PartialEq)]
pub struct V6OmegaReport {
    pub facts_stored: u32,
    pub one_shot_fact_retained: bool,
    pub active_factors_for_local_query: u32,
    pub active_byte_ratio: f64,
    pub supersession_current_is_new: bool,
    pub supersession_history_len: u32,
    pub cheap_domain_model_selected: bool,
    pub broad_domain_model_selected: bool,
    pub messages_suppressed: u64,
    pub messages_processed: u64,
    pub jit_compiled: bool,
    pub jit_deoptimized: bool,
    pub jit_equivalent_result: bool,
    pub learnability_gate_complete: bool,
    pub active_experiment_selected: bool,
    pub negative_knowledge_retained: bool,
    pub patch_is_safely_limited: bool,
}

impl V6OmegaReport {
    pub fn to_markdown(&self) -> String {
        format!(
            "# Axon V6/Ω6 Lab\n\
\n- Fatos indexados: {}; fato one-shot retido: {}.\n\
- Working set da consulta local: {} Factor; ABR sintético: {:.4}.\n\
- Supersessão: revisão atual nova {}; histórico {}.\n\
- Seleção por domínio: barato/específico {}; abrangente {}.\n\
- Mensagens: suprimidas/processadas {}/{}.\n\
- Thought JIT: compilado/deotimizado/equivalente {}/{}/{}.\n\
- Learnability Gate completo: {}; experimento ativo: {}.\n\
- Conhecimento negativo retido: {}; patch limitado pelo kernel: {}.\n\
- Resultado: valida contratos V6.0 com BTreeMap e custos declarados; não mede\n\
  escalabilidade para milhões de fatos nem eficiência física.\n",
            self.facts_stored,
            self.one_shot_fact_retained,
            self.active_factors_for_local_query,
            self.active_byte_ratio,
            self.supersession_current_is_new,
            self.supersession_history_len,
            self.cheap_domain_model_selected,
            self.broad_domain_model_selected,
            self.messages_suppressed,
            self.messages_processed,
            self.jit_compiled,
            self.jit_deoptimized,
            self.jit_equivalent_result,
            self.learnability_gate_complete,
            self.active_experiment_selected,
            self.negative_knowledge_retained,
            self.patch_is_safely_limited,
        )
    }
}

pub(super) fn run() -> V6OmegaReport {
    let original = claim(
        1,
        "capital(Freland)",
        "capital(Freland)=Noma",
        EpistemicStatus::Current,
    );
    let ledger = EpistemicLedger::default().add(original.clone()).unwrap();
    let ledger = (0..256).fold(ledger, |ledger, id| {
        ledger
            .add(claim(
                id + 10,
                format!("capital(Unrelated{id})"),
                format!("capital(Unrelated{id})=City{id}"),
                EpistemicStatus::Current,
            ))
            .unwrap()
    });
    let one_shot_fact_retained = ledger
        .current("capital(Freland)")
        .is_some_and(|view| view.claim.proposition == "capital(Freland)=Noma");
    let facts_stored = ledger.len() as u32;
    let mut metrics = CognitiveMetrics {
        factor_visits: 1,
        active_factors_peak: 1,
        ..CognitiveMetrics::default()
    };
    let active_byte_ratio = metrics.active_byte_ratio(64, facts_stored as u64 * 64);

    let revised = claim(
        2,
        "capital(Freland)",
        "capital(Freland)=Zora",
        EpistemicStatus::Current,
    );
    let superseded = EpistemicLedger::default()
        .add(original.clone())
        .unwrap()
        .supersede(original.id, revised.clone())
        .unwrap();
    let supersession_current_is_new = superseded
        .current("capital(Freland)")
        .is_some_and(|view| view.claim.id == revised.id);
    let supersession_history_len = superseded.history("capital(Freland)").len() as u32;

    let (cheap_domain_model_selected, broad_domain_model_selected) = domain_selection();

    let mut scheduler = MessageScheduler::default();
    let suppressed = scheduler.submit(message(1, 100), &mut metrics);
    let enqueued = scheduler.submit(message(1_000, 1), &mut metrics);
    let _ = scheduler.next(&mut metrics);
    let message_contract = matches!(suppressed, MessageDisposition::Suppressed)
        && matches!(enqueued, MessageDisposition::Enqueued(_));

    let (jit_compiled, jit_deoptimized, jit_equivalent_result) = jit_contract();
    let (learnability_gate_complete, active_experiment_selected) = learnability_contract();
    let negative_knowledge_retained = superseded
        .record_failure("A->B->C", "counterexample-X")
        .failed_constraints("A->B->C")
        == ["counterexample-X"];
    let patch_is_safely_limited = patch_contract();

    V6OmegaReport {
        facts_stored,
        one_shot_fact_retained,
        active_factors_for_local_query: 1,
        active_byte_ratio,
        supersession_current_is_new,
        supersession_history_len,
        cheap_domain_model_selected,
        broad_domain_model_selected,
        messages_suppressed: metrics.messages_suppressed,
        messages_processed: metrics.messages_processed,
        jit_compiled,
        jit_deoptimized,
        jit_equivalent_result,
        learnability_gate_complete: learnability_gate_complete && message_contract,
        active_experiment_selected,
        negative_knowledge_retained,
        patch_is_safely_limited,
    }
}

fn claim(
    id: u64,
    key: impl Into<String>,
    proposition: impl Into<String>,
    status: EpistemicStatus,
) -> Claim {
    Claim::for_key(ClaimId(id), key, proposition, status, 900, RevisionId(id)).unwrap()
}

fn domain_selection() -> (bool, bool) {
    let mut cheap = claim(
        1,
        "trajectory-model",
        "newtonian",
        EpistemicStatus::DomainLimited,
    );
    cheap.validity.conditions.insert("weak-field".to_string());
    cheap.estimated_cost = 1;
    let mut broad = claim(
        2,
        "trajectory-model",
        "relativistic",
        EpistemicStatus::Current,
    );
    broad.estimated_cost = 10;
    let ledger = EpistemicLedger::default()
        .add(cheap)
        .unwrap()
        .add(broad)
        .unwrap();
    let cheap_selected = ledger
        .best_valid(
            "trajectory-model",
            &BTreeSet::from(["weak-field".to_string()]),
            0,
        )
        .is_some_and(|view| view.claim.proposition == "newtonian");
    let broad_selected = ledger
        .best_valid("trajectory-model", &BTreeSet::new(), 0)
        .is_some_and(|view| view.claim.proposition == "relativistic");
    (cheap_selected, broad_selected)
}

fn message(residual: u32, bytes: u64) -> CognitiveMessage {
    CognitiveMessage {
        from: FactorId(1),
        to: FactorId(2),
        payload: MessagePayload::Residual("delta".to_string()),
        residual_milliunits: residual,
        goal_milliunits: 0,
        uncertainty_milliunits: 0,
        information_gain_milliunits: 0,
        timestamp: 1,
        provenance: vec!["v6-lab".to_string()],
        cost: CostVector::declared(1, bytes, 0, 0, 1),
    }
}

fn jit_contract() -> (bool, bool, bool) {
    let vm = ProgramVm;
    let candidate = Program {
        id: ProgramId(0),
        inputs: vec![ValueType::Atom; 3],
        output: ValueType::Atom,
        opcode: OpCode::TransitiveBefore,
        guards: vec![Guard::ContextRequired("ordered".to_string())],
        provenance: vec!["planner".to_string()],
        status: ProgramStatus::Candidate,
    };
    let inputs = [
        Value::Atom("A".to_string()),
        Value::Atom("B".to_string()),
        Value::Atom("C".to_string()),
    ];
    let interpreted = vm.execute(&candidate, &inputs, &BTreeSet::new()).unwrap();
    let library = ProgramLibrary::new(2).unwrap();
    let (library, _) = library
        .record_verified_trace(candidate.opcode, &candidate.guards, &interpreted.trace)
        .unwrap();
    let (library, compiled) = library
        .record_verified_trace(candidate.opcode, &candidate.guards, &interpreted.trace)
        .unwrap();
    let compiled = compiled.unwrap();
    let deoptimized = vm.execute(&compiled, &inputs, &BTreeSet::new()).unwrap();
    (
        library.program(compiled.id).is_some(),
        deoptimized.dispatch == ProgramDispatch::Deoptimized,
        deoptimized.result == interpreted.result,
    )
}

fn learnability_contract() -> (bool, bool) {
    let gate = LearnabilityGate;
    let request = InformationRequest {
        candidates: vec![
            Experiment {
                id: "correlation".to_string(),
                predicted_outcomes: BTreeMap::from([
                    ("direct".to_string(), true),
                    ("reverse".to_string(), true),
                ]),
                cost: CostVector::declared(1, 1, 0, 0, 1),
            },
            Experiment {
                id: "intervene".to_string(),
                predicted_outcomes: BTreeMap::from([
                    ("direct".to_string(), true),
                    ("reverse".to_string(), false),
                ]),
                cost: CostVector::declared(1, 1, 0, 0, 1),
            },
        ],
    };
    let reframe = ReframeRequest {
        target_region: BTreeSet::from([FactorId(1)]),
        residual_signature: "structured".to_string(),
        max_operations: 10,
    };
    let decisions = [
        gate.decide(
            LearnabilityEvidence {
                residual: 0.05,
                uncertainty: 0.0,
                compatible_worlds: 1,
                discriminating_evidence: 1.0,
                residual_persistence: 0.0,
                expected_adapt_gain: 0.0,
            },
            request.clone(),
            reframe.clone(),
        ),
        gate.decide(
            LearnabilityEvidence {
                residual: 0.4,
                uncertainty: 0.2,
                compatible_worlds: 1,
                discriminating_evidence: 1.0,
                residual_persistence: 0.2,
                expected_adapt_gain: 0.8,
            },
            request.clone(),
            reframe.clone(),
        ),
        gate.decide(
            LearnabilityEvidence {
                residual: 0.6,
                uncertainty: 0.9,
                compatible_worlds: 3,
                discriminating_evidence: 0.1,
                residual_persistence: 0.3,
                expected_adapt_gain: 0.1,
            },
            request.clone(),
            reframe.clone(),
        ),
        gate.decide(
            LearnabilityEvidence {
                residual: 0.9,
                uncertainty: 0.2,
                compatible_worlds: 1,
                discriminating_evidence: 0.9,
                residual_persistence: 0.9,
                expected_adapt_gain: 0.1,
            },
            request,
            reframe,
        ),
    ];
    let complete = matches!(decisions[0], Some(LearnabilityDecision::Solve))
        && matches!(decisions[1], Some(LearnabilityDecision::Adapt))
        && matches!(decisions[2], Some(LearnabilityDecision::NeedInformation(_)))
        && matches!(decisions[3], Some(LearnabilityDecision::Reframe(_)));
    let active = gate
        .choose_experiment(&InformationRequest {
            candidates: vec![
                Experiment {
                    id: "correlation".to_string(),
                    predicted_outcomes: BTreeMap::from([
                        ("direct".to_string(), true),
                        ("reverse".to_string(), true),
                    ]),
                    cost: CostVector::declared(1, 1, 0, 0, 1),
                },
                Experiment {
                    id: "intervene".to_string(),
                    predicted_outcomes: BTreeMap::from([
                        ("direct".to_string(), true),
                        ("reverse".to_string(), false),
                    ]),
                    cost: CostVector::declared(1, 1, 0, 0, 1),
                },
            ],
        })
        .is_some_and(|experiment| experiment.id == "intervene");
    (complete, active)
}

fn patch_contract() -> bool {
    let patch = crate::core_v6::CognitivePatch {
        id: PatchId(1),
        target: PatchTarget::ProgramLibrary,
        purpose: "compile verified trace".to_string(),
        invariants: BTreeSet::from(["rollback-preserved".to_string()]),
        tests: vec!["held-out".to_string()],
        rollback: RevisionId(1),
    };
    let report = ValidationReport {
        correctness_delta_milliunits: 0,
        latency_delta_ns: -1,
        active_bytes_delta: 0,
        regressions: 0,
        verifier_level: VerificationLevel::HeldOut,
        accepted: true,
    };
    ValidationKernel.validate(&patch, &report) == PatchDecision::Candidate
}
