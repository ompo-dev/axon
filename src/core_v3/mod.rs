//! V3 experimental: computador cognitivo esparso de três substratos.
//!
//! `SemanticMesh` armazena relações composicionais, `DynamicField` acompanha
//! dinâmica local e `EpisodicStore` preserva evidência exata. O módulo não
//! altera o runtime ou o formato `.axon` existente até que seus invariantes
//! tenham sido demonstrados por benchmarks.

mod abduction;
mod benchmark;
mod circuit;
mod codec;
mod core;
mod dynamic;
mod episodic;
mod event;
mod salience;
mod semantic;
mod vector;

pub use abduction::{
    AbductiveEngine, CausalModel, Contradiction, Counterfactual, NegativeArchive, ReframeKind,
};
pub use benchmark::run_jump_benchmark;
pub use circuit::{CircuitDispatch, ThoughtCompiler};
pub use codec::{AdaptiveEventCodec, CodecError};
pub use core::{CognitiveCore, CoreAction};
pub use dynamic::{BranchKind, CreditError, CreditPacket, DynamicField};
pub use episodic::EpisodicStore;
pub use event::{Event, FactorizedRepresentation, Modality, RepresentationScale};
pub use salience::{CognitiveValue, SalienceGate};
pub use semantic::{SemanticFact, SemanticMesh};
pub use vector::HyperVector;

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn vector(values: &[i8]) -> HyperVector {
        HyperVector::try_from(values.to_vec()).unwrap()
    }

    #[test]
    fn binding_is_self_inverse_for_dense_ternary_vectors() {
        let left = vector(&[1, -1, 1, -1]);
        let right = vector(&[-1, -1, 1, 1]);

        assert_eq!(left.bind(&right).unwrap().bind(&right).unwrap(), left);
    }

    #[test]
    fn salience_processes_important_predictable_events() {
        let gate = SalienceGate::default();
        let value = CognitiveValue::try_new(0.0, 1.0, 0.1, 0.1, 0.0, 0.1).unwrap();

        let decision = gate.evaluate(&value);

        assert!(decision.should_process);
        assert!(decision.score > 0.0);
    }

    #[test]
    fn semantic_mesh_binds_compositional_facts() {
        let fact = SemanticFact::new(
            "DOG",
            "IS_A",
            "MAMMAL",
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );
        let mesh = SemanticMesh::default().bind(fact);

        let facts = mesh.facts_for("DOG");

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].object, "MAMMAL");
    }

    #[test]
    fn core_keeps_semantic_facts_outside_the_dynamic_field() {
        let fact = SemanticFact::new(
            "DOG",
            "IS_A",
            "MAMMAL",
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );

        let core = CognitiveCore::default().learn_semantic(fact);

        assert_eq!(core.semantic().facts_for("DOG").len(), 1);
    }

    #[test]
    fn dynamic_field_updates_only_the_relevant_dendritic_branch() {
        let event = Event::new(
            "moving-object",
            BranchKind::Temporal,
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );
        let field = DynamicField::empty(0.20).unwrap();

        let step = field.process(&event).unwrap();
        let cell = step.field.cells().first().unwrap();

        assert!(cell.branch_active_dimensions(BranchKind::Temporal) > 0);
        assert_eq!(cell.branch_active_dimensions(BranchKind::Visual), 0);
    }

    #[test]
    fn dynamic_field_grows_for_a_new_vector_dimension() {
        let first = Event::new(
            "short",
            BranchKind::Temporal,
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );
        let second = Event::new(
            "long",
            BranchKind::Temporal,
            HyperVector::zeros(16).unwrap(),
        );
        let field = DynamicField::empty(0.20)
            .unwrap()
            .process(&first)
            .unwrap()
            .field;

        let step = field.process(&second).unwrap();

        assert_eq!(step.field.cells().len(), 2);
    }

    #[test]
    fn credit_packet_changes_only_the_eligible_local_branch() {
        let event = Event::new(
            "moving-object",
            BranchKind::Temporal,
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );
        let field = DynamicField::empty(0.20)
            .unwrap()
            .process(&event)
            .unwrap()
            .field;
        let before = field.cells().first().unwrap();
        let temporal_before = before.branch_plasticity(BranchKind::Temporal).unwrap();
        let visual_before = before.branch_plasticity(BranchKind::Visual).unwrap();

        let credited =
            field.apply_credit(CreditPacket::try_new(BranchKind::Temporal, 1.0, 1.0).unwrap());
        let after = credited.cells().first().unwrap();

        assert!(after.branch_plasticity(BranchKind::Temporal).unwrap() > temporal_before);
        assert_eq!(
            after.branch_plasticity(BranchKind::Visual).unwrap(),
            visual_before
        );
    }

    #[test]
    fn episodic_store_recovers_exact_important_experience() {
        let signature = vector(&[1, 0, -1, 1, 0, -1, 1, -1]);
        let event = Event::new("parking", BranchKind::Semantic, signature.clone()).with_metadata(
            817_251,
            1_727_000_000,
            "carro na rua X às 14:37",
        );
        let store = EpisodicStore::default().append(event, signature.clone());

        let matches = store.lookup(&signature, 0.99).unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].event.raw_detail, "carro na rua X às 14:37");
    }

    #[test]
    fn adaptive_codec_keeps_concept_and_multiscale_residuals_separate() {
        let event = Event::new(
            "spoken-command",
            BranchKind::Linguistic,
            vector(&[1, 0, -1, 1]),
        );
        let semantic_signature = vector(&[-1, 1, 1, 0, -1, 1, 0, 1]);
        let encoded = AdaptiveEventCodec::new(2).unwrap().encode(
            event,
            Modality::Audio,
            semantic_signature.clone(),
            [
                (RepresentationScale::Phoneme, vector(&[1, 0, -1, 1])),
                (RepresentationScale::Word, vector(&[1, 1, 0, -1])),
                (RepresentationScale::Intent, vector(&[1, 0, 0, 0])),
            ],
        );

        assert_eq!(encoded.semantic_signature(), &semantic_signature);
        assert!(encoded.residual(RepresentationScale::Phoneme).is_some());
        assert!(encoded.residual(RepresentationScale::Word).is_some());
        assert!(encoded.residual(RepresentationScale::Intent).is_none());
    }

    #[test]
    fn important_episode_uses_its_semantic_signature_for_lookup() {
        let semantic_signature = vector(&[-1, 1, 1, 0, -1, 1, 0, 1]);
        let event = AdaptiveEventCodec::default().encode(
            Event::new(
                "parking",
                BranchKind::Semantic,
                vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
            ),
            Modality::Text,
            semantic_signature.clone(),
            [],
        );
        let value = CognitiveValue::try_new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0).unwrap();

        let step = CognitiveCore::default().observe(event, value).unwrap();

        assert_eq!(
            step.core
                .episodes()
                .lookup(&semantic_signature, 0.99)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn core_keeps_low_value_events_dormant() {
        let event = Event::new(
            "background-noise",
            BranchKind::Temporal,
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );
        let value = CognitiveValue::try_new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0).unwrap();

        let step = CognitiveCore::default().observe(event, value).unwrap();

        assert!(!step.processed);
        assert_eq!(step.core.episodes().len(), 0);
    }

    #[test]
    fn repeated_high_value_residuals_escalate_from_adapt_to_reframe() {
        let first = Event::new(
            "first",
            BranchKind::Temporal,
            vector(&[1, 0, -1, 1, 0, -1, 1, -1]),
        );
        let second = Event::new(
            "contradiction",
            BranchKind::Temporal,
            vector(&[-1, 0, 1, -1, 0, 1, -1, 1]),
        );
        let value = CognitiveValue::try_new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0).unwrap();

        let first_step = CognitiveCore::default().observe(first, value).unwrap();
        let second_step = first_step.core.observe(second, value).unwrap();

        assert_eq!(first_step.action, CoreAction::Adapt);
        assert_eq!(second_step.action, CoreAction::Reframe);
    }

    #[test]
    fn residual_preserves_missing_predicted_dimensions() {
        let observed = vector(&[0, 1]);
        let predicted = vector(&[1, 1]);

        assert_eq!(observed.residual(&predicted).unwrap(), vector(&[-1, 0]));
    }

    #[test]
    fn abductive_engine_prefers_hidden_cause_after_interventions_falsify_direct_cause() {
        let model = CausalModel::with_direct_cause("a", "b", 256);
        let contradiction = Contradiction::between("a", "b")
            .with_shared_observations(8)
            .with_counterfactual(Counterfactual::expect("a", "b", false))
            .with_counterfactual(Counterfactual::expect("b", "a", false));
        let engine = AbductiveEngine::default();

        let islands = engine.reframe(&model, &contradiction, &NegativeArchive::default());
        let best = engine.best(&islands).unwrap();

        assert!(
            islands
                .iter()
                .any(|island| island.proposal.kind == ReframeKind::IntroduceLatentCause)
        );
        assert_eq!(best.proposal.kind, ReframeKind::IntroduceLatentCause);
        assert!(best.proposal.model.has_latent_cause_for("a", "b"));
        assert!(!best.proposal.model.has_direct_cause("a", "b"));
    }

    #[test]
    fn negative_archive_stops_a_falsified_reframe_from_returning() {
        let model = CausalModel::with_direct_cause("a", "b", 256);
        let contradiction = Contradiction::between("a", "b");
        let engine = AbductiveEngine::default();
        let archive =
            NegativeArchive::default().record(&model, ReframeKind::ReverseCausality, "a", "b");

        let islands = engine.reframe(&model, &contradiction, &archive);

        assert!(
            islands
                .iter()
                .all(|island| island.proposal.kind != ReframeKind::ReverseCausality)
        );
    }

    #[test]
    fn jump_benchmark_selects_a_reframed_model() {
        let result = run_jump_benchmark();

        assert_eq!(result.selected.kind, ReframeKind::IntroduceLatentCause);
        assert!(result.selected.counterfactual_loss < result.obvious_counterfactual_loss);
    }

    #[test]
    fn compiled_thought_deoptimizes_when_its_guard_fails() {
        let context = BTreeSet::from(["same-world".to_string()]);
        let compiler = ThoughtCompiler::new(2).unwrap();
        let first = compiler.record(&[2, 5, 9], &context).unwrap();
        let second = first.compiler.record(&[2, 5, 9], &context).unwrap();
        let circuit = second.compiled.unwrap();

        let dispatch = second
            .compiler
            .dispatch(&circuit, &BTreeSet::from(["exception".to_string()]));

        assert_eq!(
            dispatch,
            CircuitDispatch::Deoptimized {
                circuit_id: circuit.id
            }
        );
    }
}
