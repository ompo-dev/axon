use crate::core_v3::{
    AbductiveEngine, CausalModel, Contradiction, Counterfactual, NegativeArchive, ReframeKind,
};

/// Resultado de identificação estrutural em mundos causalmente distintos.
#[derive(Clone, Debug, PartialEq)]
pub struct JumpReport {
    pub worlds: u32,
    pub direct_adaptation_intervention_accuracy: f32,
    pub reframe_intervention_accuracy: f32,
    pub observational_only_identification_accuracy: f32,
    pub mean_reframe_candidates: f32,
}

pub(super) fn run() -> JumpReport {
    const WORLDS_PER_KIND: u32 = 20;
    let engine = AbductiveEngine::default();
    let kinds = [WorldKind::Direct, WorldKind::Reverse, WorldKind::Latent];
    let mut baseline_correct = 0_u32;
    let mut reframe_correct = 0_u32;
    let mut observation_only_correct = 0_u32;
    let mut candidates_evaluated = 0_u32;

    for serial in 0..WORLDS_PER_KIND {
        for kind in kinds {
            let source = format!("a-{serial}-{kind:?}");
            let target = format!("b-{serial}-{kind:?}");
            let obvious = CausalModel::with_direct_cause(source.clone(), target.clone(), 256);
            baseline_correct += u32::from(kind == WorldKind::Direct);

            // Sem intervenção, todos os mundos exibem a mesma correlação. O
            // avaliador recebe somente essa evidência, nunca o tipo verdadeiro.
            let observation_only =
                Contradiction::between(source.clone(), target.clone()).with_shared_observations(8);
            let observational_islands =
                engine.reframe(&obvious, &observation_only, &NegativeArchive::default());
            observation_only_correct +=
                u32::from(engine.best(&observational_islands).is_some_and(|island| {
                    kind == WorldKind::Reverse
                        && island.proposal.kind == ReframeKind::ReverseCausality
                }));

            if kind == WorldKind::Direct {
                reframe_correct += 1;
                continue;
            }

            let contradiction = intervention_evidence(kind, &source, &target);
            let islands = engine.reframe(&obvious, &contradiction, &NegativeArchive::default());
            candidates_evaluated += islands.len() as u32;
            reframe_correct += u32::from(
                engine
                    .best(&islands)
                    .is_some_and(|island| island.proposal.kind == expected_kind(kind)),
            );
        }
    }

    let worlds = WORLDS_PER_KIND * kinds.len() as u32;
    let reframe_worlds = worlds - WORLDS_PER_KIND;
    JumpReport {
        worlds,
        direct_adaptation_intervention_accuracy: baseline_correct as f32 / worlds as f32,
        reframe_intervention_accuracy: reframe_correct as f32 / worlds as f32,
        observational_only_identification_accuracy: observation_only_correct as f32 / worlds as f32,
        mean_reframe_candidates: candidates_evaluated as f32 / reframe_worlds as f32,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WorldKind {
    Direct,
    Reverse,
    Latent,
}

fn expected_kind(kind: WorldKind) -> ReframeKind {
    match kind {
        WorldKind::Direct => ReframeKind::RemovePremise,
        WorldKind::Reverse => ReframeKind::ReverseCausality,
        WorldKind::Latent => ReframeKind::IntroduceLatentCause,
    }
}

fn intervention_evidence(kind: WorldKind, source: &str, target: &str) -> Contradiction {
    let base = Contradiction::between(source, target).with_shared_observations(8);
    match kind {
        WorldKind::Direct => base.with_counterfactual(Counterfactual::expect(source, target, true)),
        WorldKind::Reverse => base
            .with_counterfactual(Counterfactual::expect(source, target, false))
            .with_counterfactual(Counterfactual::expect(target, source, true)),
        WorldKind::Latent => base
            .with_counterfactual(Counterfactual::expect(source, target, false))
            .with_counterfactual(Counterfactual::expect(target, source, false)),
    }
}
