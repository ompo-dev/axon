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
            let contradiction = intervention_evidence(kind, &source, &target);
            let adapted_choice = if obvious.counterfactual_loss(&contradiction) == 0 {
                ModelClass::Direct
            } else {
                ModelClass::Other
            };
            baseline_correct += u32::from(adapted_choice == kind.into());

            // Sem intervenção, todos os mundos exibem a mesma correlação. O
            // avaliador recebe somente essa evidência, nunca o tipo verdadeiro.
            let observation_only =
                Contradiction::between(source.clone(), target.clone()).with_shared_observations(8);
            let (observational_choice, _) = select_model(&engine, &obvious, &observation_only);
            observation_only_correct += u32::from(observational_choice == kind.into());

            // O tipo do mundo é usado apenas para gerar as observações do
            // simulador. A decisão é tomada exclusivamente pela perda
            // contrafactual do modelo atual e, se necessário, pela seleção de
            // hipóteses do AbductiveEngine.
            let (selected, evaluated) = select_model(&engine, &obvious, &contradiction);
            candidates_evaluated += evaluated;
            reframe_correct += u32::from(selected == kind.into());
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelClass {
    Direct,
    Reverse,
    Latent,
    Other,
}

impl From<WorldKind> for ModelClass {
    fn from(kind: WorldKind) -> Self {
        match kind {
            WorldKind::Direct => Self::Direct,
            WorldKind::Reverse => Self::Reverse,
            WorldKind::Latent => Self::Latent,
        }
    }
}

fn select_model(
    engine: &AbductiveEngine,
    current: &CausalModel,
    evidence: &Contradiction,
) -> (ModelClass, u32) {
    if current.counterfactual_loss(evidence) == 0 {
        return (ModelClass::Direct, 0);
    }

    let islands = engine.reframe(current, evidence, &NegativeArchive::default());
    let choice =
        engine
            .best(&islands)
            .map_or(ModelClass::Other, |island| match island.proposal.kind {
                ReframeKind::ReverseCausality => ModelClass::Reverse,
                ReframeKind::IntroduceLatentCause => ModelClass::Latent,
                _ => ModelClass::Other,
            });
    (choice, islands.len() as u32)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_world_is_retained_without_reframe() {
        let model = CausalModel::with_direct_cause("a", "b", 256);
        let evidence = intervention_evidence(WorldKind::Direct, "a", "b");

        assert_eq!(
            select_model(&AbductiveEngine::default(), &model, &evidence),
            (ModelClass::Direct, 0)
        );
    }

    #[test]
    fn observation_only_cannot_identify_a_structural_alternative() {
        let model = CausalModel::with_direct_cause("a", "b", 256);
        let observation = Contradiction::between("a", "b").with_shared_observations(8);

        assert_eq!(
            select_model(&AbductiveEngine::default(), &model, &observation),
            (ModelClass::Direct, 0)
        );
    }

    #[test]
    fn interventions_are_required_to_select_reverse_or_latent_models() {
        let engine = AbductiveEngine::default();

        for (world, expected) in [
            (WorldKind::Reverse, ModelClass::Reverse),
            (WorldKind::Latent, ModelClass::Latent),
        ] {
            let model = CausalModel::with_direct_cause("a", "b", 256);
            let evidence = intervention_evidence(world, "a", "b");

            assert!(model.counterfactual_loss(&evidence) > 0);
            assert_eq!(select_model(&engine, &model, &evidence), (expected, 9));
        }
    }
}
