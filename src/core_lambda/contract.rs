//! Contratos semânticos e refinamento seguro de realizações AXON-Λ.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use super::cost::{CostVector, CostWeights};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum VerificationStrength {
    Sampled,
    PropertyTested,
    Exhaustive,
}

/// A ABI semântica de uma transformação, independente de linguagem ou backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SemanticAbi {
    pub semantic_id: String,
    pub input_type: String,
    pub output_type: String,
    /// Digest da regra, para que revisão crescente não possa mascarar uma
    /// troca de semântica sob o mesmo identificador.
    pub rule_digest: u64,
}

impl SemanticAbi {
    pub fn new(
        semantic_id: impl Into<String>,
        input_type: impl Into<String>,
        output_type: impl Into<String>,
        rule_digest: u64,
    ) -> Self {
        Self {
            semantic_id: semantic_id.into(),
            input_type: input_type.into(),
            output_type: output_type.into(),
            rule_digest,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractedMorphism {
    pub abi: SemanticAbi,
    pub revision: u64,
    /// Premissas conjuntivas exigidas pela implementação.
    pub preconditions: BTreeSet<String>,
    pub guarantees: BTreeSet<String>,
    pub max_error_milliunits: u64,
    pub verification: VerificationStrength,
}

impl ContractedMorphism {
    pub fn new(
        abi: SemanticAbi,
        revision: u64,
        preconditions: BTreeSet<String>,
        guarantees: BTreeSet<String>,
        max_error_milliunits: u64,
        verification: VerificationStrength,
    ) -> Self {
        Self {
            abi,
            revision,
            preconditions,
            guarantees,
            max_error_milliunits,
            verification,
        }
    }

    pub fn with_abi(mut self, abi: SemanticAbi) -> Self {
        self.abi = abi;
        self
    }

    /// `self ⊑ required`: aceita todo caso antes aceito, entrega ao menos as
    /// garantias anteriores, não aumenta o erro e usa verificação tão forte ou
    /// mais forte. Para precondições conjuntivas, enfraquecer é remover termos.
    pub fn refines(&self, required: &Self) -> bool {
        !self.abi.semantic_id.is_empty()
            && self.abi.semantic_id == required.abi.semantic_id
            && self.abi.input_type == required.abi.input_type
            && self.abi.output_type == required.abi.output_type
            && self.abi.rule_digest == required.abi.rule_digest
            && self.revision >= required.revision
            && self.preconditions.is_subset(&required.preconditions)
            && self.guarantees.is_superset(&required.guarantees)
            && self.max_error_milliunits <= required.max_error_milliunits
            && self.verification >= required.verification
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecisionCertificate {
    pub winner_lower_bound: i64,
    pub runner_up_upper_bound: i64,
}

impl DecisionCertificate {
    pub const fn new(winner_lower_bound: i64, runner_up_upper_bound: i64) -> Self {
        Self {
            winner_lower_bound,
            runner_up_upper_bound,
        }
    }

    pub fn survives(self, error_milliunits: u64) -> bool {
        let error = i64::try_from(error_milliunits).unwrap_or(i64::MAX);
        self.winner_lower_bound.saturating_sub(error)
            > self.runner_up_upper_bound.saturating_add(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MorphismVariant {
    pub name: String,
    pub contract: ContractedMorphism,
    pub cost: CostVector,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MorphismImplementation {
    pub operation: String,
    pub required_contract: ContractedMorphism,
    pub certificate: DecisionCertificate,
    pub variants: Vec<MorphismVariant>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RealizationPlan {
    pub name: String,
    pub cost: CostVector,
    pub certificate_preserved: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealizationError {
    NoSafeRefinement,
}

impl Display for RealizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSafeRefinement => {
                write!(formatter, "no realization refines the semantic contract")
            }
        }
    }
}

impl Error for RealizationError {}

impl MorphismImplementation {
    pub fn new(
        operation: impl Into<String>,
        required_contract: ContractedMorphism,
        certificate: DecisionCertificate,
        variants: Vec<(&str, ContractedMorphism, CostVector)>,
    ) -> Self {
        Self {
            operation: operation.into(),
            required_contract,
            certificate,
            variants: variants
                .into_iter()
                .map(|(name, contract, cost)| MorphismVariant {
                    name: name.to_owned(),
                    contract,
                    cost,
                })
                .collect(),
        }
    }

    pub fn realize(&self, weights: CostWeights) -> Result<RealizationPlan, RealizationError> {
        self.variants
            .iter()
            .filter(|variant| variant.contract.refines(&self.required_contract))
            .filter(|variant| {
                self.certificate
                    .survives(variant.contract.max_error_milliunits)
            })
            .min_by(|left, right| {
                left.cost
                    .weighted_score(weights)
                    .cmp(&right.cost.weighted_score(weights))
                    .then_with(|| left.name.cmp(&right.name))
            })
            .map(|variant| RealizationPlan {
                name: variant.name.clone(),
                cost: variant.cost,
                certificate_preserved: true,
            })
            .ok_or(RealizationError::NoSafeRefinement)
    }
}
