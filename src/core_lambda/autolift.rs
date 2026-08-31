//! Auto-LIFT certificado para a classe exata de entradas source exchangeable
//! de um Factor `max` comutativo.
//!
//! Colour refinement somente propõe candidatos. A compressão é autorizada pelo
//! certificado estrutural, que verifica domínio, valor, incidência e relação
//! preservada. Portanto uma cor igual nunca é tomada como prova de equivalência.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use super::general::{GeneralGraph, GeneralRule, GraphError, stable_mix};

const MAX_COLOUR_ROUNDS: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiftCertificate {
    pub owner: usize,
    pub representative: usize,
    pub members: Vec<usize>,
    pub source_value: i64,
    pub graph_digest: u64,
    pub colour: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiftedClass {
    pub owner: usize,
    pub representative: usize,
    pub members: Vec<usize>,
    pub source_value: i64,
    pub certificate: LiftCertificate,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LiftedOwnerPlan {
    owner: usize,
    class_indices: Vec<usize>,
    direct_inputs: Vec<usize>,
    floor: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CertifiedAutoLift {
    classes: Vec<LiftedClass>,
    owner_plans: Vec<LiftedOwnerPlan>,
    member_class: Vec<Option<usize>>,
    graph_digest: u64,
    pub colour_rounds: usize,
    pub candidate_members: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalUnlift {
    class_index: usize,
    member: usize,
    replacement_value: i64,
    original_members: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AutoLiftError {
    NotLifted,
    UnknownOwner,
    InvalidCertificate,
    Graph(GraphError),
}

impl From<GraphError> for AutoLiftError {
    fn from(value: GraphError) -> Self {
        Self::Graph(value)
    }
}

impl Display for AutoLiftError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::NotLifted => "Factor is not part of a certified lifted class",
            Self::UnknownOwner => "no lifted plan exists for this Factor",
            Self::InvalidCertificate => "Auto-LIFT certificate does not verify against the graph",
            Self::Graph(error) => return Display::fmt(error, formatter),
        };
        write!(formatter, "{message}")
    }
}

impl Error for AutoLiftError {}

impl CertifiedAutoLift {
    pub fn discover(graph: &GeneralGraph) -> Result<Self, AutoLiftError> {
        let (colours, colour_rounds) = colour_refinement(graph);
        let mut classes = Vec::new();
        let mut owner_plans = Vec::new();
        let mut member_class = vec![None; graph.factors().len()];
        let mut candidate_members = 0;

        for (owner, factor) in graph.factors().iter().enumerate() {
            let GeneralRule::Max { floor } = factor.rule else {
                continue;
            };
            let mut grouped_sources = BTreeMap::<u32, Vec<usize>>::new();
            let mut direct_inputs = Vec::new();
            for input in &factor.inputs {
                if matches!(graph.factors()[*input].rule, GeneralRule::Source { .. }) {
                    grouped_sources
                        .entry(colours[*input])
                        .or_default()
                        .push(*input);
                } else {
                    direct_inputs.push(*input);
                }
            }

            let mut class_indices = Vec::new();
            for (colour, members) in grouped_sources {
                if members.len() < 2 {
                    direct_inputs.extend(members);
                    continue;
                }
                candidate_members += members.len();
                if let Some(class) = certify_source_orbit(graph, owner, colour, members.clone()) {
                    let index = classes.len();
                    for member in &class.members {
                        member_class[*member] = Some(index);
                    }
                    class_indices.push(index);
                    classes.push(class);
                } else {
                    direct_inputs.extend(members);
                }
            }
            if !class_indices.is_empty() {
                direct_inputs.sort_unstable();
                owner_plans.push(LiftedOwnerPlan {
                    owner,
                    class_indices,
                    direct_inputs,
                    floor,
                });
            }
        }

        let lift = Self {
            classes,
            owner_plans,
            member_class,
            graph_digest: graph.graph_digest(),
            colour_rounds,
            candidate_members,
        };
        if lift.verify(graph) {
            Ok(lift)
        } else {
            Err(AutoLiftError::InvalidCertificate)
        }
    }

    pub fn classes(&self) -> &[LiftedClass] {
        &self.classes
    }

    pub fn verify(&self, graph: &GeneralGraph) -> bool {
        self.classes
            .iter()
            .all(|class| verify_certificate(graph, &class.certificate))
            && self.owner_plans.iter().all(|plan| {
                matches!(
                    graph.factors().get(plan.owner).map(|factor| &factor.rule),
                    Some(GeneralRule::Max { .. })
                ) && plan.class_indices.iter().all(|index| {
                    self.classes
                        .get(*index)
                        .is_some_and(|class| class.owner == plan.owner)
                })
            })
    }

    pub fn lifted_max(&self, graph: &GeneralGraph, owner: usize) -> Result<i64, AutoLiftError> {
        let plan = self.owner_plan(owner)?;
        if self.graph_digest != graph.graph_digest() {
            return Err(AutoLiftError::InvalidCertificate);
        }
        let mut value = plan.floor;
        for class_index in &plan.class_indices {
            value = value.max(self.classes[*class_index].source_value);
        }
        for input in &plan.direct_inputs {
            value = value.max(graph.base_value(*input)?);
        }
        Ok(value)
    }

    pub fn unlift(
        &self,
        member: usize,
        replacement_value: i64,
    ) -> Result<LocalUnlift, AutoLiftError> {
        let class_index = self
            .member_class
            .get(member)
            .and_then(|class| *class)
            .ok_or(AutoLiftError::NotLifted)?;
        let original_members = self.classes[class_index].members.len();
        Ok(LocalUnlift {
            class_index,
            member,
            replacement_value,
            original_members,
        })
    }

    fn owner_plan(&self, owner: usize) -> Result<&LiftedOwnerPlan, AutoLiftError> {
        self.owner_plans
            .iter()
            .find(|plan| plan.owner == owner)
            .ok_or(AutoLiftError::UnknownOwner)
    }
}

impl LocalUnlift {
    pub const fn specialized_members(&self) -> usize {
        1
    }

    pub const fn remaining_members(&self) -> usize {
        self.original_members - 1
    }

    pub fn lifted_max(
        &self,
        lift: &CertifiedAutoLift,
        graph: &GeneralGraph,
        owner: usize,
    ) -> Result<i64, AutoLiftError> {
        let plan = lift.owner_plan(owner)?;
        if lift.graph_digest != graph.graph_digest()
            || !plan.class_indices.contains(&self.class_index)
        {
            return Err(AutoLiftError::InvalidCertificate);
        }
        let mut value = plan.floor;
        for class_index in &plan.class_indices {
            let class = &lift.classes[*class_index];
            if *class_index == self.class_index {
                if class.members.len() > 1 {
                    value = value.max(class.source_value);
                }
                value = value.max(self.replacement_value);
            } else {
                value = value.max(class.source_value);
            }
        }
        for input in &plan.direct_inputs {
            value = value.max(graph.base_value(*input)?);
        }
        Ok(value)
    }

    pub fn member(&self) -> usize {
        self.member
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum RuleKey {
    Source(i64),
    Affine(i64, i64),
    Max(i64),
    ContractiveHalf(i64),
    OpaqueConstant(i64),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct InitialColourKey {
    rule: RuleKey,
    input_count: usize,
    dependent_count: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RefinementKey {
    own: u32,
    inputs: Vec<u32>,
    dependents: Vec<u32>,
}

fn colour_refinement(graph: &GeneralGraph) -> (Vec<u32>, usize) {
    let mut initial = BTreeMap::<InitialColourKey, u32>::new();
    let mut colours = Vec::with_capacity(graph.factors().len());
    for (factor, definition) in graph.factors().iter().enumerate() {
        let key = InitialColourKey {
            rule: rule_key(&definition.rule),
            input_count: definition.inputs.len(),
            dependent_count: graph.dependents(factor).map_or(0, <[usize]>::len),
        };
        let next_colour = initial.len() as u32;
        let colour = *initial.entry(key).or_insert(next_colour);
        colours.push(colour);
    }

    for round in 1..=MAX_COLOUR_ROUNDS {
        let mut dictionary = BTreeMap::<RefinementKey, u32>::new();
        let mut next = Vec::with_capacity(colours.len());
        for (factor, definition) in graph.factors().iter().enumerate() {
            let mut inputs = definition
                .inputs
                .iter()
                .map(|input| colours[*input])
                .collect::<Vec<_>>();
            if definition.rule.is_commutative() {
                inputs.sort_unstable();
            }
            let mut dependents = graph
                .dependents(factor)
                .unwrap_or_default()
                .iter()
                .map(|dependent| colours[*dependent])
                .collect::<Vec<_>>();
            dependents.sort_unstable();
            let key = RefinementKey {
                own: colours[factor],
                inputs,
                dependents,
            };
            let next_colour = dictionary.len() as u32;
            next.push(*dictionary.entry(key).or_insert(next_colour));
        }
        if next == colours {
            return (next, round);
        }
        colours = next;
    }
    (colours, MAX_COLOUR_ROUNDS)
}

fn rule_key(rule: &GeneralRule) -> RuleKey {
    match rule {
        GeneralRule::Source { value } => RuleKey::Source(*value),
        GeneralRule::Affine {
            multiplier,
            additive,
        } => RuleKey::Affine(*multiplier, *additive),
        GeneralRule::Max { floor } => RuleKey::Max(*floor),
        GeneralRule::ContractiveHalf { target } => RuleKey::ContractiveHalf(*target),
        GeneralRule::OpaqueConstant { value } => RuleKey::OpaqueConstant(*value),
    }
}

fn certify_source_orbit(
    graph: &GeneralGraph,
    owner: usize,
    colour: u32,
    mut members: Vec<usize>,
) -> Option<LiftedClass> {
    members.sort_unstable();
    let GeneralRule::Max { .. } = graph.factors().get(owner)?.rule else {
        return None;
    };
    let source_value = match graph.factors().get(*members.first()?)?.rule {
        GeneralRule::Source { value } => value,
        _ => return None,
    };
    let owner_membership = unique_owner_membership(graph, owner)?;
    let valid = members.iter().all(|member| {
        matches!(graph.factors()[*member].rule, GeneralRule::Source { value } if value == source_value)
            && graph.factors()[*member].inputs.is_empty()
            && graph.dependents(*member) == Some(&[owner][..])
            && owner_membership[*member]
    });
    if !valid {
        return None;
    }
    let representative = *members.first()?;
    let certificate = LiftCertificate {
        owner,
        representative,
        members: members.clone(),
        source_value,
        graph_digest: graph.graph_digest(),
        colour,
    };
    Some(LiftedClass {
        owner,
        representative,
        members,
        source_value,
        certificate,
    })
}

fn verify_certificate(graph: &GeneralGraph, certificate: &LiftCertificate) -> bool {
    if certificate.graph_digest != graph.graph_digest()
        || certificate.members.len() < 2
        || certificate.members.first().copied() != Some(certificate.representative)
        || !matches!(
            graph
                .factors()
                .get(certificate.owner)
                .map(|factor| &factor.rule),
            Some(GeneralRule::Max { .. })
        )
    {
        return false;
    }
    let Some(owner_membership) = unique_owner_membership(graph, certificate.owner) else {
        return false;
    };
    certificate.members.windows(2).all(|pair| pair[0] < pair[1])
        && certificate.members.iter().all(|member| {
            matches!(graph.factors().get(*member).map(|factor| &factor.rule), Some(GeneralRule::Source { value }) if *value == certificate.source_value)
                && graph.factors()[*member].inputs.is_empty()
                && graph.dependents(*member) == Some(&[certificate.owner][..])
                && owner_membership[*member]
        })
}

fn unique_owner_membership(graph: &GeneralGraph, owner: usize) -> Option<Vec<bool>> {
    let owner_inputs = &graph.factors().get(owner)?.inputs;
    let mut membership = vec![false; graph.factors().len()];
    for input in owner_inputs {
        if membership[*input] {
            return None;
        }
        membership[*input] = true;
    }
    Some(membership)
}

#[allow(dead_code)]
fn certificate_digest(certificate: &LiftCertificate) -> u64 {
    certificate.members.iter().fold(
        stable_mix(certificate.graph_digest, certificate.owner as u64),
        |digest, member| stable_mix(digest, *member as u64),
    )
}
