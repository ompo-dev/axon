//! AXON-Λ²: cálculo incremental restrito para grafos dirigidos gerais.
//!
//! O módulo não afirma derivadas para programas arbitrários. Ele implementa
//! quatro regimes verificáveis: DAG com `DELTA`, SCC monotônica, SCC contractiva
//! e `FULL` para Factors opacos. A seleção nunca usa uma otimização quando a
//! classe estrutural não a justifica.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralFactor {
    pub inputs: Vec<usize>,
    pub rule: GeneralRule,
}

impl GeneralFactor {
    pub const fn source(value: i64) -> Self {
        Self {
            inputs: Vec::new(),
            rule: GeneralRule::Source { value },
        }
    }

    pub fn affine(input: usize, multiplier: i64, additive: i64) -> Self {
        Self {
            inputs: vec![input],
            rule: GeneralRule::Affine {
                multiplier,
                additive,
            },
        }
    }

    pub fn max(inputs: Vec<usize>, floor: i64) -> Self {
        Self {
            inputs,
            rule: GeneralRule::Max { floor },
        }
    }

    pub fn contractive_half(inputs: Vec<usize>, target: i64) -> Self {
        Self {
            inputs,
            rule: GeneralRule::ContractiveHalf { target },
        }
    }

    /// Factor deliberadamente sem derivada. O valor é independente das arestas
    /// estruturais recebidas, o que torna o fallback `FULL` exato mesmo em SCC.
    pub fn opaque_constant(inputs: Vec<usize>, value: i64) -> Self {
        Self {
            inputs,
            rule: GeneralRule::OpaqueConstant { value },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GeneralRule {
    Source {
        value: i64,
    },
    Affine {
        multiplier: i64,
        additive: i64,
    },
    /// Operação comutativa monotônica sobre inteiros ordenados.
    Max {
        floor: i64,
    },
    /// `floor((max(inputs) + target) / 2)`: Lipschitz <= 1/2 no sup-norm.
    ContractiveHalf {
        target: i64,
    },
    OpaqueConstant {
        value: i64,
    },
}

impl GeneralRule {
    pub fn supports_delta(&self) -> bool {
        !matches!(
            self,
            Self::OpaqueConstant { .. } | Self::ContractiveHalf { .. }
        )
    }

    pub fn is_commutative(&self) -> bool {
        matches!(self, Self::Max { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphDelta {
    pub factor: usize,
    pub replacement_value: i64,
}

impl GraphDelta {
    pub const fn replace_source(factor: usize, replacement_value: i64) -> Self {
        Self {
            factor,
            replacement_value,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StructuralMode {
    Reuse,
    DeltaPropagation,
    MonotoneFixpoint,
    ContractiveFixpoint,
    FullFallback,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphSlice {
    pub demanded_factors: usize,
    pub changed_factors: usize,
    pub active_factors: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VersionedDependency {
    pub factor: usize,
    pub revision: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependencyFingerprint {
    goal: usize,
    graph_digest: u64,
    dependencies: Vec<VersionedDependency>,
}

impl DependencyFingerprint {
    pub const fn goal(&self) -> usize {
        self.goal
    }

    pub const fn graph_digest(&self) -> u64 {
        self.graph_digest
    }

    pub fn dependencies(&self) -> &[VersionedDependency] {
        &self.dependencies
    }

    pub fn validates(&self, graph_digest: u64, revisions: &[u64]) -> bool {
        self.graph_digest == graph_digest
            && self.dependencies.iter().all(|dependency| {
                revisions.get(dependency.factor).copied() == Some(dependency.revision)
            })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixpointCertificate {
    pub component_size: usize,
    pub iterations: usize,
    pub residual_max: u64,
    pub lipschitz_numerator: Option<u64>,
    pub lipschitz_denominator: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphEvaluation {
    pub values: Vec<i64>,
    pub mode: StructuralMode,
    pub fixpoints: Vec<FixpointCertificate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphQueryResult {
    pub value: i64,
    pub mode: StructuralMode,
    pub slice: GraphSlice,
    pub dependency: DependencyFingerprint,
    pub fixpoints: Vec<FixpointCertificate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphError {
    EmptyGraph,
    InvalidInput,
    InvalidRuleArity,
    DeltaMustReplaceSource,
    InvalidGoal,
    ArithmeticOverflow,
    UnresolvedFixedPoint,
    UnsupportedCycle,
}

impl Display for GraphError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::EmptyGraph => "general graph cannot be empty",
            Self::InvalidInput => "factor references an unknown input",
            Self::InvalidRuleArity => "factor rule has an invalid number of inputs",
            Self::DeltaMustReplaceSource => "a graph delta can only replace a source Factor",
            Self::InvalidGoal => "goal references an unknown Factor",
            Self::ArithmeticOverflow => "graph evaluation overflowed its exact i64 domain",
            Self::UnresolvedFixedPoint => "fixed point did not converge within the certified cap",
            Self::UnsupportedCycle => "cyclic Factor family has no safe evaluator in this kernel",
        };
        write!(formatter, "{message}")
    }
}

impl Error for GraphError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ComponentClass {
    Acyclic,
    Monotone,
    Contractive,
    OpaqueFallback,
    Unsupported,
}

#[derive(Clone, Debug)]
struct Component {
    members: ComponentMembers,
    cyclic: bool,
}

#[derive(Clone, Debug)]
enum ComponentMembers {
    Single(usize),
    Many(Vec<usize>),
}

impl Component {
    fn iter(&self) -> std::slice::Iter<'_, usize> {
        match &self.members {
            ComponentMembers::Single(factor) => std::slice::from_ref(factor).iter(),
            ComponentMembers::Many(factors) => factors.iter(),
        }
    }

    fn len(&self) -> usize {
        match &self.members {
            ComponentMembers::Single(_) => 1,
            ComponentMembers::Many(factors) => factors.len(),
        }
    }

    fn first(&self) -> usize {
        match &self.members {
            ComponentMembers::Single(factor) => *factor,
            ComponentMembers::Many(factors) => factors[0],
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneralGraph {
    factors: Vec<GeneralFactor>,
    dependents: Vec<Vec<usize>>,
    components: Vec<Component>,
    component_order: Vec<usize>,
    topological_rank: Vec<usize>,
    graph_digest: u64,
}

impl GeneralGraph {
    pub fn new(factors: Vec<GeneralFactor>) -> Result<Self, GraphError> {
        if factors.is_empty() {
            return Err(GraphError::EmptyGraph);
        }
        validate_factors(&factors)?;
        let dependents = build_dependents(&factors);
        let (components, component_of) = strongly_connected_components(&factors, &dependents);
        let component_order = component_order(&factors, &component_of);
        let mut topological_rank = vec![0; factors.len()];
        for (rank, component_index) in component_order.iter().enumerate() {
            for factor in components[*component_index].iter() {
                topological_rank[*factor] = rank;
            }
        }
        let graph_digest = digest_graph(&factors);
        Ok(Self {
            factors,
            dependents,
            components,
            component_order,
            topological_rank,
            graph_digest,
        })
    }

    pub fn factors(&self) -> &[GeneralFactor] {
        &self.factors
    }

    pub fn dependents(&self, factor: usize) -> Option<&[usize]> {
        self.dependents.get(factor).map(Vec::as_slice)
    }

    pub fn graph_digest(&self) -> u64 {
        self.graph_digest
    }

    pub fn base_value(&self, factor: usize) -> Result<i64, GraphError> {
        self.validate_goal(factor)?;
        let required = self.backward_set(factor);
        Ok(self.evaluate_required(&required, None)?.values[factor])
    }

    pub fn revisions_after(&self, delta: GraphDelta) -> Result<Vec<u64>, GraphError> {
        self.validate_delta(delta)?;
        let mut revisions = vec![1; self.factors.len()];
        revisions[delta.factor] = 2;
        Ok(revisions)
    }

    pub fn evaluate(&self) -> Result<GraphEvaluation, GraphError> {
        self.evaluate_with_delta(None, None)
    }

    /// Referência semântica total para o benchmark: materializa o estado inteiro,
    /// sem incluir o custo separado de planejar slice e fingerprint.
    pub fn full_value_after(&self, goal: usize, delta: GraphDelta) -> Result<i64, GraphError> {
        self.validate_goal(goal)?;
        self.validate_delta(delta)?;
        let required = self.backward_set(goal);
        Ok(self.evaluate_required(&required, Some(delta))?.values[goal])
    }

    /// Baseline físico sem planejamento por objetivo: materializa todos os
    /// Factors para que a janela cronometrada compare execução, não construção
    /// de cone/fingerprint, contra um quotient previamente certificado.
    pub fn full_evaluation_after(&self, delta: GraphDelta) -> Result<GraphEvaluation, GraphError> {
        self.validate_delta(delta)?;
        self.evaluate_with_delta(None, Some(delta))
    }

    pub fn full_query(
        &self,
        goal: usize,
        delta: GraphDelta,
    ) -> Result<GraphQueryResult, GraphError> {
        self.validate_goal(goal)?;
        self.validate_delta(delta)?;
        let (backward, _, slice) = self.slice_sets(goal, delta.factor);
        let evaluation = self.evaluate_required(&backward, Some(delta))?;
        Ok(GraphQueryResult {
            value: evaluation.values[goal],
            mode: StructuralMode::FullFallback,
            slice,
            dependency: self.fingerprint(goal, delta)?,
            fixpoints: evaluation.fixpoints,
        })
    }

    pub fn query(&self, goal: usize, delta: GraphDelta) -> Result<GraphQueryResult, GraphError> {
        self.validate_goal(goal)?;
        self.validate_delta(delta)?;
        let (backward, forward, slice) = self.slice_sets(goal, delta.factor);
        let base_values = self.evaluate_required(&backward, None)?.values;
        let mode = self.select_mode(&backward, &forward, &slice);
        let dependency = self.fingerprint(goal, delta)?;
        if mode == StructuralMode::Reuse {
            return Ok(GraphQueryResult {
                value: base_values[goal],
                mode,
                slice,
                dependency,
                fixpoints: Vec::new(),
            });
        }
        if mode == StructuralMode::DeltaPropagation {
            return Ok(GraphQueryResult {
                value: self.delta_value(goal, delta, &backward, &forward, &base_values)?,
                mode,
                slice,
                dependency,
                fixpoints: Vec::new(),
            });
        }
        let evaluation = self.evaluate_required(&backward, Some(delta))?;
        Ok(GraphQueryResult {
            value: evaluation.values[goal],
            mode,
            slice,
            dependency,
            fixpoints: evaluation.fixpoints,
        })
    }

    /// Compara todo o cone forward afetado, não somente a decisão final.
    pub fn delta_overlay_matches_full(&self, delta: GraphDelta) -> Result<bool, GraphError> {
        self.validate_delta(delta)?;
        if self.components.iter().any(|component| component.cyclic) {
            return Ok(false);
        }
        let forward = self.forward_set(delta.factor);
        if !forward
            .iter()
            .all(|factor| self.factors[*factor].rule.supports_delta())
        {
            return Ok(false);
        }
        let required = self.required_for(&forward);
        let base_values = self.evaluate_required(&required, None)?.values;
        let overlay = self.delta_overlay(delta, &forward, &base_values)?;
        let full = self.evaluate_required(&required, Some(delta))?;
        Ok(forward.into_iter().all(|factor| {
            overlay.get(&factor).copied().unwrap_or(base_values[factor]) == full.values[factor]
        }))
    }

    fn validate_goal(&self, goal: usize) -> Result<(), GraphError> {
        if goal < self.factors.len() {
            Ok(())
        } else {
            Err(GraphError::InvalidGoal)
        }
    }

    fn validate_delta(&self, delta: GraphDelta) -> Result<(), GraphError> {
        let factor = self
            .factors
            .get(delta.factor)
            .ok_or(GraphError::InvalidInput)?;
        if matches!(factor.rule, GeneralRule::Source { .. }) {
            Ok(())
        } else {
            Err(GraphError::DeltaMustReplaceSource)
        }
    }

    fn slice_sets(
        &self,
        goal: usize,
        changed: usize,
    ) -> (BTreeSet<usize>, BTreeSet<usize>, GraphSlice) {
        let backward = self.backward_set(goal);
        let forward = self.forward_set(changed);
        let active_factors = backward.intersection(&forward).count();
        let slice = GraphSlice {
            demanded_factors: backward.len(),
            changed_factors: forward.len(),
            active_factors,
        };
        (backward, forward, slice)
    }

    fn backward_set(&self, start: usize) -> BTreeSet<usize> {
        let mut visited = BTreeSet::new();
        let mut pending = vec![start];
        while let Some(factor) = pending.pop() {
            if !visited.insert(factor) {
                continue;
            }
            pending.extend(self.factors[factor].inputs.iter().copied());
        }
        visited
    }

    fn forward_set(&self, start: usize) -> BTreeSet<usize> {
        let mut visited = BTreeSet::new();
        let mut pending = vec![start];
        while let Some(factor) = pending.pop() {
            if !visited.insert(factor) {
                continue;
            }
            pending.extend(self.dependents[factor].iter().copied());
        }
        visited
    }

    fn required_for(&self, factors: &BTreeSet<usize>) -> BTreeSet<usize> {
        factors
            .iter()
            .fold(BTreeSet::new(), |mut required, factor| {
                required.extend(self.backward_set(*factor));
                required
            })
    }

    fn select_mode(
        &self,
        backward: &BTreeSet<usize>,
        forward: &BTreeSet<usize>,
        slice: &GraphSlice,
    ) -> StructuralMode {
        if slice.active_factors == 0 {
            return StructuralMode::Reuse;
        }
        let active = backward
            .intersection(forward)
            .copied()
            .collect::<BTreeSet<_>>();
        let mut has_monotone = false;
        let mut has_contractive = false;
        let mut has_fallback = false;
        for component in &self.components {
            if !component.iter().any(|factor| active.contains(factor)) {
                continue;
            }
            match self.classify_component(component) {
                ComponentClass::Acyclic => {}
                ComponentClass::Monotone => has_monotone = true,
                ComponentClass::Contractive => has_contractive = true,
                ComponentClass::OpaqueFallback | ComponentClass::Unsupported => has_fallback = true,
            }
        }
        if has_fallback {
            StructuralMode::FullFallback
        } else if has_contractive {
            StructuralMode::ContractiveFixpoint
        } else if has_monotone {
            StructuralMode::MonotoneFixpoint
        } else if active
            .iter()
            .all(|factor| self.factors[*factor].rule.supports_delta())
            && slice.active_factors < self.factors.len()
        {
            StructuralMode::DeltaPropagation
        } else {
            StructuralMode::FullFallback
        }
    }

    fn fingerprint(
        &self,
        goal: usize,
        delta: GraphDelta,
    ) -> Result<DependencyFingerprint, GraphError> {
        let revisions = self.revisions_after(delta)?;
        Ok(DependencyFingerprint {
            goal,
            graph_digest: self.graph_digest,
            dependencies: self
                .backward_set(goal)
                .into_iter()
                .map(|factor| VersionedDependency {
                    factor,
                    revision: revisions[factor],
                })
                .collect(),
        })
    }

    fn delta_value(
        &self,
        goal: usize,
        delta: GraphDelta,
        backward: &BTreeSet<usize>,
        forward: &BTreeSet<usize>,
        base_values: &[i64],
    ) -> Result<i64, GraphError> {
        let active = backward
            .intersection(forward)
            .copied()
            .collect::<BTreeSet<_>>();
        let overlay = self.delta_overlay(delta, &active, base_values)?;
        overlay
            .get(&goal)
            .copied()
            .or_else(|| base_values.get(goal).copied())
            .ok_or(GraphError::InvalidGoal)
    }

    fn delta_overlay(
        &self,
        delta: GraphDelta,
        active: &BTreeSet<usize>,
        base_values: &[i64],
    ) -> Result<BTreeMap<usize, i64>, GraphError> {
        let mut ordered = active.iter().copied().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|factor| self.topological_rank[*factor]);
        let mut overlay = BTreeMap::new();
        for factor in ordered {
            let inputs = self.factors[factor]
                .inputs
                .iter()
                .map(|input| overlay.get(input).copied().unwrap_or(base_values[*input]))
                .collect::<Vec<_>>();
            let value = evaluate_rule(
                &self.factors[factor].rule,
                &inputs,
                Some((delta.factor, delta.replacement_value)),
                factor,
            )?;
            overlay.insert(factor, value);
        }
        Ok(overlay)
    }

    fn evaluate_required(
        &self,
        required: &BTreeSet<usize>,
        delta: Option<GraphDelta>,
    ) -> Result<GraphEvaluation, GraphError> {
        let filter = (required.len() < self.factors.len()).then_some(required);
        self.evaluate_with_delta(filter, delta)
    }

    fn evaluate_with_delta(
        &self,
        required: Option<&BTreeSet<usize>>,
        delta: Option<GraphDelta>,
    ) -> Result<GraphEvaluation, GraphError> {
        if let Some(delta) = delta {
            self.validate_delta(delta)?;
        }
        let mut values = vec![0; self.factors.len()];
        let mut fixpoints = Vec::new();
        let mut mode = StructuralMode::FullFallback;
        for component_index in &self.component_order {
            let component = &self.components[*component_index];
            if required
                .is_some_and(|needed| !component.iter().any(|factor| needed.contains(factor)))
            {
                continue;
            }
            match self.classify_component(component) {
                ComponentClass::Acyclic => {
                    let factor = component.first();
                    let inputs = input_values(&self.factors[factor], &values);
                    values[factor] = evaluate_rule(
                        &self.factors[factor].rule,
                        &inputs,
                        delta.map(|change| (change.factor, change.replacement_value)),
                        factor,
                    )?;
                }
                ComponentClass::Monotone => {
                    let certificate = self.evaluate_monotone(component, &mut values, delta)?;
                    fixpoints.push(certificate);
                    mode = StructuralMode::MonotoneFixpoint;
                }
                ComponentClass::Contractive => {
                    let certificate = self.evaluate_contractive(component, &mut values, delta)?;
                    fixpoints.push(certificate);
                    mode = StructuralMode::ContractiveFixpoint;
                }
                ComponentClass::OpaqueFallback => {
                    for factor in component.iter() {
                        let inputs = input_values(&self.factors[*factor], &values);
                        values[*factor] = evaluate_rule(
                            &self.factors[*factor].rule,
                            &inputs,
                            delta.map(|change| (change.factor, change.replacement_value)),
                            *factor,
                        )?;
                    }
                    mode = StructuralMode::FullFallback;
                }
                ComponentClass::Unsupported => return Err(GraphError::UnsupportedCycle),
            }
        }
        Ok(GraphEvaluation {
            values,
            mode,
            fixpoints,
        })
    }

    fn evaluate_monotone(
        &self,
        component: &Component,
        values: &mut [i64],
        delta: Option<GraphDelta>,
    ) -> Result<FixpointCertificate, GraphError> {
        let members = component.iter().copied().collect::<BTreeSet<_>>();
        let mut working = component.iter().map(|_| i64::MIN).collect::<Vec<_>>();
        let positions = member_positions(component);
        let cap = component.len().saturating_mul(2).saturating_add(2);
        for iteration in 1..=cap {
            let mut next = Vec::with_capacity(component.len());
            for factor in component.iter() {
                let inputs = self.factors[*factor]
                    .inputs
                    .iter()
                    .map(|input| {
                        if members.contains(input) {
                            working[positions[input]]
                        } else {
                            values[*input]
                        }
                    })
                    .collect::<Vec<_>>();
                next.push(evaluate_rule(
                    &self.factors[*factor].rule,
                    &inputs,
                    delta.map(|change| (change.factor, change.replacement_value)),
                    *factor,
                )?);
            }
            let residual_max = max_residual(&working, &next);
            if residual_max == 0 {
                for (position, factor) in component.iter().enumerate() {
                    values[*factor] = next[position];
                }
                return Ok(FixpointCertificate {
                    component_size: component.len(),
                    iterations: iteration,
                    residual_max,
                    lipschitz_numerator: None,
                    lipschitz_denominator: None,
                });
            }
            working = next;
        }
        Err(GraphError::UnresolvedFixedPoint)
    }

    fn evaluate_contractive(
        &self,
        component: &Component,
        values: &mut [i64],
        delta: Option<GraphDelta>,
    ) -> Result<FixpointCertificate, GraphError> {
        let members = component.iter().copied().collect::<BTreeSet<_>>();
        let mut working = vec![0; component.len()];
        let positions = member_positions(component);
        let cap = component.len().saturating_mul(128).saturating_add(128);
        for iteration in 1..=cap {
            let mut next = Vec::with_capacity(component.len());
            for factor in component.iter() {
                let inputs = self.factors[*factor]
                    .inputs
                    .iter()
                    .map(|input| {
                        if members.contains(input) {
                            working[positions[input]]
                        } else {
                            values[*input]
                        }
                    })
                    .collect::<Vec<_>>();
                next.push(evaluate_rule(
                    &self.factors[*factor].rule,
                    &inputs,
                    delta.map(|change| (change.factor, change.replacement_value)),
                    *factor,
                )?);
            }
            let residual_max = max_residual(&working, &next);
            if residual_max == 0 {
                for (position, factor) in component.iter().enumerate() {
                    values[*factor] = next[position];
                }
                return Ok(FixpointCertificate {
                    component_size: component.len(),
                    iterations: iteration,
                    residual_max,
                    lipschitz_numerator: Some(1),
                    lipschitz_denominator: Some(2),
                });
            }
            working = next;
        }
        Err(GraphError::UnresolvedFixedPoint)
    }

    fn classify_component(&self, component: &Component) -> ComponentClass {
        if !component.cyclic {
            return ComponentClass::Acyclic;
        }
        let mut rules = component.iter().map(|factor| &self.factors[*factor].rule);
        if rules
            .clone()
            .all(|rule| matches!(rule, GeneralRule::Max { .. }))
        {
            ComponentClass::Monotone
        } else if rules
            .clone()
            .all(|rule| matches!(rule, GeneralRule::ContractiveHalf { .. }))
        {
            ComponentClass::Contractive
        } else if rules.all(|rule| matches!(rule, GeneralRule::OpaqueConstant { .. })) {
            ComponentClass::OpaqueFallback
        } else {
            ComponentClass::Unsupported
        }
    }
}

fn validate_factors(factors: &[GeneralFactor]) -> Result<(), GraphError> {
    for factor in factors {
        if factor.inputs.iter().any(|input| *input >= factors.len()) {
            return Err(GraphError::InvalidInput);
        }
        let valid_arity = match factor.rule {
            GeneralRule::Source { .. } => factor.inputs.is_empty(),
            GeneralRule::Affine { .. } => factor.inputs.len() == 1,
            GeneralRule::Max { .. } | GeneralRule::OpaqueConstant { .. } => true,
            GeneralRule::ContractiveHalf { .. } => !factor.inputs.is_empty(),
        };
        if !valid_arity {
            return Err(GraphError::InvalidRuleArity);
        }
    }
    Ok(())
}

fn build_dependents(factors: &[GeneralFactor]) -> Vec<Vec<usize>> {
    let mut dependents = vec![Vec::new(); factors.len()];
    for (factor, definition) in factors.iter().enumerate() {
        for input in &definition.inputs {
            dependents[*input].push(factor);
        }
    }
    dependents
}

fn strongly_connected_components(
    factors: &[GeneralFactor],
    dependents: &[Vec<usize>],
) -> (Vec<Component>, Vec<usize>) {
    let mut visited = vec![false; factors.len()];
    let mut finish = Vec::with_capacity(factors.len());
    for start in 0..factors.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![(start, 0_usize)];
        while let Some((factor, next)) = stack.pop() {
            if next < dependents[factor].len() {
                stack.push((factor, next + 1));
                let successor = dependents[factor][next];
                if !visited[successor] {
                    visited[successor] = true;
                    stack.push((successor, 0));
                }
            } else {
                finish.push(factor);
            }
        }
    }

    let mut component_of = vec![usize::MAX; factors.len()];
    let mut components = Vec::new();
    for start in finish.into_iter().rev() {
        if component_of[start] != usize::MAX {
            continue;
        }
        let component_index = components.len();
        let mut nodes = Vec::new();
        let mut stack = vec![start];
        component_of[start] = component_index;
        while let Some(factor) = stack.pop() {
            nodes.push(factor);
            for input in &factors[factor].inputs {
                if component_of[*input] == usize::MAX {
                    component_of[*input] = component_index;
                    stack.push(*input);
                }
            }
        }
        let cyclic = nodes.len() > 1 || factors[nodes[0]].inputs.contains(&nodes[0]);
        let members = if nodes.len() == 1 {
            ComponentMembers::Single(nodes[0])
        } else {
            ComponentMembers::Many(nodes)
        };
        components.push(Component { members, cyclic });
    }
    (components, component_of)
}

fn component_order(factors: &[GeneralFactor], component_of: &[usize]) -> Vec<usize> {
    let component_count = component_of.iter().copied().max().unwrap_or(0) + 1;
    let mut successors = vec![Vec::new(); component_count];
    for (factor, definition) in factors.iter().enumerate() {
        let destination = component_of[factor];
        for input in &definition.inputs {
            let source = component_of[*input];
            if source != destination {
                successors[source].push(destination);
            }
        }
    }
    let mut indegree = vec![0_usize; component_count];
    for adjacent in &mut successors {
        adjacent.sort_unstable();
        adjacent.dedup();
        for successor in adjacent {
            indegree[*successor] += 1;
        }
    }
    let mut queue = VecDeque::new();
    for (component, degree) in indegree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(component);
        }
    }
    let mut order = Vec::with_capacity(component_count);
    while let Some(component) = queue.pop_front() {
        order.push(component);
        for successor in &successors[component] {
            indegree[*successor] -= 1;
            if indegree[*successor] == 0 {
                queue.push_back(*successor);
            }
        }
    }
    order
}

fn input_values(factor: &GeneralFactor, values: &[i64]) -> Vec<i64> {
    factor.inputs.iter().map(|input| values[*input]).collect()
}

fn evaluate_rule(
    rule: &GeneralRule,
    inputs: &[i64],
    source_delta: Option<(usize, i64)>,
    factor: usize,
) -> Result<i64, GraphError> {
    match rule {
        GeneralRule::Source { value } => Ok(source_delta
            .filter(|(changed, _)| *changed == factor)
            .map_or(*value, |(_, replacement)| replacement)),
        GeneralRule::Affine {
            multiplier,
            additive,
        } => inputs[0]
            .checked_mul(*multiplier)
            .and_then(|value| value.checked_add(*additive))
            .ok_or(GraphError::ArithmeticOverflow),
        GeneralRule::Max { floor } => Ok(inputs.iter().copied().fold(*floor, i64::max)),
        GeneralRule::ContractiveHalf { target } => {
            let input_max = inputs
                .iter()
                .copied()
                .max()
                .ok_or(GraphError::InvalidRuleArity)?;
            let sum = i128::from(input_max) + i128::from(*target);
            i64::try_from(sum.div_euclid(2)).map_err(|_| GraphError::ArithmeticOverflow)
        }
        GeneralRule::OpaqueConstant { value } => Ok(*value),
    }
}

fn member_positions(component: &Component) -> BTreeMap<usize, usize> {
    component
        .iter()
        .enumerate()
        .map(|(position, factor)| (*factor, position))
        .collect()
}

fn max_residual(previous: &[i64], next: &[i64]) -> u64 {
    previous
        .iter()
        .zip(next)
        .map(|(left, right)| left.abs_diff(*right))
        .max()
        .unwrap_or(0)
}

fn digest_graph(factors: &[GeneralFactor]) -> u64 {
    factors
        .iter()
        .enumerate()
        .fold(0xA11C_EC70_2026_0002, |digest, (id, factor)| {
            let with_id = stable_mix(digest, id as u64);
            let with_rule = match factor.rule {
                GeneralRule::Source { value } => stable_mix(with_id, value as u64 ^ 0x01),
                GeneralRule::Affine {
                    multiplier,
                    additive,
                } => stable_mix(
                    stable_mix(with_id, multiplier as u64 ^ 0x02),
                    additive as u64,
                ),
                GeneralRule::Max { floor } => stable_mix(with_id, floor as u64 ^ 0x03),
                GeneralRule::ContractiveHalf { target } => {
                    stable_mix(with_id, target as u64 ^ 0x04)
                }
                GeneralRule::OpaqueConstant { value } => stable_mix(with_id, value as u64 ^ 0x05),
            };
            factor.inputs.iter().fold(with_rule, |current, input| {
                stable_mix(current, *input as u64)
            })
        })
}

pub(crate) fn stable_mix(value: u64, contribution: u64) -> u64 {
    let mixed = value ^ contribution.wrapping_add(0x9E37_79B9_7F4A_7C15);
    mixed.rotate_left(17).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}
