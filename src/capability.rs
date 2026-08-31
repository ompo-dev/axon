use std::collections::BTreeSet;

/// Effects are explicit. A capability cannot acquire one at execution time.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Effect {
    ReadData,
    WriteData,
    Network,
    Actuator,
    SpawnProcess,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Authority {
    effects: BTreeSet<Effect>,
}

impl<const N: usize> From<[Effect; N]> for Authority {
    fn from(effects: [Effect; N]) -> Self {
        Self {
            effects: effects.into_iter().collect(),
        }
    }
}

impl Authority {
    pub fn allows(&self, effect: Effect) -> bool {
        self.effects.contains(&effect)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capability {
    name: String,
    effects: BTreeSet<Effect>,
}

impl Capability {
    pub fn new<const N: usize>(name: impl Into<String>, effects: [Effect; N]) -> Self {
        Self {
            name: name.into(),
            effects: effects.into_iter().collect(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Necessary conditions before a capability can be realized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Feasibility {
    observable: bool,
    identifiable: bool,
    reachable: bool,
    well_conditioned: bool,
    affordable: bool,
}

impl Feasibility {
    pub const fn new(
        observable: bool,
        identifiable: bool,
        reachable: bool,
        well_conditioned: bool,
        affordable: bool,
    ) -> Self {
        Self {
            observable,
            identifiable,
            reachable,
            well_conditioned,
            affordable,
        }
    }

    pub const fn ready() -> Self {
        Self::new(true, true, true, true, true)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateFailure {
    NotObservable,
    NotIdentifiable,
    Unreachable,
    IllConditioned,
    OverBudget,
    Unauthorized(Effect),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityGate {
    feasibility: Feasibility,
    authority: Authority,
}

impl CapabilityGate {
    pub fn new(feasibility: Feasibility, authority: Authority) -> Self {
        Self {
            feasibility,
            authority,
        }
    }

    pub fn evaluate(&self, capability: &Capability) -> Result<(), GateFailure> {
        if !self.feasibility.observable {
            return Err(GateFailure::NotObservable);
        }
        if !self.feasibility.identifiable {
            return Err(GateFailure::NotIdentifiable);
        }
        if !self.feasibility.reachable {
            return Err(GateFailure::Unreachable);
        }
        if !self.feasibility.well_conditioned {
            return Err(GateFailure::IllConditioned);
        }
        if !self.feasibility.affordable {
            return Err(GateFailure::OverBudget);
        }
        capability
            .effects
            .iter()
            .find(|&&effect| !self.authority.allows(effect))
            .copied()
            .map_or(Ok(()), |effect| Err(GateFailure::Unauthorized(effect)))
    }
}
