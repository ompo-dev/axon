use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProceduralCircuit {
    pub id: u32,
    pub skill: String,
    pub path: Vec<u32>,
    guards: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CircuitKey {
    skill: String,
    path: Vec<u32>,
    guards: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProceduralFabric {
    compile_after: u32,
    successful_runs: BTreeMap<CircuitKey, u32>,
    circuits: BTreeMap<CircuitKey, ProceduralCircuit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProceduralStep {
    pub fabric: ProceduralFabric,
    pub compiled: Option<ProceduralCircuit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProceduralDispatch {
    FastPath { circuit_id: u32 },
    Deoptimized { circuit_id: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProceduralError {
    InvalidCompileThreshold,
    EmptyPath,
}

impl Display for ProceduralError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCompileThreshold => write!(f, "compile threshold must be positive"),
            Self::EmptyPath => write!(f, "a procedural path must contain at least one step"),
        }
    }
}

impl Error for ProceduralError {}

impl ProceduralFabric {
    pub fn new(compile_after: u32) -> Result<Self, ProceduralError> {
        if compile_after == 0 {
            return Err(ProceduralError::InvalidCompileThreshold);
        }
        Ok(Self {
            compile_after,
            successful_runs: BTreeMap::new(),
            circuits: BTreeMap::new(),
        })
    }

    /// Only a verified successful execution is eligible for compilation.
    pub fn record_verified_success(
        &self,
        skill: impl Into<String>,
        path: &[u32],
        guards: &BTreeSet<String>,
    ) -> Result<ProceduralStep, ProceduralError> {
        if path.is_empty() {
            return Err(ProceduralError::EmptyPath);
        }
        let key = CircuitKey {
            skill: skill.into(),
            path: path.to_vec(),
            guards: guards.clone(),
        };
        if let Some(circuit) = self.circuits.get(&key) {
            return Ok(ProceduralStep {
                fabric: self.clone(),
                compiled: Some(circuit.clone()),
            });
        }
        let mut successful_runs = self.successful_runs.clone();
        let count = successful_runs.entry(key.clone()).or_insert(0);
        *count = count.saturating_add(1);
        let mut circuits = self.circuits.clone();
        let compiled = if *count >= self.compile_after {
            let circuit = ProceduralCircuit {
                id: circuits.len() as u32,
                skill: key.skill.clone(),
                path: key.path.clone(),
                guards: key.guards.clone(),
            };
            circuits.insert(key, circuit.clone());
            Some(circuit)
        } else {
            None
        };
        Ok(ProceduralStep {
            fabric: Self {
                compile_after: self.compile_after,
                successful_runs,
                circuits,
            },
            compiled,
        })
    }

    pub fn dispatch(
        &self,
        circuit: &ProceduralCircuit,
        context: &BTreeSet<String>,
    ) -> ProceduralDispatch {
        if circuit.guards.is_subset(context) {
            ProceduralDispatch::FastPath {
                circuit_id: circuit.id,
            }
        } else {
            ProceduralDispatch::Deoptimized {
                circuit_id: circuit.id,
            }
        }
    }
}
