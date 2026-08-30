use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Circuit {
    pub id: u32,
    pub path: Vec<u32>,
    required_context: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CircuitKey {
    path: Vec<u32>,
    required_context: BTreeSet<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThoughtCompiler {
    compile_after: u32,
    path_counts: BTreeMap<CircuitKey, u32>,
    circuits: BTreeMap<CircuitKey, Circuit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompilerStep {
    pub compiler: ThoughtCompiler,
    pub compiled: Option<Circuit>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CircuitDispatch {
    FastPath { circuit_id: u32 },
    Deoptimized { circuit_id: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CircuitError {
    InvalidCompileLimit,
    EmptyPath,
}

impl Display for CircuitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCompileLimit => write!(f, "compile limit must be positive"),
            Self::EmptyPath => write!(f, "thought paths must contain at least one cell"),
        }
    }
}

impl Error for CircuitError {}

impl ThoughtCompiler {
    pub fn new(compile_after: u32) -> Result<Self, CircuitError> {
        if compile_after == 0 {
            return Err(CircuitError::InvalidCompileLimit);
        }
        Ok(Self {
            compile_after,
            path_counts: BTreeMap::new(),
            circuits: BTreeMap::new(),
        })
    }

    pub fn record(
        &self,
        path: &[u32],
        required_context: &BTreeSet<String>,
    ) -> Result<CompilerStep, CircuitError> {
        if path.is_empty() {
            return Err(CircuitError::EmptyPath);
        }
        let key = CircuitKey {
            path: path.to_vec(),
            required_context: required_context.clone(),
        };
        if let Some(circuit) = self.circuits.get(&key) {
            return Ok(CompilerStep {
                compiler: self.clone(),
                compiled: Some(circuit.clone()),
            });
        }
        let mut path_counts = self.path_counts.clone();
        let count = path_counts.entry(key.clone()).or_insert(0);
        *count = count.saturating_add(1);
        let mut circuits = self.circuits.clone();
        let compiled = if *count >= self.compile_after {
            let circuit = Circuit {
                id: circuits.len() as u32,
                path: key.path.clone(),
                required_context: required_context.clone(),
            };
            circuits.insert(key, circuit.clone());
            Some(circuit)
        } else {
            None
        };
        Ok(CompilerStep {
            compiler: Self {
                compile_after: self.compile_after,
                path_counts,
                circuits,
            },
            compiled,
        })
    }

    pub fn dispatch(&self, circuit: &Circuit, context: &BTreeSet<String>) -> CircuitDispatch {
        if circuit.required_context.is_subset(context) {
            CircuitDispatch::FastPath {
                circuit_id: circuit.id,
            }
        } else {
            CircuitDispatch::Deoptimized {
                circuit_id: circuit.id,
            }
        }
    }
}
