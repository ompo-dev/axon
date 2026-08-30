use std::collections::BTreeMap;

use super::dynamic::BranchKind;
use super::vector::HyperVector;

/// Source-specific details stay separate from the semantic concept they express.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Modality {
    Abstract,
    Text,
    Audio,
    Vision,
    Action,
}

/// Multiple scales may coexist; none is imposed as the one canonical language unit.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum RepresentationScale {
    Byte,
    Phoneme,
    Morpheme,
    Word,
    Phrase,
    Concept,
    Intent,
}

/// Factorized event coding: a shared concept plus modality/scale-specific residuals.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorizedRepresentation {
    modality: Modality,
    semantic_signature: HyperVector,
    residuals: BTreeMap<RepresentationScale, HyperVector>,
}

impl FactorizedRepresentation {
    pub fn new(modality: Modality, semantic_signature: HyperVector) -> Self {
        Self {
            modality,
            semantic_signature,
            residuals: BTreeMap::new(),
        }
    }

    pub fn modality(&self) -> Modality {
        self.modality
    }

    pub fn semantic_signature(&self) -> &HyperVector {
        &self.semantic_signature
    }

    pub fn residual(&self, scale: RepresentationScale) -> Option<&HyperVector> {
        self.residuals.get(&scale)
    }

    pub fn with_residual(&self, scale: RepresentationScale, residual: HyperVector) -> Self {
        let mut residuals = self.residuals.clone();
        residuals.insert(scale, residual);
        Self {
            modality: self.modality,
            semantic_signature: self.semantic_signature.clone(),
            residuals,
        }
    }
}

/// Immutable event packet. Large raw data remains outside active state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Event {
    pub id: u64,
    pub timestamp_ms: u64,
    pub label: String,
    pub branch: BranchKind,
    pub signal: HyperVector,
    representation: FactorizedRepresentation,
    pub raw_detail: String,
}

impl Event {
    pub fn new(label: impl Into<String>, branch: BranchKind, signal: HyperVector) -> Self {
        Self {
            id: 0,
            timestamp_ms: 0,
            label: label.into(),
            branch,
            representation: FactorizedRepresentation::new(Modality::Abstract, signal.clone()),
            signal,
            raw_detail: String::new(),
        }
    }

    pub fn with_representation(mut self, representation: FactorizedRepresentation) -> Self {
        self.representation = representation;
        self
    }

    pub fn representation(&self) -> &FactorizedRepresentation {
        &self.representation
    }

    pub fn semantic_signature(&self) -> &HyperVector {
        self.representation.semantic_signature()
    }

    pub fn residual(&self, scale: RepresentationScale) -> Option<&HyperVector> {
        self.representation.residual(scale)
    }

    pub fn with_metadata(
        mut self,
        id: u64,
        timestamp_ms: u64,
        raw_detail: impl Into<String>,
    ) -> Self {
        self.id = id;
        self.timestamp_ms = timestamp_ms;
        self.raw_detail = raw_detail.into();
        self
    }
}
