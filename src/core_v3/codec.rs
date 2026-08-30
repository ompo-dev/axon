use std::error::Error;
use std::fmt::{Display, Formatter};

use super::event::{Event, FactorizedRepresentation, Modality, RepresentationScale};
use super::vector::HyperVector;

/// Packet boundary between external encoders and the sparse cognitive core.
///
/// It does not claim to infer a concept from raw media. An input codec supplies
/// the semantic signature and residual candidates; this component keeps only
/// residuals worth routing at the current scale.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveEventCodec {
    minimum_active_dimensions: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    ZeroMinimumActiveDimensions,
}

impl Display for CodecError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "minimum active dimensions must be positive")
    }
}

impl Error for CodecError {}

impl Default for AdaptiveEventCodec {
    fn default() -> Self {
        Self {
            minimum_active_dimensions: 1,
        }
    }
}

impl AdaptiveEventCodec {
    pub fn new(minimum_active_dimensions: usize) -> Result<Self, CodecError> {
        if minimum_active_dimensions == 0 {
            return Err(CodecError::ZeroMinimumActiveDimensions);
        }
        Ok(Self {
            minimum_active_dimensions,
        })
    }

    pub fn encode(
        &self,
        event: Event,
        modality: Modality,
        semantic_signature: HyperVector,
        residuals: impl IntoIterator<Item = (RepresentationScale, HyperVector)>,
    ) -> Event {
        let representation = residuals.into_iter().fold(
            FactorizedRepresentation::new(modality, semantic_signature),
            |representation, (scale, residual)| {
                if residual.active_dimensions() >= self.minimum_active_dimensions {
                    representation.with_residual(scale, residual)
                } else {
                    representation
                }
            },
        );
        event.with_representation(representation)
    }
}
