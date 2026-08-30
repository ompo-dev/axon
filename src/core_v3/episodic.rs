use super::event::Event;
use super::vector::{HyperVector, VectorError};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Episode {
    pub event: Event,
    pub semantic_signature: HyperVector,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EpisodeMatch {
    pub event: Event,
    pub similarity: f32,
}

/// Append-only by API: old stores remain valid references to earlier experience.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EpisodicStore {
    entries: Vec<Episode>,
}

impl EpisodicStore {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn append(&self, event: Event, semantic_signature: HyperVector) -> Self {
        let mut entries = self.entries.clone();
        entries.push(Episode {
            event,
            semantic_signature,
        });
        Self { entries }
    }

    pub fn lookup(
        &self,
        signature: &HyperVector,
        minimum_similarity: f32,
    ) -> Result<Vec<EpisodeMatch>, VectorError> {
        let mut matches = Vec::new();
        for episode in &self.entries {
            let similarity = episode.semantic_signature.similarity(signature)?;
            if similarity >= minimum_similarity {
                matches.push(EpisodeMatch {
                    event: episode.event.clone(),
                    similarity,
                });
            }
        }
        matches.sort_by(|left, right| right.similarity.total_cmp(&left.similarity));
        Ok(matches)
    }
}
