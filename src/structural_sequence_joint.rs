//! Globally updated sparse joint intent/BIO model. Its beam policy is in the codec.
use super::*;
use crate::live_store::checksum;
use compact::CompactModel;
const MAGIC: &[u8; 8] = b"AXSQ0005";
pub const CONTEXT: &str = "axon/joint-sequence-v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub learner: super::Budget,
    pub search: search::Budget,
}
#[derive(Clone, Debug)]
pub struct Model {
    base: CompactModel,
    subwords: bool,
    search: search::Budget,
}
pub use super::search::Prediction;
impl Model {
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        budget.search.validate()?;
        let base = super::Model::fit_mode(
            examples,
            budget.learner,
            None,
            compact::MAX_WEIGHTS,
            Some(budget.search),
            emit,
        )?;
        let model = Self {
            base: CompactModel::from_legacy(base)?,
            subwords: budget.learner.subwords,
            search: budget.search,
        };
        model.to_bytes()?;
        Ok(model)
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        self.probe_search(text, self.search)
    }
    pub fn measure(&self, text: &str, policy: Option<search::Budget>) -> search::Measurement {
        self.base
            .measure_search(text, self.subwords, policy.unwrap_or(self.search))
    }
    /// Decoder-only control; an override does not change the inherited default policy.
    pub fn probe_search(&self, text: &str, budget: search::Budget) -> Result<Prediction, Error> {
        self.base.search(text, self.subwords, budget)
    }
    pub fn weights(&self) -> usize {
        self.base.weights()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() - std::mem::size_of::<CompactModel>()
            + self.base.retained_bytes()
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let policy = serde_json::to_vec(&self.search).map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = MAGIC.to_vec();
        bytes.push(u8::from(self.subwords));
        bytes.extend_from_slice(&(policy.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&policy);
        bytes.extend_from_slice(&self.base.to_bytes()?);
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if !(50..=MAX_BYTES).contains(&bytes.len()) {
            return Err(Error::Invalid("joint sequence length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if body.get(..8) != Some(MAGIC.as_slice())
            || body[8] > 1
            || checksum(body).to_le_bytes() != digest
        {
            return Err(Error::Invalid("joint sequence header"));
        }
        let size = u32::from_le_bytes(
            body[9..13]
                .try_into()
                .map_err(|_| Error::Invalid("joint policy length"))?,
        ) as usize;
        let end = 13usize
            .checked_add(size)
            .filter(|end| *end <= body.len())
            .ok_or(Error::Invalid("joint policy length"))?;
        let search: search::Budget =
            serde_json::from_slice(&body[13..end]).map_err(|e| Error::Storage(e.to_string()))?;
        search.validate()?;
        Ok(Self {
            base: CompactModel::from_bytes(&body[end..])?,
            subwords: body[8] != 0,
            search,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn structured_learning_roundtrip_and_corruption_are_bounded() {
        let data = (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |intent| Example {
                    text: format!("{intent} letter to person-{i}"),
                    intent: intent.into(),
                    tags: vec!["O".into(), "O".into(), "O".into(), "B:person".into()],
                })
            })
            .collect::<Vec<_>>();
        let budget = Budget {
            learner: super::super::Budget {
                seed: 47,
                buckets: 1024,
                epochs: 6,
                weights: 8000,
                intent_weights: 4000,
                subwords: true,
                shared_roles: false,
                minimum_margin: 1,
            },
            search: search::Budget {
                beam: 4,
                intents: 2,
                max_work: 1_000_000,
            },
        };
        let model = Model::learn(&data, budget, &mut |_| Ok(())).unwrap();
        let raw = model.to_bytes().unwrap();
        let born = Model::from_bytes(&raw).unwrap();
        let p = born.predict("send letter to person-new").unwrap();
        assert_eq!(
            p.output,
            search::output(
                &tokens("send letter to person-new").unwrap(),
                "send",
                &["O", "O", "O", "B:person"]
            )
            .unwrap()
        );
        assert!(p.work > 0 && p.expansions > 0 && p.pruned);
        assert_eq!(raw, born.to_bytes().unwrap());
        assert!(model.retained_bytes() < MAX_BYTES);
        for i in 0..raw.len() {
            let mut broken = raw.clone();
            broken[i] ^= 1;
            assert!(Model::from_bytes(&broken).is_err());
        }
        for i in [0, 8, 12, 49, raw.len() - 1] {
            assert!(Model::from_bytes(&raw[..i]).is_err());
        }
    }
}
