//! CSR storage for the same learned scores, with a distinct immutable codec.
use super::*;
use crate::live_store::checksum;
const MAGIC: &[u8; 8] = b"AXSQ0002";
pub(super) const MAX_WEIGHTS: usize = 78000;
pub const CONTEXT: &str = "axon/compact-sequence-v1";
pub const ENHANCED_CONTEXT: &str = "axon/enhanced-sequence-v1";
const ENHANCED_MAGIC: &[u8; 8] = b"AXSQ0003";
#[derive(Clone, Debug)]
pub struct CompactModel {
    metadata: Model,
    offsets: Vec<u32>,
    values: Vec<(u8, i8)>,
}
#[derive(Clone, Debug)]
pub struct EnhancedModel {
    base: CompactModel,
    subwords: bool,
}
impl EnhancedModel {
    pub fn refine(
        examples: &[Example],
        budget: Budget,
        initial: &Self,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        if budget.subwords != initial.subwords {
            return Err(Error::Invalid("enhanced refinement feature schema"));
        }
        let prior = initial.base.as_legacy();
        let model = Self {
            base: CompactModel::from_legacy(Model::fit(
                examples,
                budget,
                Some(&prior),
                MAX_WEIGHTS,
                emit,
            )?)?,
            subwords: initial.subwords,
        };
        model.to_bytes()?;
        Ok(model)
    }
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        let base =
            CompactModel::from_legacy(Model::fit(examples, budget, None, MAX_WEIGHTS, emit)?)?;
        let model = Self {
            base,
            subwords: budget.subwords,
        };
        model.to_bytes()?;
        Ok(model)
    }
    pub fn weights(&self) -> usize {
        self.base.weights()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() - std::mem::size_of::<CompactModel>()
            + self.base.retained_bytes()
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        self.base.predict_subwords(text, self.subwords)
    }
    pub fn probe_search(
        &self,
        text: &str,
        budget: search::Budget,
    ) -> Result<search::Prediction, Error> {
        self.base.search(text, self.subwords, budget)
    }
    pub fn measure_search(&self, text: &str, budget: search::Budget) -> search::Measurement {
        self.base.measure_search(text, self.subwords, budget)
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut bytes = ENHANCED_MAGIC.to_vec();
        bytes.push(u8::from(self.subwords));
        bytes.extend_from_slice(&self.base.to_bytes()?);
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < 45 || bytes.len() > MAX_BYTES {
            return Err(Error::Invalid("enhanced sequence length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if body.get(..8) != Some(ENHANCED_MAGIC.as_slice())
            || body[8] > 1
            || checksum(body).to_le_bytes() != digest
        {
            return Err(Error::Invalid("enhanced sequence header"));
        }
        Ok(Self {
            base: CompactModel::from_bytes(&body[9..])?,
            subwords: body[8] != 0,
        })
    }
}
fn take<const N: usize>(source: &mut &[u8]) -> Result<[u8; N], Error> {
    let (value, rest) = source
        .split_first_chunk::<N>()
        .ok_or(Error::Invalid("compact sequence truncation"))?;
    *source = rest;
    Ok(*value)
}
impl CompactModel {
    // Expand only in training scratch; persisted descendants keep the CSR codec.
    fn as_legacy(&self) -> Model {
        let mut model = self.metadata.clone();
        let classes = model.intents.len() + model.tags.len();
        model.weights = Vec::with_capacity(self.values.len());
        for (row, span) in self.offsets.windows(2).enumerate() {
            for (class, value) in &self.values[span[0] as usize..span[1] as usize] {
                model
                    .weights
                    .push(((row * classes + *class as usize) as u32, *value));
            }
        }
        model
    }
    pub fn refine(
        examples: &[Example],
        budget: Budget,
        initial: &Self,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        if budget.subwords {
            return Err(Error::Invalid("subword model requires enhanced codec"));
        }
        let prior = initial.as_legacy();
        Self::from_legacy(Model::fit(
            examples,
            budget,
            Some(&prior),
            MAX_WEIGHTS,
            emit,
        )?)
    }
    pub fn from_legacy(mut model: Model) -> Result<Self, Error> {
        let weights = std::mem::take(&mut model.weights);
        if weights.is_empty() || weights.len() > MAX_WEIGHTS {
            return Err(Error::Budget);
        }
        let classes = model.intents.len() + model.tags.len();
        let mut offsets = vec![0; model.buckets + 1];
        let mut values = Vec::with_capacity(weights.len());
        for (key, value) in weights {
            let row = key as usize / classes;
            offsets[row + 1] += 1;
            values.push(((key as usize % classes) as u8, value));
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let model = Self {
            metadata: model,
            offsets,
            values,
        };
        model.to_bytes()?;
        Ok(model)
    }
    pub fn learn(
        examples: &[Example],
        budget: Budget,
        emit: &mut impl FnMut(&Event) -> Result<(), Error>,
    ) -> Result<Self, Error> {
        if budget.subwords {
            return Err(Error::Invalid("subword model requires enhanced codec"));
        }
        Self::from_legacy(Model::fit(examples, budget, None, MAX_WEIGHTS, emit)?)
    }
    pub fn weights(&self) -> usize {
        self.values.len()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>() - std::mem::size_of::<Model>()
            + self.metadata.retained_bytes()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.values.capacity() * std::mem::size_of::<(u8, i8)>()
    }
    pub(super) fn metadata(&self) -> &Model {
        &self.metadata
    }
    pub(super) fn scores(&self, features: &[usize], work: &mut usize) -> Vec<i32> {
        let mut scores = vec![0; self.metadata.intents.len() + self.metadata.tags.len()];
        for feature in features {
            let row =
                &self.values[self.offsets[*feature] as usize..self.offsets[*feature + 1] as usize];
            for (class, value) in row {
                scores[*class as usize] += i32::from(*value);
                *work += 1;
            }
            *work += 1;
        }
        scores
    }
    pub fn predict(&self, text: &str) -> Result<Prediction, Error> {
        self.metadata
            .predict_using(text, |f, work| self.scores(f, work))
    }
    fn predict_subwords(&self, text: &str, subwords: bool) -> Result<Prediction, Error> {
        self.metadata
            .predict_configured(text, subwords, |f, work| self.scores(f, work))
    }
    pub(super) fn search(
        &self,
        text: &str,
        subwords: bool,
        budget: search::Budget,
    ) -> Result<search::Prediction, Error> {
        search::predict(&self.metadata, text, subwords, budget, |f, work| {
            self.scores(f, work)
        })
    }
    pub(super) fn measure_search(
        &self,
        text: &str,
        subwords: bool,
        budget: search::Budget,
    ) -> search::Measurement {
        search::measure(&self.metadata, text, subwords, budget, |f, work| {
            self.scores(f, work)
        })
    }
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let header = serde_json::to_vec(&Wire::from(self.metadata.clone()))
            .map_err(|e| Error::Storage(e.to_string()))?;
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&(self.values.len() as u32).to_le_bytes());
        let rows = self.offsets.windows(2).filter(|p| p[0] != p[1]).count();
        bytes.extend_from_slice(&(rows as u32).to_le_bytes());
        for (row, span) in self
            .offsets
            .windows(2)
            .enumerate()
            .filter(|(_, p)| p[0] != p[1])
        {
            bytes.extend_from_slice(&(row as u16).to_le_bytes());
            bytes.extend_from_slice(&((span[1] - span[0]) as u16).to_le_bytes());
            for (class, value) in &self.values[span[0] as usize..span[1] as usize] {
                bytes.extend_from_slice(&[*class, *value as u8]);
            }
        }
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        if bytes.len() > MAX_BYTES {
            return Err(Error::Budget);
        }
        Ok(bytes)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() < 28 || bytes.len() > MAX_BYTES {
            return Err(Error::Invalid("compact sequence length"));
        }
        let (body, digest) = bytes.split_at(bytes.len() - 8);
        if checksum(body).to_le_bytes() != digest {
            return Err(Error::Invalid("compact sequence checksum"));
        }
        let mut input = body;
        if &take::<8>(&mut input)? != MAGIC {
            return Err(Error::Invalid("compact sequence magic"));
        }
        let size = u32::from_le_bytes(take(&mut input)?) as usize;
        let (header, rest) = input
            .split_at_checked(size)
            .ok_or(Error::Invalid("compact sequence header"))?;
        input = rest;
        let wire: Wire =
            serde_json::from_slice(header).map_err(|e| Error::Storage(e.to_string()))?;
        validate_header(&wire)?;
        if !wire.packed.is_empty() {
            return Err(Error::Invalid("compact metadata weights"));
        }
        let metadata = metadata(wire);
        let count = u32::from_le_bytes(take(&mut input)?) as usize;
        let rows = u32::from_le_bytes(take(&mut input)?) as usize;
        if !(1..=MAX_WEIGHTS).contains(&count) || rows == 0 || rows > metadata.buckets {
            return Err(Error::Invalid("compact sequence quota"));
        }
        let mut offsets = vec![0; metadata.buckets + 1];
        let mut values = Vec::with_capacity(count);
        let mut previous = None;
        for _ in 0..rows {
            let row = u16::from_le_bytes(take(&mut input)?) as usize;
            let entries = u16::from_le_bytes(take(&mut input)?) as usize;
            if row >= metadata.buckets
                || previous.is_some_and(|p| row <= p)
                || entries == 0
                || entries > metadata.intents.len() + metadata.tags.len()
                || values.len() + entries > count
            {
                return Err(Error::Invalid("compact sequence row"));
            }
            previous = Some(row);
            let mut previous_class = None;
            for _ in 0..entries {
                let [class, value] = take(&mut input)?;
                if class as usize >= metadata.intents.len() + metadata.tags.len()
                    || value == 0
                    || previous_class.is_some_and(|p| class <= p)
                {
                    return Err(Error::Invalid("compact sequence class"));
                }
                values.push((class, value as i8));
                previous_class = Some(class);
            }
            offsets[row + 1] = entries as u32;
        }
        if values.len() != count || !input.is_empty() {
            return Err(Error::Invalid("compact sequence trailing data"));
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        Ok(Self {
            metadata,
            offsets,
            values,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refinement_remaps_classes_preserves_parent_and_requires_matching_features() {
        let mut examples = (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |intent| Example {
                    text: format!("{intent} letter to person-{i}"),
                    intent: intent.into(),
                    tags: ["O", "O", "O", "B:person"].map(str::to_owned).into(),
                })
            })
            .collect::<Vec<_>>();
        let mut budget = Budget {
            seed: 47,
            buckets: 4096,
            epochs: 16,
            weights: 6000,
            intent_weights: 0,
            minimum_margin: 1,
            subwords: false,
            shared_roles: false,
        };
        let compact = CompactModel::learn(&examples, budget, &mut |_| Ok(())).unwrap();
        assert_eq!(
            compact.as_legacy().to_bytes().unwrap(),
            learned_with(&examples, budget).to_bytes().unwrap()
        );
        budget.subwords = true;
        let enhanced = EnhancedModel::learn(&examples, budget, &mut |_| Ok(())).unwrap();
        let a = compact.to_bytes().unwrap();
        let b = enhanced.to_bytes().unwrap();
        examples.extend((0..16).map(|i| Example {
            text: format!("archive letter priority level-{i}"),
            intent: "archive".into(),
            tags: ["O", "O", "O", "B:importance"].map(str::to_owned).into(),
        }));
        let child = EnhancedModel::refine(&examples, budget, &enhanced, &mut |_| Ok(())).unwrap();
        assert!(CompactModel::refine(&examples, budget, &compact, &mut |_| Ok(())).is_err());
        budget.subwords = false;
        assert!(EnhancedModel::refine(&examples, budget, &enhanced, &mut |_| Ok(())).is_err());
        let other = CompactModel::refine(&examples, budget, &compact, &mut |_| Ok(())).unwrap();
        let reference = Model::learn(&examples, budget, Some(&compact.as_legacy()), &mut |_| {
            Ok(())
        })
        .unwrap();
        let child = EnhancedModel::from_bytes(&child.to_bytes().unwrap()).unwrap();
        let other = CompactModel::from_bytes(&other.to_bytes().unwrap()).unwrap();
        assert_eq!(other.metadata.intents, ["archive", "cancel", "send"]);
        assert!(other.metadata.tags.contains(&"B:importance".to_owned()));
        for e in &examples {
            let normalize = |p: Result<Prediction, Error>| {
                p.map(|p| serde_json::to_value(p).unwrap())
                    .map_err(|e| e.to_string())
            };
            assert_eq!(
                normalize(other.predict(&e.text)),
                normalize(reference.predict(&e.text))
            );
            let value = serde_json::to_value(child.predict(&e.text).unwrap().output).unwrap();
            assert_eq!(value[1], e.intent);
            assert_eq!(
                value[2][1][0],
                if e.intent == "archive" {
                    "importance"
                } else {
                    "person"
                }
            );
        }
        assert_eq!(a, compact.to_bytes().unwrap());
        assert_eq!(b, enhanced.to_bytes().unwrap());
        budget.seed += 1;
        assert!(CompactModel::refine(&examples, budget, &compact, &mut |_| Ok(())).is_err());
        budget.subwords = true;
        assert!(EnhancedModel::refine(&examples, budget, &enhanced, &mut |_| Ok(())).is_err());
        assert!(EnhancedModel::refine(&[], budget, &enhanced, &mut |_| Ok(())).is_err());
    }
    fn learned_with(examples: &[Example], budget: Budget) -> Model {
        Model::learn(examples, budget, None, &mut |_| Ok(())).unwrap()
    }
    #[test]
    fn enhanced_codec_preserves_feature_configuration_and_rejects_forged_flags() {
        let examples = (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |label| Example {
                    text: format!("{label} letter to person-{i}"),
                    intent: label.into(),
                    tags: ["O", "O", "O", "B:person"].map(str::to_owned).into(),
                })
            })
            .collect::<Vec<_>>();
        let budget = Budget {
            seed: 47,
            buckets: 4096,
            epochs: 10,
            weights: 6000,
            intent_weights: 0,
            minimum_margin: 1,
            subwords: true,
            shared_roles: true,
        };
        assert!(Model::learn(&examples, budget, None, &mut |_| Ok(())).is_err());
        assert!(CompactModel::learn(&examples, budget, &mut |_| Ok(())).is_err());
        let model = EnhancedModel::learn(&examples, budget, &mut |_| Ok(())).unwrap();
        let bytes = model.to_bytes().unwrap();
        assert_eq!(bytes[8], 1);
        let child = EnhancedModel::from_bytes(&bytes).unwrap();
        assert_eq!(bytes, child.to_bytes().unwrap());
        assert_eq!(model.retained_bytes(), child.retained_bytes());
        for text in [
            "send letter to person-0",
            "cancel letter to person-1",
            "send letter to other-name",
        ] {
            let a = model
                .predict(text)
                .map(|p| serde_json::to_value(p).unwrap())
                .map_err(|e| e.to_string());
            let b = child
                .predict(text)
                .map(|p| serde_json::to_value(p).unwrap())
                .map_err(|e| e.to_string());
            assert_eq!(a, b);
        }
        let mut forged = bytes[..bytes.len() - 8].to_vec();
        forged[8] = 2;
        forged.extend_from_slice(&checksum(&forged).to_le_bytes());
        assert!(EnhancedModel::from_bytes(&forged).is_err());
        assert!(CompactModel::from_bytes(&bytes).is_err());
    }
    fn learned() -> Model {
        let data = (0..16)
            .flat_map(|i| {
                ["send", "cancel"].map(move |intent| Example {
                    text: format!("{intent} letter to person-{i}"),
                    intent: intent.into(),
                    tags: ["O", "O", "O", "B:person"].map(str::to_owned).into(),
                })
            })
            .collect::<Vec<_>>();
        Model::learn(
            &data,
            Budget {
                seed: 47,
                buckets: 4096,
                epochs: 10,
                weights: 6000,
                intent_weights: 0,
                subwords: false,
                shared_roles: false,
                minimum_margin: 1,
            },
            None,
            &mut |_| Ok(()),
        )
        .unwrap()
    }
    #[test]
    fn csr_preserves_predictions_and_roundtrip_without_reinterpreting_legacy_bytes() {
        let legacy = learned();
        let original = legacy.to_bytes().unwrap();
        let compact = CompactModel::from_legacy(legacy.clone()).unwrap();
        let bytes = compact.to_bytes().unwrap();
        let child = CompactModel::from_bytes(&bytes).unwrap();
        assert_eq!(child.to_bytes().unwrap(), bytes);
        assert_eq!(compact.retained_bytes(), child.retained_bytes());
        assert_eq!(compact.weights(), legacy.weights());
        for text in [
            "send letter to person-0",
            "cancel letter to person-1",
            "send letter to other-name",
            "",
            "unknown",
        ] {
            let before = legacy
                .predict(text)
                .map(|p| serde_json::to_value(p).unwrap())
                .map_err(|e| e.to_string());
            let after = child
                .predict(text)
                .map(|p| serde_json::to_value(p).unwrap())
                .map_err(|e| e.to_string());
            assert_eq!(before, after);
        }
        assert_eq!(legacy.to_bytes().unwrap(), original);
        assert!(Model::from_bytes(&bytes).is_err());
        assert!(CompactModel::from_bytes(&original).is_err());
    }
    fn dense() -> Model {
        let mut model = learned();
        let classes = model.intents.len() + model.tags.len();
        model.weights = (0..model.buckets * classes)
            .map(|key| (key as u32, (key % 126 + 1) as i8))
            .collect();
        model
    }
    #[test]
    fn csr_reduces_storage_and_heap_for_identical_many_weight_scores() {
        let model = dense();
        let compact = CompactModel::from_legacy(model.clone()).unwrap();
        assert!(compact.to_bytes().unwrap().len() * 2 < model.to_bytes().unwrap().len());
        assert!(compact.retained_bytes() * 2 < model.retained_bytes());
        let mut a = 0;
        let mut b = 0;
        for row in (0..4096).step_by(7) {
            assert_eq!(model.scores(&[row], &mut a), compact.scores(&[row], &mut b));
        }
        assert_eq!(a, b);
    }
    #[test]
    fn csr_rejects_truncation_forged_rows_and_classes_even_with_valid_checksum() {
        let original = CompactModel::from_legacy(dense())
            .unwrap()
            .to_bytes()
            .unwrap();
        for size in [0, 8, 27, original.len() - 1] {
            assert!(CompactModel::from_bytes(&original[..size]).is_err());
        }
        let header = u32::from_le_bytes(original[8..12].try_into().unwrap()) as usize;
        let row = 20 + header;
        for (offset, value) in [
            (row, 255),
            (row + 1, 255),
            (row + 4, 255),
            (row + 5, 0),
            (row + 6, original[row + 4]),
        ] {
            let mut bytes = original[..original.len() - 8].to_vec();
            bytes[offset] = value;
            bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
            assert!(
                CompactModel::from_bytes(&bytes).is_err(),
                "corruption offset {offset}"
            );
        }
        let mut bytes = original[..original.len() - 8].to_vec();
        bytes.push(0);
        bytes.extend_from_slice(&checksum(&bytes).to_le_bytes());
        assert!(CompactModel::from_bytes(&bytes).is_err());
    }
}
