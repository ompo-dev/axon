//! Supervised segment experiments reuse the numeric genome and label-free bounded batch reader.
use super::*;
use crate::structural::sequence::{
    Example,
    segment::{Budget, Model},
};
pub(super) const MODULE: &str = "semantic-segment-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: Budget,
    examples: Vec<Example>,
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let mut presentations = 0usize;
    let mut replaced_spans = 0usize;
    let model = Model::learn(&training.examples, training.budget, &mut |e| {
        presentations += e.base.sentences + e.augmented_sentences;
        replaced_spans += e.replaced_spans;
        telemetry
            .emit(
                "segment-training",
                &[
                    ("epoch", V::Count(e.base.epoch as u64)),
                    ("intent_errors", V::Count(e.base.intent_errors as u64)),
                    ("tag_errors", V::Count(e.base.tag_errors as u64)),
                    (
                        "augmented_sentences",
                        V::Count(e.augmented_sentences as u64),
                    ),
                    ("replaced_spans", V::Count(e.replaced_spans as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    })?;
    let page = model.to_bytes()?;
    fs::create_dir(output)?;
    let genome = packed_payloads(
        &[
            ("structural", ModuleKind::Structural, &before),
            (MODULE, ModuleKind::SegmentSequence, &page),
        ],
        output,
    )?;
    let report = serde_json::json!({"schema":"axon-segment-training/v1","budget":training.budget,"total":training.examples.len(),"parent_unchanged":parent.to_bytes()?==before,
        "page_bytes":page.len(),"retained_model_bytes":model.retained_bytes(),"encoded_bytes":genome.encoded_bytes(),"weights":model.weights(),"codec":"AXSQ0008","elapsed_ns":started.elapsed().as_nanos(),"sentence_presentations":presentations,"replaced_spans":replaced_spans,
        "scope":"Segment averaged perceptron, learned intent-role domains optional, exact top-two segmentation only for the selected intent/domain. Not a CRF probability, semantic certificate, independent reserve or action authority. Training workspace is not inference residency."});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn segment_genome_preserves_parent_and_uses_the_shared_label_free_reader() {
        let root = std::env::temp_dir().join(format!(
            "axon-segment-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let parent = root.join("parent");
        fs::create_dir(&parent).unwrap();
        let seed = StructuralSeed::default();
        let before = seed.to_bytes().unwrap();
        packed(&seed, &parent).unwrap();
        let examples=(0..8).flat_map(|i|["send","cancel"].map(move |intent|serde_json::json!({"text":format!("{intent} to person-{i} surname-{i} now"),"intent":intent,"tags":["O","O","B:person","I:person","O"]}))).collect::<Vec<_>>();
        let data = root.join("train.json");
        save(&data,&serde_json::json!({"budget":{"learner":{"seed":47,"buckets":512,"epochs":8,"weights":2000,"minimum_margin":1},"max_span":4,"typed_roles":true,"max_work":1000000},"examples":examples})).unwrap();
        let output = root.join("trained");
        train(
            &parent.join("structural.axnum"),
            &root.join("birth-parent"),
            &data,
            &output,
        )
        .unwrap();
        assert_eq!(
            fs::read(output.join("structural.axstruct")).unwrap(),
            before
        );
        let input = root.join("input.json");
        save(
            &input,
            &serde_json::json!([{"id":"new","text":"send to unknown person now"}]),
        )
        .unwrap();
        let receipt = root.join("predictions.json");
        encoder_lab::segment_batch(
            &output.join("structural.axnum"),
            &root.join("child"),
            &input,
            &receipt,
        )
        .unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert!(rows[0]["work"].as_u64().unwrap() > 0);
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"x","text":"y","expected":7})
            )
            .is_err()
        );
        let plain_examples=(0..8).flat_map(|i|["send","cancel"].map(move |intent|serde_json::json!({"text":format!("{intent} person-{i}"),"intent":intent,"tags":["O","O"]}))).collect::<Vec<_>>();
        let config = root.join("plain-training.json");
        save(&config,&serde_json::json!({"budget":{"learner":{"seed":47,"buckets":512,"epochs":8,"weights":2000,"minimum_margin":1},"max_span":4,"typed_roles":true,"max_work":1000000},"examples":plain_examples})).unwrap();
        let plain = root.join("plain");
        train(
            &parent.join("structural.axnum"),
            &root.join("plain-parent"),
            &config,
            &plain,
        )
        .unwrap();
        let input = root.join("plain-inputs.json");
        save(
            &input,
            &serde_json::json!([{"id":"single-path","text":"send new-person"}]),
        )
        .unwrap();
        let receipt = root.join("plain-predictions.json");
        encoder_lab::segment_batch(
            &plain.join("structural.axnum"),
            &root.join("plain-child"),
            &input,
            &receipt,
        )
        .unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert!(rows[0]["prediction"].is_array());
        assert!(rows[0]["tag_margin"].is_null());
    }
}
