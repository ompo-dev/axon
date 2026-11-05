//! Explicit supervised lexical encoder experiments. Predictions never grant authority.
use super::*;
use crate::structural::sequence::{
    Example as SequenceExample,
    lexical_encoder::{Auxiliary, Budget, CONTEXT, LexicalGraph, Model},
};
const MODULE: &str = "semantic-lexical-encoder-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: Budget,
    examples: Vec<SequenceExample>,
    #[serde(default)]
    auxiliary: Option<Auxiliary>,
    #[serde(default)]
    lexical_graph: Option<LexicalGraph>,
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, false)
}
pub fn refine(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, true)
}
fn train_mode(number: &Path, birth: &Path, data: &Path, output: &Path, refine: bool) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let original_number = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let initial = if refine {
        Some(born_page(
            number,
            &birth.join("lexical-source"),
            MODULE,
            CONTEXT,
            |m| {
                if let Module::LexicalEncoder(m) = m {
                    Some(m.as_ref().clone())
                } else {
                    None
                }
            },
        )?)
    } else {
        None
    };
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let started = Instant::now();
    let mut auxiliary_sentences = 0usize;
    let mut auxiliary_words = 0usize;
    let mut auxiliary_errors = 0usize;
    let mut optimizer_steps = 0usize;
    let mut lexical_edge_updates = 0usize;
    let mut primary_tag_normalization_terms = 0usize;
    let mut emit = |e: &crate::structural::sequence::lexical_encoder::TrainingEvent| {
        auxiliary_sentences += e.auxiliary_sentences;
        auxiliary_words += e.auxiliary_words;
        auxiliary_errors += e.auxiliary_errors;
        optimizer_steps += e.optimizer_steps;
        lexical_edge_updates += e.lexical_edge_updates;
        primary_tag_normalization_terms += e.primary_tag_normalization_terms;
        telemetry
            .emit(
                "encoder-training",
                &[
                    ("epoch", V::Count(e.primary.epoch as u64)),
                    ("intent_errors", V::Count(e.primary.intent_errors as u64)),
                    ("tag_errors", V::Count(e.primary.tag_errors as u64)),
                    ("optimizer_steps", V::Count(e.optimizer_steps as u64)),
                    (
                        "lexical_edge_updates",
                        V::Count(e.lexical_edge_updates as u64),
                    ),
                    (
                        "primary_tag_normalization_terms",
                        V::Count(e.primary_tag_normalization_terms as u64),
                    ),
                    (
                        "auxiliary_sentences",
                        V::Count(e.auxiliary_sentences as u64),
                    ),
                    ("auxiliary_words", V::Count(e.auxiliary_words as u64)),
                    ("auxiliary_errors", V::Count(e.auxiliary_errors as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    };
    let model = if let Some(m) = initial.as_ref() {
        Model::refine_with_sources(
            &training.examples,
            training.budget,
            m,
            training.auxiliary.as_ref(),
            training.lexical_graph.as_ref(),
            &mut emit,
        )?
    } else {
        Model::learn_with_sources(
            &training.examples,
            training.budget,
            training.auxiliary.as_ref(),
            training.lexical_graph.as_ref(),
            &mut emit,
        )?
    };
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original_number {
        return Err("encoder training changed parent number".into());
    }
    let page = model.to_bytes()?;
    fs::create_dir(output)?;
    let genome = packed_payloads(
        &[
            ("structural", ModuleKind::Structural, &before),
            (MODULE, ModuleKind::LexicalEncoder, &page),
        ],
        output,
    )?;
    let report = serde_json::json!({"schema":"axon-lexical-encoder-training/v1","budget":training.budget,"total":training.examples.len(),"words":model.words(),"weights":model.weights(),"page_bytes":page.len(),"retained_model_bytes":model.retained_bytes(),"encoded_bytes":genome.encoded_bytes(),"parent_unchanged":parent.to_bytes()?==before,"elapsed_ns":started.elapsed().as_nanos(),"scope":"Local supervised vocabulary and positional embeddings; no pretrained network, general language certificate, independent reserve or action authority. Training workspace is not inference residency."});
    let mut report = report;
    if let Some(m) = initial.as_ref() {
        report["refinement"] = serde_json::json!({"inherited_words":m.words(),"inherited_weights":m.weights(),"parent_number_unchanged":true,"optimizer_state_inherited":false,"vocabulary_policy":"keep every inherited word; admit frequent new words only into remaining bounded slots","scope":"Continue dequantized parameters, not private episodes or raw training examples. New training may forget; retention requires a separate audit. Ancestor cost is not zero."});
    }
    report["codec"] = serde_json::json!(model.codec());
    report["primary_tag_normalization_terms"] = serde_json::json!(primary_tag_normalization_terms);
    report["role_schema"] = serde_json::json!({"learned":training.budget.typed_roles,"source":"primary train intent/role labels only","neural_weights_exclude_categorical_tail":true});
    report["lexical_graph"] = serde_json::json!({"configured":training.lexical_graph.is_some(),"weight":training.lexical_graph.as_ref().map(|g|g.weight),"interval":training.lexical_graph.as_ref().map(|g|g.interval),"edges":training.lexical_graph.as_ref().map(|g|g.edges.len()),"updates":lexical_edge_updates,"edges_deployed":false});
    report["optimizer"] = serde_json::json!({"kind":if training.budget.minibatch==0 {"online-sgd"} else {"minibatch-adam"},"steps":optimizer_steps,"batch_sentences":training.budget.minibatch,"state_deployed":false,"gradient_clip_norm":if training.budget.minibatch==0 {None} else {Some(1.0)}});
    report["auxiliary"] = serde_json::json!({"configured":training.auxiliary.is_some(),"sentences":auxiliary_sentences,"words":auxiliary_words,"errors":auxiliary_errors,"head_deployed":false,"vocabulary_source":"primary examples only","weight":training.auxiliary.as_ref().map(|a|a.weight),"interval":training.auxiliary.as_ref().map(|a|a.interval)});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, false)
}
pub fn segment_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, true)
}
fn batch_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
    segment: bool,
) -> Result<()> {
    bounded_path(number)?;
    bounded_path(birth)?;
    bounded_path(data)?;
    bounded_path(receipt)?;
    let inputs: Vec<InterpretationInput> =
        serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&inputs.len())
        || inputs
            .iter()
            .any(|i| i.id.is_empty() || i.id.len() > 96 || !ids.insert(&i.id))
    {
        return Err("encoder input identities".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let mut rows = Vec::new();
    let mut work = 0usize;
    let mut bytes = 0usize;
    let (id, context) = if segment {
        (
            super::segment_lab::MODULE,
            crate::structural::sequence::segment::CONTEXT,
        )
    } else {
        (MODULE, CONTEXT)
    };
    born.controller.runtime_mut().with_module(id,context,&auth,|module|{
        for input in &inputs {
            let measured=match module {
                Module::LexicalEncoder(model) if !segment=>model.measure(&input.text),
                Module::SegmentSequence(model) if segment=>model.measure(&input.text),
                _=>return Err(crate::demand::DemandError::Invalid("encoder module")),
            };work=work.checked_add(measured.work).ok_or(crate::demand::DemandError::Budget("encoder work overflow"))?;
            if work>100_000_000{return Err(crate::demand::DemandError::Budget("encoder aggregate work"));}
            let predicted=measured.result;
            let row=serde_json::json!({"id":input.id,"prediction":predicted.as_ref().ok().map(|p|&p.output),"work":measured.work,"intent_margin":predicted.as_ref().ok().map(|p|p.intent_margin),"tag_margin":predicted.as_ref().ok().and_then(|p|(p.tag_margin!=i32::MAX).then_some(p.tag_margin)),"error":predicted.as_ref().err().map(ToString::to_string)});
            bytes+=serde_json::to_vec(&row).map_err(|e|crate::demand::DemandError::Module(e.to_string()))?.len();if bytes>16_000_000{return Err(crate::demand::DemandError::Budget("encoder receipt size"));}rows.push(row);
        }Ok(())
    })?;
    let unchanged = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? == original;
    if !unchanged {
        return Err("encoder changed inherited number".into());
    }
    save(receipt, &rows)?;
    let stats = born.controller.runtime().stats();
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"seed_unchanged":unchanged,"teacher_queries":0,"loads":stats.loads,"active_modules":stats.active_modules,"work":work,"scope":if segment {"One demand-loaded segment model; tag_margin compares the exact top two segmentations for the selected intent and learned role domain, not all meanings, probabilities or authority. Work includes rejected scoring/expansions, not process/IO work."} else {"One demand-loaded lexical encoder, no clone outside runtime. Work counts model scoring including refusals, not process/IO work. Scores do not authorize effects."}})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn portable_encoder(
        relative_regions: bool,
        with_auxiliary: bool,
        with_bytes: bool,
        with_ordered: bool,
        with_split: bool,
        with_refinement: bool,
    ) {
        let root = std::env::temp_dir().join(format!(
            "axon-encoder-{}-{}-{}-{}-{}-{}",
            std::process::id(),
            relative_regions,
            with_auxiliary,
            with_bytes,
            with_ordered,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let parent = root.join("parent");
        fs::create_dir(&parent).unwrap();
        let seed = StructuralSeed::default();
        let original = seed.to_bytes().unwrap();
        packed(&seed, &parent).unwrap();
        let examples=(0..8).flat_map(|i|["send","cancel"].map(move |intent|serde_json::json!({"text":format!("{intent} person-{i}"),"intent":intent,"tags":["O","B:person"]}))).collect::<Vec<_>>();
        let config = root.join("train.json");
        let mut training = serde_json::json!({"budget":{"seed":47,"vocabulary":64,"characters":256,"dimensions":8,"epochs":8,"learning_rate":0.1,"balanced":false,"minimum_margin":1,"relative_regions":relative_regions},"examples":examples});
        if with_bytes {
            training["budget"]["characters"] = serde_json::json!(0);
            training["budget"]["byte_convolution"] = serde_json::json!(true);
        }
        if with_ordered {
            training["budget"]["characters"] = serde_json::json!(0);
            training["budget"]["ordered_context"] = serde_json::json!(true);
        }
        if with_split {
            training["budget"]["characters"] = serde_json::json!(0);
            training["budget"]["hidden"] = serde_json::json!(16);
            training["budget"]["epochs"] = serde_json::json!(32);
            training["budget"]["split_heads"] = serde_json::json!(true);
        }
        if with_auxiliary {
            training["auxiliary"] = serde_json::json!({"weight":0.1,"interval":4,"examples":(0..4).map(|i|serde_json::json!({"text":format!("aux-only-{i} arrives"),"labels":["NOUN","VERB"]})).collect::<Vec<_>>()});
        }
        save(&config, &training).unwrap();
        let output = root.join("model");
        train(
            &parent.join("structural.axnum"),
            &root.join("born-parent"),
            &config,
            &output,
        )
        .unwrap();
        assert_eq!(
            fs::read(output.join("structural.axstruct")).unwrap(),
            original
        );
        let output = if with_refinement {
            let refined = root.join("refined");
            let source = output.join("structural.axnum");
            let number = fs::read(&source).unwrap();
            refine(&source, &root.join("refine-birth"), &config, &refined).unwrap();
            assert_eq!(fs::read(&source).unwrap(), number);
            assert_eq!(
                fs::read(refined.join("structural.axstruct")).unwrap(),
                original
            );
            let report: serde_json::Value =
                serde_json::from_slice(&fs::read(refined.join("training.json")).unwrap()).unwrap();
            assert!(report["refinement"]["inherited_words"].as_u64().unwrap() > 0);
            assert!(report["refinement"]["inherited_weights"].as_u64().unwrap() > 0);
            assert_eq!(report["refinement"]["optimizer_state_inherited"], false);
            refined
        } else {
            output
        };
        let input = root.join("input.json");
        save(
            &input,
            &serde_json::json!([{"id":"new","text":"send person-new"}]),
        )
        .unwrap();
        let receipt = root.join("receipt.json");
        batch(
            &output.join("structural.axnum"),
            &root.join("born-child"),
            &input,
            &receipt,
        )
        .unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), 1);
        assert!(rows[0]["work"].as_u64().unwrap() > 0);
        assert_eq!(
            rows[0]["prediction"],
            serde_json::json!(["intent", "send", ["slots", ["person", "person-new"]]])
        );
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"x","text":"y","expected":"send"})
            )
            .is_err()
        );
        let probe_input = root.join("search-input.json");
        save(&probe_input,&serde_json::json!({"format":"lexical-mean","budget":{"beam":4,"intents":2,"max_work":1_000_000},"inputs":[{"id":"new","text":"send person-new"}]})).unwrap();
        let probe_receipt = root.join("search-receipt.json");
        super::super::joint_lab::probe(
            &output.join("structural.axnum"),
            &root.join("search-child"),
            &probe_input,
            &probe_receipt,
        )
        .unwrap();
        let rows: serde_json::Value =
            serde_json::from_slice(&fs::read(probe_receipt).unwrap()).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), 1);
        assert!(rows[0]["prediction"].is_array());
        assert!(rows[0]["work"].as_u64().unwrap() > 0);
    }
    #[test]
    fn portable_encoder_preserves_parent_and_prediction_has_no_teacher_fields() {
        portable_encoder(false, false, false, false, false, false);
    }
    #[test]
    fn region_encoder_uses_the_same_numeric_birth_and_label_free_prediction_contract() {
        portable_encoder(true, false, false, false, false, false);
    }
    #[test]
    fn auxiliary_training_head_is_removed_before_numeric_birth_and_label_free_prediction() {
        portable_encoder(true, true, false, false, false, false);
    }
    #[test]
    fn byte_model_uses_the_existing_numeric_birth_runtime_and_teacher_free_contract() {
        portable_encoder(true, false, true, false, false, false);
    }
    #[test]
    fn ordered_model_uses_numeric_birth_and_auxiliary_has_no_inference_head() {
        portable_encoder(true, true, false, true, false, false);
    }
    #[test]
    fn split_mixers_use_numeric_birth_and_the_label_free_runtime() {
        portable_encoder(true, false, false, false, true, false);
    }
    #[test]
    fn lexical_refinement_uses_the_existing_seed_and_label_free_birth_contract() {
        portable_encoder(true, false, false, false, true, true);
        portable_encoder(false, false, false, false, false, true);
    }
}
