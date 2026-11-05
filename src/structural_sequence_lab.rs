//! Explicit empirical interpretation page. It never authorizes or executes a tool.
use super::*;
use crate::structural::sequence::compact::{CompactModel, EnhancedModel};
use crate::structural::sequence::{self, Model};
const MODULE: &str = "semantic-sequence-v1";
const COMPACT: &str = "semantic-compact-v1";
const ENHANCED: &str = "semantic-enhanced-v1";
const FACTORIZED: &str = "semantic-factorized-v1";
#[derive(Clone, Copy)]
enum Format {
    Legacy,
    Compact,
    Enhanced,
    Factorized,
}
impl Format {
    fn properties(self) -> (&'static str, ModuleKind, &'static str) {
        match self {
            Self::Legacy => (MODULE, ModuleKind::Sequence, sequence::CONTEXT),
            Self::Compact => (
                COMPACT,
                ModuleKind::CompactSequence,
                sequence::compact::CONTEXT,
            ),
            Self::Enhanced => (
                ENHANCED,
                ModuleKind::EnhancedSequence,
                sequence::compact::ENHANCED_CONTEXT,
            ),
            Self::Factorized => (
                FACTORIZED,
                ModuleKind::FactorizedSequence,
                sequence::factorized::CONTEXT,
            ),
        }
    }
}
enum Empirical {
    Legacy(Model),
    Compact(CompactModel),
    Enhanced(EnhancedModel),
    Factorized(sequence::factorized::Model),
}
impl Empirical {
    fn to_bytes(&self) -> std::result::Result<Vec<u8>, crate::structural::Error> {
        match self {
            Self::Legacy(m) => m.to_bytes(),
            Self::Compact(m) => m.to_bytes(),
            Self::Enhanced(m) => m.to_bytes(),
            Self::Factorized(m) => m.to_bytes(),
        }
    }
    fn predict(
        &self,
        text: &str,
    ) -> std::result::Result<sequence::Prediction, crate::structural::Error> {
        match self {
            Self::Legacy(m) => m.predict(text),
            Self::Compact(m) => m.predict(text),
            Self::Enhanced(m) => m.predict(text),
            Self::Factorized(m) => m.predict(text),
        }
    }
    fn weights(&self) -> usize {
        match self {
            Self::Legacy(m) => m.weights(),
            Self::Compact(m) => m.weights(),
            Self::Enhanced(m) => m.weights(),
            Self::Factorized(m) => m.weights(),
        }
    }
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Legacy(m) => m.retained_bytes(),
            Self::Compact(m) => m.retained_bytes(),
            Self::Enhanced(m) => m.retained_bytes(),
            Self::Factorized(m) => m.retained_bytes(),
        }
    }
}
fn born(number: &Path, birth: &Path, format: Format) -> Result<Empirical> {
    let (module, _, context) = format.properties();
    born_page(number, birth, module, context, |m| match (format, m) {
        (Format::Legacy, Module::Sequence(m)) => Some(Empirical::Legacy(m.as_ref().clone())),
        (Format::Compact, Module::CompactSequence(m)) => {
            Some(Empirical::Compact(m.as_ref().clone()))
        }
        (Format::Enhanced, Module::EnhancedSequence(m)) => {
            Some(Empirical::Enhanced(m.as_ref().clone()))
        }
        (Format::Factorized, Module::FactorizedSequence(m)) => {
            Some(Empirical::Factorized(m.as_ref().clone()))
        }
        _ => None,
    })
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: sequence::Budget,
    examples: Vec<sequence::Example>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FactorizedTraining {
    budget: sequence::factorized::Budget,
    examples: Vec<sequence::Example>,
}
pub fn factorized_train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    factorized_train_mode(number, birth, data, output, false)
}
pub fn factorized_refine(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    factorized_train_mode(number, birth, data, output, true)
}
fn factorized_train_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
    refine: bool,
) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: FactorizedTraining = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let initial = if refine {
        Some(born(
            number,
            &birth.join("factorized-source"),
            Format::Factorized,
        )?)
    } else {
        None
    };
    let mut telemetry = LabTelemetry::from_environment(FACTORIZED);
    let started = Instant::now();
    let mut emit = |event: &sequence::Event| {
        telemetry
            .emit(
                "sequence-training",
                &[
                    ("epoch", V::Count(event.epoch as u64)),
                    ("sentences", V::Count(event.sentences as u64)),
                    ("intent_errors", V::Count(event.intent_errors as u64)),
                    ("tag_errors", V::Count(event.tag_errors as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    };
    let model = match initial.as_ref() {
        Some(Empirical::Factorized(m)) => {
            sequence::factorized::Model::refine(&training.examples, training.budget, m, &mut emit)?
        }
        None => sequence::factorized::Model::learn(&training.examples, training.budget, &mut emit)?,
        _ => return Err("factorized refinement format".into()),
    };
    if parent.to_bytes()? != before {
        return Err("factorized training changed parent".into());
    }
    publish_model(
        Empirical::Factorized(model),
        &before,
        output,
        Format::Factorized,
        serde_json::json!({"refined":refine,"total":training.examples.len(),"budget":training.budget}),
        started,
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CompressionProbe {
    training: Training,
    inputs: Vec<InterpretationInput>,
}
pub fn compression_probe(data: &Path, output: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let probe: CompressionProbe = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&probe.inputs.len())
        || probe
            .inputs
            .iter()
            .any(|r| r.id.is_empty() || r.id.len() > 96 || !ids.insert(&r.id))
    {
        return Err("compression probe identity or quota".into());
    }
    let mut telemetry = LabTelemetry::from_environment("sequence-compression-probe");
    let started = Instant::now();
    let (wide, pruned) = Model::compression_models(
        &probe.training.examples,
        probe.training.budget,
        &mut |event| {
            telemetry
                .emit(
                    "compression-training",
                    &[
                        ("epoch", V::Count(event.epoch as u64)),
                        ("sentences", V::Count(event.sentences as u64)),
                    ],
                )
                .map_err(|e| crate::structural::Error::Storage(e.to_string()))
        },
    )?;
    let mut results = Vec::new();
    for row in &probe.inputs {
        let predictions = [&wide, &pruned].map(|model| {
            let prediction = model.probe_predict(&row.text, probe.training.budget.subwords);
            serde_json::json!({"id":row.id,"prediction":prediction.as_ref().ok().map(|p|&p.output),"error":prediction.as_ref().err().map(ToString::to_string)})
        });
        results.push(
            serde_json::json!({"id":row.id,"unpruned":predictions[0],"pruned":predictions[1]}),
        );
    }
    fs::create_dir(output)?;
    save(&output.join("predictions.json"), &results)?;
    let summary = serde_json::json!({"schema":"axon-sequence-compression-probe/v1","unpruned_weights":wide.weights(),"pruned_weights":pruned.weights(),"unpruned_model_bytes":wide.retained_bytes(),"pruned_model_bytes":pruned.retained_bytes(),"training_rows":probe.training.examples.len(),"prediction_rows":probe.inputs.len(),"teacher_queries_at_prediction":0,"elapsed_ns":started.elapsed().as_nanos(),"genome_published":false,"scope":"Nonportable laboratory scratch model; compares the same quantized coefficients before/after pruning, not full-precision loss or M1 certification. Unpruned memory exceeds deployment quotas and is never admitted."});
    save(&output.join("summary.json"), &summary)?;
    println!("{summary}");
    Ok(())
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, false, Format::Legacy)
}
pub fn refine(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, true, Format::Legacy)
}
pub fn compact_train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, false, Format::Compact)
}
pub fn enhanced_train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, false, Format::Enhanced)
}
pub fn compact_refine(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, true, Format::Compact)
}
pub fn enhanced_refine(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, true, Format::Enhanced)
}
fn train_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
    refine: bool,
    format: Format,
) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let initial = if refine {
        Some(born(number, &birth.join("sequence-source"), format)?)
    } else {
        None
    };
    let prior = initial.as_ref();
    let (module, _, _) = format.properties();
    let mut telemetry = LabTelemetry::from_environment(module);
    let started = Instant::now();
    let mut emit = |event: &sequence::Event| {
        telemetry
            .emit(
                "sequence-training",
                &[
                    ("epoch", V::Count(event.epoch as u64)),
                    ("sentences", V::Count(event.sentences as u64)),
                    ("intent_errors", V::Count(event.intent_errors as u64)),
                    ("tag_errors", V::Count(event.tag_errors as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    };
    let model =
        match (format, prior) {
            (Format::Compact, Some(Empirical::Compact(m))) => Empirical::Compact(
                CompactModel::refine(&training.examples, training.budget, m, &mut emit)?,
            ),
            (Format::Enhanced, Some(Empirical::Enhanced(m))) => Empirical::Enhanced(
                EnhancedModel::refine(&training.examples, training.budget, m, &mut emit)?,
            ),
            (Format::Compact, None) => Empirical::Compact(CompactModel::learn(
                &training.examples,
                training.budget,
                &mut emit,
            )?),
            (Format::Enhanced, None) => Empirical::Enhanced(EnhancedModel::learn(
                &training.examples,
                training.budget,
                &mut emit,
            )?),
            (Format::Legacy, None | Some(Empirical::Legacy(_))) => Empirical::Legacy(Model::learn(
                &training.examples,
                training.budget,
                match prior {
                    Some(Empirical::Legacy(m)) => Some(m),
                    _ => None,
                },
                &mut emit,
            )?),
            _ => return Err("sequence refinement format".into()),
        };
    if parent.to_bytes()? != before {
        return Err("sequence training changed parent".into());
    }
    publish_model(
        model,
        &before,
        output,
        format,
        serde_json::json!({"refined":refine,"total":training.examples.len(),"budget":training.budget}),
        started,
    )
}
fn publish_model(
    model: Empirical,
    parent_page: &[u8],
    output: &Path,
    format: Format,
    mut report: serde_json::Value,
    started: Instant,
) -> Result<()> {
    let (module, kind, _) = format.properties();
    let weights = model.weights();
    let retained = model.retained_bytes();
    let page = model.to_bytes()?;
    fs::create_dir(output)?;
    let genome = packed_payloads(
        &[
            ("structural", ModuleKind::Structural, parent_page),
            (module, kind, &page),
        ],
        output,
    )?;
    report.as_object_mut().ok_or("training report shape")?.extend(serde_json::json!({"schema":"axon-sequence-training/v1","weights":weights,"retained_model_bytes":retained,"page_bytes":page.len(),"encoded_bytes":genome.encoded_bytes(),"parent_unchanged":true,"elapsed_ns":started.elapsed().as_nanos(),"module":module,"scope":"Empirical intent and BIO sequence predictor, not an exact certificate, full M1 approval, or action authority."}).as_object().unwrap().clone());
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Format::Legacy)
}
pub fn compact_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Format::Compact)
}
pub fn enhanced_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Format::Enhanced)
}
pub fn factorized_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Format::Factorized)
}
fn batch_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
    format: Format,
) -> Result<()> {
    bounded_path(data)?;
    bounded_path(receipt)?;
    let rows: Vec<InterpretationInput> = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&rows.len())
        || rows
            .iter()
            .any(|r| r.id.is_empty() || r.id.len() > 96 || !ids.insert(&r.id))
    {
        return Err("sequence batch identity or quota".into());
    }
    let (module, _, _) = format.properties();
    let model = born(number, birth, format)?;
    let before = model.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment(module);
    let started = Instant::now();
    let mut answered = 0;
    let mut predictions = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let result = model.predict(&row.text);
        answered += usize::from(result.is_ok());
        let prediction = serde_json::json!({"id":row.id,"prediction":result.as_ref().ok().map(|p|&p.output),"intent_margin":result.as_ref().ok().map(|p|p.intent_margin),"tag_margin":result.as_ref().ok().map(|p|p.tag_margin),"work":result.as_ref().ok().map(|p|p.work),"error":result.as_ref().err().map(ToString::to_string)});
        let encoded = serde_json::to_string(&prediction)?;
        telemetry.emit(
            "sequence-prediction",
            &[
                ("case", V::Count(index as u64)),
                ("actual", V::Text(&encoded)),
            ],
        )?;
        predictions.push(prediction);
    }
    if model.to_bytes()? != before {
        return Err("sequence inference changed seed".into());
    }
    save(receipt, &predictions)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"answered":answered,"abstained":rows.len()-answered,"seed_unchanged":true,"teacher_queries":0,"elapsed_ns":started.elapsed().as_nanos(),"module":module,"scope":"Only id/text reach prediction; margins are not probabilities or permission to execute a tool."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_genome_preserves_symbolic_page_and_loads_empirical_page_only_on_demand() {
        portable_page(false);
    }
    #[test]
    fn compact_numeric_genome_preserves_symbolic_page_and_loads_only_on_demand() {
        portable_page(true);
    }
    #[test]
    fn enhanced_numeric_genome_preserves_symbolic_page_and_demand_contracts() {
        portable_format(Format::Enhanced);
    }
    #[test]
    fn factorized_numeric_genome_preserves_pages_refinement_and_demand_contracts() {
        portable_format(Format::Factorized);
    }
    fn portable_page(compact: bool) {
        portable_format(if compact {
            Format::Compact
        } else {
            Format::Legacy
        });
    }
    fn portable_format(format: Format) {
        let root = std::env::temp_dir().join(format!(
            "axon-sequence-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let parent = root.join("parent");
        fs::create_dir(&parent).unwrap();
        let symbolic = StructuralSeed::default();
        packed(&symbolic, &parent).unwrap();
        let examples=(0..16).flat_map(|i|["send","cancel"].map(move |intent|serde_json::json!({"text":format!("{intent} letter to person-{i}"),"intent":intent,"tags":["O","O","O","B:person"]}))).collect::<Vec<_>>();
        let data = root.join("train.json");
        let budget = if matches!(format, Format::Factorized) {
            serde_json::json!({"seed":47,"buckets":256,"dimensions":16,"epochs":32,"learning_rate":0.3,"subwords":false,"minimum_margin":1})
        } else {
            serde_json::json!({"seed":47,"buckets":4096,"epochs":10,"weights":6000,"minimum_margin":1})
        };
        fs::write(
            &data,
            serde_json::to_vec(&serde_json::json!({"budget":budget,"examples":examples})).unwrap(),
        )
        .unwrap();
        let output = root.join("model");
        let training = match format {
            Format::Compact => compact_train,
            Format::Enhanced => enhanced_train,
            Format::Legacy => train,
            Format::Factorized => factorized_train,
        };
        training(
            &parent.join("structural.axnum"),
            &root.join("training-birth"),
            &data,
            &output,
        )
        .unwrap();
        assert_eq!(
            fs::read(output.join("structural.axstruct")).unwrap(),
            symbolic.to_bytes().unwrap()
        );
        let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
        let genome = Genome::open_number(&output.join("structural.axnum"), &auth).unwrap();
        let mut child = genome
            .birth(&root.join("child"), BirthLimits::default(), &auth)
            .unwrap();
        assert_eq!(child.controller.runtime().stats().loads, 0);
        let (module, _, context) = format.properties();
        child
            .controller
            .runtime_mut()
            .with_module(module, context, &auth, |m| {
                let prediction = match (format, m) {
                    (Format::Legacy, Module::Sequence(model)) => {
                        model.predict("send letter to person-0")
                    }
                    (Format::Compact, Module::CompactSequence(model)) => {
                        model.predict("send letter to person-0")
                    }
                    (Format::Enhanced, Module::EnhancedSequence(model)) => {
                        model.predict("send letter to person-0")
                    }
                    (Format::Factorized, Module::FactorizedSequence(model)) => {
                        model.predict("send letter to person-0")
                    }
                    _ => panic!("wrong empirical page"),
                };
                assert!(prediction.is_ok());
                Ok(())
            })
            .unwrap();
        assert_eq!(child.controller.runtime().stats().loads, 1);
        assert_eq!(child.controller.runtime().stats().active_modules, 1);
        child
            .controller
            .runtime_mut()
            .with_module("structural", crate::structural::CONTEXT, &auth, |m| {
                let Module::Structural(seed) = m else {
                    panic!("wrong symbolic page")
                };
                assert_eq!(seed.to_bytes().unwrap(), symbolic.to_bytes().unwrap());
                Ok(())
            })
            .unwrap();
        assert_eq!(child.controller.runtime().stats().loads, 2);
        assert_eq!(child.controller.runtime().stats().active_modules, 1);
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"x","text":"raw","intent":"teacher-answer"})
            )
            .is_err()
        );
        drop(child);
        let before = fs::read(output.join("structural.axnum")).unwrap();
        let refinement = match format {
            Format::Legacy => refine,
            Format::Compact => compact_refine,
            Format::Enhanced => enhanced_refine,
            Format::Factorized => factorized_refine,
        };
        let refined = root.join("refined");
        refinement(
            &output.join("structural.axnum"),
            &root.join("refine-birth"),
            &data,
            &refined,
        )
        .unwrap();
        assert_eq!(before, fs::read(output.join("structural.axnum")).unwrap());
        assert_eq!(
            fs::read(refined.join("structural.axstruct")).unwrap(),
            symbolic.to_bytes().unwrap()
        );
        let model = born(
            &refined.join("structural.axnum"),
            &root.join("refined-born"),
            format,
        )
        .unwrap();
        assert!(model.predict("cancel letter to person-1").is_ok());
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(refined.join("training.json")).unwrap()).unwrap();
        assert_eq!(report["refined"], true);
        let absolute = root.canonicalize().unwrap();
        assert!(absolute.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        fs::remove_dir_all(absolute).unwrap();
    }
}
