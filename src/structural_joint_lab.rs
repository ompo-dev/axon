//! Label-free inference and explicit training of the separate joint sequence page.
use super::*;
use crate::structural::sequence::{self, joint, search};
const MODULE: &str = "semantic-joint-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: joint::Budget,
    examples: Vec<sequence::Example>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    text: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    format: String,
    budget: search::Budget,
    inputs: Vec<Input>,
}

pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let model = joint::Model::learn(&training.examples, training.budget, &mut |e| {
        telemetry
            .emit(
                "joint-training",
                &[
                    ("epoch", V::Count(e.epoch as u64)),
                    ("intent_errors", V::Count(e.intent_errors as u64)),
                    ("tag_errors", V::Count(e.tag_errors as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    })?;
    let page = model.to_bytes()?;
    fs::create_dir(output)?;
    let genome = packed_payloads(
        &[
            ("structural", ModuleKind::Structural, &before),
            (MODULE, ModuleKind::JointSequence, &page),
        ],
        output,
    )?;
    let report = serde_json::json!({"schema":"axon-joint-training/v1","budget":training.budget,"total":training.examples.len(),"parent_unchanged":parent.to_bytes()?==before,"page_bytes":page.len(),"encoded_bytes":genome.encoded_bytes(),"retained_model_bytes":model.retained_bytes(),"weights":model.weights(),"elapsed_ns":started.elapsed().as_nanos(),"scope":"Joint structured perceptron, bounded approximate beam, not a CRF, exact search, calibrated confidence or tool authority. Training workspace is not inference residency."});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    let inputs: Vec<Input> = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    evaluate(
        number,
        birth,
        receipt,
        &inputs,
        MODULE,
        joint::CONTEXT,
        None,
    )
}
pub fn probe(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    let probe: Probe = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let (id, context, mean_tags) = match probe.format.as_str() {
        "factorized" => (
            "semantic-factorized-v1",
            sequence::factorized::CONTEXT,
            false,
        ),
        "enhanced" => (
            "semantic-enhanced-v1",
            sequence::compact::ENHANCED_CONTEXT,
            false,
        ),
        "joint" => (MODULE, joint::CONTEXT, false),
        "lexical-sum" => (
            "semantic-lexical-encoder-v1",
            sequence::lexical_encoder::CONTEXT,
            false,
        ),
        "lexical-mean" => (
            "semantic-lexical-encoder-v1",
            sequence::lexical_encoder::CONTEXT,
            true,
        ),
        _ => return Err("search format".into()),
    };
    probe.budget.validate()?;
    evaluate(
        number,
        birth,
        receipt,
        &probe.inputs,
        id,
        context,
        Some((probe.budget, mean_tags)),
    )
}
fn evaluate(
    number: &Path,
    birth: &Path,
    receipt: &Path,
    inputs: &[Input],
    id: &str,
    context: &str,
    policy: Option<(search::Budget, bool)>,
) -> Result<()> {
    bounded_path(number)?;
    bounded_path(birth)?;
    bounded_path(receipt)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&inputs.len())
        || inputs
            .iter()
            .any(|i| i.id.is_empty() || i.id.len() > 96 || !ids.insert(&i.id))
    {
        return Err("joint inference identities".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let mut rows = Vec::new();
    let mut bytes = 0usize;
    let mut work = 0usize;
    born.controller.runtime_mut().with_module(id,context,&auth,|module|{
        for input in inputs {
            let measured=match (module,policy) {
                (Module::JointSequence(m),p)=>m.measure(&input.text,p.map(|(p,_)|p)),
                (Module::FactorizedSequence(m),Some((p,_)))=>m.measure_search(&input.text,p),
                (Module::EnhancedSequence(m),Some((p,_)))=>m.measure_search(&input.text,p),
                (Module::LexicalEncoder(m),Some((p,mean)))=>m.measure_search(&input.text,p,mean),
                _=>return Err(crate::demand::DemandError::Invalid("joint inference module")),
            };
            work=work.checked_add(measured.work).ok_or(crate::demand::DemandError::Budget("joint aggregate work"))?;
            if work>100_000_000{return Err(crate::demand::DemandError::Budget("joint aggregate work"));}
            let predicted=measured.result;
            let row=serde_json::json!({"id":input.id,"prediction":predicted.as_ref().ok().map(|p|&p.output),"detail":predicted.as_ref().ok(),"work":measured.work,"error":predicted.as_ref().err().map(ToString::to_string)});
            bytes+=serde_json::to_vec(&row).map_err(|e|crate::demand::DemandError::Module(e.to_string()))?.len();
            if bytes>16_000_000{return Err(crate::demand::DemandError::Budget("joint receipt bytes"));}
            rows.push(row);
        } Ok(())
    })?;
    let unchanged = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? == original;
    if !unchanged {
        return Err("joint inference changed number".into());
    }
    save(receipt, &rows)?;
    let stats = born.controller.runtime().stats();
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"teacher_queries":0,"seed_unchanged":unchanged,"loads":stats.loads,"active_modules":stats.active_modules,"work":work,"scope":"Only id/text at inference; margins compare retained beam paths, not probabilities or global ambiguity certificates. One demand-loaded model, no model clone outside runtime. Work counts model scoring and expansions including rejections, not process/IO work."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn born_joint_model_loads_once_keeps_parent_bytes_and_rejects_teacher_fields() {
        let root = std::env::temp_dir().join(format!(
            "axon-joint-{}-{}",
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
        let bytes = seed.to_bytes().unwrap();
        packed(&seed, &parent).unwrap();
        let examples=(0..8).flat_map(|i| ["a","b"].map(move |label|serde_json::json!({"text":format!("{label} item-{i}"),"intent":label,"tags":["O","B:item"]}))).collect::<Vec<_>>();
        let data = root.join("training.json");
        save(&data,&serde_json::json!({"budget":{"learner":{"seed":47,"buckets":512,"epochs":4,"weights":4000,"intent_weights":2000,"minimum_margin":1},"search":{"beam":2,"intents":2,"max_work":1000000}},"examples":examples})).unwrap();
        let output = root.join("trained");
        train(
            &parent.join("structural.axnum"),
            &root.join("birth-parent"),
            &data,
            &output,
        )
        .unwrap();
        assert_eq!(fs::read(output.join("structural.axstruct")).unwrap(), bytes);
        let input = root.join("inputs.json");
        save(
            &input,
            &serde_json::json!([{"id":"new","text":"a item-new"}]),
        )
        .unwrap();
        let receipt = root.join("receipt.json");
        batch(
            &output.join("structural.axnum"),
            &root.join("child"),
            &input,
            &receipt,
        )
        .unwrap();
        let rows: serde_json::Value = serde_json::from_slice(&fs::read(receipt).unwrap()).unwrap();
        assert_eq!(rows.as_array().unwrap().len(), 1);
        assert!(rows[0]["work"].as_u64().unwrap() > 0);
        assert!(
            serde_json::from_value::<Input>(serde_json::json!({"id":"x","text":"a","label":7}))
                .is_err()
        );
        assert!(serde_json::from_value::<Probe>(serde_json::json!({"format":"joint","budget":{"beam":2,"intents":2,"max_work":100},"inputs":[],"expected":7})).is_err());
    }
}
