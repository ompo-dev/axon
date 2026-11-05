//! Experimental syntax page; no inferred relation can authorize effects.
use super::*;
use crate::structural::sequence::dependency::{self, Model};
const MODULE: &str = "learned-dependency-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: dependency::Budget,
    examples: Vec<dependency::Example>,
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let data: Training = serde_json::from_slice(&read_bounded(data, 32_000_000)?)?;
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let started = Instant::now();
    let model = Model::learn(&data.examples, data.budget, &mut |event| {
        telemetry
            .emit(
                "dependency-training",
                &[
                    ("epoch", V::Count(event.epoch as u64)),
                    ("updates", V::Count(event.updates as u64)),
                    ("errors", V::Count(event.errors as u64)),
                    ("nonprojective", V::Count(event.nonprojective as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    })?;
    if parent.to_bytes()? != before {
        return Err("dependency changed parent".into());
    }
    let page = model.to_bytes()?;
    let decoded = Model::from_bytes(&page)?;
    let resident = Module::Dependency(Box::new(decoded)).retained_bytes();
    if resident > 262144 {
        return Err("seed budget exhausted: dependency module residency".into());
    }
    fs::create_dir(output)?;
    let genome = packed_payloads(
        &[
            ("structural", ModuleKind::Structural, &before),
            (MODULE, ModuleKind::Dependency, &page),
        ],
        output,
    )?;
    let report = serde_json::json!({"schema":"axon-dependency-training/v1","total":data.examples.len(),"budget":data.budget,
        "page_bytes":page.len(),"retained_model_bytes_upper_estimate":model.retained_bytes(),"born_module_bytes_upper_estimate":resident,"encoded_bytes":genome.encoded_bytes(),
        "parent_unchanged":true,"elapsed_ns":started.elapsed().as_nanos(),"module":MODULE,
        "scope":"Supervised segmentation, POS and projective syntax. Nonprojective trees teach segmentation/POS only. Not semantics, full M1 or tool permission; inherited operational page is unchanged."});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(receipt)?;
    let rows: Vec<InterpretationInput> = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&rows.len())
        || rows
            .iter()
            .any(|r| r.id.is_empty() || r.id.len() > 96 || !ids.insert(&r.id))
    {
        return Err("dependency batch identity/quota".into());
    }
    let model = born_page(number, birth, MODULE, dependency::CONTEXT, |m| match m {
        Module::Dependency(m) => Some(m.as_ref().clone()),
        _ => None,
    })?;
    let before = model.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let started = Instant::now();
    let mut results = vec![];
    let mut answered = 0;
    for (index, row) in rows.iter().enumerate() {
        let result = model.predict(&row.text);
        answered += usize::from(result.is_ok());
        let value = serde_json::json!({"id":row.id,"prediction":result.as_ref().ok(),"error":result.as_ref().err().map(ToString::to_string)});
        let actual = serde_json::to_string(&value)?;
        telemetry.emit(
            "dependency-prediction",
            &[
                ("case", V::Count(index as u64)),
                ("actual", V::Text(&actual)),
            ],
        )?;
        results.push(value);
    }
    if before != model.to_bytes()? {
        return Err("dependency prediction mutated seed".into());
    }
    save(receipt, &results)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"answered":answered,"seed_unchanged":true,"teacher_queries":0,"elapsed_ns":started.elapsed().as_nanos(),"module":MODULE,"scope":"Only id/text reach inference; no gold segmentation, POS, heads, labels or semantic authority."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numeric_child_preserves_operational_page_and_loads_syntax_only_on_demand() {
        let root = std::env::temp_dir().join(format!(
            "axon-dependency-{}-{}",
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
        packed(&seed, &parent).unwrap();
        let model = Model::learn(
            &dependency::tests::data(),
            dependency::tests::budget(),
            &mut |_| Ok(()),
        )
        .unwrap();
        let page = model.to_bytes().unwrap();
        let out = root.join("child");
        fs::create_dir(&out).unwrap();
        let before = seed.to_bytes().unwrap();
        packed_payloads(
            &[
                ("structural", ModuleKind::Structural, &before),
                (MODULE, ModuleKind::Dependency, &page),
            ],
            &out,
        )
        .unwrap();
        let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
        let genome = Genome::open_number(&out.join("structural.axnum"), &auth).unwrap();
        let mut birth = genome
            .birth(&root.join("birth"), BirthLimits::default(), &auth)
            .unwrap();
        assert_eq!(birth.controller.runtime().stats().loads, 0);
        birth
            .controller
            .runtime_mut()
            .with_module(MODULE, dependency::CONTEXT, &auth, |m| {
                let Module::Dependency(m) = m else {
                    panic!("wrong page")
                };
                assert_eq!(m.to_bytes().unwrap(), page);
                assert!(m.predict("alice move bola").is_ok());
                Ok(())
            })
            .unwrap();
        assert_eq!(birth.controller.runtime().stats().loads, 1);
        birth
            .controller
            .runtime_mut()
            .with_module("structural", crate::structural::CONTEXT, &auth, |m| {
                let Module::Structural(m) = m else {
                    panic!("wrong operational page")
                };
                assert_eq!(m.to_bytes().unwrap(), before);
                Ok(())
            })
            .unwrap();
        assert_eq!(birth.controller.runtime().stats().loads, 2);
        assert_eq!(birth.controller.runtime().stats().active_modules, 1);
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"x","text":"raw","words":[]})
            )
            .is_err()
        );
    }
}
