//! Serial demand-paged syntax -> feature bridge -> empirical relation. No tools.
use super::*;
use crate::structural::sequence::{
    dependency,
    relation_network::{self as network, Budget, Model},
};
const MODULE: &str = "relation-network-v1";
const SYNTAX: &str = "learned-dependency-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    id: String,
    premise: String,
    hypothesis: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use network::Mode;
    #[test]
    fn whole_number_pipeline_preserves_parent_and_uses_one_active_page_without_teacher_fields() {
        let root = std::env::temp_dir().join(format!(
            "axon-relation-net-{}-{}",
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
        let symbolic_bytes = symbolic.to_bytes().unwrap();
        let syntax = dependency::Model::learn(
            &dependency::tests::data(),
            dependency::tests::budget(),
            &mut |_| Ok(()),
        )
        .unwrap();
        let syntax_bytes = syntax.to_bytes().unwrap();
        packed_payloads(
            &[
                ("structural", ModuleKind::Structural, &symbolic_bytes),
                (SYNTAX, ModuleKind::Dependency, &syntax_bytes),
            ],
            &parent,
        )
        .unwrap();
        let b = Budget {
            seed: 47,
            buckets: 256,
            dimensions: 8,
            epochs: 4,
            learning_rate: 0.1,
            mode: Mode::Graph,
            minimum_margin: 0.0,
        };
        let examples=(0..12).map(|i|serde_json::json!({"premise":"alice move bola","hypothesis":(["bruno move caixa","alice move bola","alice move caixa"][i%3]),"label":i%3})).collect::<Vec<_>>();
        let training = root.join("training.json");
        save(
            &training,
            &serde_json::json!({"budget":b,"examples":examples}),
        )
        .unwrap();
        let output = root.join("model");
        train(
            &parent.join("structural.axnum"),
            &root.join("train-birth"),
            &training,
            &output,
        )
        .unwrap();
        assert_eq!(
            fs::read(output.join("structural.axstruct")).unwrap(),
            symbolic_bytes
        );
        assert_eq!(
            fs::read(output.join(format!("{SYNTAX}.axseq"))).unwrap(),
            syntax_bytes
        );
        let inputs = root.join("inputs.json");
        save(&inputs,&serde_json::json!([{"id":"new","premise":"bruno move bola","hypothesis":"alice move caixa"}])).unwrap();
        batch(
            &output.join("structural.axnum"),
            &root.join("new-birth"),
            &inputs,
            &root.join("predictions.json"),
        )
        .unwrap();
        let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
        let g = Genome::open_number(&output.join("structural.axnum"), &auth).unwrap();
        let mut born = g
            .birth(&root.join("check-demand"), BirthLimits::default(), &auth)
            .unwrap();
        let rt = born.controller.runtime_mut();
        for (id, context) in [
            (SYNTAX, dependency::CONTEXT),
            (MODULE, network::CONTEXT),
            (SYNTAX, dependency::CONTEXT),
        ] {
            rt.with_module(id, context, &auth, |_| Ok(())).unwrap();
            assert_eq!(rt.stats().active_modules, 1);
        }
        assert_eq!(rt.stats().peak_active_modules, 1);
        assert_eq!(rt.stats().evictions, 2);
        assert!(
            serde_json::from_value::<Pair>(
                serde_json::json!({"id":"x","premise":"a","hypothesis":"b","label":1})
            )
            .is_err()
        );
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Example {
    premise: String,
    hypothesis: String,
    label: u8,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: Budget,
    examples: Vec<Example>,
}
fn observed_features(
    b: Budget,
    p: &str,
    h: &str,
    syntax: &dependency::Model,
) -> std::result::Result<(Vec<usize>, bool, usize), crate::structural::Error> {
    let a = syntax.predict(p);
    let c = syntax.predict(h);
    let work = a.as_ref().map_or(0, |p| p.work) + c.as_ref().map_or(0, |p| p.work);
    let graphs = a.as_ref().ok().zip(c.as_ref().ok());
    Ok((network::features(b, p, h, graphs)?, graphs.is_some(), work))
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let data: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    if !(4..=10000).contains(&data.examples.len()) {
        return Err("relation training quota".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut child = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = child.controller.runtime_mut();
    let mut parent = None;
    let mut syntax_page = None;
    runtime.with_module("structural", crate::structural::CONTEXT, &auth, |m| {
        let Module::Structural(m) = m else {
            return Err(crate::demand::DemandError::Invalid("structural parent"));
        };
        parent = Some(
            m.to_bytes()
                .map_err(|e| crate::demand::DemandError::Module(e.to_string()))?,
        );
        Ok(())
    })?;
    let mut prepared = vec![];
    let mut syntax_unavailable = 0;
    let started = Instant::now();
    runtime.with_module(SYNTAX, dependency::CONTEXT, &auth, |m| {
        let Module::Dependency(m) = m else {
            return Err(crate::demand::DemandError::Invalid("syntax parent"));
        };
        syntax_page = Some(
            m.to_bytes()
                .map_err(|e| crate::demand::DemandError::Module(e.to_string()))?,
        );
        for e in &data.examples {
            let (f, available, _) = if data.budget.mode.uses_syntax() {
                observed_features(data.budget, &e.premise, &e.hypothesis, m)
            } else {
                network::features(data.budget, &e.premise, &e.hypothesis, None)
                    .map(|f| (f, true, 0))
            }
            .map_err(|e| crate::demand::DemandError::Module(e.to_string()))?;
            syntax_unavailable += usize::from(!available);
            prepared.push((f, e.label));
        }
        Ok(())
    })?;
    runtime.unload_all();
    let preprocessing_ns = started.elapsed().as_nanos();
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let fit = Instant::now();
    let model = Model::learn(&prepared, data.budget, &mut |e| {
        telemetry
            .emit(
                "relation-network-training",
                &[
                    ("epoch", V::Count(e.epoch as u64)),
                    ("examples", V::Count(e.examples as u64)),
                    ("errors", V::Count(e.errors as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    })?;
    let fit_ns = fit.elapsed().as_nanos();
    let page = model.to_bytes()?;
    if original != read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? {
        return Err("relation training changed parent".into());
    }
    fs::create_dir(output)?;
    let genome = packed_payloads(
        &[
            (
                "structural",
                ModuleKind::Structural,
                parent.as_ref().ok_or("missing parent")?,
            ),
            (
                SYNTAX,
                ModuleKind::Dependency,
                syntax_page.as_ref().ok_or("missing syntax page")?,
            ),
            (MODULE, ModuleKind::RelationNetwork, &page),
        ],
        output,
    )?;
    let report = serde_json::json!({"schema":"axon-relation-network-training/v1","budget":data.budget,"examples":data.examples.len(),"page_bytes":page.len(),"model_retained_bytes":model.retained_bytes(),"encoded_bytes":genome.encoded_bytes(),"syntax_used":data.budget.mode.uses_syntax(),"syntax_unavailable_pairs":syntax_unavailable,"preprocessing_ns":preprocessing_ns,"fit_ns":fit_ns,"parent_unchanged":true,"teacher_source":"explicit train labels only; graph modes use predictions of frozen inherited syntax page","scope":"Empirical NLI, not symbolic proof, autonomous architecture evolution, action authority or M1 completion. Training workspace is separate from inference residency."});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(receipt)?;
    let rows: Vec<Pair> = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&rows.len())
        || rows
            .iter()
            .any(|r| r.id.is_empty() || r.id.len() > 96 || !ids.insert(&r.id))
    {
        return Err("relation batch identity/quota".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut child = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = child.controller.runtime_mut();
    let mut budget = None;
    runtime.with_module(MODULE, network::CONTEXT, &auth, |m| {
        let Module::RelationNetwork(m) = m else {
            return Err(crate::demand::DemandError::Invalid("relation page"));
        };
        budget = Some(m.budget());
        Ok(())
    })?;
    let b = budget.ok_or("missing relation budget")?;
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let started = Instant::now();
    let mut results = vec![];
    let mut answered = 0;
    let mut max_bridge_bytes = 0;
    for (index, row) in rows.iter().enumerate() {
        let mut observed = None;
        if b.mode.uses_syntax() {
            runtime.with_module(SYNTAX, dependency::CONTEXT, &auth, |m| {
                let Module::Dependency(m) = m else {
                    return Err(crate::demand::DemandError::Invalid("syntax page"));
                };
                observed = Some(observed_features(b, &row.premise, &row.hypothesis, m));
                Ok(())
            })?;
        } else {
            observed = Some(
                network::features(b, &row.premise, &row.hypothesis, None).map(|f| (f, true, 0)),
            );
        }
        let observation = observed.ok_or("missing features")?;
        let mut prediction = None;
        if let Ok((f, _, _)) = &observation {
            max_bridge_bytes = max_bridge_bytes.max(f.capacity() * size_of::<usize>());
            runtime.with_module(MODULE, network::CONTEXT, &auth, |m| {
                let Module::RelationNetwork(m) = m else {
                    return Err(crate::demand::DemandError::Invalid("relation page"));
                };
                prediction = Some(m.predict(f));
                Ok(())
            })?;
        }
        let value = prediction.as_ref().and_then(|p| p.as_ref().ok());
        answered += usize::from(value.is_some_and(|p| p.label.is_some()));
        let result = serde_json::json!({"id":row.id,"prediction":value.and_then(|p|p.label),"details":value,"syntax_used":b.mode.uses_syntax(),"syntax_available":if b.mode.uses_syntax() {observation.as_ref().ok().map(|(_,a,_)|*a)}else{None},"syntax_work":if b.mode.uses_syntax() {observation.as_ref().ok().map(|(_,_,w)|*w)}else{None},"error":observation.as_ref().err().map(ToString::to_string).or_else(||prediction.as_ref().and_then(|p|p.as_ref().err()).map(ToString::to_string))});
        let encoded = serde_json::to_string(&result)?;
        telemetry.emit(
            "relation-network-prediction",
            &[
                ("case", V::Count(index as u64)),
                ("actual", V::Text(&encoded)),
            ],
        )?;
        results.push(result);
    }
    if original != read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? {
        return Err("relation batch changed seed".into());
    }
    let stats = runtime.stats();
    if stats.peak_active_modules > 1 {
        return Err("relation pipeline retained concurrent brain pages".into());
    }
    save(receipt, &results)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"answered":answered,"seed_unchanged":true,"teacher_queries":0,"elapsed_ns":started.elapsed().as_nanos(),"peak_active_modules":stats.peak_active_modules,"peak_active_bytes":stats.peak_active_bytes,"loads":stats.loads,"evictions":stats.evictions,"bytes_read":stats.bytes_read,"maximum_bridge_vector_bytes":max_bridge_bytes,"scope":"Raw id/premise/hypothesis only. Syntax is dropped before relation page loads. Residency counts pages, not process RSS or all temporary parser allocations."})
    );
    Ok(())
}
