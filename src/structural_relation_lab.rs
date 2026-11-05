//! Supervised relation experiments reuse structural induction and the demand-loaded genome.
use super::*;
use crate::structural::{agent::utterance, induction};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: induction::Budget,
    examples: Vec<TrainingPair>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingPair {
    premise: String,
    hypothesis: String,
    label: i64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    id: String,
    premise: String,
    hypothesis: String,
}
fn input(premise: &str, hypothesis: &str) -> std::result::Result<Term, crate::structural::Error> {
    let term = Term::Tree(vec![
        Term::Symbol("relation".into()),
        utterance(premise)?,
        utterance(hypothesis)?,
    ]);
    term.validate()?;
    Ok(term)
}
#[derive(Clone, Copy)]
enum Representation {
    Raw,
    Edit,
    Memory,
    Topology,
}
impl Representation {
    fn module(self) -> &'static str {
        match self {
            Self::Raw => "relations",
            Self::Edit => "relations-edit-v1",
            Self::Memory => "relations-memory-v1",
            Self::Topology => "relations-topology-v1",
        }
    }
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, Representation::Raw)
}
pub fn train_aligned(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, Representation::Edit)
}
pub fn train_memory(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, Representation::Memory)
}
pub fn train_topology(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    train_mode(number, birth, data, output, Representation::Topology)
}
fn represented(
    premise: &str,
    hypothesis: &str,
    representation: Representation,
) -> std::result::Result<Term, crate::structural::Error> {
    match representation {
        Representation::Raw => input(premise, hypothesis),
        Representation::Edit => crate::structural::alignment::unique(premise, hypothesis),
        Representation::Memory | Representation::Topology => {
            let contrast = if matches!(representation, Representation::Topology) {
                crate::structural::alignment::topology(premise, hypothesis)?
            } else {
                crate::structural::alignment::unique(premise, hypothesis)?
            };
            let key = serde_json::to_string(&contrast)
                .map_err(|e| crate::structural::Error::Storage(e.to_string()))?;
            let term = Term::Tree(vec![
                Term::Symbol("contrast-key".into()),
                Term::Symbol(key),
                Term::Symbol(premise.to_lowercase()),
                Term::Symbol(hypothesis.to_lowercase()),
            ]);
            term.validate()?;
            Ok(term)
        }
    }
}
fn train_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
    representation: Representation,
) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    if !(4..=10000).contains(&training.examples.len()) {
        return Err("training pair quota".into());
    }
    let mut examples = Vec::new();
    let mut excluded = 0;
    for pair in &training.examples {
        if !(0..=2).contains(&pair.label) {
            return Err("relation training label".into());
        }
        match represented(&pair.premise, &pair.hypothesis, representation) {
            Ok(input) => examples.push(Example {
                input,
                output: Term::Integer(pair.label),
            }),
            Err(_) => excluded += 1,
        }
    }
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment("structural-relations-v1");
    let started = Instant::now();
    let mut emit = |event: &induction::Event| {
        telemetry
            .emit(
                "relation-training",
                &[
                    ("method", V::Text("StructuralSeed::induce")),
                    ("phase", V::Text(event.phase)),
                    ("candidate", V::Count(event.candidate as u64)),
                    ("coverage", V::Count(event.coverage as u64)),
                    ("checked", V::Count(event.checked as u64)),
                ],
            )
            .map_err(|e| crate::structural::Error::Storage(e.to_string()))
    };
    let (relations, statistics) = if matches!(
        representation,
        Representation::Memory | Representation::Topology
    ) {
        induction::learn_closed(&examples, training.budget, &[1], &mut emit)?
    } else {
        induction::learn(&examples, training.budget, &mut emit)?
    };
    if parent.to_bytes()? != before {
        return Err("relation training changed parent".into());
    }
    fs::create_dir(output)?;
    let module = representation.module();
    let genome = packed_modules(&[("structural", &parent), (module, &relations)], output)?;
    let report = serde_json::json!({"schema":"axon-relation-training/v1","examples":examples.len(),"excluded":excluded,
        "budget":training.budget,"statistics":statistics,"parent_unchanged":true,"modules":2,"representation":module,
        "parent_bytes":before.len(),"relation_bytes":relations.to_bytes()?.len(),"encoded_bytes":genome.encoded_bytes(),
        "elapsed_ns":started.elapsed().as_nanos(),"scope":"Supervised structural classifier; not a conversational model or M1 completion."});
    save(&output.join("training.json"), &report)?;
    println!("{report}");
    Ok(())
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Representation::Raw)
}
pub fn batch_aligned(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Representation::Edit)
}
pub fn batch_memory(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Representation::Memory)
}
pub fn batch_topology(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, Representation::Topology)
}
fn batch_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
    representation: Representation,
) -> Result<()> {
    bounded_path(data)?;
    bounded_path(receipt)?;
    let pairs: Vec<Pair> = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    if !(1..=10000).contains(&pairs.len()) {
        return Err("relation batch quota".into());
    }
    let mut ids = BTreeSet::new();
    for pair in &pairs {
        if pair.id.is_empty() || pair.id.len() > 96 || !ids.insert(&pair.id) {
            return Err("relation identity".into());
        }
    }
    let module = representation.module();
    let seed = born_module(number, birth, module)?;
    let before = seed.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment("structural-relations-v1");
    let mut rows = Vec::new();
    let mut answered = 0;
    let started = Instant::now();
    for (index, pair) in pairs.iter().enumerate() {
        let result = represented(&pair.premise, &pair.hypothesis, representation)
            .and_then(|t| seed.transform(&t, 512));
        let prediction = result
            .as_ref()
            .ok()
            .filter(|p| !p.rules.is_empty())
            .and_then(|p| match p.output {
                Term::Integer(n) if (0..=2).contains(&n) => Some(n),
                _ => None,
            });
        answered += usize::from(prediction.is_some());
        let rules = result
            .as_ref()
            .ok()
            .map(|p| p.rules.clone())
            .unwrap_or_default();
        let error = result.as_ref().err().map(ToString::to_string);
        let row =
            serde_json::json!({"id":pair.id,"prediction":prediction,"rules":rules,"error":error});
        let actual = serde_json::to_string(&row)?;
        telemetry.emit(
            "relation-prediction",
            &[
                ("method", V::Text("StructuralSeed::relation")),
                ("case", V::Count(index as u64)),
                ("actual", V::Text(&actual)),
            ],
        )?;
        rows.push(row);
    }
    if seed.to_bytes()? != before {
        return Err("relation inference changed seed".into());
    }
    save(receipt, &rows)?;
    println!(
        "{}",
        serde_json::json!({"total":pairs.len(),"answered":answered,"abstained":pairs.len()-answered,"seed_unchanged":true,"teacher_queries":0,"elapsed_ns":started.elapsed().as_nanos(),"module":module,"scope":"Only requested module loaded; no truth labels in inference input."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inference_contract_rejects_labels_and_malformed_text() {
        assert!(
            serde_json::from_value::<Pair>(
                serde_json::json!({"id":"x","premise":"a b","hypothesis":"c d","label":1})
            )
            .is_err()
        );
        assert!(input("", "anything").is_err());
        assert!(input("a\nb", "anything").is_err());
        assert!(input(&"word ".repeat(100), "anything").is_err());
        assert!(input("Something changes.", "Something changes.").is_ok());
    }
    #[test]
    fn relation_page_preserves_parent_and_birth_loads_only_requested_module() {
        let root = std::env::temp_dir().join(format!(
            "axon-relations-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let parent = StructuralSeed::default();
        let parent_dir = root.join("parent");
        fs::create_dir(&parent_dir).unwrap();
        packed(&parent, &parent_dir).unwrap();
        let examples=(0..12).flat_map(|i|[false,true].map(move |opposite|TrainingPair {
            premise:format!("entity-{i} sees object-{i}"),
            hypothesis:format!("entity-{i} {} object-{i}",if opposite {"avoids"} else {"sees"}),label:i64::from(opposite),
        })).map(|e|serde_json::json!({"premise":e.premise,"hypothesis":e.hypothesis,"label":e.label})).collect::<Vec<_>>();
        let training = root.join("train.json");
        fs::write(&training,serde_json::to_vec(&serde_json::json!({"budget":{"seed":47,"proposals":64,"rules":16,"checks":100000},"examples":examples})).unwrap()).unwrap();
        let output = root.join("trained");
        train(
            &parent_dir.join("structural.axnum"),
            &root.join("parent-birth"),
            &training,
            &output,
        )
        .unwrap();
        assert_eq!(
            fs::read(output.join("structural.axstruct")).unwrap(),
            parent.to_bytes().unwrap()
        );
        let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
        let genome = Genome::open_number(&output.join("structural.axnum"), &auth).unwrap();
        let mut child = genome
            .birth(&root.join("child"), BirthLimits::default(), &auth)
            .unwrap();
        assert_eq!(child.controller.runtime().stats().loads, 0);
        child
            .controller
            .runtime_mut()
            .with_module("relations", crate::structural::CONTEXT, &auth, |module| {
                let Module::Structural(s) = module else {
                    panic!("wrong module");
                };
                assert_eq!(
                    s.transform(&input("new sees other", "new avoids other").unwrap(), 32)
                        .unwrap()
                        .output,
                    Term::Integer(1)
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(child.controller.runtime().stats().loads, 1);
        assert_eq!(child.controller.runtime().stats().active_modules, 1);
        child
            .controller
            .runtime_mut()
            .with_module("structural", crate::structural::CONTEXT, &auth, |module| {
                let Module::Structural(s) = module else {
                    panic!("wrong module");
                };
                assert_eq!(s.to_bytes().unwrap(), parent.to_bytes().unwrap());
                Ok(())
            })
            .unwrap();
        assert_eq!(child.controller.runtime().stats().loads, 2);
        assert_eq!(child.controller.runtime().stats().active_modules, 1);
        drop(child);
        fs::remove_dir_all(root).unwrap();
    }
}
