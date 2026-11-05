//! Structured language learning; inference accepts text, never annotations or family labels.
use super::*;
use crate::structural::{Error as StructuralError, induction};

const MODULE: &str = "semantic-grammar-v1";
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    budget: induction::Budget,
    families: Vec<Vec<TeachingRow>>,
    #[serde(default)]
    lexical_domains: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TeachingRow {
    text: String,
    target: Term,
}
fn input(text: &str) -> std::result::Result<Term, StructuralError> {
    if text.trim().is_empty() || text.len() > 4096 || text.chars().any(char::is_control) {
        return Err(StructuralError::Invalid("grammar text bounds"));
    }
    let mut words = vec![Term::Symbol("utterance".into())];
    words.extend(
        text.to_lowercase()
            .trim()
            .trim_end_matches(['.', '?', '!'])
            .split_whitespace()
            .map(|s| Term::Symbol(s.into())),
    );
    if words.len() == 1 {
        return Err(StructuralError::Invalid("empty grammar text"));
    }
    let term = Term::Tree(words);
    term.validate()?;
    Ok(term)
}
pub fn train(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    bounded_path(data)?;
    bounded_path(output)?;
    let training: Training = serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let total = training.families.iter().map(Vec::len).sum::<usize>();
    if training.families.is_empty()
        || training.families.len() > 10000
        || !(4..=10000).contains(&total)
        || training.families.iter().any(Vec::is_empty)
    {
        return Err("grammar training quota".into());
    }
    let mut excluded = 0;
    let mut families = Vec::new();
    for family in training.families {
        let mut examples = Vec::new();
        for row in family {
            row.target.validate()?;
            match input(&row.text) {
                Ok(input) => examples.push(Example {
                    input,
                    output: row.target,
                }),
                Err(_) => excluded += 1,
            }
        }
        if !examples.is_empty() {
            families.push(examples);
        }
    }
    let parent = born_seed(number, birth)?;
    let before = parent.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let started = Instant::now();
    let mut emit = |event: &induction::Event| {
        telemetry
            .emit(
                "grammar-training",
                &[
                    ("phase", V::Text(event.phase)),
                    ("candidate", V::Count(event.candidate as u64)),
                    ("coverage", V::Count(event.coverage as u64)),
                    ("checked", V::Count(event.checked as u64)),
                ],
            )
            .map_err(|e| StructuralError::Storage(e.to_string()))
    };
    let (grammar, statistics) = if training.lexical_domains {
        induction::learn_grouped_domains(&families, training.budget, &mut emit)?
    } else {
        induction::learn_grouped(&families, training.budget, &mut emit)?
    };
    if parent.to_bytes()? != before {
        return Err("grammar training changed parent".into());
    }
    fs::create_dir(output)?;
    let genome = packed_modules(&[("structural", &parent), (MODULE, &grammar)], output)?;
    let report = serde_json::json!({"schema":"axon-grammar-training/v1", "total":total,"excluded":excluded,
        "budget":training.budget,"lexical_domains":training.lexical_domains,"statistics":statistics,"parent_unchanged":true,"modules":2,
        "parent_bytes":before.len(),"grammar_bytes":grammar.to_bytes()?.len(),"encoded_bytes":genome.encoded_bytes(),
        "elapsed_ns":started.elapsed().as_nanos(),"module":MODULE,
        "scope":"Supervised tree transduction; not general language understanding or M1 approval."});
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
        return Err("grammar batch identity or quota".into());
    }
    let grammar = born_module(number, birth, MODULE)?;
    let before = grammar.to_bytes()?;
    let mut telemetry = LabTelemetry::from_environment(MODULE);
    let mut predictions = Vec::new();
    let started = Instant::now();
    let mut answered = 0;
    for (index, row) in rows.iter().enumerate() {
        let result = input(&row.text).and_then(|t| grammar.transform(&t, 512));
        let prediction = result.as_ref().ok().filter(|p| !p.rules.is_empty());
        answered += usize::from(prediction.is_some());
        let actual = serde_json::json!({"id":row.id,"prediction":prediction.map(|p| &p.output),
            "rules":prediction.map(|p| &p.rules),"error":result.as_ref().err().map(ToString::to_string)});
        let encoded = serde_json::to_string(&actual)?;
        telemetry.emit(
            "grammar-prediction",
            &[
                ("case", V::Count(index as u64)),
                ("actual", V::Text(&encoded)),
            ],
        )?;
        predictions.push(actual);
    }
    if grammar.to_bytes()? != before {
        return Err("grammar inference changed seed".into());
    }
    save(receipt, &predictions)?;
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"answered":answered,"abstained":rows.len()-answered,
        "seed_unchanged":true,"teacher_queries":0,"elapsed_ns":started.elapsed().as_nanos(),"module":MODULE})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_text_contract_preserves_digits_and_rejects_answers() {
        assert_eq!(
            input(" Send 003 to A. ").unwrap(),
            serde_json::from_value::<Term>(serde_json::json!([
                "utterance",
                "send",
                "003",
                "to",
                "a"
            ]))
            .unwrap()
        );
        for text in ["", " ", "!", "a\nb"] {
            assert!(input(text).is_err());
        }
        assert!(input(&"x ".repeat(40)).is_err());
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"a","text":"raw","target":["answer"]})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"a","text":"raw","family":"label"})
            )
            .is_err()
        );
    }
}
