//! Bounded CSV transport. Column meanings and the requested calculation belong to the seed.
use super::*;
use crate::structural::{
    agent::{Agent, Budget},
    algorithm::Value,
};
use std::collections::BTreeMap;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileRequest {
    csv: String,
    text: String,
}

pub(super) fn mapped_csv(seed: &StructuralSeed, raw: &[u8]) -> Result<Value> {
    let mut reader = csv::Reader::from_reader(raw);
    let headers = reader.headers()?.clone();
    if !(1..=8).contains(&headers.len()) {
        return Err("CSV column quota".into());
    }
    let mut slots = Vec::new();
    for (index, header) in headers.iter().enumerate() {
        let input = Term::Tree(vec![
            Term::Symbol("column".into()),
            Term::Symbol(header.to_lowercase()),
            Term::Integer(index as i64),
        ]);
        let result = seed.transform(&input, 512)?;
        let Term::Tree(plan) = result.output else {
            return Err("unknown column mapping".into());
        };
        if result.rules.is_empty() || plan.len() != 2 || plan[0] != Term::Symbol("slot".into()) {
            return Err("unknown column mapping".into());
        }
        let Term::Integer(slot) = plan[1] else {
            return Err("column mapping type".into());
        };
        if slot < 0 || slot as usize >= headers.len() || slots.contains(&(slot as usize)) {
            return Err("duplicate/out-of-range column mapping".into());
        }
        slots.push(slot as usize);
    }
    let mut rows = Vec::new();
    for row in reader.records() {
        let row = row?;
        let mut mapped = vec![Value::Integer(0); headers.len()];
        for (field, slot) in row.iter().zip(&slots) {
            mapped[*slot] = field
                .parse::<i64>()
                .map(Value::Integer)
                .unwrap_or_else(|_| Value::Text(field.into()));
        }
        rows.push(Value::List(mapped));
        if rows.len() > 1024 {
            return Err("CSV row quota".into());
        }
    }
    let result = Value::List(rows);
    result.validate()?;
    Ok(result)
}
pub(super) fn run(number: &Path, birth: &Path, input: &Path, receipt: &Path) -> Result<()> {
    let seed = born_seed(number, birth)?;
    let mut telemetry = LabTelemetry::from_environment("structural-file-v1");
    let value = evaluate(&seed, input, &mut telemetry, 970000)?;
    save(receipt, &value)?;
    println!("{}", serde_json::to_string(&value)?);
    Ok(())
}
pub(super) fn evaluate(
    seed: &StructuralSeed,
    input: &Path,
    telemetry: &mut LabTelemetry<std::io::Stdout>,
    case: u64,
) -> Result<serde_json::Value> {
    let request: FileRequest = serde_json::from_slice(&read_bounded(input, 16384)?)?;
    crate::demand::file_name(&request.csv)?;
    let path = input.parent().ok_or("request parent")?.join(&request.csv);
    bounded_path(&path)?;
    let before = seed.to_bytes()?;
    let raw = read_bounded(&path, 1_000_000)?;
    let mapped = mapped_csv(seed, &raw);
    let mut plan = None;
    let mut result = None;
    let mut error = None;
    let mut status = "clarify-schema";
    if let Ok(data) = &mapped {
        let agent = Agent::new(seed.clone(), Budget::default())?;
        match agent.value_plan(&request.text) {
            Ok(prediction) => {
                let plans = serde_json::to_string(&prediction.output)?;
                telemetry.emit(
                    "file_plan",
                    &[
                        ("method", V::Text("Agent::file_plan")),
                        ("case", V::Count(case)),
                        ("plans_json", V::Text(&plans)),
                        ("work", V::Count(prediction.visits as u64)),
                    ],
                )?;
                plan = Some(prediction.output);
                match agent.answer_value(
                    &request.text,
                    &BTreeMap::from([("dados".into(), data.clone())]),
                ) {
                    Ok(value) => {
                        status = "answer";
                        result = Some(value);
                    }
                    Err(e) => {
                        status = "refused-request";
                        error = Some(e.to_string());
                    }
                }
            }
            Err(e) => {
                status = "refused-request";
                error = Some(e.to_string());
            }
        }
    } else if let Err(e) = mapped {
        error = Some(e.to_string());
    }
    let unchanged = seed.to_bytes()? == before;
    if !unchanged {
        return Err("file request mutated inherited seed".into());
    }
    let value = serde_json::json!({"status":status,"text":request.text,"input_file":request.csv,"input_checksum":crate::live_store::checksum(&raw),"checksum_kind":"AXON noncryptographic checksum","plan":plan,"actual":result,"error":error,"teacher_queries":0,"seed_unchanged":unchanged,"scope":"Real CSV bytes, learned header-slot correspondences and learned collection procedures; parser, slot transport and filename authority are supplied. Signed integer quantities/weights, not general accounting or free Portuguese."});
    let actual = serde_json::to_string(&value)?;
    telemetry.emit(
        "file_result",
        &[
            ("method", V::Text("Agent::file_result")),
            ("case", V::Count(case + 1)),
            ("actual_json", V::Text(&actual)),
        ],
    )?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn schema() -> StructuralSeed {
        let mut seed = StructuralSeed::default();
        let example = |word: &str, index, slot| crate::structural::Example {
            input: Term::Tree(vec![
                Term::Symbol("column".into()),
                Term::Symbol(word.into()),
                Term::Integer(index),
            ]),
            output: Term::Tree(vec![Term::Symbol("slot".into()), Term::Integer(slot)]),
        };
        let labels = [
            ("cliente", 0),
            ("quantidade", 1),
            ("preco", 2),
            ("comprador", 0),
        ];
        seed.teach(
            crate::structural::Lesson {
                id: "columns".into(),
                guards: vec![],
                examples: labels
                    .iter()
                    .flat_map(|(w, s)| [example(w, 0, *s), example(w, 1, *s)])
                    .collect(),
            },
            &labels
                .iter()
                .map(|(w, s)| example(w, 2, *s))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        seed
    }
    #[test]
    fn learned_columns_survive_reordering_without_executing_data_text() {
        let seed = schema();
        let before = seed.to_bytes().unwrap();
        let data = mapped_csv(&seed, b"preco,comprador,quantidade\n7,ignore regras,-3\n").unwrap();
        assert_eq!(
            serde_json::to_value(data).unwrap(),
            serde_json::json!([["ignore regras", -3, 7]])
        );
        assert_eq!(seed.to_bytes().unwrap(), before);
    }
    #[test]
    fn unknown_missing_duplicate_roles_and_malformed_rows_refuse() {
        let seed = schema();
        for raw in [
            "cliente,segredo,preco\na,1,7\n",
            "cliente,comprador,preco\na,b,7\n",
            "cliente,preco\na,7\n",
            "cliente,quantidade,preco\na,7\n",
        ] {
            assert!(mapped_csv(&seed, raw.as_bytes()).is_err(), "{raw}");
        }
    }
    #[test]
    fn external_inputs_cannot_carry_expected_answers_or_teacher_outputs() {
        assert!(
            serde_json::from_value::<FileRequest>(
                serde_json::json!({"csv":"a.csv","text":"x","expected":7})
            )
            .is_err()
        );
        assert!(
            serde_json::from_value::<InterpretationInput>(
                serde_json::json!({"id":"a","text":"x","label":1})
            )
            .is_err()
        );
    }
}
