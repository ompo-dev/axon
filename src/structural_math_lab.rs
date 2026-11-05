//! Pure question -> empirical bindings -> inherited numeric program diagnostic.
//! Program predictions never confer tool authority or certify physical units.
use super::*;
use crate::numeric::evolution::constructive::CircuitDna;
use crate::structural::sequence::lexical_encoder::{CONTEXT, Model};
const MODULE: &str = "math-lexical-encoder-v1";
#[path = "structural_math_actor_lab.rs"]
pub mod actor;
#[path = "structural_pointer_lab.rs"]
pub mod pointer;
use crate::structural::sequence::lexical_encoder::relational_pointer as pointer_model;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    encoder: String,
    circuits: Vec<Circuit>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Circuit {
    id: String,
    file: String,
}
fn program_id(id: &str) -> bool {
    id.len() == 9 && id.starts_with("math-p") && id.as_bytes()[6..].iter().all(u8::is_ascii_digit)
}
pub fn pack(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    pack_mode(number, birth, data, output, None)
}
fn pack_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
    actor_layout: Option<actor::Layout>,
) -> Result<()> {
    pack_using(number, birth, data, output, actor_layout, None)
}
pub fn pointer_pack(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    pointer_pack_layout(number, birth, data, output, pointer::Layout::Legacy)
}
fn pointer_pack_layout(
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
    layout: pointer::Layout,
) -> Result<()> {
    pack_using(number, birth, data, output, None, Some(layout))
}
fn pack_using(
    number: &Path,
    birth: &Path,
    data: &Path,
    output: &Path,
    actor_layout: Option<actor::Layout>,
    pointer: Option<pointer::Layout>,
) -> Result<()> {
    let actor_mode = actor_layout.is_some() || pointer.is_some();
    for p in [number, birth, data, output] {
        bounded_path(p)?;
    }
    let manifest: Manifest = serde_json::from_slice(&read_bounded(data, 65536)?)?;
    if manifest.circuits.is_empty() || manifest.circuits.len() > if actor_mode { 4 } else { 32 } {
        return Err("math circuit count".into());
    }
    let encoder_path = Path::new(&manifest.encoder);
    bounded_path(encoder_path)?;
    let encoder = read_bounded(encoder_path, 196608)?;
    if let Some(layout) = pointer {
        if let Some(read) = layout.read() {
            if pointer_model::context_read::ContextModel::from_bytes(&encoder)?.read() != read {
                return Err("pointer encoder read mode".into());
            }
        } else {
            pointer_model::Model::from_bytes(&encoder)?;
        }
    } else {
        Model::from_bytes(&encoder)?;
    }
    let mut pages = Vec::new();
    let mut ids = BTreeSet::new();
    for circuit in &manifest.circuits {
        if !(if actor_mode {
            actor::program_id(&circuit.id)
        } else {
            program_id(&circuit.id)
        }) || !ids.insert(&circuit.id)
        {
            return Err("math circuit identity".into());
        }
        let file = Path::new(&circuit.file);
        bounded_path(file)?;
        let page = read_bounded(file, 65536)?;
        let program = CircuitDna::from_bytes(&page)?.program()?;
        if actor_mode && program.arity() != 2 {
            return Err("actor primitive arity".into());
        }
        pages.push(page);
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let parent = Genome::open_number(number, &auth)?;
    let born = parent.birth(birth, BirthLimits::default(), &auth)?;
    fs::create_dir(output)?;
    let mut catalog = Catalog::new(output);
    for entry in born.controller.runtime().catalog().entries() {
        let page = read_bounded(&birth.join(&entry.file), 262144)?;
        publish_new(&output.join(&entry.file), &page)?;
        catalog.register(
            ModuleSpec {
                id: &entry.id,
                kind: entry.kind,
                file: &entry.file,
                input: &entry.input,
                output: &entry.output,
                cost: entry.cost,
            },
            &auth,
        )?;
    }
    let encoder_file = if let Some(layout) = pointer {
        layout.file()
    } else {
        actor_layout.map_or("math-language.axseq", actor::Layout::file)
    };
    publish_new(&output.join(encoder_file), &encoder)?;
    catalog.register(
        ModuleSpec {
            id: if let Some(layout) = pointer {
                layout.module()
            } else {
                actor_layout.map_or(MODULE, actor::Layout::module)
            },
            kind: if let Some(layout) = pointer {
                layout.kind()
            } else {
                ModuleKind::LexicalEncoder
            },
            file: encoder_file,
            input: if pointer.is_some() {
                "position-candidates"
            } else {
                "text"
            },
            output: if pointer.is_some() {
                "selection"
            } else {
                "interpretation"
            },
            cost: 1,
        },
        &auth,
    )?;
    for (circuit, page) in manifest.circuits.iter().zip(&pages) {
        let file = format!("{}.axcg", circuit.id);
        let input = format!(
            "numbers/{}",
            CircuitDna::from_bytes(page)?.program()?.arity()
        );
        publish_new(&output.join(&file), page)?;
        catalog.register(
            ModuleSpec {
                id: &circuit.id,
                kind: ModuleKind::Circuit,
                file: &file,
                input: &input,
                output: "number",
                cost: 1,
            },
            &auth,
        )?;
    }
    let mut residency = parent.residency();
    if actor_mode {
        residency.active_modules = residency.active_modules.max(2);
    }
    let mut runtime = DemandRuntime::new(catalog, residency)?;
    let child = parent.inherit_runtime(&mut runtime, &[], &auth)?;
    child.save_number_new(&output.join("structural.axnum"), &auth)?;
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("math pack changed parent".into());
    }
    let mut report = serde_json::json!({"parent_unchanged":true,"old_modules_preserved":parent.module_count(),"new_circuits":pages.len(),"modules":child.module_count(),"encoded_bytes":child.encoded_bytes(),"generation":child.generation(),"parent_hex":format!("{:016X}",child.parent()),"scope":"Checked additive hereditary extension; no private episodes or equation labels in the math language page."});
    if actor_mode {
        report["actor_active_module_quota"] = serde_json::json!(residency.active_modules);
    }
    println!("{report}");
    Ok(())
}
fn scalar(token: &str) -> std::result::Result<f64, &'static str> {
    let s = token.strip_prefix('$').unwrap_or(token);
    let s = s.strip_suffix(['.', ',', '?', '!', ';', ':']).unwrap_or(s);
    let digits = s.strip_prefix(['+', '-']).unwrap_or(s);
    let mut pieces = digits.split('.');
    let integer = pieces.next().unwrap_or("");
    let fraction = pieces.next();
    if integer.is_empty()
        || !integer.bytes().all(|b| b.is_ascii_digit())
        || (integer.len() > 1 && integer.starts_with('0'))
        || fraction.is_some_and(|p| p.is_empty() || !p.bytes().all(|b| b.is_ascii_digit()))
        || pieces.next().is_some()
    {
        return Err("decimal operand");
    }
    let n: f64 = s.parse().map_err(|_| "decimal operand")?;
    if !n.is_finite() || n.abs() > 1e9 {
        return Err("operand bounds");
    }
    Ok(if n == 0.0 { 0.0 } else { n })
}
fn bindings(
    prediction: &Term,
    text: &str,
) -> std::result::Result<(String, Vec<f64>), &'static str> {
    let Term::Tree(root) = prediction else {
        return Err("math interpretation");
    };
    let [Term::Symbol(head), Term::Symbol(program), Term::Tree(slots)] = root.as_slice() else {
        return Err("math interpretation");
    };
    if head != "intent"
        || !program_id(program)
        || !matches!(slots.first(),Some(Term::Symbol(s)) if s=="slots")
        || !(2..=5).contains(&slots.len())
    {
        return Err("math slots/program");
    }
    let words = crate::structural::sequence::tokens(text).map_err(|_| "question tokens")?;
    let mut values = [None; 4];
    for slot in &slots[1..] {
        let Term::Tree(pair) = slot else {
            return Err("math operand");
        };
        let [Term::Symbol(role), Term::Symbol(value)] = pair.as_slice() else {
            return Err("math operand");
        };
        if role.len() != 2
            || !role.starts_with('x')
            || !(b'0'..=b'3').contains(&role.as_bytes()[1])
            || !words
                .iter()
                .any(|w| w.trim_end_matches(['.', '?', '!']) == value)
        {
            return Err("math operand source/role");
        }
        let index = usize::from(role.as_bytes()[1] - b'0');
        if values[index].replace(scalar(value)?).is_some() {
            return Err("duplicate math operand");
        }
    }
    let count = slots.len() - 1;
    if values[..count].iter().any(Option::is_none) || values[count..].iter().any(Option::is_some) {
        return Err("missing math operand");
    }
    Ok((
        program.clone(),
        values[..count].iter().map(|v| v.unwrap()).collect(),
    ))
}
fn operand_path(
    words: &[String],
    arity: Option<usize>,
    previous: &[&str],
    index: usize,
    next: &str,
) -> bool {
    let Some(arity) = arity.filter(|n| (1..=4).contains(n)) else {
        return false;
    };
    if previous.len() != index || index >= words.len() {
        return false;
    }
    let mut seen = [false; 4];
    for tag in previous.iter().copied().chain(std::iter::once(next)) {
        if tag == "O" {
            continue;
        }
        let b = tag.as_bytes();
        if b.len() != 4 || &b[..3] != b"B:x" || !(b'0'..=b'3').contains(&b[3]) {
            return false;
        }
        let role = usize::from(b[3] - b'0');
        if role >= arity || seen[role] {
            return false;
        }
        seen[role] = true;
    }
    if next != "O" && scalar(&words[index]).is_err() {
        return false;
    }
    let missing = seen[..arity].iter().filter(|v| !**v).count();
    missing
        <= words[index + 1..]
            .iter()
            .filter(|w| scalar(w).is_ok())
            .count()
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, false)
}
pub fn search_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_mode(number, birth, data, receipt, true)
}
fn batch_mode(
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
    constrained: bool,
) -> Result<()> {
    for p in [number, birth, data, receipt] {
        bounded_path(p)?;
    }
    let inputs: Vec<InterpretationInput> =
        serde_json::from_slice(&read_bounded(data, 16_000_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=10000).contains(&inputs.len())
        || inputs
            .iter()
            .any(|i| i.id.is_empty() || i.id.len() > 96 || !ids.insert(&i.id))
    {
        return Err("math batch identities".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = born.controller.runtime_mut();
    let arities = runtime
        .catalog()
        .entries()
        .iter()
        .filter(|e| program_id(&e.id) && e.kind == ModuleKind::Circuit)
        .filter_map(|e| {
            e.input
                .strip_prefix("numbers/")
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|n| (1..=4).contains(n))
                .map(|n| (e.id.clone(), n))
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut rows = Vec::new();
    let mut prepared = Vec::new();
    let mut work = 0usize;
    runtime.with_module(MODULE,CONTEXT,&auth,|module|{
        let Module::LexicalEncoder(model)=module else { return Err(crate::demand::DemandError::Invalid("math encoder kind")); };
        for input in &inputs {
            let (predicted,measured_work)=if constrained {
                let words=crate::structural::sequence::tokens(&input.text).unwrap_or_default();
                let filter=|intent:&str,previous:&[&str],index,next:&str|operand_path(&words,arities.get(intent).copied(),previous,index,next);
                let measured=model.measure_search_constrained(&input.text,crate::structural::sequence::search::Budget{beam:16,intents:8,max_work:5_000_000},true,&filter);
                (measured.result.map(|p|p.output),measured.work)
            }else{ let measured=model.measure(&input.text);(measured.result.map(|p|p.output),measured.work) };
            work=work.checked_add(measured_work).ok_or(crate::demand::DemandError::Budget("math work overflow"))?;
            if work>100_000_000 { return Err(crate::demand::DemandError::Budget("math aggregate work")); }
            let parsed=predicted.map_err(|e|e.to_string()).and_then(|p|bindings(&p,&input.text).map_err(str::to_owned));
            let (program,values,error)=match parsed { Ok((p,v))=>(Some(p),Some(v),None),Err(e)=>(None,None,Some(e)) };
            prepared.push(program.clone().zip(values.clone()));
            rows.push(serde_json::json!({"id":input.id,"program":program,"values":values,"answer":null,"error":error,"language_work":measured_work}));
        }Ok(())
    })?;
    // Group only the empirical requests, never labels; one active module remains enough.
    let mut grouped = std::collections::BTreeMap::<String, Vec<(usize, Vec<f64>)>>::new();
    for (index, item) in prepared.into_iter().enumerate() {
        if let Some((p, v)) = item {
            grouped.entry(p).or_default().push((index, v));
        }
    }
    for (program, requests) in grouped {
        let kind_ok = runtime
            .catalog()
            .get(&program)
            .is_some_and(|e| e.kind == ModuleKind::Circuit);
        for (index, values) in requests {
            let result = if kind_ok {
                runtime
                    .predict_numeric(
                        &program,
                        crate::numeric::evolution::constructive::CONTEXT,
                        &values,
                        &auth,
                    )
                    .map_err(|e| e.to_string())
            } else {
                Err("math circuit unavailable".into())
            };
            match result {
                Ok(n) => rows[index]["answer"] = serde_json::json!(n),
                Err(e) => rows[index]["error"] = serde_json::json!(e),
            }
        }
    }
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("math inference changed seed".into());
    }
    let bytes = serde_json::to_vec(&rows)?;
    if bytes.len() > 16_000_000 {
        return Err("math receipt quota".into());
    }
    publish_new(receipt, &bytes)?;
    let stats = runtime.stats();
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"seed_unchanged":true,"teacher_queries":0,"loads":stats.loads,"active_modules":stats.active_modules,"ending_active_payload_bytes":stats.active_bytes,"peak_active_payload_bytes":stats.peak_active_bytes,"peak_active_modules":stats.peak_active_modules,"language_work":work,"constrained":constrained,"scope":"Empirical program/operand predictions, pure arithmetic, no expected answers, semantic certificate, physical units or action authority. Work counts model scoring and constraint consultations, not constraint CPU instructions, IO or compiled arithmetic; active bytes exclude eager genome/process/scratch."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inherited_math_genome_retains_old_pages_and_executes_empirical_bindings() {
        use crate::numeric::{NumericBudget, NumericExample, NumericSkill, NumericTolerance};
        let root = std::env::temp_dir().join(format!(
            "axon-math-{}-{}",
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
        let number = parent.join("structural.axnum");
        let original = fs::read(&number).unwrap();
        let config = root.join("training.json");
        let examples=(0..8).flat_map(|i|["add","take"].map(move|verb|serde_json::json!({"text":format!("{verb} {} {}",i+2,i+11),"intent":if verb=="add" {"math-p000"}else{"math-p001"},"tags":["O","B:x0","B:x1"]}))).collect::<Vec<_>>();
        save(&config,&serde_json::json!({"budget":{"seed":47,"vocabulary":64,"characters":256,"dimensions":8,"epochs":32,"learning_rate":0.1,"balanced":false,"minimum_margin":1,"hidden":16,"relative_regions":true},"examples":examples})).unwrap();
        let trained = root.join("trained");
        encoder_lab::train(&number, &root.join("training-birth"), &config, &trained).unwrap();
        let mut circuits = Vec::new();
        for (index, add) in [true, false].into_iter().enumerate() {
            let rows = [
                (1, 4),
                (2, 11),
                (4, 2),
                (7, 13),
                (11, 3),
                (6, 19),
                (13, 5),
                (3, 8),
            ]
            .into_iter()
            .map(|(a, b)| {
                let a = f64::from(a);
                let b = f64::from(b);
                NumericExample::new(vec![a, b], if add { a + b } else { a - b })
            })
            .collect::<Vec<_>>();
            let skill = NumericSkill::learn(
                "operation",
                "math/test",
                rows,
                &[],
                NumericBudget::default(),
                NumericTolerance::default(),
            )
            .unwrap();
            let file = root.join(format!("source-{index}.axcg"));
            publish_new(
                &file,
                &CircuitDna::from_program(skill.program())
                    .unwrap()
                    .to_bytes(),
            )
            .unwrap();
            circuits.push(serde_json::json!({"id":format!("math-p{index:03}"),"file":file}));
        }
        let manifest = root.join("manifest.json");
        save(&manifest,&serde_json::json!({"encoder":trained.join("semantic-lexical-encoder-v1.axseq"),"circuits":circuits})).unwrap();
        let out = root.join("combined");
        pack(&number, &root.join("pack-birth"), &manifest, &out).unwrap();
        assert_eq!(
            fs::read(out.join("structural.axstruct")).unwrap(),
            seed.to_bytes().unwrap()
        );
        assert_eq!(fs::read(&number).unwrap(), original);
        let input = root.join("inputs.json");
        save(&input,&serde_json::json!([{"id":"add","text":"add 2 11"},{"id":"take","text":"take 2 11"},{"id":"oversize","text":vec!["a";65].join(" ")}])).unwrap();
        let receipt = root.join("results.json");
        batch(
            &out.join("structural.axnum"),
            &root.join("inference-birth"),
            &input,
            &receipt,
        )
        .unwrap();
        let results: serde_json::Value =
            serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(results[0]["answer"], 13.0);
        assert_eq!(results[1]["answer"], -9.0);
        assert!(results[2]["answer"].is_null());
        assert!(results[2]["error"].is_string());
        let searched = root.join("searched.json");
        search_batch(
            &out.join("structural.axnum"),
            &root.join("search-birth"),
            &input,
            &searched,
        )
        .unwrap();
        let searched: serde_json::Value =
            serde_json::from_slice(&fs::read(searched).unwrap()).unwrap();
        assert_eq!(searched[0]["answer"], 13.0);
        assert_eq!(searched[1]["answer"], -9.0);
        assert!(searched[2]["answer"].is_null());
        assert!(
            batch(
                &out.join("structural.axnum"),
                &root.join("overwrite-birth"),
                &input,
                &receipt
            )
            .is_err()
        );
        let poisoned = root.join("poisoned.json");
        save(
            &poisoned,
            &serde_json::json!([{"id":"leak","text":"add 2 11","answer":13}]),
        )
        .unwrap();
        assert!(
            batch(
                &out.join("structural.axnum"),
                &root.join("poisoned-birth"),
                &poisoned,
                &root.join("unused.json")
            )
            .is_err()
        );
        assert!(!root.join("poisoned-birth").exists());
        assert!(pack(&number, &root.join("new-pack-birth"), &manifest, &out).is_err());
    }
    #[test]
    fn selected_operands_must_be_contiguous_unique_and_copied_from_question() {
        let parse = |json: &str, text: &str| bindings(&serde_json::from_str(json).unwrap(), text);
        assert_eq!(
            parse(
                r#"["intent","math-p007",["slots",["x1","5"],["x0","$12,"]]]"#,
                "Needs $12, has 5"
            )
            .unwrap(),
            ("math-p007".into(), vec![12., 5.])
        );
        assert!(
            parse(
                r#"["intent","math-p001",["slots",["x0","5"]]]"#,
                "Has 5. Continues here"
            )
            .is_ok()
        );
        for json in [
            r#"["intent","../x",["slots",["x0","5"]]]"#,
            r#"["intent","math-p001",["slots",["x0","99"]]]"#,
            r#"["intent","math-p001",["slots",["x1","5"]]]"#,
            r#"["intent","math-p001",["slots",["x0","5"],["x0","5"]]]"#,
            r#"["intent","math-p001",["slots",["x4","5"]]]"#,
            r#"["intent","math-p001",["slots",["x0","5 12"]]]"#,
        ] {
            assert!(parse(json, "5 12").is_err(), "{json}");
        }
    }
    #[test]
    fn decimal_contract_refuses_implicit_unit_conversion_and_nonfinite_values() {
        for (s, n) in [("$5.20,", 5.2), ("-3", -3.), ("+0.3?", 0.3), ("12.", 12.)] {
            assert_eq!(scalar(s).unwrap(), n);
        }
        for s in [
            "12%",
            "1,200",
            "2e3",
            "01",
            "NaN",
            "$",
            "3/2",
            "1..",
            "1_000",
            "1000000001",
            "",
        ] {
            assert!(scalar(s).is_err(), "{s}");
        }
    }
    #[test]
    fn operand_path_enforces_exact_cardinality_without_using_answers() {
        let words = vec!["has".into(), "3".into(), "and".into(), "5".into()];
        assert!(!operand_path(&words, Some(2), &[], 0, "B:x0"));
        assert!(operand_path(&words, Some(2), &["O"], 1, "B:x1"));
        assert!(!operand_path(
            &words,
            Some(2),
            &["O", "B:x1", "O"],
            3,
            "B:x1"
        ));
        assert!(!operand_path(&words, Some(2), &["O", "B:x1", "O"], 3, "O"));
        assert!(operand_path(
            &words,
            Some(2),
            &["O", "B:x1", "O"],
            3,
            "B:x0"
        ));
        assert!(!operand_path(&words, Some(1), &["O"], 1, "I:x0"));
        assert!(!operand_path(&words, None, &[], 0, "O"));
    }
}
