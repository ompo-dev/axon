//! A learned, bounded controller over pure inherited arithmetic, not a supplied solver.
use super::*;
pub(super) const MODULE: &str = "math-actor-v1";
#[derive(Clone, Copy)]
pub(super) enum Layout {
    Legacy,
    Focused,
}
impl Layout {
    pub(super) fn module(self) -> &'static str {
        match self {
            Self::Legacy => MODULE,
            Self::Focused => "math-actor-focused-v1",
        }
    }
    pub(super) fn file(self) -> &'static str {
        match self {
            Self::Legacy => "math-actor.axseq",
            Self::Focused => "math-actor-focused.axseq",
        }
    }
}
const STOP: &str = "math-done";
const MAX_ACTIONS: usize = 8;
const DECISION_WORK: usize = 2_000_000;
const REQUEST_WORK: usize = 18_000_000;

pub(super) fn program_id(id: &str) -> bool {
    matches!(id, "math-p032" | "math-p033" | "math-p034" | "math-p035")
}
pub fn pack(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    super::pack_mode(number, birth, data, output, Some(Layout::Legacy))
}
pub fn focused_pack(number: &Path, birth: &Path, data: &Path, output: &Path) -> Result<()> {
    super::pack_mode(number, birth, data, output, Some(Layout::Focused))
}
#[derive(Debug)]
enum Action {
    Stop(f64),
    Apply(String, [f64; 2]),
}
struct State {
    words: Vec<String>,
    registers: Vec<f64>,
}
impl State {
    fn new(text: &str) -> std::result::Result<Self, String> {
        Ok(Self {
            words: crate::structural::sequence::tokens(text).map_err(|e| e.to_string())?,
            registers: Vec::new(),
        })
    }
    #[cfg(test)]
    fn frame(&self) -> std::result::Result<String, String> {
        self.frame_using(Layout::Legacy)
    }
    fn frame_using(&self, layout: Layout) -> std::result::Result<String, String> {
        let mut words = vec!["axon-question".to_owned()];
        words.extend(self.words.iter().cloned());
        words.push(format!("axon-step-{}", self.registers.len()));
        words.push("axon-memory".into());
        for (i, n) in self.registers.iter().enumerate() {
            if !n.is_finite() || n.abs() > 1e9 {
                return Err("actor register bounds".into());
            }
            words.push(format!("r{i}"));
            words.push(if *n == 0. { "0".into() } else { n.to_string() });
        }
        if matches!(layout, Layout::Focused) {
            let step = words.remove(self.words.len() + 1);
            words.insert(0, step);
        }
        let text = words.join(" ");
        crate::structural::sequence::tokens(&text).map_err(|e| e.to_string())?;
        Ok(text)
    }
}
fn path_allowed(
    words: &[String],
    intent: &str,
    previous: &[&str],
    index: usize,
    next: &str,
) -> bool {
    if (intent != STOP && !program_id(intent)) || previous.len() != index || index >= words.len() {
        return false;
    }
    let mut mask = 0u8;
    for role in previous.iter().copied().chain(std::iter::once(next)) {
        let bit = match role {
            "O" => 0,
            "B:a" => 1,
            "B:b" => 2,
            "B:both" => 3,
            _ => return false,
        };
        if bit & mask != 0 {
            return false;
        }
        mask |= bit;
    }
    if intent == STOP && mask & 2 != 0 {
        return false;
    }
    if next != "O" && scalar(&words[index]).is_err() {
        return false;
    }
    let remaining = words[index + 1..]
        .iter()
        .filter(|w| scalar(w).is_ok())
        .count();
    // One remaining scalar can fill both arguments; exact shape is still not meaning.
    if remaining == 0 {
        mask == if intent == STOP { 1 } else { 3 }
    } else {
        true
    }
}
fn action(prediction: &Term, frame: &str) -> std::result::Result<Action, String> {
    let Term::Tree(root) = prediction else {
        return Err("actor interpretation".into());
    };
    let [Term::Symbol(head), Term::Symbol(program), Term::Tree(slots)] = root.as_slice() else {
        return Err("actor interpretation".into());
    };
    if head != "intent"
        || (program != STOP && !program_id(program))
        || !matches!(slots.first(),Some(Term::Symbol(s)) if s=="slots")
        || !(2..=3).contains(&slots.len())
    {
        return Err("actor action shape".into());
    }
    let words = crate::structural::sequence::tokens(frame).map_err(|e| e.to_string())?;
    let mut a = None;
    let mut b = None;
    for slot in &slots[1..] {
        let Term::Tree(pair) = slot else {
            return Err("actor operand".into());
        };
        let [Term::Symbol(role), Term::Symbol(value)] = pair.as_slice() else {
            return Err("actor operand".into());
        };
        if !words
            .iter()
            .any(|w| w.trim_end_matches(['.', '?', '!']) == value)
        {
            return Err("actor operand source".into());
        }
        let n = scalar(value).map_err(str::to_owned)?;
        match role.as_str() {
            "a" if a.is_none() => a = Some(n),
            "b" if b.is_none() => b = Some(n),
            "both" if a.is_none() && b.is_none() => {
                a = Some(n);
                b = Some(n);
            }
            _ => return Err("actor duplicate/unknown operand".into()),
        }
    }
    if program == STOP {
        if let (Some(n), None) = (a, b) {
            return Ok(Action::Stop(n));
        }
    } else if let (Some(a), Some(b)) = (a, b) {
        return Ok(Action::Apply(program.clone(), [a, b]));
    }
    Err("actor missing/extra operand".into())
}
fn measure_frame(
    runtime: &mut DemandRuntime,
    frame: &str,
    auth: &Authority,
    layout: Layout,
) -> Result<crate::structural::sequence::search::Measurement> {
    let words = crate::structural::sequence::tokens(frame)?;
    let filter = |intent: &str, previous: &[&str], index, next: &str| {
        path_allowed(&words, intent, previous, index, next)
    };
    Ok(
        runtime.with_module(layout.module(), CONTEXT, auth, |module| {
            let Module::LexicalEncoder(model) = module else {
                return Err(crate::demand::DemandError::Invalid("actor page kind"));
            };
            Ok(model.measure_search_constrained(
                frame,
                crate::structural::sequence::search::Budget {
                    beam: 8,
                    intents: 5,
                    max_work: DECISION_WORK,
                },
                true,
                &filter,
            ))
        })?,
    )
}
fn solve(
    runtime: &mut DemandRuntime,
    text: &str,
    auth: &Authority,
    total_work: &mut usize,
    layout: Layout,
) -> serde_json::Value {
    let mut trace = Vec::new();
    let mut work = 0usize;
    let mut calculations = 0usize;
    let result = (|| -> std::result::Result<f64, String> {
        let mut state = State::new(text)?;
        for step in 0..=MAX_ACTIONS {
            let frame = state.frame_using(layout)?;
            let measured =
                measure_frame(runtime, &frame, auth, layout).map_err(|e| e.to_string())?;
            work = work
                .checked_add(measured.work)
                .ok_or("actor work overflow")?;
            *total_work = total_work
                .checked_add(measured.work)
                .ok_or("actor batch work overflow")?;
            if work > REQUEST_WORK || *total_work > 100_000_000 {
                return Err("actor aggregate work quota".into());
            }
            let prediction = match measured.result {
                Ok(p) => p.output,
                Err(e) => {
                    trace.push(serde_json::json!({"step":step,"frame":frame,"work":measured.work,"error":e.to_string()}));
                    return Err(e.to_string());
                }
            };
            let selected = action(&prediction, &frame);
            let mut event = serde_json::json!({"step":step,"frame":frame,"prediction":prediction,"work":measured.work});
            match selected {
                Ok(Action::Stop(n)) => {
                    event["answer"] = serde_json::json!(n);
                    trace.push(event);
                    return Ok(n);
                }
                Ok(Action::Apply(id, values)) => {
                    if state.registers.len() >= MAX_ACTIONS {
                        event["error"] = serde_json::json!("actor action quota");
                        trace.push(event);
                        return Err("actor action quota".into());
                    }
                    if !runtime
                        .catalog()
                        .get(&id)
                        .is_some_and(|e| e.kind == ModuleKind::Circuit && e.input == "numbers/2")
                    {
                        return Err("actor circuit unavailable".into());
                    }
                    calculations += 1;
                    match runtime.predict_numeric(
                        &id,
                        crate::numeric::evolution::constructive::CONTEXT,
                        &values,
                        auth,
                    ) {
                        Ok(n) if n.is_finite() && n.abs() <= 1e9 => {
                            event["computed"] = serde_json::json!(n);
                            state.registers.push(n);
                        }
                        Ok(_) => {
                            event["error"] = serde_json::json!("actor result bounds");
                            trace.push(event);
                            return Err("actor result bounds".into());
                        }
                        Err(e) => {
                            event["error"] = serde_json::json!(e.to_string());
                            trace.push(event);
                            return Err(e.to_string());
                        }
                    }
                }
                Err(e) => {
                    event["error"] = serde_json::json!(e);
                    trace.push(event);
                    return Err(e);
                }
            }
            trace.push(event);
        }
        Err("actor did not stop".into())
    })();
    let (answer, error) = match result {
        Ok(n) => (Some(n), None),
        Err(e) => (None, Some(e)),
    };
    serde_json::json!({"answer":answer,"error":error,"language_work":work,"arithmetic_evaluations":calculations,"trace":trace})
}
pub fn batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_using(number, birth, data, receipt, Layout::Legacy)
}
pub fn focused_batch(number: &Path, birth: &Path, data: &Path, receipt: &Path) -> Result<()> {
    batch_using(number, birth, data, receipt, Layout::Focused)
}
pub fn frames_batch(
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
    focused: bool,
) -> Result<()> {
    for file in [number, birth, data, receipt] {
        bounded_path(file)?;
    }
    let inputs: Vec<InterpretationInput> = serde_json::from_slice(&read_bounded(data, 150_000)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=32).contains(&inputs.len())
        || inputs.iter().any(|i| {
            i.id.is_empty()
                || i.id.len() > 96
                || !ids.insert(&i.id)
                || crate::structural::sequence::tokens(&i.text).is_err()
        })
    {
        return Err("actor frame diagnostic quota/identity".into());
    }
    let layout = if focused {
        Layout::Focused
    } else {
        Layout::Legacy
    };
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = born.controller.runtime_mut();
    let mut rows = Vec::new();
    let mut work = 0usize;
    for input in &inputs {
        let measured = measure_frame(runtime, &input.text, &auth, layout)?;
        work = work
            .checked_add(measured.work)
            .ok_or("actor frame work overflow")?;
        if work > 100_000_000 {
            return Err("actor frame aggregate work quota".into());
        }
        let (prediction, error) = match measured.result {
            Ok(p) => (Some(p.output), None),
            Err(e) => (None, Some(e.to_string())),
        };
        rows.push(serde_json::json!({"id":input.id,"frame":input.text,"prediction":prediction,"error":error,"work":measured.work}));
    }
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("actor frame diagnostic mutated genome".into());
    }
    publish_new(receipt, &serde_json::to_vec(&rows)?)?;
    let stats = runtime.stats();
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"seed_unchanged":true,"supplied_states":true,"additional_teacher_queries":0,"arithmetic_evaluations":0,"language_work":work,"loads":stats.loads,"peak_active_modules":stats.peak_active_modules,"peak_active_payload_bytes":stats.peak_active_bytes,"scope":"Diagnostic ONLY: supplied teacher states, same constrained decision as own-state rollout, no arithmetic execution, answers or autonomous task success."})
    );
    Ok(())
}
fn batch_using(
    number: &Path,
    birth: &Path,
    data: &Path,
    receipt: &Path,
    layout: Layout,
) -> Result<()> {
    for file in [number, birth, data, receipt] {
        bounded_path(file)?;
    }
    let inputs: Vec<InterpretationInput> = serde_json::from_slice(&read_bounded(data, 65536)?)?;
    let mut ids = BTreeSet::new();
    if !(1..=4).contains(&inputs.len())
        || inputs
            .iter()
            .any(|i| i.id.is_empty() || i.id.len() > 96 || !ids.insert(&i.id))
    {
        return Err("actor batch quota/identity".into());
    }
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    let original = read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)?;
    let genome = Genome::open_number(number, &auth)?;
    let mut born = genome.birth(birth, BirthLimits::default(), &auth)?;
    let runtime = born.controller.runtime_mut();
    let mut rows = Vec::new();
    let mut work = 0usize;
    for input in &inputs {
        let mut row = solve(runtime, &input.text, &auth, &mut work, layout);
        row["id"] = serde_json::json!(input.id);
        rows.push(row);
    }
    if read_bounded(number, crate::demand::genome::MAX_DIGITS as u64)? != original {
        return Err("actor inference mutated genome".into());
    }
    let bytes = serde_json::to_vec(&rows)?;
    if bytes.len() > 1_000_000 {
        return Err("actor receipt quota".into());
    }
    publish_new(receipt, &bytes)?;
    let stats = runtime.stats();
    println!(
        "{}",
        serde_json::json!({"total":rows.len(),"seed_unchanged":true,"teacher_queries":0,"loads":stats.loads,"peak_active_modules":stats.peak_active_modules,"peak_active_payload_bytes":stats.peak_active_bytes,"language_work":work,"scope":"Own predicted decisions/results only, no equation or future teacher state. Pure arithmetic, provided shape/action quotas, not semantic/units/AGI certificate. Logical scoring work excludes IO, arithmetic and physical callback CPU; active bytes exclude process/eager genome/scratch."})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learned_actor_computes_its_own_intermediate_before_selecting_stop() {
        use crate::numeric::{NumericBudget, NumericExample, NumericSkill, NumericTolerance};
        let root = std::env::temp_dir().join(format!(
            "axon-actor-{}-{}",
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
        super::super::packed(&seed, &parent).unwrap();
        let number = parent.join("structural.axnum");
        let original = fs::read(&number).unwrap();
        let mut examples = Vec::new();
        for i in 0i32..24 {
            let (a, b) = (i + 2, i * 3 + 11);
            let words = vec![
                "combine".into(),
                a.to_string(),
                "with".into(),
                b.to_string(),
            ];
            for done in [false, true] {
                let state = State {
                    words: words.clone(),
                    registers: if done {
                        vec![f64::from(a + b)]
                    } else {
                        Vec::new()
                    },
                };
                let text = state.frame().unwrap();
                let mut tags = vec!["O"; text.split_whitespace().count()];
                if done {
                    *tags.last_mut().unwrap() = "B:a";
                } else {
                    tags[2] = "B:a";
                    tags[4] = "B:b";
                }
                examples.push(serde_json::json!({"text":text,"intent":if done{STOP}else{"math-p032"},"tags":tags}));
            }
        }
        let config = root.join("teaching.json");
        save(&config,&serde_json::json!({"budget":{"seed":47,"vocabulary":256,"characters":0,"dimensions":8,"epochs":32,"learning_rate":0.1,"balanced":false,"minimum_margin":1,"hidden":16,"relative_regions":true},"examples":examples})).unwrap();
        let trained = root.join("trained");
        encoder_lab::train(&number, &root.join("teaching-birth"), &config, &trained).unwrap();
        let rows = [
            (1., 4.),
            (2., 11.),
            (4., 2.),
            (7., 13.),
            (11., 3.),
            (6., 19.),
            (13., 5.),
            (3., 8.),
        ]
        .into_iter()
        .map(|(a, b)| NumericExample::new(vec![a, b], a + b))
        .collect();
        let skill = NumericSkill::learn(
            "primitive-add",
            "actor/test",
            rows,
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )
        .unwrap();
        let circuit = root.join("primitive.axcg");
        publish_new(
            &circuit,
            &CircuitDna::from_program(skill.program())
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
        let manifest = root.join("manifest.json");
        save(&manifest,&serde_json::json!({"encoder":trained.join("semantic-lexical-encoder-v1.axseq"),"circuits":[{"id":"math-p032","file":circuit}]})).unwrap();
        let combined = root.join("combined");
        pack(&number, &root.join("packing-birth"), &manifest, &combined).unwrap();
        let child = combined.join("structural.axnum");
        let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
        let genome = Genome::open_number(&child, &auth).unwrap();
        assert_eq!(genome.residency().active_modules, 2);
        assert_eq!(genome.module_count(), 3);
        assert_eq!(
            fs::read(combined.join("structural.axstruct")).unwrap(),
            seed.to_bytes().unwrap()
        );
        assert_eq!(fs::read(&number).unwrap(), original);
        let input = root.join("inputs.json");
        save(
            &input,
            &serde_json::json!([{"id":"own","text":"combine 2 with 11"}]),
        )
        .unwrap();
        let receipt = root.join("receipt.json");
        batch(&child, &root.join("fresh-birth"), &input, &receipt).unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(result[0]["answer"], 13.0);
        assert_eq!(result[0]["arithmetic_evaluations"], 1);
        assert_eq!(result[0]["trace"][0]["computed"], 13.0);
        assert_eq!(result[0]["trace"].as_array().unwrap().len(), 2);
        let frames = root.join("frames.json");
        save(
            &frames,
            &serde_json::json!([
                {"id":"frame0","text":result[0]["trace"][0]["frame"]},
                {"id":"frame1","text":result[0]["trace"][1]["frame"]}
            ]),
        )
        .unwrap();
        let frame_receipt = root.join("frame-receipt.json");
        frames_batch(
            &child,
            &root.join("frame-birth"),
            &frames,
            &frame_receipt,
            false,
        )
        .unwrap();
        let frame_rows: serde_json::Value =
            serde_json::from_slice(&fs::read(frame_receipt).unwrap()).unwrap();
        for i in 0..2 {
            assert_eq!(
                frame_rows[i]["prediction"],
                result[0]["trace"][i]["prediction"]
            );
            assert_eq!(frame_rows[i]["work"], result[0]["trace"][i]["work"]);
        }
        let bad = root.join("bad.json");
        save(
            &bad,
            &serde_json::json!([{"id":"bad","text":"combine 2 with 11","expected":13}]),
        )
        .unwrap();
        assert!(
            batch(
                &child,
                &root.join("refused-birth"),
                &bad,
                &root.join("unused.json")
            )
            .is_err()
        );
        assert!(!root.join("refused-birth").exists());
        assert!(
            frames_batch(
                &child,
                &root.join("bad-frame-birth"),
                &bad,
                &root.join("bad-frame-receipt.json"),
                false
            )
            .is_err()
        );
        assert!(!root.join("bad-frame-birth").exists());
        let many = root.join("many.json");
        save(
            &many,
            &serde_json::json!(
                (0..5)
                    .map(|i| serde_json::json!({"id":format!("q{i}"),"text":"combine 2 with 11"}))
                    .collect::<Vec<_>>()
            ),
        )
        .unwrap();
        assert!(
            batch(
                &child,
                &root.join("many-birth"),
                &many,
                &root.join("many-results.json")
            )
            .is_err()
        );
        assert!(!root.join("many-birth").exists());
    }
    #[test]
    fn frames_only_include_previous_computations_and_preserve_decimal_values() {
        let mut state = State::new("Has 3 and 5?").unwrap();
        assert_eq!(
            state.frame().unwrap(),
            "axon-question has 3 and 5 axon-step-0 axon-memory"
        );
        state.registers.push(1e-7);
        assert!(state.frame().unwrap().ends_with("r0 0.0000001"));
        assert_eq!(
            state.frame_using(Layout::Focused).unwrap(),
            "axon-step-1 axon-question has 3 and 5 axon-memory r0 0.0000001"
        );
        state.registers.push(-1.25);
        assert!(state.frame().unwrap().ends_with("r0 0.0000001 r1 -1.25"));
        let large = State::new(&vec!["x"; 63].join(" ")).unwrap();
        assert!(large.frame().is_err());
    }
    #[test]
    fn action_operands_are_copied_and_halt_selects_its_value_explicitly() {
        let parse = |s: &str| {
            action(
                &serde_json::from_str(s).unwrap(),
                "axon-question has 3 and 5. axon-step-1 axon-memory r0 8",
            )
        };
        assert!(
            matches!(parse(r#"["intent","math-p034",["slots",["both","8"]]]"#),Ok(Action::Apply(_,v)) if v==[8.,8.])
        );
        assert!(matches!(
            parse(r#"["intent","math-done",["slots",["a","3"]]]"#),
            Ok(Action::Stop(3.))
        ));
        for s in [
            r#"["intent","math-done",["slots",["a","8"],["b","5"]]]"#,
            r#"["intent","math-p032",["slots",["a","99"],["b","5"]]]"#,
            r#"["intent","math-p032",["slots",["both","8"],["a","3"]]]"#,
            r#"["intent","open-file",["slots",["a","3"]]]"#,
        ] {
            assert!(parse(s).is_err());
        }
    }
    #[test]
    fn shape_guard_handles_both_and_never_allows_invalid_or_duplicated_roles() {
        let words = vec!["has".into(), "3".into(), "and".into(), "5".into()];
        assert!(!path_allowed(&words, "math-p032", &[], 0, "B:a"));
        assert!(path_allowed(
            &words,
            "math-p032",
            &["O", "O", "O"],
            3,
            "B:both"
        ));
        assert!(!path_allowed(&words, STOP, &["O", "O", "O"], 3, "B:both"));
        assert!(!path_allowed(
            &words,
            "math-p032",
            &["O", "B:a", "O"],
            3,
            "B:a"
        ));
        assert!(path_allowed(
            &words,
            "math-p032",
            &["O", "B:a", "O"],
            3,
            "B:b"
        ));
        assert!(!path_allowed(&words, "math-p032", &["O", "O", "O"], 3, "O"));
    }
}
