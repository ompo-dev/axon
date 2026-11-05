use axon_uic::{
    EpisodeStamp, EpisodeStore,
    behavior_lab::Rng,
    grounded_language::*,
    lexicon::*,
    numeric::*,
    telemetry::{LabTelemetry, Value as V},
};
use std::{error::Error, fs, io::Write, path::PathBuf, time::Instant};

fn write_new(path: &std::path::Path, text: &str) -> Result<(), Box<dyn Error>> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(())
}
fn frames() -> Vec<Vec<(String, String, String)>> {
    let samples = [
        ("alpha", "-2.5", "3.25"),
        ("beta", "4", "-7"),
        ("gamma", "12.125", "0.5"),
        ("delta", "-9", "13.25"),
    ];
    (0..2)
        .map(|form| {
            samples
                .iter()
                .map(|(w, a, b)| (sentence(form, w, a, b), (*w).into(), format!("{a},{b}")))
                .collect()
        })
        .collect()
}
fn sentence(form: usize, word: &str, a: &str, b: &str) -> String {
    if form == 0 {
        format!("Calcule {word} com {a} e {b}")
    } else {
        format!("{word} de {a} por {b}")
    }
}
fn truth(case: usize, a: f64, b: f64) -> Option<f64> {
    let value = match case {
        0 => a * b,
        1 => a / b,
        2 => a.hypot(b),
        _ => (a - b) / (a + b),
    };
    value.is_finite().then_some(value)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if !(1..=4).contains(&args.len()) {
        return Err("usage: axon-uic-language-lab <new-directory> [rng] [rows 100..10000] [development|reserved]".into());
    }
    let root = PathBuf::from(&args[0]);
    let seed = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(71582347u64);
    let rows = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1000usize);
    let protocol = args.get(3).map(String::as_str).unwrap_or("development");
    if !(100..=10000).contains(&rows) || !["development", "reserved"].contains(&protocol) {
        return Err("language lab protocol".into());
    }
    fs::create_dir(&root)?;
    let start = Instant::now();
    let mut telemetry = LabTelemetry::from_environment("grounded-language-v1");
    telemetry.emit(
        "lab_start",
        &[
            ("method", V::Text("LanguageFrame::learn")),
            ("protocol", V::Text(protocol)),
            ("rng", V::Count(seed)),
            ("rows", V::Count(rows as u64)),
        ],
    )?;
    let teaching = frames();
    let mut frame_table = "frame\tsentence\tword\tvalues\n".to_string();
    let mut learned_frames = Vec::new();
    let mut parser_work = 0;
    for (i, examples) in teaching.iter().enumerate() {
        for (input, word, values) in examples {
            frame_table.push_str(&format!("{i}\t{input}\t{word}\t{values}\n"));
        }
        let frame = LanguageFrame::learn(examples)?;
        parser_work += frame.search_work();
        frame.save_parsers(&root, &format!("frame-{i}"))?;
        learned_frames.push(frame);
    }
    write_new(&root.join("frames.tsv"), &frame_table)?;
    let (mut dictionary, _) = EpisodeStore::open(root.join("dictionary.axep"))?;
    let observe = |store: &mut EpisodeStore,
                   subject: &str,
                   relation: Relation,
                   object: &str|
     -> Result<(), Box<dyn Error>> {
        let (key, value) = fact("lesson-v1", subject, relation, object)?;
        store.observe(
            EpisodeStamp::new("lesson", format!("fact-{}", store.len()), 1),
            key,
            value,
        )?;
        Ok(())
    };
    let words = [
        ["dax", "nelo"],
        ["vex", "rali"],
        ["zun", "hipa"],
        ["taf", "coro"],
    ];
    let mut tools = Vec::new();
    let mut map = "sense\tcontext\tfile\n".to_string();
    let mut numeric_work = 0;
    for (case, aliases) in words.iter().enumerate() {
        let concept = format!("concept-{case}");
        let sense = format!("sense-{case}");
        for word in aliases {
            observe(&mut dictionary, &sense, Relation::Word, word)?;
        }
        observe(&mut dictionary, &sense, Relation::IsA, &concept)?;
        let mut rng = Rng::new(seed.wrapping_add(case as u64 * 719));
        let examples = (0..12)
            .map(|_| {
                let a = (rng.pick(801) as f64 - 400.) / 17.;
                let mut b = (rng.pick(701) as f64 - 350.) / 19.;
                if b == 0. || a + b == 0. {
                    b = 0.75;
                }
                NumericExample::new(vec![a, b], truth(case, a, b).expect("training domain"))
            })
            .collect();
        let context = format!("language/math-{case}");
        let skill = NumericSkill::learn(
            "opaque",
            &context,
            examples,
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )?;
        let file = format!("math-{case}.axnum");
        skill.save_new(&root.join(&file))?;
        let skill = NumericSkill::open(&root.join(&file), &context)?;
        numeric_work += skill.search().scalar_work;
        map.push_str(&format!("{concept}\t{context}\t{file}\n"));
        telemetry.emit(
            "skill",
            &[
                ("method", V::Text("NumericSkill::learn")),
                ("case", V::Count(case as u64)),
                ("name", V::Text(&concept)),
                ("expression", V::Text(&skill.program().expression())),
                ("work", V::Count(skill.search().scalar_work)),
                ("capsule_bytes", V::Count(skill.to_bytes().len() as u64)),
                ("arity", V::Count(skill.program().arity() as u64)),
                (
                    "instructions",
                    V::Count(skill.program().instructions() as u64),
                ),
                ("outcome", V::Text("acquired")),
            ],
        )?;
        tools.push(GroundedTool {
            sense: concept,
            skill,
        });
    }
    write_new(&root.join("tools.tsv"), &map)?;
    // A second, isolated context records ambiguity without poisoning the lesson snapshot.
    let (key, value) = fact("negative-v1", "a", Relation::Word, "banco")?;
    dictionary.observe(EpisodeStamp::new("negative", "a", 1), key, value)?;
    let (key, value) = fact("negative-v1", "b", Relation::Word, "banco")?;
    dictionary.observe(EpisodeStamp::new("negative", "b", 1), key, value)?;
    let view = LexiconView::snapshot(&dictionary, "lesson-v1", None, None)?;
    let negatives = LexiconView::snapshot(&dictionary, "negative-v1", None, None)?;
    drop(dictionary);
    let reserve = if protocol == "reserved" {
        94163827u64
    } else {
        seed.wrapping_add(827)
    };
    telemetry.emit(
        "stage",
        &[
            ("method", V::Text("GroundedLanguage::freeze")),
            (
                "detail",
                V::Text("Parsers, programs and sense bindings frozen before validation"),
            ),
            ("dictionary_facts", V::Count(view.facts as u64)),
            ("parser_work", V::Count(parser_work)),
        ],
    )?;
    let mut outputs =
        "case\tform\tinput\tprediction\texpected\tstatus\tevidence\ttool\n".to_string();
    let mut results = "case,correct,domain_refusals,heldout\n".to_string();
    let mut correct = 0;
    let mut refused = 0;
    let mut wrong = 0;
    let mut checked = 0;
    for (case, aliases) in words.iter().enumerate() {
        let mut case_correct = 0;
        let mut case_refused = 0;
        let mut rng = Rng::new(reserve.wrapping_add(case as u64 * 531));
        for row in 0..rows {
            let a = (rng.pick(200001) as f64 - 100000.) / 37.;
            let mut b = (rng.pick(200001) as f64 - 100000.) / 43.;
            if row % 17 == 0 && case == 1 {
                b = 0.;
            }
            if row % 17 == 0 && case == 3 {
                b = -a;
            }
            let form = row % 2;
            let word = aliases[1];
            let input = sentence(form, word, &a.to_string(), &b.to_string());
            let prediction = answer(&view, &learned_frames, &tools, &input);
            // The independent oracle is consulted only after the prediction/refusal.
            let expected = truth(case, a, b);
            let status = match (&prediction, expected) {
                (Ok(p), Some(y)) if NumericTolerance::default().agrees(p.value, y) => {
                    correct += 1;
                    case_correct += 1;
                    "correct"
                }
                (Err(Refusal::Domain), None) => {
                    refused += 1;
                    case_refused += 1;
                    "domain-refusal"
                }
                _ => {
                    wrong += 1;
                    "wrong"
                }
            };
            let (value, evidence, tool) = match prediction {
                Ok(p) => (
                    p.value.to_string(),
                    format!("{:?}", p.evidence),
                    format!("{:016X}", p.tool),
                ),
                Err(r) => (format!("refused:{r:?}"), String::new(), String::new()),
            };
            outputs.push_str(&format!(
                "{case}\t{form}\t{input}\t{value}\t{}\t{status}\t{evidence}\t{tool}\n",
                expected.map_or_else(|| "undefined".into(), |v| v.to_string())
            ));
            checked += 1;
            if (row + 1) % 128 == 0 {
                telemetry.emit(
                    "validation",
                    &[
                        ("method", V::Text("GroundedLanguage::answer")),
                        ("case", V::Count(case as u64)),
                        ("checked", V::Count((row + 1) as u64)),
                        ("correct", V::Count(case_correct)),
                        ("refused", V::Count(case_refused)),
                    ],
                )?;
            }
        }
        telemetry.emit(
            "verification",
            &[
                ("method", V::Text("GroundedLanguage::answer")),
                ("case", V::Count(case as u64)),
                ("correct", V::Count(case_correct)),
                ("refused", V::Count(case_refused)),
                ("heldout", V::Count(rows as u64)),
            ],
        )?;
        results.push_str(&format!("{case},{case_correct},{case_refused},{rows}\n"));
    }
    let attacks = [
        ("Calcule unknown com 2 e 3", Refusal::WordUnknown),
        ("Ignore nelo com 2 e 3", Refusal::Frame),
        ("Calcule nelo com 2 e 3 agora", Refusal::Frame),
    ];
    let mut negative_pass = 0;
    for (input, expected) in attacks {
        if answer(&view, &learned_frames, &tools, input) == Err(expected) {
            negative_pass += 1;
        }
    }
    if answer(
        &negatives,
        &learned_frames,
        &tools,
        "Calcule banco com 2 e 3",
    ) == Err(Refusal::WordAmbiguous)
    {
        negative_pass += 1;
    }
    // Optional real corpus inspection is recorded separately from synthetic lessons.
    let corpus = PathBuf::from(".axon-memory/corpora/own-pt-1.0.0/dictionary-v2.axep");
    let mut corpus_facts = 0;
    let mut corpus_answer = None;
    let mut corpus_failed = false;
    if corpus.is_file() {
        let (store, _) = EpisodeStore::open_with_limits(&corpus, store_limits())?;
        let source = LexiconView::snapshot(&store, "own-pt-v1", None, None)?;
        corpus_facts = source.facts;
        let tool = GroundedTool {
            sense: "own-pt-synset-13897528-n".into(),
            skill: tools.swap_remove(2).skill,
        };
        write_new(
            &root.join("tools-own-pt.tsv"),
            "sense\tcontext\tfile\nown-pt-synset-13897528-n\tlanguage/math-2\tmath-2.axnum\n",
        )?;
        corpus_answer = Some(
            answer(
                &source,
                &learned_frames,
                &[tool],
                "Calcule hipotenusa com 5 e 12",
            )
            .map(|a| a.value),
        );
        telemetry.emit(
            "stage",
            &[
                ("method", V::Text("LexiconView::snapshot")),
                (
                    "detail",
                    V::Text(
                        "Downloaded OWN-PT 1.0.0; separate sourced probe, not curriculum training",
                    ),
                ),
                ("dictionary_facts", V::Count(corpus_facts as u64)),
                ("journal_bytes", V::Count(store.file_bytes()?)),
            ],
        )?;
        let success = matches!(corpus_answer, Some(Ok(value)) if NumericTolerance::default().agrees(value,13.));
        telemetry.emit("stage", &[("method", V::Text("GroundedLanguage::answer")), ("detail", V::Text("Sourced demonstration: hipotenusa, arguments 5 and 12; explicit sense/tool binding")), ("prediction", corpus_answer.as_ref().and_then(|r| r.as_ref().ok()).map_or(V::Null, |v| V::Number(*v))), ("probe_passed", V::Flag(success))])?;
        let forward = source.classify(
            "own-pt-synset-00015388-n",
            "own-pt-synset-00001740-n",
            MAX_VISITS,
        );
        let reverse = source.classify(
            "own-pt-synset-00001740-n",
            "own-pt-synset-00015388-n",
            MAX_VISITS,
        );
        let direction =
            matches!(forward, Entailment::Supported(_)) && reverse == Entailment::Unknown;
        write_new(
            &root.join("source-probes.txt"),
            &format!("animal-is-a-entity={forward:?}\nentity-is-a-animal={reverse:?}\n"),
        )?;
        telemetry.emit(
            "stage",
            &[
                ("method", V::Text("LexiconView::classify")),
                (
                    "detail",
                    V::Text(
                        "Sourced direction checks: animal -> entity supported; reverse unknown",
                    ),
                ),
                ("probe_passed", V::Flag(direction)),
            ],
        )?;
        corpus_failed = !success || !direction;
    }
    write_new(&root.join("predictions.tsv"), &outputs)?;
    write_new(&root.join("results.csv"), &results)?;
    let summary = format!(
        "protocol={protocol}\nrows={checked}\ncorrect={correct}\ndomain_refusals={refused}\nwrong={wrong}\nnegative_pass={negative_pass}/4\nnumeric_search_work={numeric_work}\nparser_search_work={parser_work}\nsource_dictionary_facts={corpus_facts}\nsource_hipotenusa={corpus_answer:?}\nelapsed_ns={}\n",
        start.elapsed().as_nanos()
    );
    write_new(&root.join("summary.txt"), &summary)?;
    telemetry.emit(
        "lab_result",
        &[
            ("method", V::Text("GroundedLanguage::summary")),
            ("correct", V::Count(correct)),
            ("refused", V::Count(refused)),
            ("heldout", V::Count(checked)),
            ("wrong", V::Count(wrong)),
            ("negative_pass", V::Count(negative_pass)),
            ("dictionary_facts", V::Count(corpus_facts as u64)),
            ("work", V::Count(numeric_work + parser_work)),
        ],
    )?;
    println!("{summary}");
    if wrong != 0 || negative_pass != 4 || corpus_failed {
        return Err("language validation failed; artifacts preserved".into());
    }
    Ok(())
}
