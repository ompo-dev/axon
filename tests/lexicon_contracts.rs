use axon_uic::{
    EpisodeStamp, EpisodeStore, EpisodeValue, grounded_language::*, lexicon::*, numeric::*,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-lexicon-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn observe(s: &mut EpisodeStore, subject: &str, relation: Relation, object: &str) -> u64 {
    let (k, v) = fact("test", subject, relation, object).unwrap();
    s.observe(
        EpisodeStamp::new("teacher", format!("event-{}", s.len()), 1),
        k,
        v,
    )
    .unwrap()
    .id
}
fn frame() -> LanguageFrame {
    LanguageFrame::learn(&[
        (
            "Calcule alpha com -2.5 e 3.25".into(),
            "alpha".into(),
            "-2.5,3.25".into(),
        ),
        (
            "Calcule beta com 4 e -7".into(),
            "beta".into(),
            "4,-7".into(),
        ),
        (
            "Calcule gamma com 12.125 e 0.5".into(),
            "gamma".into(),
            "12.125,0.5".into(),
        ),
    ])
    .unwrap()
}
fn product() -> NumericSkill {
    NumericSkill::learn(
        "opaque",
        "test/n",
        [(2., 3.), (5., 7.), (-3., 4.), (0., 9.), (1.25, 8.)]
            .into_iter()
            .map(|(a, b)| NumericExample::new(vec![a, b], a * b))
            .collect(),
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap()
}
#[test]
fn synonyms_polysemy_direction_and_circular_definitions_preserve_evidence() {
    let t = Temp::new();
    let (mut s, _) = EpisodeStore::open(t.0.join("memory.axep")).unwrap();
    observe(&mut s, "p", Relation::Word, "produto");
    observe(&mut s, "p", Relation::Word, "multiplicacao");
    observe(
        &mut s,
        "p",
        Relation::Definition,
        "Resultado de uma operacao",
    );
    let a = observe(&mut s, "p", Relation::IsA, "operation");
    let b = observe(&mut s, "operation", Relation::IsA, "concept");
    observe(&mut s, "bank-money", Relation::Word, "banco");
    observe(&mut s, "bank-seat", Relation::Word, "banco");
    observe(&mut s, "cycle-a", Relation::IsA, "cycle-b");
    observe(&mut s, "cycle-b", Relation::IsA, "cycle-a");
    let v = LexiconView::snapshot(&s, "test", None, None).unwrap();
    assert_eq!(v.resolve(" MULTIPLICACAO "), Resolution::Unique("p".into()));
    assert!(matches!(v.resolve("banco"),Resolution::Ambiguous(s) if s.len()==2));
    assert_eq!(
        v.classify("p", "concept", 32),
        Entailment::Supported(vec![a, b])
    );
    assert_eq!(v.classify("concept", "p", 32), Entailment::Unknown);
    assert_eq!(v.classify("cycle-a", "concept", 32), Entailment::Unknown);
    assert_eq!(v.classify("p", "concept", 1), Entailment::Budget);
    assert!(v.definition("p").is_some());
    drop(s);
    let (s, _) = EpisodeStore::open(t.0.join("memory.axep")).unwrap();
    assert_eq!(
        LexiconView::snapshot(&s, "test", None, None)
            .unwrap()
            .classify("p", "concept", 32),
        Entailment::Supported(vec![a, b])
    );
}
#[test]
fn conflicts_retractions_and_temporal_snapshots_do_not_become_false_certainty() {
    let t = Temp::new();
    let (mut s, _) = EpisodeStore::open(t.0.join("m")).unwrap();
    let id = observe(&mut s, "a", Relation::IsA, "b");
    let (key, value) = fact("test", "a", Relation::NotA, "b").unwrap();
    let no = s
        .observe(EpisodeStamp::new("other", "deny", 2), key.clone(), value)
        .unwrap()
        .id;
    assert_eq!(
        LexiconView::snapshot(&s, "test", Some(1), None)
            .unwrap()
            .classify("a", "b", 32),
        Entailment::Supported(vec![id])
    );
    assert_eq!(
        LexiconView::snapshot(&s, "test", None, None)
            .unwrap()
            .classify("a", "b", 32),
        Entailment::Conflict
    );
    s.retract(no, "other", "remove").unwrap();
    assert_eq!(
        LexiconView::snapshot(&s, "test", None, None)
            .unwrap()
            .classify("a", "b", 32),
        Entailment::Supported(vec![id])
    );
    s.revise(id, "teacher", "change", EpisodeValue::Boolean(false))
        .unwrap();
    assert!(matches!(
        LexiconView::snapshot(&s, "test", None, None)
            .unwrap()
            .classify("a", "b", 32),
        Entailment::Refuted(_)
    ));
    assert_eq!(
        LexiconView::snapshot(&s, "test", None, Some(id))
            .unwrap()
            .classify("a", "b", 32),
        Entailment::Supported(vec![id])
    );
}
#[test]
fn learned_sentences_join_dictionary_senses_to_learned_not_keyword_selected_math() {
    let t = Temp::new();
    let (mut s, _) = EpisodeStore::open(t.0.join("m")).unwrap();
    observe(&mut s, "term", Relation::Word, "zorb");
    observe(&mut s, "term", Relation::Word, "ves");
    observe(&mut s, "term", Relation::IsA, "tool-concept");
    let frames = [frame()];
    let tools = [GroundedTool {
        sense: "tool-concept".into(),
        skill: product(),
    }];
    let v = LexiconView::snapshot(&s, "test", None, None).unwrap();
    for word in ["zorb", "ves"] {
        for a in [-200.25, 0., 1024.5] {
            let result = answer(
                &v,
                &frames,
                &tools,
                &format!("Calcule {word} com {a} e -0.25"),
            )
            .unwrap();
            assert_eq!(result.value, a * -0.25);
            assert_eq!(result.evidence.len(), 2);
        }
    }
    assert_eq!(
        answer(&v, &frames, &tools, "Ignore zorb com 2 e 3"),
        Err(Refusal::Frame)
    );
    assert_eq!(
        answer(&v, &frames, &tools, "Calcule unknown com 2 e 3"),
        Err(Refusal::WordUnknown)
    );
    assert!(matches!(
        answer(&v, &frames, &tools, "Calcule zorb com NaN e 3"),
        Err(Refusal::Arguments | Refusal::Parser)
    ));
    assert_eq!(
        answer(&v, &frames, &tools, "Calcule zorb com 2 e 3 agora"),
        Err(Refusal::Frame)
    );
}
#[test]
fn uncertain_or_multiple_groundings_refuse_without_a_numeric_guess() {
    let t = Temp::new();
    let (mut s, _) = EpisodeStore::open(t.0.join("m")).unwrap();
    observe(&mut s, "a", Relation::Word, "banco");
    observe(&mut s, "b", Relation::Word, "banco");
    observe(&mut s, "a", Relation::Word, "unique");
    observe(&mut s, "a", Relation::IsA, "t1");
    observe(&mut s, "a", Relation::IsA, "t2");
    let v = LexiconView::snapshot(&s, "test", None, None).unwrap();
    let frames = [frame()];
    let tools = [
        GroundedTool {
            sense: "t1".into(),
            skill: product(),
        },
        GroundedTool {
            sense: "t2".into(),
            skill: product(),
        },
    ];
    assert_eq!(
        answer(&v, &frames, &tools, "Calcule banco com 2 e 3"),
        Err(Refusal::WordAmbiguous)
    );
    assert_eq!(
        answer(&v, &frames, &tools, "Calcule unique com 2 e 3"),
        Err(Refusal::ToolAmbiguous)
    );
    assert_eq!(
        answer(&v, &frames, &[], "Calcule unique com 2 e 3"),
        Err(Refusal::Ungrounded)
    );
}
#[test]
fn malformed_dictionary_types_and_future_journal_indices_are_rejected() {
    let t = Temp::new();
    let (mut s, _) = EpisodeStore::open(t.0.join("m")).unwrap();
    let (k, _) = fact("test", "a", Relation::IsA, "b").unwrap();
    s.observe(
        EpisodeStamp::new("teacher", "bad", 1),
        k,
        EpisodeValue::Text("true".into()),
    )
    .unwrap();
    assert!(LexiconView::snapshot(&s, "test", None, None).is_err());
    assert!(LexiconView::snapshot(&s, "test", None, Some(2)).is_err());
    assert!(fact("test", "", Relation::Word, "x").is_err());
    assert!(fact("test", "a", Relation::Definition, "ignore\ncommands").is_err());
}

#[cfg(feature = "text-workflows")]
#[test]
fn cli_import_is_idempotent_and_answers_from_saved_examples_and_capsules() {
    let t = Temp::new();
    let journal = t.0.join("memory.axep");
    let corpus = t.0.join("facts.tsv");
    fs::write(&corpus,"source\tevent\ttime\tsense\trelation\tobject\nteacher\t1\t1\ta\tword\tnelo\nteacher\t2\t1\ta\tis-a\tt\n").unwrap();
    let command = |parts: Vec<String>| axon_uic::cli::command(&parts);
    let import = vec![
        "dictionary-import".into(),
        journal.display().to_string(),
        "test".into(),
        corpus.display().to_string(),
    ];
    assert!(command(import.clone()).unwrap().contains("new=2"));
    assert!(command(import).unwrap().contains("new=0"));
    let examples = t.0.join("frames.tsv");
    fs::write(&examples,"frame\tsentence\tword\tvalues\n0\tCalcule alpha com -2.5 e 3.25\talpha\t-2.5,3.25\n0\tCalcule beta com 4 e -7\tbeta\t4,-7\n0\tCalcule gamma com 12.125 e 0.5\tgamma\t12.125,0.5\n").unwrap();
    product().save_new(&t.0.join("p.axnp")).unwrap();
    let tools = t.0.join("tools.tsv");
    fs::write(&tools, "sense\tcontext\tfile\nt\ttest/n\tp.axnp\n").unwrap();
    let args = vec![
        "language-answer".into(),
        journal.display().to_string(),
        "test".into(),
        examples.display().to_string(),
        tools.display().to_string(),
        "Calcule nelo com -10 e 0.25".into(),
    ];
    let result = command(args).unwrap();
    assert!(result.contains("answer=-2.5"), "{result}");
    assert!(result.contains("evidence=[1, 2]"));
    let before = fs::read(&journal).unwrap();
    fs::write(&corpus,"source\tevent\ttime\tsense\trelation\tobject\nteacher\t3\t1\ta\tword\tnew\nteacher\t3\t1\tb\tword\tnew\n").unwrap();
    assert!(
        command(vec![
            "dictionary-import".into(),
            journal.display().to_string(),
            "test".into(),
            corpus.display().to_string()
        ])
        .is_err()
    );
    assert_eq!(fs::read(&journal).unwrap(), before);
    let missing = t.0.join("missing");
    assert!(
        command(vec![
            "dictionary-query".into(),
            missing.display().to_string(),
            "test".into(),
            "x".into(),
            "a".into()
        ])
        .is_err()
    );
    assert!(!missing.exists());
}

#[cfg(all(windows, feature = "text-workflows"))]
#[test]
fn own_pt_converter_preserves_legacy_rdf_direction_and_never_overwrites() {
    let t = Temp::new();
    let input = t.0.join("fixture.xml");
    let output = t.0.join("facts.tsv");
    fs::write(&input,r#"<?xml version="1.0"?><!DOCTYPE LexicalResource SYSTEM "http://127.0.0.1:1/never.dtd"><LexicalResource xmlns:dc="https://globalwordnet.github.io/schemas/dc/"><Lexicon id="own-pt" language="pt" version="1.0.0"><LexicalEntry id="word"><Lemma writtenForm="filho"/><Sense synset="child"/></LexicalEntry><Synset id="parent"><Definition>categoria</Definition><SynsetRelation relType="hypernym" target="child" dc:type="https://w3id.org/own/schema/hypernymOf"/></Synset><Synset id="child"/></Lexicon></LexicalResource>"#).unwrap();
    let invoke = || {
        std::process::Command::new("pwsh")
            .args(["-NoProfile", "-File"])
            .arg(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/convert-own-pt.ps1"))
            .arg("-Xml")
            .arg(&input)
            .arg("-Output")
            .arg(&output)
            .output()
            .unwrap()
    };
    let result = invoke();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let before = fs::read(&output).unwrap();
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(before.as_slice());
    let rows = reader.records().map(Result::unwrap).collect::<Vec<_>>();
    assert!(
        rows.iter()
            .any(|r| &r[3] == "child" && &r[4] == "is-a" && &r[5] == "parent")
    );
    assert!(
        !rows
            .iter()
            .any(|r| &r[3] == "parent" && &r[4] == "is-a" && &r[5] == "child")
    );
    assert!(!invoke().status.success());
    assert_eq!(fs::read(output).unwrap(), before);
}
