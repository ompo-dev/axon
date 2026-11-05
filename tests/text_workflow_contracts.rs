#![cfg(feature = "text-workflows")]

use axon_uic::{TextBudget, TextExample, TextPrediction, TextSkill, TextSkillError, cli};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

struct Temp(PathBuf);
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "axon-text-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn skill() -> TextSkill {
    TextSkill::learn(
        "contacts",
        "test/contacts",
        vec![
            TextExample::new("Ana Maria|Silva Costa", "Silva Costa, Ana Maria"),
            TextExample::new("Bruno|Lima Rocha", "Lima Rocha, Bruno"),
        ],
        TextBudget::default(),
    )
    .unwrap()
}
fn command(args: &[&str]) -> Result<String, cli::CliError> {
    cli::command(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}
fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn token(preview: &str) -> &str {
    preview
        .lines()
        .find_map(|line| line.strip_prefix("review_token="))
        .unwrap()
}
fn table(path: &Path, headers: &[&str], rows: &[&[&str]]) {
    let mut writer = csv::Writer::from_path(path).unwrap();
    writer.write_record(headers).unwrap();
    for row in rows {
        writer.write_record(*row).unwrap();
    }
    writer.flush().unwrap();
}

#[test]
fn persisted_program_is_reloaded_and_reused_on_new_values_without_search() {
    let temp = Temp::new();
    let path = temp.path("contacts.axtext");
    let original = skill();
    original.save_new(&path).unwrap();
    let loaded = TextSkill::load(&path).unwrap();
    assert_eq!(original, loaded);
    assert!(
        matches!(loaded.predict("test/contacts", "Carla Sofia|Pereira Alves").unwrap(), TextPrediction::Agreed(w) if w.output == "Pereira Alves, Carla Sofia")
    );
    assert!(matches!(
        loaded.predict("other", "anything"),
        Err(TextSkillError::ContextMismatch)
    ));
}

#[test]
fn every_corruption_and_truncation_is_refused_without_rewriting() {
    let temp = Temp::new();
    let original = temp.path("original.axtext");
    skill().save_new(&original).unwrap();
    let bytes = fs::read(original).unwrap();
    let probe = temp.path("probe.axtext");
    for index in 0..bytes.len() {
        let mut corrupt = bytes.clone();
        corrupt[index] ^= 1;
        fs::write(&probe, &corrupt).unwrap();
        assert!(TextSkill::load(&probe).is_err(), "corruption {index}");
        assert_eq!(fs::read(&probe).unwrap(), corrupt);
        fs::write(&probe, &bytes[..index]).unwrap();
        assert!(TextSkill::load(&probe).is_err(), "truncation {index}");
        assert_eq!(fs::read(&probe).unwrap(), bytes[..index]);
    }
}

#[test]
fn additional_demonstrations_resolve_ambiguity_and_preserve_the_parent() {
    let temp = Temp::new();
    let old = TextSkill::learn(
        "surname",
        "names",
        vec![
            TextExample::new("Ana Silva", "Silva"),
            TextExample::new("Bruno Costa", "Costa"),
        ],
        TextBudget::default(),
    )
    .unwrap();
    let old_path = temp.path("old.axtext");
    old.save_new(&old_path).unwrap();
    let original = fs::read(&old_path).unwrap();
    assert!(matches!(
        old.predict("names", "Carla de Souza").unwrap(),
        TextPrediction::Ambiguous { .. }
    ));
    let new = old
        .refine(
            "names",
            &[TextExample::new("Carla de Souza", "Souza")],
            false,
        )
        .unwrap();
    new.save_new(temp.path("new.axtext")).unwrap();
    assert_eq!(new.revision(), 2);
    assert_eq!(new.parent(), Some(old.fingerprint()));
    assert!(
        matches!(new.predict("names", "Daniel da Rocha").unwrap(), TextPrediction::Agreed(w) if w.output == "Rocha")
    );
    assert_eq!(fs::read(&old_path).unwrap(), original);
}

#[test]
fn changing_a_demonstration_requires_explicit_replacement() {
    let old = TextSkill::learn(
        "case",
        "letters",
        vec![
            TextExample::new("Alpha", "ALPHA"),
            TextExample::new("Beta", "BETA"),
        ],
        TextBudget::default(),
    )
    .unwrap();
    let corrected = [
        TextExample::new("Alpha", "alpha"),
        TextExample::new("Beta", "beta"),
    ];
    assert!(old.refine("letters", &corrected, false).is_err());
    let new = old.refine("letters", &corrected, true).unwrap();
    assert!(
        matches!(new.predict("letters", "Gamma").unwrap(), TextPrediction::Agreed(w) if w.output == "gamma")
    );
    assert!(
        matches!(old.predict("letters", "Gamma").unwrap(), TextPrediction::Agreed(w) if w.output == "GAMMA")
    );
}

#[test]
fn csv_workflow_preserves_fields_and_requires_a_matching_preview_confirmation() {
    let temp = Temp::new();
    let model = temp.path("contacts.axtext");
    skill().save_new(&model).unwrap();
    let input = temp.path("input.csv");
    let output = temp.path("output.csv");
    table(
        &input,
        &["contact", "note"],
        &[
            &["Carla Sofia|Pereira Alves", "comma, quote \" and\nnewline"],
            &["Daniel|Rocha Santos", "second"],
        ],
    );
    let before = fs::read(&input).unwrap();
    let preview = command(&[
        "text-preview",
        s(&model),
        s(&input),
        "--column",
        "contact",
        "--result-column",
        "normalized",
        "--context",
        "test/contacts",
    ])
    .unwrap();
    assert!(preview.contains("agreed=2"));
    assert!(preview.contains("Pereira Alves, Carla Sofia"));
    let mut args = vec![
        "text-apply",
        s(&model),
        s(&input),
        "--column",
        "contact",
        "--result-column",
        "normalized",
        "--context",
        "test/contacts",
        "--output",
        s(&output),
    ];
    assert!(command(&args).is_err());
    assert!(!output.exists());
    args.extend(["--confirm", token(&preview)]);
    assert!(command(&args).unwrap().contains("status=published"));
    let mut reader = csv::Reader::from_path(&output).unwrap();
    assert_eq!(
        reader.headers().unwrap(),
        &csv::StringRecord::from(vec!["contact", "note", "normalized"])
    );
    let rows = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(&rows[0][1], "comma, quote \" and\nnewline");
    assert_eq!(&rows[0][2], "Pereira Alves, Carla Sofia");
    assert_eq!(fs::read(&input).unwrap(), before);
    let published = fs::read(&output).unwrap();
    assert!(command(&args).is_err());
    assert_eq!(fs::read(&output).unwrap(), published);
}

#[test]
fn changed_input_invalidates_confirmation_and_ambiguous_batches_publish_nothing() {
    let temp = Temp::new();
    let model = temp.path("name.axtext");
    TextSkill::learn(
        "surname",
        "names",
        vec![
            TextExample::new("Ana Silva", "Silva"),
            TextExample::new("Bruno Costa", "Costa"),
        ],
        TextBudget::default(),
    )
    .unwrap()
    .save_new(&model)
    .unwrap();
    let input = temp.path("input.csv");
    let output = temp.path("output.csv");
    table(&input, &["name"], &[&["Carla Souza"]]);
    let preview = command(&[
        "text-preview",
        s(&model),
        s(&input),
        "--column",
        "name",
        "--context",
        "names",
        "--result-column",
        "surname",
    ])
    .unwrap();
    table(&input, &["name"], &[&["Daniel Rocha"], &["Carla de Souza"]]);
    assert!(
        command(&[
            "text-apply",
            s(&model),
            s(&input),
            "--column",
            "name",
            "--context",
            "names",
            "--result-column",
            "surname",
            "--output",
            s(&output),
            "--confirm",
            token(&preview)
        ])
        .is_err()
    );
    assert!(!output.exists());
    let preview = command(&[
        "text-preview",
        s(&model),
        s(&input),
        "--column",
        "name",
        "--context",
        "names",
    ])
    .unwrap();
    assert!(preview.contains("needs_example=1"));
    assert!(!preview.contains("review_token="));
}

#[test]
fn skill_and_output_publication_never_clobber_existing_paths_or_hardlinks() {
    let temp = Temp::new();
    let path = temp.path("keep.axtext");
    fs::write(&path, b"user data").unwrap();
    let alias = temp.path("alias.axtext");
    fs::hard_link(&path, &alias).unwrap();
    assert!(skill().save_new(&alias).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"user data");
    assert!(fs::read_dir(&temp.0).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".axon-text-stage")
    }));
}

#[test]
fn duplicate_headers_bad_utf8_and_unknown_options_are_rejected() {
    let temp = Temp::new();
    let model = temp.path("model.axtext");
    skill().save_new(&model).unwrap();
    let input = temp.path("input.csv");
    for bytes in [
        b"contact,contact\na,b\n".as_slice(),
        b"contact,note\n\xff,b\n".as_slice(),
        b"contact,note\na,b,c\n".as_slice(),
    ] {
        fs::write(&input, bytes).unwrap();
        assert!(
            command(&[
                "text-preview",
                s(&model),
                s(&input),
                "--column",
                "contact",
                "--context",
                "test/contacts"
            ])
            .is_err()
        );
    }
    assert!(
        command(&[
            "text-fit",
            "x.csv",
            "--name",
            "x",
            "--context",
            "x",
            "--skill",
            s(&model),
            "--unknown",
            "x"
        ])
        .is_err()
    );
}

#[test]
fn executable_learns_and_reopens_a_skill_across_processes() {
    let temp = Temp::new();
    let examples = temp.path("examples.csv");
    let model = temp.path("model.axtext");
    table(
        &examples,
        &["input", "output"],
        &[
            &["  Alpha  Beta ", "alpha beta"],
            &[" Gamma   Delta  ", "gamma delta"],
        ],
    );
    let fitted = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .args([
            "text-fit",
            s(&examples),
            "--name",
            "normalize",
            "--context",
            "strings",
            "--skill",
            s(&model),
        ])
        .output()
        .unwrap();
    assert!(
        fitted.status.success(),
        "{}",
        String::from_utf8_lossy(&fitted.stderr)
    );
    let inspected = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .args(["text-inspect", s(&model)])
        .output()
        .unwrap();
    assert!(inspected.status.success());
    assert!(
        String::from_utf8(inspected.stdout)
            .unwrap()
            .contains("program_pieces=1")
    );
}

#[test]
fn formula_like_exports_are_blocked_without_modifying_originals() {
    let temp = Temp::new();
    let model = temp.path("contacts.axtext");
    skill().save_new(&model).unwrap();
    let input = temp.path("input.csv");
    let output = temp.path("output.csv");
    for suspicious in ["=1+2", " +cmd", "@formula", "\tvalue", "\u{ff1d}SUM(A1)"] {
        table(
            &input,
            &["contact", "note"],
            &[&["Carla Sofia|Pereira Alves", suspicious]],
        );
        let before = fs::read(&input).unwrap();
        let preview = command(&[
            "text-preview",
            s(&model),
            s(&input),
            "--column",
            "contact",
            "--result-column",
            "normalized",
            "--context",
            "test/contacts",
        ])
        .unwrap();
        assert!(
            command(&[
                "text-apply",
                s(&model),
                s(&input),
                "--column",
                "contact",
                "--result-column",
                "normalized",
                "--context",
                "test/contacts",
                "--output",
                s(&output),
                "--confirm",
                token(&preview)
            ])
            .is_err()
        );
        assert!(!output.exists());
        assert_eq!(fs::read(&input).unwrap(), before);
    }
}
