#![cfg(feature = "curriculum")]

use axon_uic::curriculum::{Budget, Example, Kind, Prepared, Stage, Value};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "axon-curriculum-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let path = self.0.canonicalize().unwrap();
        assert!(path.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        fs::remove_dir_all(path).unwrap();
    }
}
fn fixture() -> Prepared {
    let numeric = |offset: u32, count: u32| {
        (offset..offset + count)
            .map(|i| {
                let x = f64::from(i) / 7.0;
                Example {
                    input: Value::Numbers(vec![x]),
                    output: Value::Number(x * x),
                    arguments: vec![],
                }
            })
            .collect()
    };
    Prepared {
        version: 1,
        id: "contract".into(),
        source_sha256: "a".repeat(64),
        budget: Budget {
            candidates_per_stage: 2000,
            total_candidates: 8000,
            work_per_stage: 1_000_000,
            total_numeric_work: 4_000_000,
            text_work_per_stage: 2_000_000,
            semantic_work_per_stage: 2_000_000,
        },
        stages: vec![Stage {
            id: "square".into(),
            family: "math".into(),
            kind: Kind::Numeric,
            required: true,
            parents: vec![],
            train: numeric(0, 12),
            admission: numeric(20, 12),
            test: numeric(40, 20),
            transfer: numeric(80, 10),
        }],
    }
}
#[test]
fn json_boundary_rejects_leakage_forward_parents_types_and_unknown_fields() {
    let p = fixture();
    assert!(p.validate().is_ok());
    let mut bad = p.clone();
    bad.stages[0].admission[0] = bad.stages[0].train[0].clone();
    assert!(bad.validate().is_err());
    let mut bad = p.clone();
    bad.stages[0].parents.push("future".into());
    assert!(bad.validate().is_err());
    let mut bad = p.clone();
    bad.stages[0].test[0].input = Value::Text("wrong".into());
    assert!(bad.validate().is_err());
    let mut bad = p.clone();
    bad.stages[0].test[0].input = Value::Numbers(vec![f64::NAN]);
    assert!(bad.validate().is_err());
    let mut bad = serde_json::to_value(p).unwrap();
    bad["stages"][0]["oracle"] = serde_json::json!(["x0"]);
    assert!(serde_json::from_value::<Prepared>(bad).is_err());
}
#[test]
fn portable_birth_exact_resume_append_and_tamper_detection() {
    let temp = Temp::new();
    let prepared = temp.0.join("prepared.json");
    let mut p = fixture();
    let run = |name: &str, p: &Prepared| {
        fs::write(&prepared, serde_json::to_vec(p).unwrap()).unwrap();
        let output = temp.0.join(name);
        fs::create_dir(&output).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_axon-uic-curriculum-lab"))
            .env("AXON_CURRICULUM_ROOT", temp.0.join("registry"))
            .args([
                "run",
                prepared.to_str().unwrap(),
                output.to_str().unwrap(),
                "91",
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(output.join("curriculum-results.json")).unwrap())
                .unwrap();
        (output, receipt)
    };
    let (first, receipt) = run("first", &p);
    assert_eq!(receipt["stages"][0]["admitted"], true);
    assert_eq!(receipt["stages"][0]["test_correct"], 20);
    assert_eq!(receipt["pages"][0]["initial_loads"], 0);
    assert_eq!(receipt["pages"][0]["active_modules"], 1);
    assert_eq!(receipt["pages"][0]["post_selection_rechecks"], 30);
    assert!(receipt["new_candidates"].as_u64().unwrap() > 0);
    let predict = Command::new(env!("CARGO_BIN_EXE_axon-uic-curriculum-lab"))
        .args([
            "predict",
            first.join("page-0.axnum").to_str().unwrap(),
            temp.0.join("newborn").to_str().unwrap(),
            "square",
            "[3.5]",
        ])
        .output()
        .unwrap();
    assert!(
        predict.status.success(),
        "{}",
        String::from_utf8_lossy(&predict.stderr)
    );
    let answer: serde_json::Value = serde_json::from_slice(&predict.stdout).unwrap();
    assert_eq!(answer["answer"], 12.25);
    assert_eq!(answer["training_required"], false);
    let (second, resumed) = run("second", &p);
    assert_eq!(resumed["new_candidates"], 0);
    assert_eq!(resumed["stages"][0]["cached"], true);
    assert_eq!(
        fs::read(first.join("page-0.axnum")).unwrap(),
        fs::read(second.join("page-0.axnum")).unwrap()
    );
    let mut extra = p.stages[0].clone();
    extra.id = "square-again".into();
    extra.parents = vec!["square".into()];
    p.stages.push(extra);
    let (_, appended) = run("appended", &p);
    assert_eq!(appended["stages"][0]["cached"], true);
    assert_eq!(appended["stages"][1]["cached"], false);
    let registry = temp.0.join("registry/contract-91");
    let stage = fs::read_dir(registry)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("square-")
                && !p
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .starts_with("square-again")
        })
        .unwrap();
    fs::write(stage.join("square.axevo"), b"tampered").unwrap();
    let output = temp.0.join("tampered-run");
    fs::create_dir(&output).unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_axon-uic-curriculum-lab"))
        .env("AXON_CURRICULUM_ROOT", temp.0.join("registry"))
        .args([
            "run",
            prepared.to_str().unwrap(),
            output.to_str().unwrap(),
            "91",
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("checksum"));
}

#[test]
fn curriculum_pages_preserve_core_limits_and_all_inherited_modules() {
    let temp = Temp::new();
    let mut p = fixture();
    p.id = "paged-contract".into();
    p.budget.total_candidates = 1_000_000;
    p.budget.total_numeric_work = 100_000_000;
    for i in 1..129 {
        let mut stage = p.stages[0].clone();
        stage.id = format!("square-{i}");
        stage.parents = vec!["square".into()];
        p.stages.push(stage);
    }
    let prepared = temp.0.join("prepared.json");
    fs::write(&prepared, serde_json::to_vec(&p).unwrap()).unwrap();
    let output = temp.0.join("output");
    fs::create_dir(&output).unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_axon-uic-curriculum-lab"))
        .env("AXON_CURRICULUM_ROOT", temp.0.join("registry"))
        .args([
            "run",
            prepared.to_str().unwrap(),
            output.to_str().unwrap(),
            "91",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("curriculum-results.json")).unwrap()).unwrap();
    let pages = result["pages"].as_array().unwrap();
    assert_eq!(pages.len(), 2);
    assert!(
        result["stages"]
            .as_array()
            .unwrap()
            .iter()
            .all(|s| s["admitted"] == true)
    );
    assert_eq!(pages[0]["modules"].as_array().unwrap().len(), 128);
    assert_eq!(pages[1]["modules"].as_array().unwrap().len(), 1);
    for page in pages {
        assert!(page["encoded_bytes"].as_u64().unwrap() <= 262144);
        assert_eq!(page["initial_loads"], 0);
        assert_eq!(page["active_modules"], 1);
    }
}

#[test]
fn learned_routing_and_inventory_report_run_in_new_process_without_module_ids() {
    let temp = Temp::new();
    let mut p = fixture();
    for (name, left, right, multiply) in [
        ("invoice", "preco", "quantidade", true),
        ("stock", "necessidade", "estoque", false),
    ] {
        let rows = |start: usize, count: usize| {
            (start..start + count)
                .map(|i| {
                    let x = ((i * 17 + 3) % 97) as f64;
                    let y = ((i * 23 + 7) % 89) as f64;
                    let input = if i % 2 == 0 {
                        format!("{name} {left} {x} {right} {y}")
                    } else {
                        format!("{name} {right} {y} {left} {x}")
                    };
                    Example {
                        input: Value::Text(input),
                        output: Value::Number(if multiply { x * y } else { x - y }),
                        arguments: vec![x, y],
                    }
                })
                .collect()
        };
        p.stages.push(Stage {
            id: name.into(),
            family: "operational".into(),
            kind: Kind::Relational,
            required: true,
            parents: vec![],
            train: rows(0, 12),
            admission: rows(30, 12),
            test: rows(50, 20),
            transfer: rows(80, 10),
        });
    }
    let prepared = temp.0.join("prepared.json");
    fs::write(&prepared, serde_json::to_vec(&p).unwrap()).unwrap();
    let output = temp.0.join("trained");
    fs::create_dir(&output).unwrap();
    let bin = env!("CARGO_BIN_EXE_axon-uic-curriculum-lab");
    let run = Command::new(bin)
        .env("AXON_CURRICULUM_ROOT", temp.0.join("registry"))
        .args([
            "run",
            prepared.to_str().unwrap(),
            output.to_str().unwrap(),
            "91",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let page = output.join("page-0.axnum");
    let ask = Command::new(bin)
        .args([
            "ask",
            page.to_str().unwrap(),
            temp.0.join("ask").to_str().unwrap(),
            "\"quantidade 13 invoice preco 7\"",
        ])
        .output()
        .unwrap();
    assert!(
        ask.status.success(),
        "{}",
        String::from_utf8_lossy(&ask.stderr)
    );
    let reply: serde_json::Value = serde_json::from_slice(&ask.stdout).unwrap();
    assert_eq!(reply["answer"]["value"], 91.0);
    assert_eq!(reply["answer"]["module"], "invoice");
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("evals/fixtures/transfer-inventory-v1");
    let report = temp.0.join("report");
    let execute = || {
        Command::new(bin)
            .args([
                "report",
                page.to_str().unwrap(),
                root.to_str().unwrap(),
                root.join("recipe.json").to_str().unwrap(),
                report.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let completed = execute();
    assert!(
        completed.status.success(),
        "{}",
        String::from_utf8_lossy(&completed.stderr)
    );
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(report.join("audit-inventory-v1.json")).unwrap()).unwrap();
    assert_eq!(receipt["rows"][0]["needed"], 8);
    assert_eq!(receipt["rows"][0]["cost_cents"], 5600);
    assert_eq!(receipt["rows"][1]["needed"], 0);
    assert_eq!(receipt["rows"][2]["cost_cents"], 120);
    let before = fs::read(report.join("report.csv")).unwrap();
    assert!(!execute().status.success());
    assert_eq!(before, fs::read(report.join("report.csv")).unwrap());
}

#[test]
fn integrated_exercise_preserves_pre_feedback_misses_and_cold_controls() {
    let temp = Temp::new();
    let output = temp.0.join("exercise");
    let run = Command::new(env!("CARGO_BIN_EXE_axon-uic-curriculum-lab"))
        .args(["exercise", output.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let r: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("audit-integration-v1.json")).unwrap())
            .unwrap();
    assert_eq!(
        r["meta"]["cold_succeeded"],
        serde_json::json!([false, false])
    );
    assert_eq!(r["meta_passed"], true);
    assert_eq!(r["continuous_passed"], true);
    assert_eq!(r["language"]["unknown_before_teaching"], true);
    assert_eq!(r["language"]["old_binding_retained"], true);
    let records = r["records"].as_array().unwrap();
    assert_eq!(records.len(), 360);
    assert_eq!(
        r["continuous"]["correct"].as_u64().unwrap() as usize,
        records
            .iter()
            .filter(|r| r["scored"] == true && r["correct"] == true)
            .count()
    );
    assert!(
        records
            .iter()
            .any(|r| r["scored"] == true && r["correct"] == false)
    );
}
