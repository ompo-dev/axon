use axon_uic::numeric::{library::*, *};
use std::io::Write;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

fn sources() -> &'static Vec<NumericSkill> {
    static SOURCES: OnceLock<Vec<NumericSkill>> = OnceLock::new();
    SOURCES.get_or_init(|| {
        let two = NumericSkill::learn(
            "opaque_a",
            "past/a",
            [
                (3.0_f64, 4.0),
                (5.0, 12.0),
                (8.0, 15.0),
                (7.0, 24.0),
                (0.0, 9.0),
                (-3.0, 4.0),
                (4.0, -3.0),
                (1.0, 1.0),
            ]
            .into_iter()
            .map(|(a, b)| NumericExample::new(vec![a, b], a.hypot(b)))
            .collect(),
            &[],
            NumericBudget::default(),
            NumericTolerance::default(),
        )
        .unwrap();
        let rows = [
            vec![2.0_f64, 3.0, 6.0],
            vec![1.0, 2.0, 2.0],
            vec![3.0, 4.0, 12.0],
            vec![8.0, 9.0, 12.0],
            vec![0.0, 0.0, 5.0],
            vec![-2.0, 3.0, 6.0],
            vec![3.0, -4.0, -12.0],
            vec![1.0, 1.0, 1.0],
        ]
        .into_iter()
        .map(|x| {
            let y = x[0].hypot(x[1]).hypot(x[2]);
            NumericExample::new(x, y)
        })
        .collect();
        let three = NumericSkill::learn(
            "opaque_b",
            "past/b",
            rows,
            &[&two],
            NumericBudget {
                max_cost: 5,
                ..NumericBudget::default()
            },
            NumericTolerance::default(),
        )
        .unwrap();
        vec![two, three]
    })
}
fn mined() -> NumericLibrary {
    NumericLibrary::mine(
        "library/experience",
        &sources().iter().collect::<Vec<_>>(),
        MiningBudget::default(),
    )
    .unwrap()
}
fn changed_curriculum() -> NumericLibrary {
    let mut history = sources().clone();
    for case in 0..6 {
        let rows = [-7.0_f64, -3.25, -0.5, 0.0, 1.3, 2.25, 11.0]
            .into_iter()
            .map(|x| {
                let q = x * x;
                let y = match case {
                    0 => q,
                    1 => q + 1.0,
                    2 => q - 1.0,
                    3 => q * 2.0,
                    4 => q / 2.0,
                    _ => q + 2.0,
                };
                NumericExample::new(vec![x], y)
            })
            .collect();
        history.push(
            NumericSkill::learn(
                format!("past_{case}"),
                format!("past/{case}"),
                rows,
                &[],
                NumericBudget::default(),
                NumericTolerance::default(),
            )
            .unwrap(),
        );
    }
    NumericLibrary::mine(
        "library/changed",
        &history.iter().collect::<Vec<_>>(),
        MiningBudget::default(),
    )
    .unwrap()
}
#[test]
fn demand_recovers_a_useful_mechanism_rejected_by_retrospective_compression() {
    let library = changed_curriculum();
    let budget = NumericBudget {
        max_cost: 5,
        ..NumericBudget::default()
    };
    assert!(
        NumericSkill::learn_from_library(
            "new",
            "new/task",
            transfer_rows(),
            &library,
            budget,
            NumericTolerance::default()
        )
        .is_err()
    );
    let (skill, trace) = library
        .develop(
            "new",
            "new/task",
            transfer_rows(),
            budget,
            DevelopmentBudget::default(),
            NumericTolerance::default(),
        )
        .unwrap();
    assert!(trace.attempts.len() > 1);
    assert!(!trace.attempts[0].compatible);
    assert!(trace.attempts.last().unwrap().compatible);
    assert_eq!(
        trace.scalar_work,
        trace
            .attempts
            .iter()
            .map(|a| a.search.scalar_work)
            .sum::<u64>()
    );
    assert!(trace.proposal_patterns > 0 && trace.proposal_visits > 0);
    let admitted = trace.admitted.unwrap();
    assert!(admitted.sources().len() >= 2);
    assert!(skill.dependencies().contains(&admitted.fingerprint()));
    let restored = NumericSkill::from_bytes(&skill.to_bytes()).unwrap();
    for i in -1000..=1000 {
        let x = [i as f64 / 37.0, -9.25, 0.75];
        assert!(restored.tolerance().agrees(
            restored.program().evaluate(&x).unwrap(),
            x[0].hypot(x[1]) / x[2]
        ));
    }
}
#[test]
fn development_enforces_aggregate_budgets_and_preserves_input_errors() {
    let library = changed_curriculum();
    let budget = NumericBudget {
        max_cost: 5,
        ..NumericBudget::default()
    };
    let result = library.develop(
        "new",
        "new/task",
        transfer_rows(),
        budget,
        DevelopmentBudget {
            max_attempts: 1,
            ..DevelopmentBudget::default()
        },
        NumericTolerance::default(),
    );
    assert!(matches!(
        result,
        Err(NumericError::Development { attempts: 1, .. })
    ));
    match library.develop(
        "new",
        "new/task",
        transfer_rows(),
        budget,
        DevelopmentBudget {
            max_work: 100,
            ..DevelopmentBudget::default()
        },
        NumericTolerance::default(),
    ) {
        Err(NumericError::Development { scalar_work, .. }) => assert!(scalar_work <= 100),
        other => panic!("unexpected development result: {other:?}"),
    }
    let conflicting = vec![
        NumericExample::new(vec![1.0], 2.0),
        NumericExample::new(vec![1.0], 9.0),
    ];
    assert!(matches!(
        library.develop(
            "bad",
            "bad/context",
            conflicting,
            budget,
            DevelopmentBudget::default(),
            NumericTolerance::default()
        ),
        Err(NumericError::Contradiction)
    ));
}
#[test]
fn the_complete_lab_preserves_three_policy_comparisons_and_durable_outputs() {
    let temp = Temp::new();
    let history = temp.0.join("history");
    fs::create_dir(&history).unwrap();
    for (i, skill) in sources().iter().enumerate() {
        skill
            .save_new(&history.join(format!("source-{i}.axnum")))
            .unwrap();
    }
    let output = temp.0.join("campaign");
    let result = Command::new(env!("CARGO_BIN_EXE_axon-uic-abstraction-lab"))
        .arg(&history)
        .arg(&output)
        .args(["30719581", "100"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let summary = fs::read_to_string(output.join("summary.txt")).unwrap();
    assert!(summary.contains("demand_verified=7/7"));
    assert!(summary.contains("shared_execution_bit_identical=700"));
    let results = fs::read_to_string(output.join("results.csv")).unwrap();
    assert!(results.contains(",primitive,refused,"));
    assert!(results.contains(",compression,refused,"));
    assert!(results.contains(",developing,passed,"));
    assert!(output.join("development.csv").exists());
    let cli_model = temp.0.join("cli-point.axnum");
    let training = output.join("point_distance.training.tsv");
    let library = output.join("mined.axlib");
    let cli = Command::new(env!("CARGO_BIN_EXE_axon"))
        .args([
            "numeric-develop",
            training.to_str().unwrap(),
            "--name",
            "points",
            "--context",
            "user/points",
            "--mined-library",
            library.to_str().unwrap(),
            "--library-context",
            "abstraction-lab/experience/v1",
            "--max-cost",
            "7",
            "--max-terms",
            "200000",
            "--skill",
            cli_model.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        cli.status.success(),
        "{}",
        String::from_utf8_lossy(&cli.stderr)
    );
    assert!(
        NumericSkill::open(&cli_model, "user/points")
            .unwrap()
            .tolerance()
            .agrees(
                NumericSkill::open(&cli_model, "user/points")
                    .unwrap()
                    .predict("user/points", &[0.0, 0.0, 3.0, 4.0])
                    .unwrap(),
                5.0
            )
    );
    assert!(
        NumericLibrary::open(
            &output.join("next-epoch.axlib"),
            "abstraction-lab/experience/v2"
        )
        .is_ok()
    );
}
fn transfer_rows() -> Vec<NumericExample> {
    [
        vec![1.0_f64, 2.0, 3.0],
        vec![4.0, 7.0, 2.0],
        vec![5.0, 12.0, -3.0],
        vec![-2.0, 3.0, 7.0],
        vec![0.0, 9.0, 4.0],
        vec![11.0, -6.0, 5.0],
        vec![1.0, 1.0, 2.0],
    ]
    .into_iter()
    .map(|x| {
        let y = x[0].hypot(x[1]) / x[2];
        NumericExample::new(x, y)
    })
    .collect()
}
#[test]
fn reusable_abstractions_are_selected_without_names_or_task_recipes() {
    let library = mined();
    assert!(!library.modules().is_empty());
    let s = library.statistics();
    assert_eq!(s.unique_sources, 2);
    assert!(s.net_saved_nodes > 0);
    assert_eq!(
        s.net_saved_nodes,
        s.original_nodes - s.rewritten_nodes - s.definition_nodes
    );
    assert_eq!(library.modules()[0].sources().len(), 2);
    let budget = NumericBudget {
        max_cost: 5,
        ..NumericBudget::default()
    };
    assert!(
        NumericSkill::learn(
            "new",
            "new/task",
            transfer_rows(),
            &[],
            budget,
            NumericTolerance::default()
        )
        .is_err()
    );
    let skill = NumericSkill::learn_from_library(
        "new",
        "new/task",
        transfer_rows(),
        &library,
        budget,
        NumericTolerance::default(),
    )
    .unwrap();
    assert!(
        skill
            .dependencies()
            .iter()
            .all(|id| library.modules().iter().any(|m| m.fingerprint() == *id))
    );
    let restored = NumericSkill::from_bytes(&skill.to_bytes()).unwrap();
    for i in -100..=100 {
        let x = [i as f64 / 7.0, -3.25, 0.125];
        assert!(restored.tolerance().agrees(
            restored.predict("new/task", &x).unwrap(),
            x[0].hypot(x[1]) / x[2]
        ));
    }
    assert!(matches!(
        restored.predict("new/task", &[3.0, 4.0, 0.0]),
        Err(NumericError::Domain(_))
    ));
}
#[test]
fn duplicate_programs_and_input_renaming_do_not_fake_independent_support() {
    let original = &sources()[0];
    let renamed = NumericSkill::learn(
        "different_name",
        "different_context",
        original.examples().to_vec(),
        &[],
        NumericBudget::default(),
        original.tolerance(),
    )
    .unwrap();
    let many: Vec<_> = (0..30)
        .map(|i| if i % 2 == 0 { original } else { &renamed })
        .collect();
    let library =
        NumericLibrary::mine("library/duplicates", &many, MiningBudget::default()).unwrap();
    assert_eq!(library.statistics().supplied_sources, 30);
    assert_eq!(library.statistics().unique_sources, 1);
    assert!(library.modules().is_empty());
    assert_eq!(library.statistics().net_saved_nodes, 0);
}
#[test]
fn selection_is_deterministic_under_source_order_and_storage_roundtrips() {
    let first = mined();
    let second = NumericLibrary::mine(
        "library/experience",
        &[&sources()[1], &sources()[0]],
        MiningBudget::default(),
    )
    .unwrap();
    assert_eq!(first.fingerprint(), second.fingerprint());
    assert_eq!(
        first.statistics().net_saved_nodes,
        second.statistics().net_saved_nodes
    );
    let reopened = NumericLibrary::from_bytes(&first.to_bytes()).unwrap();
    assert_eq!(first.fingerprint(), reopened.fingerprint());
    assert_eq!(first.to_bytes(), reopened.to_bytes());
}
#[test]
fn mining_budgets_are_enforced_and_never_publish_partial_inventions() {
    for budget in [
        MiningBudget {
            max_visits: 1,
            ..MiningBudget::default()
        },
        MiningBudget {
            max_patterns: 1,
            ..MiningBudget::default()
        },
    ] {
        assert!(matches!(
            NumericLibrary::mine("tiny", &sources().iter().collect::<Vec<_>>(), budget),
            Err(NumericError::LibraryBudget { .. })
        ));
    }
    assert!(NumericLibrary::mine("bad", &[], MiningBudget::default()).is_err());
    assert!(
        NumericLibrary::mine(
            "bad",
            &[&sources()[0]],
            MiningBudget {
                max_modules: 0,
                ..MiningBudget::default()
            }
        )
        .is_err()
    );
}
#[test]
fn library_corruption_truncation_and_forged_accounting_are_rejected() {
    let bytes = mined().to_bytes();
    for i in 0..bytes.len() {
        assert!(NumericLibrary::from_bytes(&bytes[..i]).is_err());
    }
    for i in [0, 8, bytes.len() / 2, bytes.len() - 1] {
        let mut altered = bytes.clone();
        altered[i] ^= 1;
        assert!(NumericLibrary::from_bytes(&altered).is_err());
    }
    let mut altered = bytes.clone();
    let total = altered.len();
    altered[total - 28] ^= 1;
    let hash = altered[..total - 8]
        .iter()
        .fold(0xcbf29ce484222325_u64, |h, b| {
            (h ^ *b as u64).wrapping_mul(0x100000001b3)
        });
    altered[total - 8..].copy_from_slice(&hash.to_le_bytes());
    assert!(NumericLibrary::from_bytes(&altered).is_err());
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "axon-library-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let (Ok(path), Ok(root)) = (self.0.canonicalize(), std::env::temp_dir().canonicalize()) {
            assert!(path.starts_with(root));
            let _ = fs::remove_dir_all(path);
        }
    }
}
#[test]
fn publication_is_no_clobber_and_contexts_are_checked() {
    let temp = Temp::new();
    let path = temp.0.join("learned.axlib");
    let library = mined();
    library.save_new(&path).unwrap();
    assert!(library.save_new(&path).is_err());
    assert!(matches!(
        NumericLibrary::open(&path, "wrong"),
        Err(NumericError::Context)
    ));
    assert_eq!(
        NumericLibrary::open(&path, library.context())
            .unwrap()
            .fingerprint(),
        library.fingerprint()
    );
}
#[test]
fn real_cli_mines_then_uses_and_revises_the_library_with_explicit_context() {
    let temp = Temp::new();
    for (i, skill) in sources().iter().enumerate() {
        skill.save_new(&temp.0.join(format!("{i}.axnum"))).unwrap();
    }
    fs::write(
        temp.0.join("sources.tsv"),
        "skill\tcontext\n0.axnum\tpast/a\n1.axnum\tpast/b\n",
    )
    .unwrap();
    let manifest = temp.0.join("sources.tsv");
    let library = temp.0.join("new.axlib");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_axon"))
            .args(args)
            .output()
            .unwrap()
    };
    let first = run(&[
        "numeric-mine",
        manifest.to_str().unwrap(),
        "--context",
        "library/cli",
        "--library",
        library.to_str().unwrap(),
    ]);
    assert!(
        first.status.success(),
        "{}",
        String::from_utf8_lossy(&first.stderr)
    );
    assert!(String::from_utf8_lossy(&first.stdout).contains("net_saved_tree_nodes="));
    assert!(
        !run(&[
            "numeric-library-inspect",
            library.to_str().unwrap(),
            "--context",
            "wrong"
        ])
        .status
        .success()
    );
    let training = temp.0.join("data.tsv");
    let model = temp.0.join("result.axnum");
    let mut text = String::from("x0\tx1\tx2\ty\n");
    for e in transfer_rows() {
        text.push_str(&format!(
            "{}\t{}\t{}\t{:.17}\n",
            e.input[0], e.input[1], e.input[2], e.output
        ));
    }
    fs::write(&training, text).unwrap();
    let fit = run(&[
        "numeric-fit",
        training.to_str().unwrap(),
        "--name",
        "opaque",
        "--context",
        "task/cli",
        "--skill",
        model.to_str().unwrap(),
        "--mined-library",
        library.to_str().unwrap(),
        "--library-context",
        "library/cli",
        "--max-cost",
        "5",
    ]);
    assert!(
        fit.status.success(),
        "{}",
        String::from_utf8_lossy(&fit.stderr)
    );
    let pred = run(&[
        "numeric-predict",
        model.to_str().unwrap(),
        "--context",
        "task/cli",
        "--values",
        "3,4,2",
    ]);
    assert!(pred.status.success());
    assert!(String::from_utf8_lossy(&pred.stdout).contains("value=2.500"));
    let developed = temp.0.join("developed.axnum");
    let developed_result = run(&[
        "numeric-develop",
        training.to_str().unwrap(),
        "--name",
        "adaptive",
        "--context",
        "task/cli",
        "--skill",
        developed.to_str().unwrap(),
        "--mined-library",
        library.to_str().unwrap(),
        "--library-context",
        "library/cli",
        "--max-cost",
        "5",
    ]);
    assert!(developed_result.status.success());
    assert!(String::from_utf8_lossy(&developed_result.stdout).contains("development_searches=1"));
    assert!(
        run(&[
            "numeric-eval",
            developed.to_str().unwrap(),
            training.to_str().unwrap(),
            "--context",
            "task/cli",
            "--backend",
            "shared"
        ])
        .status
        .success()
    );
    assert!(
        !run(&[
            "numeric-predict",
            developed.to_str().unwrap(),
            "--context",
            "task/cli",
            "--values",
            "3,4,2",
            "--backend",
            "invalid"
        ])
        .status
        .success()
    );
    let next = temp.0.join("revision.axnum");
    assert!(
        run(&[
            "numeric-revise",
            model.to_str().unwrap(),
            training.to_str().unwrap(),
            "--context",
            "task/cli",
            "--skill",
            next.to_str().unwrap(),
            "--mined-library",
            library.to_str().unwrap(),
            "--library-context",
            "library/cli",
            "--max-cost",
            "5"
        ])
        .status
        .success()
    );
    assert_eq!(NumericSkill::open(&next, "task/cli").unwrap().revision(), 2);
    assert!(
        !run(&[
            "numeric-fit",
            training.to_str().unwrap(),
            "--name",
            "bad",
            "--context",
            "task/cli",
            "--skill",
            next.to_str().unwrap(),
            "--mined-library",
            library.to_str().unwrap(),
            "--max-cost",
            "5"
        ])
        .status
        .success()
    );
}

#[test]
fn streaming_evaluation_replays_large_files_without_collecting_the_dataset() {
    let temp = Temp::new();
    let model = temp.0.join("model.axnum");
    sources()[0].save_new(&model).unwrap();
    let data = temp.0.join("large.tsv");
    let mut file = std::io::BufWriter::new(fs::File::create(&data).unwrap());
    writeln!(file, "x0\tx1\ty").unwrap();
    for _ in 0..50000 {
        writeln!(
            file,
            "3.00000000000000000\t4.00000000000000000\t5.00000000000000000"
        )
        .unwrap();
    }
    file.flush().unwrap();
    drop(file);
    assert!(fs::metadata(&data).unwrap().len() > 2097152);
    for backend in ["stack", "shared"] {
        let result = Command::new(env!("CARGO_BIN_EXE_axon"))
            .args([
                "numeric-eval",
                model.to_str().unwrap(),
                data.to_str().unwrap(),
                "--context",
                "past/a",
                "--backend",
                backend,
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout).contains("correct=50000"));
    }
}
#[test]
fn streaming_evaluation_refuses_oversized_lines_and_malformed_final_records() {
    let temp = Temp::new();
    let model = temp.0.join("model.axnum");
    sources()[0].save_new(&model).unwrap();
    let data = temp.0.join("invalid.tsv");
    for text in [
        format!("x0\tx1\ty\n{}\t4\t5\n", "0".repeat(4097)),
        "x0\tx1\ty\n3\t4\t5\n3\t4".to_owned(),
        "x0\tx1\ty\n3\t4\tNaN\n".to_owned(),
        "x0\tx1\ty\n3\t4\t5\textra\n".to_owned(),
        "x0\tx1\ty\n".to_owned(),
    ] {
        fs::write(&data, text).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_axon"))
            .args([
                "numeric-eval",
                model.to_str().unwrap(),
                data.to_str().unwrap(),
                "--context",
                "past/a",
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
    fs::write(&data, b"x0\tx1\ty\r\n3\t4\t5\r\n3\t4\t5").unwrap();
    assert!(
        Command::new(env!("CARGO_BIN_EXE_axon"))
            .args([
                "numeric-eval",
                model.to_str().unwrap(),
                data.to_str().unwrap(),
                "--context",
                "past/a"
            ])
            .output()
            .unwrap()
            .status
            .success()
    );
}
