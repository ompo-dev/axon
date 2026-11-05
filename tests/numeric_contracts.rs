use axon_uic::numeric::*;
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

fn calibration(scale: f64) -> Vec<NumericExample> {
    [-7.0, 0.0, 0.25, 3.0, 16.0]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], scale * x + 1.0))
        .collect()
}
fn learn(examples: Vec<NumericExample>) -> NumericSkill {
    NumericSkill::learn(
        "test",
        "numeric/test",
        examples,
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap()
}
fn distance2() -> &'static NumericSkill {
    static SKILL: OnceLock<NumericSkill> = OnceLock::new();
    SKILL.get_or_init(|| {
        learn(
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
        )
    })
}

#[test]
fn induction_generalizes_calibration_outside_the_demonstration_range() {
    let skill = learn(calibration(2.0));
    for x in [-1000.0, -9.5, 0.1, 6.0, 25.0, 9999.0] {
        assert!(
            skill
                .tolerance()
                .agrees(skill.predict("numeric/test", &[x]).unwrap(), 2.0 * x + 1.0)
        );
    }
    assert!(skill.search().search_cost <= 5);
    assert!(skill.search().scalar_work > 0);
}
#[test]
fn noninteger_calibration_parameters_are_learned_from_data_not_supplied_as_literals() {
    let rows = [-11.0, 0.0, 0.25, 7.0, 92.0]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], 1.75 * x - 4.25))
        .collect();
    let skill = learn(rows);
    for x in [-1000.0, -0.33, 0.01, 14.7, 9999.0] {
        assert!(skill.tolerance().agrees(
            skill.predict("numeric/test", &[x]).unwrap(),
            1.75 * x - 4.25
        ));
    }
    assert!(skill.program().expression().contains("1.75"));
}
#[test]
fn geometry_is_synthesized_from_primitives_and_checked_on_unseen_inputs() {
    let skill = distance2();
    assert_eq!(skill.search().library_expansions, 0);
    assert!(skill.program().expression().contains("sqrt"));
    for a in -15..=15 {
        for b in -17..=17 {
            let (a, b) = (a as f64 / 3.0, b as f64 / 5.0);
            assert!(
                skill
                    .tolerance()
                    .agrees(skill.predict("numeric/test", &[a, b]).unwrap(), a.hypot(b))
            );
        }
    }
}
#[test]
fn acquired_binary_program_becomes_a_tool_for_a_three_input_task() {
    let rows = [
        (2.0_f64, 3.0, 6.0),
        (1.0, 2.0, 2.0),
        (3.0, 4.0, 12.0),
        (8.0, 9.0, 12.0),
        (0.0, 0.0, 5.0),
        (-2.0, 3.0, 6.0),
        (3.0, -4.0, -12.0),
        (1.0, 1.0, 1.0),
    ]
    .into_iter()
    .map(|(a, b, c)| NumericExample::new(vec![a, b, c], a.hypot(b).hypot(c)))
    .collect::<Vec<_>>();
    let budget = NumericBudget {
        max_cost: 5,
        ..NumericBudget::default()
    };
    assert!(matches!(
        NumericSkill::learn(
            "raw",
            "geometry/3",
            rows.clone(),
            &[],
            budget,
            NumericTolerance::default()
        ),
        Err(NumericError::NoProgram(_) | NumericError::Budget(_))
    ));
    let skill = NumericSkill::learn(
        "reused",
        "geometry/3",
        rows,
        &[distance2()],
        budget,
        NumericTolerance::default(),
    )
    .unwrap();
    // The stack compiler expands the inner call twice because its output is squared.
    assert_eq!(skill.search().library_expansions, 3);
    assert_eq!(skill.dependencies(), &[distance2().fingerprint()]);
    let restored = NumericSkill::from_bytes(&skill.to_bytes()).unwrap();
    for a in -4..=4 {
        for b in -4..=4 {
            for c in -4..=4 {
                let (a, b, c) = (a as f64 / 2.0, b as f64 / 3.0, c as f64 / 7.0);
                assert!(restored.tolerance().agrees(
                    restored.predict("geometry/3", &[a, b, c]).unwrap(),
                    a.hypot(b).hypot(c)
                ));
            }
        }
    }
}
#[test]
fn an_acquired_three_input_capability_can_be_reused_in_a_four_input_task() {
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
    let acquired = NumericSkill::learn(
        "three",
        "geometry/three",
        rows,
        &[distance2()],
        NumericBudget {
            max_cost: 5,
            ..NumericBudget::default()
        },
        NumericTolerance::default(),
    )
    .unwrap();
    let rows = [
        vec![1.0_f64, 2.0, 3.0, 4.0],
        vec![-2.0, 3.0, 6.0, 9.0],
        vec![3.0, -4.0, 12.0, -2.0],
        vec![0.0, 0.0, 5.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![9.0, 8.0, 4.0, 3.0],
    ]
    .into_iter()
    .map(|x| {
        let y = x[0].hypot(x[1]).hypot(x[2]).hypot(x[3]);
        NumericExample::new(x, y)
    })
    .collect();
    let fourth = NumericSkill::learn(
        "four",
        "geometry/four",
        rows,
        &[&acquired],
        NumericBudget {
            max_cost: 7,
            ..NumericBudget::default()
        },
        NumericTolerance::default(),
    )
    .unwrap();
    assert!(fourth.dependencies().contains(&acquired.fingerprint()));
    let restored = NumericSkill::from_bytes(&fourth.to_bytes()).unwrap();
    for a in -3..=3 {
        for b in -3..=3 {
            let input = [a as f64 / 3.0, b as f64 / 7.0, 3.5, -12.5];
            let expected = input[0].hypot(input[1]).hypot(input[2]).hypot(input[3]);
            assert!(
                restored
                    .tolerance()
                    .agrees(restored.predict("geometry/four", &input).unwrap(), expected)
            );
        }
    }
}
#[test]
fn domain_guards_refuse_zero_divisors_nonfinite_inputs_and_overflow() {
    let reciprocal = learn(
        [1.0, 2.0, 4.0, -2.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], 1.0 / x))
            .collect(),
    );
    assert!(matches!(
        reciprocal.predict("numeric/test", &[0.0]),
        Err(NumericError::Domain(_))
    ));
    for input in [vec![], vec![1.0, 2.0], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(matches!(
            reciprocal.predict("numeric/test", &input),
            Err(NumericError::Domain(_))
        ));
    }
    assert!(matches!(
        distance2().predict("numeric/test", &[1e300, 1e300]),
        Err(NumericError::Domain(_))
    ));
    assert!(matches!(
        distance2().predict("numeric/test", &[1e-300, 1e-300]),
        Err(NumericError::Domain(_))
    ));
    assert!(matches!(
        reciprocal.predict("wrong", &[1.0]),
        Err(NumericError::Context)
    ));
    let square_root = learn(
        [0.0_f64, 1.0, 4.0, 9.0, 16.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], x.sqrt()))
            .collect(),
    );
    assert!(matches!(
        square_root.predict("numeric/test", &[-1.0]),
        Err(NumericError::Domain(_))
    ));
}
#[test]
fn conflicting_data_is_not_silently_averaged() {
    let result = NumericSkill::learn(
        "bad",
        "ctx",
        vec![
            NumericExample::new(vec![2.0], 3.0),
            NumericExample::new(vec![2.0], 9.0),
        ],
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    );
    assert!(matches!(result, Err(NumericError::Contradiction)));
}
#[test]
fn unsupported_task_and_resource_limits_are_explicit_not_fake_predictions() {
    let rows = [0.2_f64, 0.7, 1.3, 2.1, 3.2]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x.sin()))
        .collect::<Vec<_>>();
    let budget = NumericBudget {
        max_cost: 5,
        max_terms: 4000,
        max_work: 100_000,
    };
    assert!(matches!(
        NumericSkill::learn(
            "sine",
            "ctx",
            rows.clone(),
            &[],
            budget,
            NumericTolerance::default()
        ),
        Err(NumericError::NoProgram(_) | NumericError::Budget(_))
    ));
    let result = NumericSkill::learn(
        "sine",
        "ctx",
        rows,
        &[],
        NumericBudget {
            max_work: 1,
            ..budget
        },
        NumericTolerance::default(),
    );
    assert!(matches!(result, Err(NumericError::Budget(_))));
}
#[test]
fn tiny_ambiguous_data_is_not_claimed_as_general_correctness() {
    let skill = learn(vec![
        NumericExample::new(vec![0.0], 0.0),
        NumericExample::new(vec![1.0], 1.0),
    ]);
    assert_eq!(skill.predict("numeric/test", &[3.0]).unwrap(), 3.0);
    let mut monitor = NumericMonitor::new(&skill);
    assert!(matches!(
        monitor.observe(&[3.0], 9.0),
        Err(NumericError::Drift { .. })
    ));
}
#[test]
fn feedback_suspends_then_an_immutable_revision_recovers_without_forgetting_the_old_model() {
    let old = learn(calibration(2.0));
    let mut monitor = NumericMonitor::new(&old);
    monitor.observe(&[10.0], 21.0).unwrap();
    assert!(matches!(
        monitor.observe(&[10.0], 31.0),
        Err(NumericError::Drift { .. })
    ));
    assert!(monitor.suspended());
    assert!(matches!(
        monitor.predict(&[10.0]),
        Err(NumericError::Suspended)
    ));
    assert_eq!(monitor.observations(), 2);
    assert!(matches!(
        monitor.observe(&[10.0], 21.0),
        Err(NumericError::Suspended)
    ));
    let next = old
        .revise(
            "numeric/test",
            calibration(3.0),
            &[],
            NumericBudget::default(),
        )
        .unwrap();
    assert_eq!(next.parent(), Some(old.fingerprint()));
    assert_eq!(next.revision(), 2);
    assert_eq!(old.predict("numeric/test", &[10.0]).unwrap(), 21.0);
    assert_eq!(next.predict("numeric/test", &[10.0]).unwrap(), 31.0);
    let restored = NumericSkill::from_bytes(&next.to_bytes()).unwrap();
    assert_eq!(restored.parent(), Some(old.fingerprint()));
    NumericMonitor::new(&restored)
        .observe(&[10.0], 31.0)
        .unwrap();
}
#[test]
fn metadata_and_tolerances_are_validated() {
    for tolerance in [
        NumericTolerance {
            absolute: f64::NAN,
            relative: 0.0,
        },
        NumericTolerance {
            absolute: 0.0,
            relative: 1.0,
        },
        NumericTolerance {
            absolute: -1.0,
            relative: 0.0,
        },
    ] {
        assert!(matches!(
            NumericSkill::learn(
                "a",
                "ctx",
                calibration(2.0),
                &[],
                NumericBudget::default(),
                tolerance
            ),
            Err(NumericError::Invalid(_))
        ));
    }
    for budget in [
        NumericBudget {
            max_cost: 0,
            ..NumericBudget::default()
        },
        NumericBudget {
            max_terms: 0,
            ..NumericBudget::default()
        },
    ] {
        assert!(matches!(
            NumericSkill::learn(
                "a",
                "ctx",
                calibration(2.0),
                &[],
                budget,
                NumericTolerance::default()
            ),
            Err(NumericError::Invalid(_))
        ));
    }
}
#[test]
fn capsule_roundtrip_truncations_and_bit_corruption_are_checked() {
    let skill = distance2();
    let bytes = skill.to_bytes();
    let restored = NumericSkill::from_bytes(&bytes).unwrap();
    assert_eq!(restored.fingerprint(), skill.fingerprint());
    assert_eq!(restored.program(), skill.program());
    for length in 0..bytes.len() {
        assert!(NumericSkill::from_bytes(&bytes[..length]).is_err());
    }
    for index in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[index] ^= 1;
        assert!(NumericSkill::from_bytes(&bad).is_err());
    }
}
#[test]
fn signed_zero_evidence_preserves_its_exact_storage_identity() {
    let skill = learn(
        [-0.0, 1.0, -1.0, 2.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], 2.0 * x + 1.0))
            .collect(),
    );
    let restored = NumericSkill::from_bytes(&skill.to_bytes()).unwrap();
    assert_eq!(restored.fingerprint(), skill.fingerprint());
    assert_eq!(
        restored.examples()[0].input[0].to_bits(),
        (-0.0_f64).to_bits()
    );
}
fn reseal(bytes: &mut [u8]) {
    let n = bytes.len() - 8;
    let mut hash = 0xcbf29ce484222325_u64;
    for b in &bytes[..n] {
        hash = (hash ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    bytes[n..].copy_from_slice(&hash.to_le_bytes());
}
fn opcode_offset(bytes: &[u8]) -> usize {
    let mut offset = 8;
    for _ in 0..2 {
        let n = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4 + n;
    }
    offset += 4 + 1 + 8 + 16;
    let arity = bytes[offset] as usize;
    offset += 1;
    let examples = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4 + examples * (arity + 1) * 8;
    let dependencies = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4 + dependencies * 8;
    offset + 4
}
#[test]
fn forged_checksum_does_not_bypass_bytecode_validation() {
    let mut bytes = distance2().to_bytes();
    let offset = opcode_offset(&bytes);
    bytes[offset] = 255;
    reseal(&mut bytes);
    assert!(NumericSkill::from_bytes(&bytes).is_err());
    let mut bytes = distance2().to_bytes();
    bytes[offset] = 2;
    reseal(&mut bytes);
    assert!(NumericSkill::from_bytes(&bytes).is_err());
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "axon-numeric-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let path = fs::canonicalize(&self.0).unwrap();
        let root = fs::canonicalize(std::env::temp_dir()).unwrap();
        assert!(
            path.starts_with(root)
                && path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("axon-numeric-")
        );
        fs::remove_dir_all(path).unwrap();
    }
}
#[test]
fn cli_exports_a_witness_free_circuit_and_reuses_the_existing_predictor() {
    let temp = Temp::new();
    let source = temp.0.join("source.axnum");
    let skill = learn(calibration(2.0));
    skill.save_new(&source).unwrap();
    let original = fs::read(&source).unwrap();
    let destination = temp.0.join("exported");
    let args = vec![
        "circuit-export".to_owned(),
        source.to_string_lossy().into_owned(),
        "--context".into(),
        "numeric/test".into(),
        "--output".into(),
        destination.to_string_lossy().into_owned(),
    ];
    let report = axon_uic::cli::command(&args).unwrap();
    assert!(report.contains("status=circuit_exported"));
    assert!(report.contains("source_examples_omitted=5"));
    assert!(axon_uic::cli::command(&args).is_err());
    assert_eq!(fs::read(&source).unwrap(), original);
    let mut wrong = args.clone();
    wrong[3] = "wrong-context".into();
    wrong[5] = temp.0.join("not-created").to_string_lossy().into_owned();
    assert!(axon_uic::cli::command(&wrong).is_err());
    assert!(!temp.0.join("not-created").exists());
    let raw = fs::read(destination.join("circuit.axcg")).unwrap();
    let number = fs::read_to_string(destination.join("circuit-number.txt")).unwrap();
    let dna = evolution::constructive::CircuitDna::from_number(&number).unwrap();
    assert_eq!(dna.to_bytes(), raw);
    assert!(raw.len() < original.len());
    fs::remove_file(source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_axon"))
        .arg("circuit-predict")
        .arg(destination.join("circuit-number.txt"))
        .arg("123.25")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("value=247.5"));
}
#[test]
fn durable_storage_preserves_identity_and_refuses_overwrite_or_wrong_context() {
    let temp = Temp::new();
    let path = temp.0.join("skill.axnum");
    let skill = distance2();
    skill.save_new(&path).unwrap();
    assert_eq!(
        NumericSkill::open(&path, "numeric/test")
            .unwrap()
            .fingerprint(),
        skill.fingerprint()
    );
    assert!(skill.save_new(&path).is_err());
    assert!(matches!(
        NumericSkill::open(&path, "wrong"),
        Err(NumericError::Context)
    ));
    assert_eq!(fs::read(path).unwrap(), skill.to_bytes());
}
#[test]
fn cli_fits_inspects_reopens_predicts_evaluates_and_refuses_bad_inputs() {
    let temp = Temp::new();
    let path = temp.0.join("skill.axnum");
    let run = |args: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_axon"))
            .args(args)
            .output()
            .unwrap()
    };
    let output = run(&[
        "numeric-fit",
        "examples/numeric/calibration.tsv",
        "--name",
        "calibrator",
        "--context",
        "sensor/test",
        "--skill",
        path.to_str().unwrap(),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("NOT A GENERAL PROOF"));
    let output = run(&[
        "numeric-predict",
        path.to_str().unwrap(),
        "--context",
        "sensor/test",
        "--values",
        "100",
    ]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("201.000000"));
    let output = run(&[
        "numeric-eval",
        path.to_str().unwrap(),
        "examples/numeric/calibration.tsv",
        "--context",
        "sensor/test",
    ]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("correct=5"));
    let wrong = temp.0.join("wrong.tsv");
    fs::write(&wrong, "x0\ty\n3\t99\n").unwrap();
    let output = run(&[
        "numeric-eval",
        path.to_str().unwrap(),
        wrong.to_str().unwrap(),
        "--context",
        "sensor/test",
    ]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("REFUTED"));
    assert!(
        !run(&[
            "numeric-inspect",
            path.to_str().unwrap(),
            "--context",
            "wrong"
        ])
        .status
        .success()
    );
    assert!(
        !run(&[
            "numeric-predict",
            path.to_str().unwrap(),
            "--context",
            "sensor/test",
            "--values",
            "NaN"
        ])
        .status
        .success()
    );
}
