#[allow(dead_code)]
#[path = "../src/bin/controller_support.rs"]
mod support;
use axon_uic::{
    Authority, Effect,
    demand::{controller::*, genome::*, *},
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
            "axon-genome-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!p.exists());
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let p = self.0.canonicalize().unwrap();
        assert!(p.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        fs::remove_dir_all(p).unwrap();
    }
}
fn fixture(root: &Temp) -> (Genome, DemandRuntime, Authority) {
    let auth = Authority::from([Effect::ReadData, Effect::WriteData, Effect::Simulate]);
    support::bootstrap(&root.0.join("source"), support::DEV_SEED, &auth).unwrap();
    let mut runtime = support::runtime(
        support::catalog(&root.0.join("source"), 0, &auth).unwrap(),
        2,
    )
    .unwrap();
    let p = Procedure::discover(
        support::GOALS[1],
        support::traces(1, 7139161, 3),
        support::traces(1, 37918713, 2),
        &mut runtime,
        &auth,
        CompositionBudget::default(),
    )
    .unwrap()
    .procedure
    .unwrap();
    let g = Genome::capture(
        "genome/test",
        &mut runtime,
        CompositionBudget::default(),
        &[p],
        65536,
        &auth,
    )
    .unwrap();
    (g, runtime, auth)
}
#[test]
fn decimal_birth_is_self_contained_typed_and_does_not_inherit_live_state() {
    let root = Temp::new();
    let (g, mut runtime, auth) = fixture(&root);
    let number = g.number();
    assert!(number.len() > 10000 && number.bytes().all(|b| b.is_ascii_digit()));
    let copy = Genome::from_number(&number).unwrap();
    assert_eq!(g.to_bytes(), copy.to_bytes());
    // The source is moved away; birth receives only the complete number.
    fs::rename(root.0.join("source"), root.0.join("not-available-to-birth")).unwrap();
    let mut first = copy
        .birth(&root.0.join("first"), BirthLimits::default(), &auth)
        .unwrap();
    assert!(root.0.join("first/birth.receipt").exists());
    assert_eq!(
        first
            .controller
            .propose(support::GOALS[1], 1, "reading|1,2", &auth)
            .unwrap()
            .value,
        0
    );
    assert_eq!(
        first
            .controller
            .propose(support::GOALS[1], 2, "reading|50,50", &auth)
            .unwrap()
            .value,
        1
    );
    let mut second = copy
        .birth(&root.0.join("second"), BirthLimits::default(), &auth)
        .unwrap();
    assert_eq!(second.controller.pending_count(), 0);
    assert_eq!(
        second
            .controller
            .propose(support::GOALS[1], 1, "reading|50,50", &auth)
            .unwrap()
            .value,
        0
    );
    assert!(
        copy.birth(&root.0.join("second"), BirthLimits::default(), &auth)
            .is_err()
    );
    // An old runtime still pointing at the moved source cannot consolidate by guessing.
    runtime.unload_all();
    assert!(
        g.consolidate(&g.procedures().unwrap(), &mut runtime, &auth)
            .is_err()
    );
    let inherited = copy.offspring(128).unwrap();
    assert_eq!(inherited.parent(), copy.fingerprint());
    assert_eq!(inherited.generation(), 1);
    assert_eq!(
        inherited.procedures().unwrap()[0].to_bytes(),
        copy.procedures().unwrap()[0].to_bytes()
    );
    assert_eq!(
        inherited.composition().max_work,
        copy.composition().max_work
    );
}
#[test]
fn malformed_numbers_semantics_ceiling_authority_and_publication_fail_closed() {
    let root = Temp::new();
    let (g, _, auth) = fixture(&root);
    let b = g.to_bytes();
    for i in 0..b.len() {
        assert!(Genome::from_bytes(&b[..i]).is_err());
    }
    for i in (0..b.len()).step_by(13) {
        let mut bad = b.clone();
        bad[i] ^= 1;
        assert!(Genome::from_bytes(&bad).is_err());
    }
    for text in ["", "1", "001", "1-00", "1.000", "1e123"] {
        assert!(Genome::from_number(text).is_err());
    }
    assert!(Genome::from_number(&(g.number() + "\n")).is_err());
    let mut number = g.number();
    number.replace_range(1..4, "999");
    assert!(Genome::from_number(&number).is_err());
    let mut bad = b.clone();
    let n = u16::from_le_bytes(b[8..10].try_into().unwrap()) as usize;
    // A resealed invalid policy is not admitted merely because its checksum is valid.
    bad[10 + n + 9 * 8..10 + n + 10 * 8].copy_from_slice(&0u64.to_le_bytes());
    let end = bad.len() - 8;
    let seal = bad[..end].iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    });
    bad[end..].copy_from_slice(&seal.to_le_bytes());
    assert!(Genome::from_bytes(&bad).is_err());
    let denied = Authority::from([Effect::ReadData]);
    let target = root.0.join("denied");
    assert!(g.birth(&target, BirthLimits::default(), &denied).is_err());
    assert!(!target.exists());
    let limited = BirthLimits {
        bytes: g.encoded_bytes() - 1,
        ..BirthLimits::default()
    };
    assert!(g.birth(&target, limited, &auth).is_err());
    assert!(!target.exists());
    let path = root.0.join("seed.axnum");
    g.save_number_new(&path, &auth).unwrap();
    assert!(g.save_number_new(&path, &auth).is_err());
    assert_eq!(
        Genome::open_number(&path, &auth).unwrap().fingerprint(),
        g.fingerprint()
    );
}

#[cfg(feature = "text-workflows")]
#[test]
fn portable_seed_cli_applies_temporal_procedures_without_labels_or_source_files() {
    let root = Temp::new();
    let (g, _, auth) = fixture(&root);
    let number = root.0.join("genome.axnum");
    g.save_number_new(&number, &auth).unwrap();
    fs::rename(root.0.join("source"), root.0.join("unavailable")).unwrap();
    let input = root.0.join("events.tsv");
    let output = root.0.join("predictions.tsv");
    let born = root.0.join("cli-birth");
    fs::write(
        &input,
        "goal\tinput\ncontroller/goal-1\treading|1,2\ncontroller/goal-1\treading|50,50\n",
    )
    .unwrap();
    let args = vec![
        "genome-apply".into(),
        number.display().to_string(),
        born.display().to_string(),
        input.display().to_string(),
        output.display().to_string(),
    ];
    let report = axon_uic::cli::command(&args).unwrap();
    assert!(report.contains("validation=unscored"));
    assert_eq!(
        fs::read_to_string(&output).unwrap(),
        "goal\tinput\tprediction\ncontroller/goal-1\treading|1,2\t0\ncontroller/goal-1\treading|50,50\t1\n"
    );
    assert!(axon_uic::cli::command(&args).is_err());
    assert_eq!(
        Genome::open_number(&number, &auth).unwrap().fingerprint(),
        g.fingerprint()
    );
}
#[cfg(feature = "text-workflows")]
#[test]
fn real_population_workers_reopen_numbers_and_archive_partial_capabilities() {
    let root = Temp::new();
    let target = root.0.join("lab");
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_axon-uic-genome-lab"))
        .arg(&target)
        .args(["84371921", "100", "development", "4", "2"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let summary = fs::read_to_string(target.join("summary.txt")).unwrap();
    assert!(summary.contains("candidates=8\n"));
    assert!(summary.contains("composition_attempts=48\n"));
    assert!(target.join("birth-0/birth.receipt").exists());
    assert!(target.join("frozen.csv").exists());
    let g = Genome::open_number(
        &target.join("candidate-0.axnum"),
        &Authority::from([Effect::ReadData]),
    )
    .unwrap();
    assert_eq!(g.module_count(), 7);
    assert!(g.procedure_count() > 0);
    let ledger = fs::read_to_string(target.join("population.csv")).unwrap();
    assert_eq!(ledger.lines().count(), 9);
}

#[test]
fn executable_circuit_birth_without_witnesses() {
    use axon_uic::numeric::{
        NumericBudget, NumericExample, NumericSkill, NumericTolerance,
        evolution::constructive::{CONTEXT, CircuitDna},
    };
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    if let Some(root) = std::env::var_os("AXON_CIRCUIT_TEST_CHILD") {
        let root = PathBuf::from(root);
        let resolved = root.canonicalize().unwrap();
        assert!(resolved.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("axon-genome-")
        );
        let number = root.join("circuit.axnum");
        let before = fs::read(&number).unwrap();
        let genome = Genome::open_number(&number, &auth).unwrap();
        let mut born = genome
            .birth(&root.join("child-birth"), BirthLimits::default(), &auth)
            .unwrap();
        assert_eq!(born.controller.runtime().stats().active_modules, 0);
        let runtime = born.controller.runtime_mut();
        let mut called = false;
        assert!(
            runtime
                .with_module("acquired", "wrong-context", &auth, |_| {
                    called = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!called);
        assert!(
            runtime
                .with_module("acquired", CONTEXT, &Authority::from([]), |_| {
                    called = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!called);
        for input in [vec![], vec![f64::NAN], vec![f64::INFINITY], vec![1.0; 5]] {
            assert!(
                runtime
                    .predict_numeric("acquired", CONTEXT, &input, &auth)
                    .is_err()
            );
        }
        assert!(
            runtime
                .predict_numeric("acquired", "wrong-context", &[1.0], &auth)
                .is_err()
        );
        assert!(
            runtime
                .predict_numeric("acquired", CONTEXT, &[1.0], &Authority::from([]))
                .is_err()
        );
        assert_eq!(runtime.stats().loads, 0);
        assert_eq!(
            runtime
                .predict_numeric("acquired", CONTEXT, &[123.25], &auth)
                .unwrap(),
            247.5
        );
        runtime
            .with_module("acquired", CONTEXT, &auth, |module| {
                let Module::Circuit(execution) = module else {
                    panic!("wrong born module")
                };
                for value in [-1000.25, -0.0, 0.125, 93.5, 99999.0] {
                    assert_eq!(execution.evaluate(&[value]).unwrap(), 2.0 * value + 1.0);
                }
                assert!(execution.evaluate(&[]).is_err());
                assert!(execution.evaluate(&[f64::MAX]).is_err());
                assert!(execution.retained_bytes() < 1024);
                assert!(execution.scratch_bytes() > 0);
                Ok(())
            })
            .unwrap();
        let stats = runtime.stats();
        assert_eq!(stats.loads, 1);
        assert_eq!(stats.active_modules, 1);
        assert!(stats.active_bytes <= 1024);
        assert_eq!(fs::read(number).unwrap(), before);
        return;
    }
    let root = Temp::new();
    let source = root.0.join("source");
    fs::create_dir(&source).unwrap();
    let skill = NumericSkill::learn(
        "private-witness-name",
        "private/witness-context",
        [-7.0, 0.0, 0.25, 3.0, 16.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], 2.0 * x + 1.0))
            .collect(),
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )
    .unwrap();
    let dna = CircuitDna::from_program(skill.program()).unwrap();
    fs::write(source.join("program.axcg"), dna.to_bytes()).unwrap();
    let mut catalog = Catalog::new(&source);
    catalog
        .register(
            ModuleSpec {
                id: "acquired",
                kind: ModuleKind::Circuit,
                file: "program.axcg",
                input: "numbers",
                output: "number",
                cost: skill.program().instructions() as u32,
            },
            &auth,
        )
        .unwrap();
    let mut runtime = DemandRuntime::new(
        catalog,
        Seed {
            active_modules: 1,
            active_bytes: 1024,
            ..Seed::default()
        },
    )
    .unwrap();
    let genome = Genome::capture(
        "circuit/test",
        &mut runtime,
        CompositionBudget::default(),
        &[],
        65536,
        &auth,
    )
    .unwrap();
    genome
        .save_number_new(&root.0.join("circuit.axnum"), &auth)
        .unwrap();
    fs::remove_file(source.join("program.axcg")).unwrap();
    drop(skill);
    drop(runtime);
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "executable_circuit_birth_without_witnesses",
            "--nocapture",
        ])
        .env("AXON_CIRCUIT_TEST_CHILD", &root.0)
        .output()
        .unwrap();
    assert!(
        child.status.success(),
        "child failed: {} {}",
        String::from_utf8_lossy(&child.stdout),
        String::from_utf8_lossy(&child.stderr)
    );
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .arg("genome-number")
        .arg(root.0.join("circuit.axnum"))
        .arg(root.0.join("cli-birth"))
        .args(["acquired", CONTEXT, "123.25"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(text.contains("value=247.5"));
    assert!(text.contains("module_loads=1"));
    assert!(text.contains("teacher_queries=0"));
}
