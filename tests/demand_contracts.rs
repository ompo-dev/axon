use axon_uic::{
    Authority, Effect, TextBudget, TextExample, TextSkill,
    behavior::{BehaviorBudget, BehaviorError, BehaviorLearner, SequenceOracle},
    demand::*,
    world::{ObservationKind, Transition, WorldLearner, WorldShape},
};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "axon-demand-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn file(&self, n: &str) -> PathBuf {
        self.0.join(n)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn authority() -> Authority {
    Authority::from([Effect::ReadData, Effect::WriteData])
}
fn save_text(temp: &Temp, file: &str, name: &str, examples: &[(&str, &str)]) {
    TextSkill::learn(
        name,
        format!("test/{name}"),
        examples
            .iter()
            .map(|(i, o)| TextExample::new(*i, *o))
            .collect(),
        TextBudget::default(),
    )
    .unwrap()
    .save_new(temp.file(file))
    .unwrap();
}
fn register(c: &mut Catalog, id: &str, file: &str, input: &str, output: &str, cost: u32) {
    c.register(
        ModuleSpec {
            id,
            kind: ModuleKind::Text,
            file,
            input,
            output,
            cost,
        },
        &authority(),
    )
    .unwrap();
}
fn catalog(temp: &Temp) -> Catalog {
    save_text(
        temp,
        "normalize.axtext",
        "normalize",
        &[
            (" Ana Maria | 1250 ", "ANA MARIA|1250"),
            ("bruno | 902", "BRUNO|902"),
            (" Carla Sofia Rocha | 34567 ", "CARLA SOFIA ROCHA|34567"),
        ],
    );
    save_text(
        temp,
        "amount.axtext",
        "amount",
        &[
            ("ANA MARIA|1250", "1250"),
            ("BRUNO|902", "902"),
            ("CARLA SOFIA ROCHA|34567", "34567"),
        ],
    );
    let mut c = Catalog::new(&temp.0);
    register(
        &mut c,
        "normalize",
        "normalize.axtext",
        "invoice/raw",
        "invoice/canonical",
        2,
    );
    register(
        &mut c,
        "amount",
        "amount.axtext",
        "invoice/canonical",
        "invoice/cents",
        3,
    );
    c.save_new(temp.file("seed.axseed"), &authority()).unwrap();
    Catalog::load(temp.file("seed.axseed"), &authority()).unwrap()
}
fn runtime(c: Catalog) -> DemandRuntime {
    DemandRuntime::new(c, Seed::default()).unwrap()
}

#[test]
fn composes_new_goal_and_keeps_one_module_resident() {
    let t = Temp::new();
    let mut r = runtime(catalog(&t));
    let plan = r.plan_text("invoice/raw", "invoice/cents").unwrap();
    assert_eq!(plan.modules, ["normalize", "amount"]);
    assert_eq!(plan.cost, 5);
    assert_eq!(r.stats().file_reads, 0);
    let outputs = r
        .execute_text(
            &plan,
            &[" dora luisa | 8751 ".into(), "  Edson | 220 ".into()],
            &authority(),
        )
        .unwrap();
    assert_eq!(outputs, ["8751", "220"]);
    assert_eq!(exact_decimal_sum(&outputs).unwrap(), 8971);
    assert_eq!(r.stats().loads, 2);
    assert_eq!(r.stats().evictions, 1);
    assert_eq!(r.stats().peak_active_modules, 1);
    assert_eq!(r.active_ids(), ["amount"]);
    assert!(r.stats().peak_active_bytes <= r.seed().active_bytes);
}
#[test]
fn missing_dormant_module_is_not_opened_by_catalog_or_planner() {
    let t = Temp::new();
    let c = catalog(&t);
    drop(c);
    fs::remove_file(t.file("normalize.axtext")).unwrap();
    let mut r = runtime(Catalog::load(t.file("seed.axseed"), &authority()).unwrap());
    let p = r.plan_text("invoice/canonical", "invoice/cents").unwrap();
    assert_eq!(
        r.execute_text(&p, &["FELIPE|817".into()], &authority())
            .unwrap(),
        ["817"]
    );
    assert_eq!(r.stats().file_reads, 1);
}
#[test]
fn oversize_module_refuses_before_io_and_does_not_evict() {
    let t = Temp::new();
    let c = catalog(&t);
    let limit = c.get("amount").unwrap().retained_bytes;
    assert!(c.get("normalize").unwrap().retained_bytes > limit);
    let mut r = DemandRuntime::new(
        c,
        Seed {
            active_bytes: limit,
            ..Seed::default()
        },
    )
    .unwrap();
    r.with_module("amount", "test/amount", &authority(), |_| Ok(()))
        .unwrap();
    assert!(matches!(
        r.with_module("normalize", "test/normalize", &authority(), |_| Ok(())),
        Err(DemandError::Budget(_))
    ));
    assert_eq!(r.active_ids(), ["amount"]);
    assert_eq!(r.stats().file_reads, 1);
}
#[test]
fn lru_eviction_is_deterministic_and_hot_reuse_does_not_read() {
    let t = Temp::new();
    let mut c = catalog(&t);
    save_text(
        &t,
        "extra.axtext",
        "extra",
        &[("ABC", "abc"), ("DEF", "def")],
    );
    register(&mut c, "extra", "extra.axtext", "x", "y", 1);
    let mut r = DemandRuntime::new(
        c,
        Seed {
            active_modules: 2,
            ..Seed::default()
        },
    )
    .unwrap();
    for id in ["normalize", "amount", "normalize", "extra"] {
        r.with_module(id, &format!("test/{id}"), &authority(), |_| Ok(()))
            .unwrap();
    }
    assert_eq!(r.active_ids(), ["normalize", "extra"]);
    assert_eq!(r.stats().loads, 3);
    assert_eq!(r.stats().hits, 1);
    assert_eq!(r.stats().evictions, 1);
    r.unload_all();
    assert_eq!(r.stats().active_bytes, 0);
    assert_eq!(r.stats().active_modules, 0);
}
#[test]
fn context_and_authority_fail_before_opening_files() {
    let t = Temp::new();
    let mut r = runtime(catalog(&t));
    assert!(matches!(
        r.with_module("normalize", "wrong", &authority(), |_| Ok(())),
        Err(DemandError::Context)
    ));
    assert!(matches!(
        r.with_module("normalize", "test/normalize", &Authority::from([]), |_| Ok(
            ()
        )),
        Err(DemandError::Unauthorized(Effect::ReadData))
    ));
    assert_eq!(r.stats().file_reads, 0);
}
#[test]
fn changed_and_corrupt_payloads_are_refused_without_rewriting() {
    let t = Temp::new();
    let mut r = runtime(catalog(&t));
    let path = t.file("amount.axtext");
    let mut b = fs::read(&path).unwrap();
    b[30] ^= 1;
    fs::write(&path, &b).unwrap();
    assert!(matches!(
        r.with_module("amount", "test/amount", &authority(), |_| Ok(())),
        Err(DemandError::Identity)
    ));
    assert_eq!(fs::read(&path).unwrap(), b);
    assert_eq!(r.stats().loads, 0);
    assert_eq!(r.stats().active_bytes, 0);
}
#[test]
fn catalog_truncation_and_each_single_byte_corruption_are_refused() {
    let t = Temp::new();
    let _ = catalog(&t);
    let b = fs::read(t.file("seed.axseed")).unwrap();
    for i in 0..b.len() {
        let mut changed = b.clone();
        changed[i] ^= 1;
        fs::write(t.file("probe.axseed"), &changed).unwrap();
        assert!(
            Catalog::load(t.file("probe.axseed"), &authority()).is_err(),
            "byte {i}"
        );
        fs::write(t.file("probe.axseed"), &b[..i]).unwrap();
        assert!(Catalog::load(t.file("probe.axseed"), &authority()).is_err());
    }
}
#[test]
fn cheaper_route_and_cycles_are_resolved_using_metadata_only() {
    let t = Temp::new();
    let mut c = catalog(&t);
    register(
        &mut c,
        "expensive",
        "amount.axtext",
        "invoice/raw",
        "invoice/cents",
        99,
    );
    register(
        &mut c,
        "cycle",
        "amount.axtext",
        "invoice/canonical",
        "invoice/raw",
        1,
    );
    let r = runtime(c);
    assert_eq!(
        r.plan_text("invoice/raw", "invoice/cents").unwrap().modules,
        ["normalize", "amount"]
    );
    assert!(matches!(
        r.plan_text("invoice/raw", "unknown"),
        Err(DemandError::NoRoute)
    ));
    assert_eq!(r.stats().file_reads, 0);
}
#[test]
fn stale_or_modified_plan_cannot_execute() {
    let t = Temp::new();
    let mut r = runtime(catalog(&t));
    let mut p = r.plan_text("invoice/raw", "invoice/cents").unwrap();
    p.cost = 1;
    assert!(r.execute_text(&p, &[], &authority()).is_err());
    assert_eq!(r.stats().file_reads, 0);
    p.cost = 5;
    p.modules.reverse();
    assert!(r.execute_text(&p, &[], &authority()).is_err());
}
#[test]
fn ambiguous_stage_refuses_instead_of_selecting_an_arbitrary_program() {
    let t = Temp::new();
    save_text(
        &t,
        "surname.axtext",
        "surname",
        &[("Ana Silva", "Silva"), ("Bruno Costa", "Costa")],
    );
    let mut c = Catalog::new(&t.0);
    register(&mut c, "surname", "surname.axtext", "name", "surname", 1);
    let mut r = runtime(c);
    let p = r.plan_text("name", "surname").unwrap();
    assert!(matches!(
        r.execute_text(&p, &["Carla Sofia Rocha".into()], &authority()),
        Err(DemandError::NeedsExample { row: 1, .. })
    ));
}
#[test]
fn row_value_batch_and_plan_budgets_are_enforced() {
    let t = Temp::new();
    let mut r = DemandRuntime::new(
        catalog(&t),
        Seed {
            max_rows: 1,
            max_value_bytes: 30,
            batch_bytes: 40,
            ..Seed::default()
        },
    )
    .unwrap();
    let p = r.plan_text("invoice/raw", "invoice/cents").unwrap();
    assert!(
        r.execute_text(&p, &["a".into(), "b".into()], &authority())
            .is_err()
    );
    assert!(r.execute_text(&p, &["x".repeat(31)], &authority()).is_err());
    assert!(
        r.execute_text(&p, &["dora | 817".into()], &authority())
            .is_err()
    );
    let r = DemandRuntime::new(
        catalog_again(&t),
        Seed {
            max_plan_cost: 4,
            ..Seed::default()
        },
    )
    .unwrap();
    assert!(matches!(
        r.plan_text("invoice/raw", "invoice/cents"),
        Err(DemandError::NoRoute)
    ));
}
fn catalog_again(t: &Temp) -> Catalog {
    Catalog::load(t.file("seed.axseed"), &authority()).unwrap()
}
#[test]
fn empty_batch_never_activates_a_learned_module() {
    let t = Temp::new();
    let mut r = runtime(catalog(&t));
    let p = r.plan_text("invoice/raw", "invoice/cents").unwrap();
    assert!(r.execute_text(&p, &[], &authority()).unwrap().is_empty());
    assert_eq!(r.stats().file_reads, 0);
}
#[test]
fn registration_refuses_device_names_and_nonportable_paths_before_io() {
    let t = Temp::new();
    let c = Catalog::new(&t.0);
    for file in [
        "NUL",
        "aux.axtext",
        "COM1.axtext",
        "lpt9.axtext",
        "../x",
        "C:\\x",
        ".hidden",
        "trailing.",
    ] {
        assert!(
            c.check_registration(&ModuleSpec {
                id: "x",
                kind: ModuleKind::Text,
                file,
                input: "a",
                output: "b",
                cost: 1
            })
            .is_err(),
            "{file}"
        );
    }
}
#[test]
fn exact_reducer_never_rounds_or_silently_wraps() {
    assert_eq!(
        exact_decimal_sum(&[u128::MAX.to_string(), "0".into()]).unwrap(),
        u128::MAX
    );
    for values in [
        vec!["-1".into()],
        vec!["12.5".into()],
        vec![" 1".into()],
        vec![u128::MAX.to_string(), "1".into()],
    ] {
        assert!(exact_decimal_sum(&values).is_err());
    }
}
struct Toggle;
impl SequenceOracle for Toggle {
    fn query(&mut self, word: &[u8]) -> Result<Vec<u16>, BehaviorError> {
        let mut state = false;
        Ok(word
            .iter()
            .map(|&a| {
                if a == 0 {
                    state = !state;
                }
                u16::from(state)
            })
            .collect())
    }
}
fn behavior_catalog(t: &Temp) -> Catalog {
    let mut learner = BehaviorLearner::new(2, "test/toggle", BehaviorBudget::default()).unwrap();
    let model = learner.hypothesize(&mut Toggle).unwrap();
    model.save_new(t.file("toggle.axbehavior")).unwrap();
    let mut c = catalog(t);
    c.register(
        ModuleSpec {
            id: "toggle",
            kind: ModuleKind::Behavior,
            file: "toggle.axbehavior",
            input: "event",
            output: "prediction",
            cost: 1,
        },
        &authority(),
    )
    .unwrap();
    c
}
#[test]
fn behavior_state_survives_eviction_and_process_style_checkpoint() {
    let t = Temp::new();
    let c = behavior_catalog(&t);
    c.save_new(t.file("full.axseed"), &authority()).unwrap();
    let mut r = runtime(c);
    let mut cursor = r.behavior_cursor("toggle", "test/toggle").unwrap();
    r.observe_behavior("toggle", "test/toggle", &mut cursor, 0, 1, &authority())
        .unwrap();
    let state = cursor.state();
    assert_ne!(state, 0);
    r.with_module("amount", "test/amount", &authority(), |_| Ok(()))
        .unwrap();
    cursor
        .save_new(t.file("cursor.axcursor"), &authority())
        .unwrap();
    drop(r);
    let mut r = runtime(Catalog::load(t.file("full.axseed"), &authority()).unwrap());
    let mut cursor = DormantCursor::load(t.file("cursor.axcursor"), &authority()).unwrap();
    assert_eq!(cursor.state(), state);
    let plan = r
        .plan_behavior(
            "toggle",
            "test/toggle",
            &cursor,
            0,
            &BTreeSet::new(),
            &authority(),
        )
        .unwrap();
    assert_eq!(plan, Some(vec![0]));
    r.observe_behavior("toggle", "test/toggle", &mut cursor, 0, 0, &authority())
        .unwrap();
    assert_eq!(cursor.observations(), 2);
}
#[test]
fn drift_suspends_cursor_and_remains_suspended_after_reload() {
    let t = Temp::new();
    let mut r = runtime(behavior_catalog(&t));
    let mut cursor = r.behavior_cursor("toggle", "test/toggle").unwrap();
    assert!(matches!(
        r.observe_behavior("toggle", "test/toggle", &mut cursor, 0, 99, &authority()),
        Err(DemandError::Drift {
            expected: 1,
            actual: 99
        })
    ));
    assert!(cursor.suspended());
    assert_eq!(cursor.state(), 0);
    cursor
        .save_new(t.file("cursor.axcursor"), &authority())
        .unwrap();
    let cursor = DormantCursor::load(t.file("cursor.axcursor"), &authority()).unwrap();
    assert!(matches!(
        r.plan_behavior(
            "toggle",
            "test/toggle",
            &cursor,
            0,
            &BTreeSet::new(),
            &authority()
        ),
        Err(DemandError::Suspended)
    ));
}
#[test]
fn cursor_cannot_be_rebound_to_a_different_model() {
    let t = Temp::new();
    let mut r = runtime(behavior_catalog(&t));
    let mut cursor = r.behavior_cursor("toggle", "test/toggle").unwrap();
    assert!(matches!(
        r.observe_behavior("amount", "test/amount", &mut cursor, 0, 1, &authority()),
        Err(DemandError::Identity)
    ));
}
#[test]
fn learned_world_model_can_also_be_loaded_then_unloaded() {
    let t = Temp::new();
    let shape = WorldShape::new(1, 1).unwrap();
    let mut w = WorldLearner::new(shape, "test/world").unwrap();
    for before in 0..2 {
        w.observe(Transition {
            before,
            action: 0,
            after: before ^ 1,
            kind: ObservationKind::Intervention,
        })
        .unwrap();
    }
    w.compile()
        .unwrap()
        .save_new(t.file("world.axworld"))
        .unwrap();
    let mut c = Catalog::new(&t.0);
    c.register(
        ModuleSpec {
            id: "world",
            kind: ModuleKind::World,
            file: "world.axworld",
            input: "state",
            output: "state-next",
            cost: 1,
        },
        &authority(),
    )
    .unwrap();
    let mut r = runtime(c);
    r.with_module("world", "test/world", &authority(), |m| {
        let Module::World(w) = m else { panic!() };
        assert_eq!(w.predict(0, 0).unwrap(), 1);
        Ok(())
    })
    .unwrap();
    r.unload_all();
    assert_eq!(r.stats().active_bytes, 0);
}
#[test]
fn catalog_and_cursor_publication_never_clobber_existing_files() {
    let t = Temp::new();
    let c = catalog(&t);
    let old = fs::read(t.file("seed.axseed")).unwrap();
    assert!(c.save_new(t.file("seed.axseed"), &authority()).is_err());
    assert_eq!(fs::read(t.file("seed.axseed")).unwrap(), old);
    let mut c = Catalog::new(&t.0);
    assert!(
        c.register(
            ModuleSpec {
                id: "bad",
                kind: ModuleKind::Text,
                file: "../amount.axtext",
                input: "x",
                output: "y",
                cost: 1
            },
            &authority()
        )
        .is_err()
    );
}

#[cfg(feature = "text-workflows")]
#[test]
fn composed_csv_requires_bound_confirmation_and_preserves_originals() {
    let t = Temp::new();
    let _ = catalog(&t);
    fs::write(
        t.file("input.csv"),
        "raw,note\n dora luisa | 8751 ,original\n edson | 220 ,second\n",
    )
    .unwrap();
    let mut args = vec![
        "seed-preview".to_owned(),
        t.file("seed.axseed").display().to_string(),
        t.file("input.csv").display().to_string(),
        "--from".into(),
        "invoice/raw".into(),
        "--goal".into(),
        "invoice/cents".into(),
        "--column".into(),
        "raw".into(),
        "--result-column".into(),
        "cents".into(),
        "--reduce".into(),
        "sum".into(),
    ];
    let preview = axon_uic::cli::command(&args).unwrap();
    assert!(preview.contains("exact_unsigned_sum=8971"));
    let token = preview
        .lines()
        .find_map(|s| s.strip_prefix("review_token="))
        .unwrap()
        .to_owned();
    args[0] = "seed-apply".into();
    args.extend([
        "--output".into(),
        t.file("output.csv").display().to_string(),
        "--confirm".into(),
        "wrong".into(),
    ]);
    assert!(axon_uic::cli::command(&args).is_err());
    assert!(!t.file("output.csv").exists());
    *args.last_mut().unwrap() = token;
    axon_uic::cli::command(&args).unwrap();
    let result = fs::read_to_string(t.file("output.csv")).unwrap();
    assert!(result.contains("original,8751"));
    assert!(axon_uic::cli::command(&args).is_err());
    assert_eq!(
        fs::read_to_string(t.file("input.csv")).unwrap(),
        "raw,note\n dora luisa | 8751 ,original\n edson | 220 ,second\n"
    );
}

#[cfg(feature = "text-workflows")]
#[test]
fn teaching_adds_missing_capability_and_feedback_refines_immutable_versions() {
    let t = Temp::new();
    Catalog::new(&t.0)
        .save_new(t.file("empty.axseed"), &authority())
        .unwrap();
    fs::write(
        t.file("first.csv"),
        "input,output\nAna Silva,Silva\nBruno Costa,Costa\n",
    )
    .unwrap();
    let teach = |old: &str, examples: &str, module: &str, output: &str, context: &str| {
        axon_uic::cli::command(&vec![
            "seed-teach".into(),
            t.file(old).display().to_string(),
            t.file(examples).display().to_string(),
            "--id".into(),
            "surname".into(),
            "--from".into(),
            "name".into(),
            "--goal".into(),
            "surname".into(),
            "--context".into(),
            context.into(),
            "--module".into(),
            t.file(module).display().to_string(),
            "--output".into(),
            t.file(output).display().to_string(),
        ])
    };
    teach(
        "empty.axseed",
        "first.csv",
        "surname-v1.axtext",
        "v1.axseed",
        "names",
    )
    .unwrap();
    let before = fs::read(t.file("v1.axseed")).unwrap();
    let old_model = fs::read(t.file("surname-v1.axtext")).unwrap();
    let mut old = runtime(Catalog::load(t.file("v1.axseed"), &authority()).unwrap());
    let p = old.plan_text("name", "surname").unwrap();
    assert!(matches!(
        old.execute_text(&p, &["Carla Sofia Rocha".into()], &authority()),
        Err(DemandError::NeedsExample { .. })
    ));
    fs::write(
        t.file("feedback.csv"),
        "input,output\nCarla Sofia Rocha,Rocha\n",
    )
    .unwrap();
    assert!(
        teach(
            "v1.axseed",
            "feedback.csv",
            "wrong.axtext",
            "wrong.axseed",
            "wrong-context"
        )
        .is_err()
    );
    assert!(!t.file("wrong.axtext").exists());
    let report = teach(
        "v1.axseed",
        "feedback.csv",
        "surname-v2.axtext",
        "v2.axseed",
        "names",
    )
    .unwrap();
    assert!(report.contains("revision=2"));
    let mut new = runtime(Catalog::load(t.file("v2.axseed"), &authority()).unwrap());
    assert!(matches!(
        new.execute_text(&p, &["Dora Luisa Maria Melo".into()], &authority()),
        Err(DemandError::Identity)
    ));
    let plan = new.plan_text("name", "surname").unwrap();
    assert_eq!(
        new.execute_text(&plan, &["Dora Luisa Maria Melo".into()], &authority())
            .unwrap(),
        ["Melo"]
    );
    assert_eq!(fs::read(t.file("v1.axseed")).unwrap(), before);
    assert_eq!(fs::read(t.file("surname-v1.axtext")).unwrap(), old_model);
}
#[cfg(feature = "text-workflows")]
#[test]
fn manifest_build_and_goal_plan_work_through_public_cli() {
    let t = Temp::new();
    let _ = catalog(&t);
    fs::write(t.file("modules.tsv"),"id\tkind\tfile\tinput\toutput\tcost\nnormalize\ttext\tnormalize.axtext\tinvoice/raw\tinvoice/canonical\t2\namount\ttext\tamount.axtext\tinvoice/canonical\tinvoice/cents\t3\n").unwrap();
    axon_uic::cli::command(&[
        "seed-build".into(),
        t.file("modules.tsv").display().to_string(),
        t.file("built.axseed").display().to_string(),
    ])
    .unwrap();
    let p = axon_uic::cli::command(&[
        "seed-plan".into(),
        t.file("built.axseed").display().to_string(),
        "--from".into(),
        "invoice/raw".into(),
        "--goal".into(),
        "invoice/cents".into(),
    ])
    .unwrap();
    assert!(p.contains("route=[\"normalize\", \"amount\"]"));
    assert!(p.contains("module_file_reads=0"));
}

#[cfg(feature = "text-workflows")]
#[test]
fn bare_filenames_work_without_changing_the_test_process_current_directory() {
    let t = Temp::new();
    let _ = catalog(&t);
    fs::write(t.file("modules.tsv"),"id\tkind\tfile\tinput\toutput\tcost\nnormalize\ttext\tnormalize.axtext\tinvoice/raw\tinvoice/canonical\t1\n").unwrap();
    fs::write(
        t.file("feedback.csv"),
        "input,output\ndora luisa | 8751,DORA LUISA|8751\n",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .current_dir(&t.0)
        .args(["seed-build", "modules.tsv", "bare.axseed"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_axon"))
        .current_dir(&t.0)
        .args([
            "seed-teach",
            "bare.axseed",
            "feedback.csv",
            "--id",
            "normalize",
            "--from",
            "invoice/raw",
            "--goal",
            "invoice/canonical",
            "--context",
            "test/normalize",
            "--module",
            "next.axtext",
            "--output",
            "next.axseed",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(t.file("next.axtext").exists());
    assert!(t.file("next.axseed").exists());
}
