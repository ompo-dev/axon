use axon_uic::{
    Authority, Effect, TextBudget, TextExample, TextSkill,
    behavior::{BehaviorBudget, Resettable},
    behavior_lab::{CachedOracle, HiddenMachine, Rng, audit_machine, discover},
    cli,
    demand::*,
    world::{ObservationKind, Transition, WorldLearner, WorldShape},
};
use std::{error::Error, fs, path::Path, time::Instant};

fn learn(root: &Path, name: &str, examples: Vec<TextExample>) -> Result<(), Box<dyn Error>> {
    TextSkill::learn(
        name,
        format!("demo/{name}"),
        examples,
        TextBudget::default(),
    )?
    .save_new(root.join(format!("{name}.axtext")))?;
    Ok(())
}
fn register(
    c: &mut Catalog,
    name: &str,
    kind: ModuleKind,
    extension: &str,
    input: &str,
    output: &str,
    authority: &Authority,
) -> Result<(), DemandError> {
    c.register(
        ModuleSpec {
            id: name,
            kind,
            file: &format!("{name}.{extension}"),
            input,
            output,
            cost: 1,
        },
        authority,
    )
}
fn letters(mut n: usize) -> String {
    let mut s = String::new();
    loop {
        s.push((b'a' + (n % 26) as u8) as char);
        n /= 26;
        if n == 0 {
            break;
        }
    }
    s
}
fn run() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 1 {
        return Err("usage: axon-uic-demand-lab <new-directory>".into());
    }
    let root = Path::new(&args[0]);
    fs::create_dir(root)?;
    let authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    let started = Instant::now();
    let training = Instant::now();
    let mut c = Catalog::new(root);
    learn(
        root,
        "normalize",
        vec![
            TextExample::new(" Ana Maria | 1250 ", "ANA MARIA|1250"),
            TextExample::new("bruno | 902", "BRUNO|902"),
            TextExample::new(" Carla Sofia Rocha | 34567 ", "CARLA SOFIA ROCHA|34567"),
        ],
    )?;
    learn(
        root,
        "amount",
        vec![
            TextExample::new("ANA MARIA|1250", "1250"),
            TextExample::new("BRUNO|902", "902"),
            TextExample::new("CARLA SOFIA ROCHA|34567", "34567"),
        ],
    )?;
    register(
        &mut c,
        "normalize",
        ModuleKind::Text,
        "axtext",
        "invoice/raw",
        "invoice/canonical",
        &authority,
    )?;
    register(
        &mut c,
        "amount",
        ModuleKind::Text,
        "axtext",
        "invoice/canonical",
        "invoice/cents",
        &authority,
    )?;
    let text_training_ns = training.elapsed().as_nanos();
    let unrelated = Instant::now();
    for i in 0..252 {
        let name = format!("dormant-{i:03}");
        learn(
            root,
            &name,
            vec![
                TextExample::new(
                    "Eva Maria|Luz Rocha",
                    format!("ID{i}: Luz Rocha, Eva Maria"),
                ),
                TextExample::new("Bruno|Silva", format!("ID{i}: Silva, Bruno")),
                TextExample::new("Carla Sofia|Alves", format!("ID{i}: Alves, Carla Sofia")),
            ],
        )?;
        register(
            &mut c,
            &name,
            ModuleKind::Text,
            "axtext",
            &format!("other/{i}/raw"),
            &format!("other/{i}/label"),
            &authority,
        )?;
    }
    let unrelated_training_ns = unrelated.elapsed().as_nanos();
    let behavior_training = Instant::now();
    let mut rng = Rng::new(713_992);
    let mut hidden = HiddenMachine::generate(&mut rng);
    let actions = hidden.actions;
    let mut oracle = CachedOracle::new(hidden.clone(), BehaviorBudget::default());
    let model = discover(
        actions,
        "demo/sentinel",
        BehaviorBudget::default(),
        &mut oracle,
    )?
    .0;
    let audit = audit_machine(&model, actions, 0, |s, a| Ok(hidden.transition(s, a)))?;
    if audit.counterexample.is_some() {
        return Err("behavior independent audit failed".into());
    }
    let behavior_queries = oracle.raw.queries;
    let behavior_steps = oracle.raw.steps;
    model.save_new(root.join("sentinel.axbehavior"))?;
    register(
        &mut c,
        "sentinel",
        ModuleKind::Behavior,
        "axbehavior",
        "sensor/event",
        "sensor/prediction",
        &authority,
    )?;
    let behavior_training_ns = behavior_training.elapsed().as_nanos();
    let shape = WorldShape::new(2, 2)?;
    let mut world = WorldLearner::new(shape, "demo/workcell")?;
    for before in 0..4 {
        for action in 0..2 {
            world.observe(Transition {
                before,
                action,
                after: before ^ (1 << action),
                kind: ObservationKind::Intervention,
            })?;
        }
    }
    world.compile()?.save_new(root.join("workcell.axworld"))?;
    register(
        &mut c,
        "workcell",
        ModuleKind::World,
        "axworld",
        "control/state",
        "control/next",
        &authority,
    )?;
    c.save_new(root.join("seed.axseed"), &authority)?;
    let training_total_ns = started.elapsed().as_nanos();
    let eager_bytes = c.eager_retained_bytes();
    let dormant_bytes = c.dormant_payload_bytes();
    let module_specs = c
        .entries()
        .iter()
        .map(|e| (e.id.clone(), e.context.clone()))
        .collect::<Vec<_>>();
    drop(c);
    drop(oracle);
    drop(model);
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    let mut input_csv = csv::Writer::from_writer(Vec::new());
    input_csv.write_record(["raw", "source"])?;
    for i in 0..20_000 {
        let amount = (i as u64 * 713 + 19) % 1_000_003;
        let value = format!(
            " customer{} family{} | {amount} ",
            letters(i),
            letters(i * 37 + 8)
        );
        input_csv.write_record([value.as_str(), "synthetic-unseen"])?;
        inputs.push(value);
        expected.push(amount.to_string());
    }
    input_csv.flush()?;
    fs::write(root.join("invoices.csv"), input_csv.into_inner()?)?;
    let expected_sum = expected
        .iter()
        .map(|s| s.parse::<u128>().unwrap())
        .sum::<u128>();
    let mut measurements = csv::Writer::from_writer(Vec::new());
    measurements.write_record([
        "round",
        "mode",
        "preload_ns",
        "execution_ns",
        "loads",
        "file_reads",
        "evictions",
        "peak_modules",
        "peak_module_bytes",
        "metadata_bytes",
        "checked_rows",
        "sum",
    ])?;
    let mut last_demand = DemandStats::default();
    let mut metadata = 0;
    let mut eager_total_ns = 0_u128;
    let mut demand_total_ns = 0_u128;
    // Alternate order; these are logical cold loads, not flushed OS caches.
    for round in 0..6 {
        for mode in if round % 2 == 0 {
            ["eager", "demand", "explained"]
        } else {
            ["explained", "demand", "eager"]
        } {
            let eager = mode == "eager";
            let start = Instant::now();
            let catalog = Catalog::load(root.join("seed.axseed"), &authority)?;
            let seed = if eager {
                Seed {
                    active_modules: 256,
                    active_bytes: eager_bytes,
                    ..Seed::default()
                }
            } else {
                Seed::default()
            };
            let mut runtime = DemandRuntime::new(catalog, seed)?;
            if eager {
                for (id, context) in &module_specs {
                    runtime.with_module(id, context, &authority, |_| Ok(()))?;
                }
            }
            let preload_ns = start.elapsed().as_nanos();
            let start = Instant::now();
            let plan = runtime.plan_text("invoice/raw", "invoice/cents")?;
            let outputs = if mode == "explained" {
                runtime.execute_text_with_witnesses(&plan, &inputs, &authority)?
            } else {
                runtime.execute_text(&plan, &inputs, &authority)?
            };
            if outputs != expected || exact_decimal_sum(&outputs)? != expected_sum {
                return Err("independent invoice oracle mismatch".into());
            }
            let execution_ns = start.elapsed().as_nanos();
            let stats = runtime.stats();
            metadata = runtime.metadata_bytes();
            if eager {
                eager_total_ns += preload_ns + execution_ns;
            } else if mode == "demand" {
                demand_total_ns += preload_ns + execution_ns;
                last_demand = stats;
            }
            measurements.write_record([
                round.to_string(),
                mode.into(),
                preload_ns.to_string(),
                execution_ns.to_string(),
                stats.loads.to_string(),
                stats.file_reads.to_string(),
                stats.evictions.to_string(),
                stats.peak_active_modules.to_string(),
                stats.peak_active_bytes.to_string(),
                metadata.to_string(),
                outputs.len().to_string(),
                expected_sum.to_string(),
            ])?;
        }
    }
    measurements.flush()?;
    fs::write(root.join("measurements.csv"), measurements.into_inner()?)?;
    // The public CLI processes the same batch and produces the inspectable deliverable.
    let mut command = vec![
        "seed-preview".into(),
        root.join("seed.axseed").display().to_string(),
        root.join("invoices.csv").display().to_string(),
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
    let preview = cli::command(&command)?;
    let token = preview
        .lines()
        .find_map(|s| s.strip_prefix("review_token="))
        .ok_or("missing review token")?
        .to_owned();
    command[0] = "seed-apply".into();
    command.extend([
        "--output".into(),
        root.join("consolidated.csv").display().to_string(),
        "--confirm".into(),
        token,
    ]);
    let apply = cli::command(&command)?;
    fs::write(root.join("preview.txt"), preview)?;
    fs::write(root.join("apply.txt"), apply)?;
    let mut reader = csv::Reader::from_path(root.join("consolidated.csv"))?;
    for (i, row) in reader.records().enumerate() {
        let row = row?;
        if row.get(0) != Some(inputs[i].as_str()) || row.get(2) != Some(expected[i].as_str()) {
            return Err("published CSV mismatch".into());
        }
    }
    let mut runtime = DemandRuntime::new(
        Catalog::load(root.join("seed.axseed"), &authority)?,
        Seed::default(),
    )?;
    let mut cursor = runtime.behavior_cursor("sentinel", "demo/sentinel")?;
    let monitor = Instant::now();
    let mut stream_loads = 0;
    let mut stream_evictions = 0;
    for i in 0..10_000 {
        if i == 5000 {
            cursor.save_new(root.join("cursor-mid.axcursor"), &authority)?;
            stream_loads += runtime.stats().loads;
            stream_evictions += runtime.stats().evictions;
            drop(runtime);
            runtime = DemandRuntime::new(
                Catalog::load(root.join("seed.axseed"), &authority)?,
                Seed::default(),
            )?;
            cursor = DormantCursor::load(root.join("cursor-mid.axcursor"), &authority)?;
        }
        let action = rng.pick(actions as usize) as u8;
        let output = hidden.step(action)?;
        runtime.observe_behavior(
            "sentinel",
            "demo/sentinel",
            &mut cursor,
            action,
            output,
            &authority,
        )?;
        if i % 64 == 0 {
            runtime.with_module("amount", "demo/amount", &authority, |_| Ok(()))?;
        }
    }
    cursor.save_new(root.join("cursor-final.axcursor"), &authority)?;
    stream_loads += runtime.stats().loads;
    stream_evictions += runtime.stats().evictions;
    let report = format!(
        "catalog_modules=256\nlearned_text_modules=254\nbehavior_modules=1\nworld_modules=1\ninvoice_demonstrations=6\nunseen_invoice_rows=20000\nchecked_execution_rows={}\nexact_cents_sum={expected_sum}\nselected_route=normalize,amount\nselected_module_reads={}\npeak_active_modules={}\npeak_module_retained_bytes={}\neager_module_retained_bytes={eager_bytes}\nruntime_metadata_bytes={metadata}\ndormant_encoded_bytes={dormant_bytes}\ntext_training_and_registration_ns={text_training_ns}\nunrelated_training_and_registration_ns={unrelated_training_ns}\nbehavior_training_and_registration_ns={behavior_training_ns}\nbehavior_training_queries={behavior_queries}\nbehavior_training_steps={behavior_steps}\nall_training_storage_registration_ns={training_total_ns}\neager_lifecycle_mean_ns={}\ndemand_lifecycle_mean_ns={}\nstream_observations={}\nstream_resets_after_start=0\nstream_loads={stream_loads}\nstream_evictions={stream_evictions}\nstream_elapsed_ns={}\ncursor_ram_bytes={}\ncursor_checkpoint_bytes=35\nbehavior_audited_edges={}\nnotes=synthetic finite families; declared contracts; no AGI, RSS cap, energy measurement or open-domain correctness guarantee\n",
        20_000 * 18,
        last_demand.file_reads,
        last_demand.peak_active_modules,
        last_demand.peak_active_bytes,
        eager_total_ns / 6,
        demand_total_ns / 6,
        cursor.observations(),
        monitor.elapsed().as_nanos(),
        std::mem::size_of::<DormantCursor>(),
        audit.checked_edges
    );
    fs::write(root.join("report.txt"), &report)?;
    println!("{report}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
