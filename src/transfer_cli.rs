use crate::{
    Authority, Effect, EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue,
    behavior::*,
    behavior_lab::{Rng, audit_machine},
    cli::CliError,
    live_store::checksum,
    text_skill::{publish_new, read_bounded},
    transfer::*,
    transfer_lab::*,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write,
    fs,
    path::Path,
    time::Instant,
};
pub(crate) const USAGE: &str = "\n  axon transfer-demo <new-directory> [--payload <document.md>]\n  axon transfer-replay <demo-directory>";
const SOURCE_FILES: [&str; 3] = [
    "source-control.axbehavior",
    "source-protocol.axbehavior",
    "source-workflow.axbehavior",
];
const CHANNEL: &str = "representation/adaptation/v1";
impl From<TransferError> for CliError {
    fn from(e: TransferError) -> Self {
        Self::Transfer(e)
    }
}
fn atlas_digest(models: &[BehaviorModel]) -> u64 {
    let bytes = models
        .iter()
        .flat_map(|m| m.fingerprint().to_le_bytes())
        .collect::<Vec<_>>();
    checksum(&bytes)
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let path = args.get(1).ok_or(CliError::Usage)?;
    match args[0].as_str() {
        "transfer-demo" if args.len() == 2 || (args.len() == 4 && args[2] == "--payload") => {
            let payload = if let Some(p) = args.get(3) {
                read_bounded(Path::new(p), 65_536).map_err(BehaviorError::from)?
            } else {
                include_bytes!("../examples/behavior_document.md").to_vec()
            };
            if payload.len() < 20
                || !payload.starts_with(b"# ")
                || payload.contains(&0)
                || std::str::from_utf8(&payload).is_err()
            {
                return Err(TransferError::Invalid(
                    "payload: titled UTF-8 Markdown, 20..65536 bytes, no NUL",
                )
                .into());
            }
            demo(Path::new(path), payload)
        }
        "transfer-replay" if args.len() == 2 => replay(Path::new(path)),
        _ => Err(CliError::Usage),
    }
}
fn demo(directory: &Path, payload: Vec<u8>) -> Result<String, CliError> {
    fs::create_dir(directory).map_err(|source| CliError::ReadTask {
        path: directory.into(),
        source,
    })?;
    let started = Instant::now();
    let authority = Authority::from([Effect::Simulate, Effect::WriteData]);
    let mut sources = Vec::new();
    let mut source_queries = 0;
    let mut source_steps = 0;
    for (i, family) in [Family::Grid, Family::Protocol, Family::Pipeline]
        .into_iter()
        .enumerate()
    {
        let context = format!("demo/prototype/v1/{}", family.name());
        let extent = if family == Family::Protocol { 3 } else { 2 };
        let mut store = EpisodeStore::open(directory.join(format!("source-{i}.axep")))?.0;
        let model;
        {
            let mut memory = BehaviorMemory::restore(
                &mut store,
                &context,
                4,
                BehaviorBudget::default(),
                authority.clone(),
                Environment::new(family, extent, 0),
            )?;
            model = discover(4, &context, &mut memory)?;
            source_queries += memory.queries();
            source_steps += memory.steps();
        }
        let truth = Environment::new(family, extent, 0);
        if audit_machine(&model, 4, 0, |s, a| truth.transition(s, a))?
            .counterexample
            .is_some()
        {
            return Err(TransferError::Invalid("source independent audit failed").into());
        }
        model.save_new(directory.join(SOURCE_FILES[i]))?;
        sources.push(model);
    }
    let source_learning_ms = started.elapsed().as_secs_f64() * 1000.0;
    let context = format!("demo/adaptation/v1/{:016x}", checksum(&payload));
    let mut rng = Rng::new(771_311);
    let target = Rewired::random(Environment::new(Family::Pipeline, 2, 0), &mut rng);
    let goal = target.output_for(201);
    let learned;
    let queries;
    let steps;
    let transfer_ms;
    {
        let mut store = EpisodeStore::open(directory.join("target.axep"))?.0;
        let started = Instant::now();
        {
            let mut memory = BehaviorMemory::restore(
                &mut store,
                &context,
                4,
                BehaviorBudget::default(),
                authority.clone(),
                target.clone(),
            )?;
            learned = adapt(
                &sources.iter().collect::<Vec<_>>(),
                4,
                &context,
                TransferBudget::default(),
                &mut memory,
            )?;
            queries = memory.queries();
            steps = memory.steps();
        }
        transfer_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut fields = BTreeMap::new();
        fields.insert(
            "atlas".into(),
            EpisodeValue::Text(format!("{:016x}", atlas_digest(&sources))),
        );
        fields.insert(
            "model".into(),
            EpisodeValue::Text(format!("{:016x}", learned.model.fingerprint())),
        );
        fields.insert(
            "mode".into(),
            EpisodeValue::Text(format!("{:?}", learned.mode)),
        );
        fields.insert("queries".into(), EpisodeValue::Integer(queries as i64));
        let id = store.next_id();
        store.record(
            EpisodeStamp::new("axon.transfer/v1", format!("adaptation-{id}"), id),
            EpisodeData::Trace {
                context: context.clone(),
                channel: CHANNEL.into(),
                fields,
            },
        )?;
    }
    let audit = audit_machine(&learned.model, 4, 0, |s, a| target.transition(s, a))?;
    if audit.counterexample.is_some() {
        return Err(TransferError::Invalid("target independent audit failed").into());
    }
    learned
        .model
        .save_new(directory.join("target.axbehavior"))?;
    // Install the learned artifact into the existing demand-driven runtime.
    let install_authority = Authority::from([Effect::ReadData, Effect::WriteData]);
    let mut catalog = crate::demand::Catalog::new(directory);
    for (i, file) in SOURCE_FILES.iter().enumerate() {
        catalog.register(
            crate::demand::ModuleSpec {
                id: &format!("prototype-{i}"),
                kind: crate::demand::ModuleKind::Behavior,
                file,
                input: "events",
                output: "predictions",
                cost: 1,
            },
            &install_authority,
        )?;
    }
    catalog.register(
        crate::demand::ModuleSpec {
            id: "adapted",
            kind: crate::demand::ModuleKind::Behavior,
            file: "target.axbehavior",
            input: "target-events",
            output: "target-predictions",
            cost: 1,
        },
        &install_authority,
    )?;
    catalog.save_new(directory.join("atlas.axseed"), &install_authority)?;
    let mut runtime = crate::demand::DemandRuntime::new(catalog, crate::demand::Seed::default())?;
    runtime.with_module("adapted", &context, &install_authority, |_| Ok(()))?;
    let retained = runtime.stats().active_bytes;
    runtime.unload_all();
    let replay_report = replay(directory)?;
    let mut store = EpisodeStore::open(directory.join("target.axep"))?.0;
    let plan;
    let reply;
    {
        let environment = Environment::publishing(payload.clone(), directory.join("published.md"));
        let mut rng = Rng::new(771_311);
        let physical = Rewired::random(environment, &mut rng);
        let mut memory = BehaviorMemory::restore(
            &mut store,
            &context,
            4,
            BehaviorBudget::default(),
            authority,
            physical,
        )?;
        plan = learned
            .model
            .plan_output(&[], goal, &BTreeSet::new())?
            .ok_or(TransferError::Invalid("publication goal unreachable"))?;
        reply = memory.execute_goal(&learned.model, &[], goal, &BTreeSet::new())?;
    }
    drop(store);
    let published =
        read_bounded(&directory.join("published.md"), 65_536).map_err(BehaviorError::from)?;
    if published != payload {
        return Err(TransferError::Invalid("publication content mismatch").into());
    }
    let mut report = format!(
        "mode={:?}\nsource_models={}\nsource_learning_ms={source_learning_ms:.3}\nsource_queries={source_queries}\nsource_steps={source_steps}\nadaptation_with_durable_validation_ms={transfer_ms:.3}\nadaptation_queries={queries}\nadaptation_steps={steps}\nlearning_probes={}\ninitial_hypotheses={}\nsurviving_hypotheses={}\nsearch_work={}\npeak_hypothesis_bytes={}\nmodel_bytes={}\nmodel_retained_bytes={retained}\naudited_edges={}\npublication_plan={plan:?}\nobserved_reply={reply:?}\npublished_bytes={}\npublished_digest={:016x}\n{}",
        learned.mode,
        sources.len(),
        learned.transfer_stats.queries,
        learned.transfer_stats.initial_hypotheses,
        learned.transfer_stats.surviving_hypotheses,
        learned.transfer_stats.work,
        learned.transfer_stats.peak_hypothesis_bytes,
        learned.model.encoded_bytes(),
        audit.checked_edges,
        published.len(),
        checksum(&published),
        replay_report
    );
    writeln!(report,"limits=finite deterministic resettable interfaces; supplied alphabet and goal; no physical robotics, natural-language fluency or AGI").expect("String");
    publish_new(&directory.join("report.txt"), report.as_bytes()).map_err(BehaviorError::from)?;
    Ok(report)
}
struct NoContact;
impl Resettable for NoContact {
    fn reset(&mut self) -> Result<(), BehaviorError> {
        Err(BehaviorError::Invalid("replay attempted a new reset"))
    }
    fn step(&mut self, _: u8) -> Result<u16, BehaviorError> {
        Err(BehaviorError::Invalid("replay attempted a new action"))
    }
}
fn replay(directory: &Path) -> Result<String, CliError> {
    let sources = SOURCE_FILES
        .iter()
        .map(|file| BehaviorModel::load(directory.join(file)))
        .collect::<Result<Vec<_>, _>>()?;
    let expected = BehaviorModel::load(directory.join("target.axbehavior"))?;
    let mut store = EpisodeStore::open(directory.join("target.axep"))?.0;
    let records = store.traces(expected.context(), CHANNEL)?;
    let last = records
        .last()
        .ok_or(TransferError::Invalid("missing adaptation receipt"))?;
    let EpisodeData::Trace { fields, .. } = last.data() else {
        unreachable!()
    };
    if fields.get("atlas")
        != Some(&EpisodeValue::Text(format!(
            "{:016x}",
            atlas_digest(&sources)
        )))
        || fields.get("model")
            != Some(&EpisodeValue::Text(format!(
                "{:016x}",
                expected.fingerprint()
            )))
    {
        return Err(TransferError::Invalid("receipt/atlas/model identity mismatch").into());
    }
    let started = Instant::now();
    let mut memory = BehaviorMemory::restore(
        &mut store,
        expected.context(),
        4,
        BehaviorBudget::default(),
        Authority::from([]),
        NoContact,
    )?;
    let before = memory.queries();
    let learned = adapt(
        &sources.iter().collect::<Vec<_>>(),
        4,
        expected.context(),
        TransferBudget::default(),
        &mut memory,
    )?;
    if learned.model != expected || memory.queries() != before {
        return Err(TransferError::Invalid("replay/model mismatch").into());
    }
    Ok(format!(
        "replay_ms={:.3}\nnew_environment_queries=0\nreplay_model_matches=true\n",
        started.elapsed().as_secs_f64() * 1000.0
    ))
}
