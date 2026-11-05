use crate::{Authority, Effect, EpisodeStore, behavior::*, behavior_lab::*, cli::CliError};
use std::{collections::BTreeSet, fmt::Write, fs, path::Path, time::Instant};
pub(crate) const USAGE: &str = "\n  axon behavior-demo <new-directory> [--payload <document.md>]\n  axon behavior-inspect <model.axbehavior>\n  axon behavior-plan <model.axbehavior> --context <context> --goal <output-token> [--history <comma-separated-actions>]";
impl From<BehaviorError> for CliError {
    fn from(e: BehaviorError) -> Self {
        Self::Behavior(e)
    }
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let path = args.get(1).ok_or(CliError::Usage)?;
    match args[0].as_str() {
        "behavior-demo" if args.len() == 2 || (args.len() == 4 && args[2] == "--payload") => {
            let payload = if let Some(file) = args.get(3) {
                crate::text_skill::read_bounded(Path::new(file), 65_536)
                    .map_err(BehaviorError::from)?
            } else {
                include_bytes!("../examples/behavior_document.md").to_vec()
            };
            if payload.len() < 20
                || !payload.starts_with(b"# ")
                || payload.contains(&0)
                || std::str::from_utf8(&payload).is_err()
            {
                return Err(BehaviorError::Invalid(
                    "payload must be UTF-8 Markdown with a title, 20..65536 bytes, no NUL",
                )
                .into());
            }
            demo(Path::new(path), payload)
        }
        "behavior-inspect" if args.len() == 2 => {
            let model = BehaviorModel::load(path)?;
            let mut out = format!(
                "context={} latent_states={} actions={} bytes={} fingerprint={:016x}\n",
                model.context(),
                model.states(),
                model.actions(),
                model.encoded_bytes(),
                model.fingerprint()
            );
            for (state, access) in model.access_sequences().iter().enumerate() {
                writeln!(out, "state={state} access={access:?}").expect("String");
                for action in 0..model.actions() {
                    writeln!(
                        out,
                        "  action={action} {:?}",
                        model.edge(state as u16, action)?
                    )
                    .expect("String");
                }
            }
            Ok(out)
        }
        "behavior-plan" if args.len() == 6 || args.len() == 8 => {
            let mut context = None;
            let mut goal = None;
            let mut history = None;
            for pair in args[2..].as_chunks::<2>().0 {
                match pair[0].as_str() {
                    "--context" if context.is_none() => context = Some(pair[1].as_str()),
                    "--goal" if goal.is_none() => {
                        goal = Some(pair[1].parse::<u16>().map_err(|_| CliError::Usage)?)
                    }
                    "--history" if history.is_none() => {
                        history = Some(
                            pair[1]
                                .split(',')
                                .map(|s| s.parse::<u8>().map_err(|_| CliError::Usage))
                                .collect::<Result<Vec<_>, _>>()?,
                        )
                    }
                    _ => return Err(CliError::Usage),
                }
            }
            let model = BehaviorModel::load(path)?;
            if Some(model.context()) != context {
                return Err(BehaviorError::Invalid("context mismatch").into());
            }
            Ok(format!(
                "candidate_plan={:?}\n",
                model.plan_output(
                    &history.unwrap_or_default(),
                    goal.ok_or(CliError::Usage)?,
                    &BTreeSet::new()
                )?
            ))
        }
        _ => Err(CliError::Usage),
    }
}
fn demo(directory: &Path, payload: Vec<u8>) -> Result<String, CliError> {
    fs::create_dir(directory).map_err(|source| CliError::ReadTask {
        path: directory.into(),
        source,
    })?;
    let context = format!(
        "demo/document-v1/{:016x}",
        crate::live_store::checksum(&payload)
    );
    let journal = directory.join("experience.axep");
    let artifact = directory.join("document.axbehavior");
    let budget = BehaviorBudget::default();
    let authority = Authority::from([Effect::Simulate, Effect::WriteData]);
    let started = Instant::now();
    let model;
    let revisions;
    let suffixes;
    let queries;
    let steps;
    let trie;
    {
        let mut store = EpisodeStore::open(&journal)?.0;
        let mut memory = BehaviorMemory::restore(
            &mut store,
            &context,
            WORKFLOW_ACTIONS,
            budget,
            authority.clone(),
            DocumentWorkflow::new(payload.clone(), None),
        )?;
        let first =
            BehaviorLearner::new(WORKFLOW_ACTIONS, &context, budget)?.hypothesize(&mut memory)?;
        first.save_new(directory.join("hypothesis-0.axbehavior"))?;
        (model, revisions, suffixes) = discover(WORKFLOW_ACTIONS, &context, budget, &mut memory)?;
        queries = memory.queries();
        steps = memory.steps();
        trie = memory.prefix_trie_nodes();
        model.save_new(&artifact)?;
    }
    let learn_ms = started.elapsed().as_secs_f64() * 1000.0;
    let audit = audit_machine(&model, WORKFLOW_ACTIONS, 0, |state, action| {
        let mut sut = DocumentWorkflow::new(payload.clone(), None);
        sut.set_audit_state(state as u16);
        let reply = sut.step(action)?;
        Ok((sut.hidden_state() as usize, reply))
    })?;
    if audit.counterexample.is_some() {
        return Err(
            BehaviorError::Invalid("post-training exact audit failed; no file published").into(),
        );
    }
    let loaded = BehaviorModel::load(&artifact)?;
    let resumed = Instant::now();
    let mut store = EpisodeStore::open(&journal)?.0;
    {
        let mut memory = BehaviorMemory::restore(
            &mut store,
            &context,
            WORKFLOW_ACTIONS,
            budget,
            Authority::from([]),
            DocumentWorkflow::new(payload.clone(), None),
        )?;
        let rebuilt = discover(WORKFLOW_ACTIONS, &context, budget, &mut memory)?.0;
        if rebuilt != loaded || memory.queries() != queries {
            return Err(BehaviorError::Invalid("replay needed new experiments").into());
        }
    }
    let replay_ms = resumed.elapsed().as_secs_f64() * 1000.0;
    let forbidden = BTreeSet::from([401, DENIED]);
    let plan = loaded
        .plan_output(&[], PUBLISHED, &forbidden)?
        .ok_or(BehaviorError::Invalid(
            "publication not predicted reachable",
        ))?;
    let mut memory = BehaviorMemory::restore(
        &mut store,
        &context,
        WORKFLOW_ACTIONS,
        budget,
        authority,
        DocumentWorkflow::new(payload.clone(), Some(directory.join("published.md"))),
    )?;
    let reply = memory
        .execute_goal(&loaded, &[], PUBLISHED, &forbidden)?
        .ok_or(BehaviorError::Invalid("no plan"))?;
    drop(memory);
    if fs::read(directory.join("published.md")).map_err(|source| CliError::ReadTask {
        path: directory.join("published.md"),
        source,
    })? != payload
    {
        return Err(BehaviorError::Invalid("published content mismatch").into());
    }
    let mut mutant = DocumentWorkflow::new(payload, None);
    mutant.mutant = true;
    let mut bad = RawOracle::new(mutant);
    let witness = regression_probe(&loaded, &mut bad, 65_536)?.ok_or(BehaviorError::Invalid(
        "mutation not detected by black-box probes",
    ))?;
    let regression_queries = bad.queries;
    let actual_bad = bad.query(&witness)?;
    let mut out = format!(
        "context={context}\nlatent_state_revisions={revisions:?} distinguishing_suffixes={suffixes} learned_states={}\nqueries={queries} commanded_steps={steps} cached_trie_nodes={trie} model_bytes={}\nlearn_with_sync_ms={learn_ms:.3} replay_ms={replay_ms:.3} replay_new_queries=0\nexact_product_pairs={} exact_transition_checks={}\n",
        model.states(),
        model.encoded_bytes(),
        audit.checked_pairs,
        audit.checked_edges
    );
    writeln!(
        out,
        "same_visible_ack=true different_future_states={} vs {}",
        loaded.state_after(&[0, 1, 2])?,
        loaded.state_after(&[0, 1, 2, 3, 4, 5, 6, 7])?
    )
    .expect("String");
    writeln!(
        out,
        "new_goal_plan={plan:?} actions={:?}",
        plan.iter()
            .map(|&a| DocumentWorkflow::action_name(a))
            .collect::<Vec<_>>()
    )
    .expect("String");
    writeln!(
        out,
        "goal_observed={:?} published_content_exact=true output={}",
        reply.last(),
        directory.join("published.md").display()
    )
    .expect("String");
    writeln!(
        out,
        "black_box_regression_queries={regression_queries} regression_witness={witness:?} predicted={:?} mutant_observed={actual_bad:?}",
        loaded.predict(&witness)?
    )
    .expect("String");
    writeln!(
        out,
        "journal_bytes={} model={}",
        store.file_bytes()?,
        artifact.display()
    )
    .expect("String");
    crate::text_skill::publish_new(&directory.join("demo.txt"), out.as_bytes())
        .map_err(BehaviorError::from)?;
    Ok(out)
}
