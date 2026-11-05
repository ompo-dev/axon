use axon_uic::{
    EpisodeData, EpisodeStamp, EpisodeStore, EpisodeValue,
    numeric::{continuous::*, library::NumericLibrary},
};
use std::{
    collections::BTreeMap,
    error::Error,
    io::{self, BufRead, Read, Write},
    path::Path,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

fn json(text: &str) -> String {
    let mut s = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if c.is_control() => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
    s
}
fn emit(line: &str) -> io::Result<()> {
    let mut out = io::stdout().lock();
    writeln!(out, "{line}")?;
    out.flush()
}
fn status(a: &ContinuousAgent, kind: &str) -> String {
    let s = a.statistics();
    format!(
        "{{\"kind\":{},\"context\":{},\"phase\":{},\"epoch\":{},\"model\":{},\"observations\":{},\"feedback\":{},\"predictions\":{},\"scored\":{},\"correct\":{},\"drifts\":{},\"promotions\":{},\"reuses\":{},\"jobs\":{},\"failed_jobs\":{},\"stale_jobs\":{},\"search_work\":{},\"pending\":{},\"training_window\":{},\"archive_models\":{},\"archive_bytes\":{},\"retained_bytes\":{}}}",
        json(kind),
        json(a.context()),
        json(a.phase().as_str()),
        a.epoch(),
        a.model_identity()
            .map(|id| json(&format!("{id:016x}")))
            .unwrap_or_else(|| "null".into()),
        s.observations,
        s.feedback,
        s.predictions,
        s.scored,
        s.correct,
        s.drifts,
        s.promotions,
        s.reuses,
        s.jobs,
        s.failed_jobs,
        s.stale_jobs,
        s.search_work,
        a.pending_count(),
        a.window_count(),
        a.archived_models(),
        a.archived_bytes(),
        a.retained_bytes()
    )
}
fn milestone(
    journal: &mut Option<EpisodeStore>,
    a: &ContinuousAgent,
    kind: &str,
) -> Result<(), Box<dyn Error>> {
    if let Some(journal) = journal {
        let mut fields = BTreeMap::new();
        fields.insert("kind".into(), EpisodeValue::Text(kind.into()));
        fields.insert(
            "phase".into(),
            EpisodeValue::Text(a.phase().as_str().into()),
        );
        fields.insert("epoch".into(), EpisodeValue::Text(a.epoch().to_string()));
        fields.insert(
            "model".into(),
            EpisodeValue::Text(
                a.model_identity()
                    .map(|id| format!("{id:016x}"))
                    .unwrap_or_else(|| "none".into()),
            ),
        );
        fields.insert(
            "statistics".into(),
            EpisodeValue::Text(status(a, "milestone")),
        );
        journal.record(
            EpisodeStamp::new(
                "axon.continuous/v1",
                format!("event-{}", journal.next_id()),
                a.statistics().observations,
            ),
            EpisodeData::Trace {
                context: a.context().into(),
                channel: "numeric.continuous/v1".into(),
                fields,
            },
        )?;
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let usage = "usage: axon-uic-live-agent --context <context> --arity <1..4> [--restore <checkpoint.axcont>] [--library <file.axlib> --library-context <context>] [--episodes <journal.axep>]\nstdin TSV events: O<TAB>id<TAB>x0,x1,... ; F<TAB>id<TAB>actual ; T<TAB>id<TAB>raw-frame (predict next x0) ; S ; C<TAB>new-checkpoint-path ; Q. Do not mix T with O/F. Events are flushed as NDJSON before EOF.";
    if args == ["--help"] {
        println!("{usage}");
        return Ok(());
    }
    let mut options = BTreeMap::new();
    for pair in args.chunks(2) {
        if pair.len() != 2
            || ![
                "--context",
                "--arity",
                "--restore",
                "--library",
                "--library-context",
                "--episodes",
            ]
            .contains(&pair[0].as_str())
            || options.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err(usage.into());
        }
    }
    let context = options.get("--context").ok_or(usage)?;
    let arity = options.get("--arity").ok_or(usage)?.parse::<usize>()?;
    if options.contains_key("--library") != options.contains_key("--library-context") {
        return Err(usage.into());
    }
    let library = options
        .get("--library")
        .map(|file| {
            NumericLibrary::open(Path::new(file), &options["--library-context"]).map(Arc::new)
        })
        .transpose()?;
    let identity = library.as_ref().map(|l| l.fingerprint());
    let mut agent = if let Some(file) = options.get("--restore") {
        ContinuousAgent::open_checkpoint(Path::new(file), context, identity)?
    } else {
        ContinuousAgent::new(context, arity, identity, ContinuousPolicy::default())?
    };
    if agent.arity() != arity {
        return Err("restored arity mismatch".into());
    }
    let mut journal = options
        .get("--episodes")
        .map(|file| EpisodeStore::open(file).map(|(store, _)| store))
        .transpose()?;
    emit(&status(&agent, "ready"))?;
    let (input_tx, input_rx) = mpsc::sync_channel::<Option<String>>(128);
    thread::spawn(move || {
        let stdin = io::stdin();
        let mut reader = stdin.lock();
        loop {
            let mut line = String::new();
            match reader.by_ref().take(4097).read_line(&mut line) {
                Ok(0) => break,
                Ok(n) if n <= 4096 => {
                    if input_tx.send(Some(line)).is_err() {
                        return;
                    }
                }
                _ => {
                    let _ = input_tx.send(Some("INVALID_INPUT_LINE".into()));
                    break;
                }
            }
        }
        let _ = input_tx.send(None);
    });
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    let mut worker = None;
    let receive_result = |agent: &mut ContinuousAgent,
                          result: LearningResult,
                          journal: &mut Option<EpisodeStore>|
     -> Result<(), Box<dyn Error>> {
        let time = result.elapsed_ns;
        let work = result.search_work;
        let error = result.error.clone();
        let disposition = agent.submit_learning_result(result);
        emit(&format!(
            "{{\"kind\":\"learning_result\",\"status\":{},\"elapsed_ns\":{},\"search_work\":{},\"error\":{},\"phase\":{}}}",
            json(disposition.as_str()),
            time,
            work,
            error
                .as_ref()
                .map(|e| json(e))
                .unwrap_or_else(|| "null".into()),
            json(agent.phase().as_str())
        ))?;
        milestone(journal, agent, disposition.as_str())?;
        Ok(())
    };
    loop {
        if let Ok(result) = result_rx.try_recv() {
            receive_result(&mut agent, result, &mut journal)?;
            if let Some(handle) = worker.take() {
                let handle: thread::JoinHandle<()> = handle;
                handle.join().map_err(|_| "learning worker panicked")?;
            }
        }
        if worker.is_none()
            && let Some(job) = agent.take_learning_job()
        {
            emit(&format!(
                "{{\"kind\":\"learning_started\",\"token\":{},\"observations\":{}}}",
                job.token(),
                agent.statistics().observations
            ))?;
            let library = library.clone();
            let tx = result_tx.clone();
            worker = Some(thread::spawn(move || {
                let _ = tx.send(job.run(library.as_deref()));
            }));
        }
        let line = match input_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(Some(line)) => line,
            Ok(None) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
        };
        let fields: Vec<_> = line.trim_end_matches(['\r', '\n']).split('\t').collect();
        let outcome = (|| -> Result<bool, Box<dyn Error>> {
            match fields.as_slice() {
                ["T", id, values] => {
                    let values = values
                        .split(',')
                        .map(str::parse::<f64>)
                        .collect::<Result<Vec<_>, _>>()?;
                    let r = agent.advance_frame(id.parse()?, &values, 0)?;
                    emit(&format!(
                        "{{\"kind\":\"frame\",\"id\":{},\"next_prediction\":{},\"previous_status\":{},\"previous_correct\":{},\"duplicate\":{},\"phase\":{}}}",
                        json(&r.prediction.id.to_string()),
                        r.prediction
                            .value
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "null".into()),
                        r.previous
                            .map(|f| json(f.status.as_str()))
                            .unwrap_or_else(|| "null".into()),
                        r.previous.map(|f| f.correct).unwrap_or(false),
                        r.prediction.duplicate,
                        json(agent.phase().as_str())
                    ))?;
                    if let Some(f) = r.previous
                        && matches!(
                            f.status,
                            FeedbackStatus::Drift
                                | FeedbackStatus::Promoted
                                | FeedbackStatus::CandidateRejected
                        )
                    {
                        milestone(&mut journal, &agent, f.status.as_str())?;
                    }
                }
                ["O", id, values] => {
                    let id = id.parse()?;
                    let values = values
                        .split(',')
                        .map(str::parse::<f64>)
                        .collect::<Result<Vec<_>, _>>()?;
                    let p = agent.observe(id, &values)?;
                    emit(&format!(
                        "{{\"kind\":\"prediction\",\"id\":{},\"sequence\":{},\"epoch\":{},\"value\":{},\"model\":{},\"duplicate\":{},\"phase\":{}}}",
                        json(&p.id.to_string()),
                        p.sequence,
                        p.epoch,
                        p.value
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "null".into()),
                        p.model
                            .map(|id| json(&format!("{id:016x}")))
                            .unwrap_or_else(|| "null".into()),
                        p.duplicate,
                        json(agent.phase().as_str())
                    ))?;
                }
                ["F", id, value] => {
                    let r = agent.feedback(id.parse()?, value.parse()?)?;
                    emit(&format!(
                        "{{\"kind\":\"feedback\",\"id\":{},\"status\":{},\"scored\":{},\"correct\":{},\"phase\":{}}}",
                        json(&r.id.to_string()),
                        json(r.status.as_str()),
                        r.scored,
                        r.correct,
                        json(agent.phase().as_str())
                    ))?;
                    if matches!(
                        r.status,
                        FeedbackStatus::Drift
                            | FeedbackStatus::Promoted
                            | FeedbackStatus::CandidateRejected
                    ) {
                        milestone(&mut journal, &agent, r.status.as_str())?;
                    }
                }
                ["S"] => emit(&status(&agent, "status"))?,
                ["C", path] => {
                    agent.save_checkpoint_new(Path::new(path))?;
                    emit(&format!(
                        "{{\"kind\":\"checkpoint\",\"path\":{},\"bytes\":{}}}",
                        json(path),
                        agent.to_bytes().len()
                    ))?;
                }
                ["Q"] => return Ok(true),
                _ => return Err("invalid live event".into()),
            }
            Ok(false)
        })();
        match outcome {
            Ok(true) => break,
            Ok(false) => {}
            Err(error) => emit(&format!(
                "{{\"kind\":\"error\",\"message\":{}}}",
                json(&error.to_string())
            ))?,
        }
    }
    if let Some(handle) = worker {
        handle.join().map_err(|_| "learning worker panicked")?;
        if let Ok(result) = result_rx.try_recv() {
            receive_result(&mut agent, result, &mut journal)?;
        }
    }
    emit(&status(&agent, "stopped"))?;
    Ok(())
}
