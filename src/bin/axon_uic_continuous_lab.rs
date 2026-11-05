use axon_uic::{
    behavior_lab::Rng,
    numeric::{NumericSkill, NumericTolerance, continuous::*},
    telemetry::{LabTelemetry, Value as V},
};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::Instant,
};

fn truth(channel: usize, regime: usize, x: &[f64]) -> f64 {
    if channel == 0 {
        match regime % 4 {
            0 => 2.0 * x[0] + 1.0,
            1 => -1.75 * x[0] + 4.25,
            2 => x[0] * x[0],
            _ => x[0] / (1.0 + x[0] * x[0]),
        }
    } else {
        match regime % 3 {
            0 => x[0].hypot(x[1]),
            1 => x[0] + x[1],
            _ => x[0] * x[1],
        }
    }
}
fn write_new(path: impl AsRef<std::path::Path>, data: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(data)?;
    f.sync_all()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=3).contains(&args.len()) {
        return Err("usage: axon-uic-continuous-lab <new-output-directory> [seed] [observations-per-channel 1000..1000000]".into());
    }
    let output = PathBuf::from(&args[0]);
    let seed = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(41973621_u64);
    let count = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(100000_usize);
    if !(1000..=1000000).contains(&count) {
        return Err("observation limit".into());
    }
    fs::create_dir(&output)?;
    let started = Instant::now();
    let mut telemetry = LabTelemetry::from_environment("continuous");
    telemetry.emit(
        "lab_start",
        &[
            ("seed", V::Count(seed)),
            ("observations_per_channel", V::Count(count as u64)),
            ("channels", V::Count(3)),
            ("telemetry", V::Text("prequential-checkpoints")),
        ],
    )?;
    let mut rng = Rng::new(seed);
    let segment = (count / 20).max(100);
    let policy = ContinuousPolicy {
        tolerance: NumericTolerance {
            absolute: 1e-7,
            relative: 1e-8,
        },
        ..ContinuousPolicy::default()
    };
    let mut agents = [
        ContinuousAgent::new("continuous-lab/calibration", 1, None, policy)?,
        ContinuousAgent::new("continuous-lab/frame", 2, None, policy)?,
    ];
    let mut labels: [Vec<(u64, f64, usize)>; 2] = [Vec::new(), Vec::new()];
    let mut jobs: [Option<(LearningJob, usize)>; 2] = [None, None];
    let mut frozen: [Option<NumericSkill>; 2] = [None, None];
    let mut baseline = [(0_u64, 0_u64); 2];
    let mut paired = [(0_u64, 0_u64); 2];
    let mut max_retained = [0_usize; 2];
    let mut maximum_ingest_ns = 0_u128;
    let mut ingest_ns = 0_u128;
    let mut learning_ns = 0_u64;
    let mut missing = [0_u64; 2];
    let mut regime_promotions = [0_usize; 2];
    let mut trace =
        String::from("channel,observation,kind,phase,epoch,model,scored,correct,search_work\n");
    let mut transitions = String::from(
        "channel,regime,boundary,first_twenty_correct_predictions_at,latency_observations\n",
    );
    let mut streak = [0_usize; 2];
    let mut settled = [false; 2];
    let mut last_regime = [usize::MAX; 2];
    for tick in 1..=count {
        for channel in 0..2 {
            let regime = (tick - 1) / segment;
            if last_regime[channel] != regime {
                streak[channel] = 0;
                settled[channel] = false;
                last_regime[channel] = regime;
            }
            let x: Vec<_> = (0..agents[channel].arity())
                .map(|_| (rng.pick(20001) as f64 - 10000.0) / 137.0)
                .collect();
            let expected = truth(channel, regime, &x);
            let before = Instant::now();
            let prediction = agents[channel].observe(tick as u64, &x)?;
            let elapsed = before.elapsed().as_nanos();
            ingest_ns += elapsed;
            maximum_ingest_ns = maximum_ingest_ns.max(elapsed);
            if prediction
                .value
                .is_some_and(|v| policy.tolerance.agrees(v, expected))
            {
                streak[channel] += 1;
            } else {
                streak[channel] = 0;
            }
            if streak[channel] >= 20 && !settled[channel] {
                settled[channel] = true;
                regime_promotions[channel] += 1;
                transitions.push_str(&format!(
                    "{channel},{regime},{},{tick},{}\n",
                    regime * segment + 1,
                    tick - regime * segment
                ));
            }
            if let Some(frozen) = &frozen[channel] {
                baseline[channel].1 += 1;
                paired[channel].1 += u64::from(prediction.value.is_some());
                paired[channel].0 += u64::from(
                    prediction
                        .value
                        .is_some_and(|v| policy.tolerance.agrees(v, expected)),
                );
                if frozen
                    .program()
                    .evaluate(&x)
                    .is_ok_and(|v| policy.tolerance.agrees(v, expected))
                {
                    baseline[channel].0 += 1;
                }
            }
            // Labels may be absent or arrive out of order; the agent receives no regime/boundary marker.
            if rng.pick(101) == 0 {
                missing[channel] += 1;
            } else {
                labels[channel].push((tick as u64, expected, tick + 1 + rng.pick(7)));
            }
            let mut index = 0;
            while index < labels[channel].len() {
                if labels[channel][index].2 <= tick {
                    let (id, y, _) = labels[channel].remove(index);
                    let receipt = agents[channel].feedback(id, y)?;
                    if !matches!(
                        receipt.status,
                        FeedbackStatus::Accepted | FeedbackStatus::Historical
                    ) {
                        let a = &agents[channel];
                        trace.push_str(&format!(
                            "{channel},{tick},{},{},{},{:?},{},{},{}\n",
                            receipt.status.as_str(),
                            a.phase().as_str(),
                            a.epoch(),
                            a.model_identity(),
                            receipt.scored,
                            receipt.correct,
                            a.statistics().search_work
                        ));
                    }
                } else {
                    index += 1;
                }
            }
            if let Some((_, due)) = &jobs[channel]
                && *due <= tick
            {
                let (job, _) = jobs[channel].take().expect("due job");
                let result = job.run(None);
                learning_ns += result.elapsed_ns;
                let status = agents[channel].submit_learning_result(result);
                let a = &agents[channel];
                trace.push_str(&format!(
                    "{channel},{tick},{},{},{},{:?},,,{}\n",
                    status.as_str(),
                    a.phase().as_str(),
                    a.epoch(),
                    a.model_identity(),
                    a.statistics().search_work
                ));
            }
            if jobs[channel].is_none()
                && let Some(job) = agents[channel].take_learning_job()
            {
                jobs[channel] = Some((job, tick + 3));
            }
            if frozen[channel].is_none() {
                frozen[channel] = agents[channel].active_skill().cloned();
            }
            max_retained[channel] = max_retained[channel].max(agents[channel].retained_bytes());
            if tick.is_multiple_of(128) || tick == count {
                let a = &agents[channel];
                let s = a.statistics();
                telemetry.emit(
                    "continuous_progress",
                    &[
                        ("channel", V::Count(channel as u64)),
                        ("observation", V::Count(tick as u64)),
                        ("phase", V::Text(a.phase().as_str())),
                        ("correct", V::Count(s.correct)),
                        ("scored", V::Count(s.scored)),
                        ("observations", V::Count(s.observations)),
                        ("promotions", V::Count(s.promotions)),
                        ("drifts", V::Count(s.drifts)),
                        ("work", V::Count(s.search_work)),
                    ],
                )?;
            }
            if tick == count / 2 {
                let path = output.join(format!("channel-{channel}.midpoint.axcont"));
                agents[channel].save_checkpoint_new(&path)?;
                let context = agents[channel].context().to_owned();
                agents[channel] = ContinuousAgent::open_checkpoint(&path, &context, None)?;
                // Keep the old worker result to verify that restart refuses its token if one was in flight.
            }
        }
    }
    let mut summary = format!(
        "AXON CONTINUOUS PREQUENTIAL LAB\nseed={seed}\nobservations_per_channel={count}\nchannels=3\nregime_segment={segment}\nsupervised_observe_total_ns={ingest_ns}\nmax_single_supervised_observe_ns={maximum_ingest_ns}\nsupervised_learning_total_ns={learning_ns}\nNo future labels, boundary markers or complete dataset supplied to the agent. Job latency is simulated as three arrivals; real concurrent worker is exercised separately by the live CLI.\n"
    );
    let mut results = String::from(
        "channel,observations,labels,predictions,scored,correct,errors,coverage,conditional_accuracy,drifts,promotions,reuses,jobs,failed_jobs,stale_jobs,late_for_learning,expired,missing_labels,archived_models,archive_bytes,max_retained_bytes,frozen_correct,frozen_total,settled_regimes,paired_adaptive_correct,paired_adaptive_predictions\n",
    );
    for channel in 0..2 {
        let a = &agents[channel];
        let s = a.statistics();
        let coverage = s.scored as f64 / s.feedback.max(1) as f64;
        let accuracy = s.correct as f64 / s.scored.max(1) as f64;
        results.push_str(&format!("{channel},{},{},{},{},{},{},{coverage},{accuracy},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",s.observations,s.feedback,s.predictions,s.scored,s.correct,s.scored-s.correct,s.drifts,s.promotions,s.reuses,s.jobs,s.failed_jobs,s.stale_jobs,s.late_for_learning,s.expired,missing[channel],a.archived_models(),a.archived_bytes(),max_retained[channel],baseline[channel].0,baseline[channel].1,regime_promotions[channel],paired[channel].0,paired[channel].1));
        summary.push_str(&format!("channel_{channel}: coverage={coverage:.6}, conditional_accuracy={accuracy:.6}, errors={}, drifts={}, promotions={}, reuses={}, archived={}, max_retained_bytes={}, settled_regimes={}\n",s.scored-s.correct,s.drifts,s.promotions,s.reuses,a.archived_models(),max_retained[channel],regime_promotions[channel]));
        a.save_checkpoint_new(&output.join(format!("channel-{channel}.final.axcont")))?;
        if let Some(skill) = a.active_skill() {
            skill.save_new(&output.join(format!("channel-{channel}.active.axnum")))?;
        }
    }
    let mut raw = ContinuousAgent::new("continuous-lab/raw-dynamics", 2, None, policy)?;
    let mut raw_rng = Rng::new(seed ^ 0x725e394ffb19);
    let mut state = 13.0;
    let mut job: Option<(LearningJob, usize)> = None;
    let mut frozen_raw = None::<NumericSkill>;
    let mut raw_base = (0_u64, 0_u64);
    let mut raw_paired = (0_u64, 0_u64);
    let mut raw_max = 0;
    let mut raw_streak = 0;
    let mut raw_settled = false;
    let mut raw_regime = usize::MAX;
    let mut raw_settled_count = 0;
    let mut raw_ingest_ns = 0_u128;
    let mut raw_learning_ns = 0_u64;
    for tick in 1..=count {
        let regime = (tick - 1) / segment;
        if regime != raw_regime {
            raw_regime = regime;
            raw_streak = 0;
            raw_settled = false;
        }
        let ambient = (raw_rng.pick(20001) as f64 - 10000.0) / 137.0;
        let current = [state, ambient];
        // The learner receives only this raw frame. Its target comes from the following frame.
        let before = Instant::now();
        let receipt = raw.advance_frame(tick as u64, &current, 0)?;
        raw_ingest_ns += before.elapsed().as_nanos();
        let next = match regime % 3 {
            0 => (state + ambient) / 2.0,
            1 => ((state + ambient) / 2.0 + ambient) / 2.0,
            _ => state,
        };
        if receipt
            .prediction
            .value
            .is_some_and(|v| policy.tolerance.agrees(v, next))
        {
            raw_streak += 1;
        } else {
            raw_streak = 0;
        }
        if raw_streak >= 20 && !raw_settled {
            raw_settled = true;
            raw_settled_count += 1;
            transitions.push_str(&format!(
                "2,{regime},{},{tick},{}\n",
                regime * segment + 1,
                tick - regime * segment
            ));
        }
        if let Some(frozen) = &frozen_raw {
            raw_base.1 += 1;
            raw_paired.1 += u64::from(receipt.prediction.value.is_some());
            raw_paired.0 += u64::from(
                receipt
                    .prediction
                    .value
                    .is_some_and(|v| policy.tolerance.agrees(v, next)),
            );
            if frozen
                .program()
                .evaluate(&current)
                .is_ok_and(|v| policy.tolerance.agrees(v, next))
            {
                raw_base.0 += 1;
            }
        }
        if let Some(f) = receipt.previous
            && matches!(
                f.status,
                FeedbackStatus::Drift
                    | FeedbackStatus::Promoted
                    | FeedbackStatus::CandidateRejected
            )
        {
            trace.push_str(&format!(
                "2,{tick},{},{},{},{:?},{},{},{}\n",
                f.status.as_str(),
                raw.phase().as_str(),
                raw.epoch(),
                raw.model_identity(),
                f.scored,
                f.correct,
                raw.statistics().search_work
            ));
        }
        if let Some((_, due)) = &job
            && *due <= tick
        {
            let (work, _) = job.take().expect("due job");
            let result = work.run(None);
            raw_learning_ns += result.elapsed_ns;
            let disposition = raw.submit_learning_result(result);
            trace.push_str(&format!(
                "2,{tick},{},{},{},{:?},,,{}\n",
                disposition.as_str(),
                raw.phase().as_str(),
                raw.epoch(),
                raw.model_identity(),
                raw.statistics().search_work
            ));
        }
        if job.is_none()
            && let Some(work) = raw.take_learning_job()
        {
            job = Some((work, tick + 3));
        }
        if frozen_raw.is_none() {
            frozen_raw = raw.active_skill().cloned();
        }
        raw_max = raw_max.max(raw.retained_bytes());
        if tick == count / 2 {
            raw.save_checkpoint_new(&output.join("channel-2.midpoint.axcont"))?;
            raw = ContinuousAgent::open_checkpoint(
                &output.join("channel-2.midpoint.axcont"),
                "continuous-lab/raw-dynamics",
                None,
            )?;
        }
        state = next;
        if tick.is_multiple_of(128) || tick == count {
            let s = raw.statistics();
            telemetry.emit(
                "continuous_progress",
                &[
                    ("channel", V::Count(2)),
                    ("observation", V::Count(tick as u64)),
                    ("phase", V::Text(raw.phase().as_str())),
                    ("correct", V::Count(s.correct)),
                    ("scored", V::Count(s.scored)),
                    ("observations", V::Count(s.observations)),
                    ("promotions", V::Count(s.promotions)),
                    ("drifts", V::Count(s.drifts)),
                    ("work", V::Count(s.search_work)),
                ],
            )?;
        }
    }
    let s = raw.statistics();
    let coverage = s.scored as f64 / s.feedback.max(1) as f64;
    let accuracy = s.correct as f64 / s.scored.max(1) as f64;
    results.push_str(&format!(
        "2,{},{},{},{},{},{},{coverage},{accuracy},{},{},{},{},{},{},{},{},0,{},{},{},{},{},{},{},{}\n",
        s.observations,
        s.feedback,
        s.predictions,
        s.scored,
        s.correct,
        s.scored - s.correct,
        s.drifts,
        s.promotions,
        s.reuses,
        s.jobs,
        s.failed_jobs,
        s.stale_jobs,
        s.late_for_learning,
        s.expired,
        raw.archived_models(),
        raw.archived_bytes(),
        raw_max,
        raw_base.0,
        raw_base.1,
        raw_settled_count,
        raw_paired.0,
        raw_paired.1
    ));
    summary.push_str(&format!("channel_2_self_supervised: coverage={coverage:.6}, conditional_accuracy={accuracy:.6}, errors={}, drifts={}, promotions={}, reuses={}, archived={}, max_retained_bytes={}, settled_regimes={}, ingest_ns={raw_ingest_ns}, learning_ns={raw_learning_ns}\n",s.scored-s.correct,s.drifts,s.promotions,s.reuses,raw.archived_models(),raw_max,raw_settled_count));
    raw.save_checkpoint_new(&output.join("channel-2.final.axcont"))?;
    if let Some(skill) = raw.active_skill() {
        skill.save_new(&output.join("channel-2.active.axnum"))?;
    }
    summary.push_str(&format!("whole_lab_ns={}\nRaw observations are not retained as a complete dataset. No claim of perception, autonomy, watts or AGI.\n",started.elapsed().as_nanos()));
    write_new(output.join("results.csv"), results.as_bytes())?;
    write_new(output.join("transitions.csv"), transitions.as_bytes())?;
    write_new(output.join("events.csv"), trace.as_bytes())?;
    write_new(output.join("summary.txt"), summary.as_bytes())?;
    print!("{summary}");
    let passed = !agents
        .iter()
        .any(|a| a.statistics().promotions < 3 || a.statistics().reuses == 0)
        && raw.statistics().promotions >= 3
        && raw.statistics().reuses > 0;
    let statistics = agents
        .iter()
        .map(|a| a.statistics())
        .chain([raw.statistics()]);
    let (correct, scored, promotions, drifts) = statistics.fold((0, 0, 0, 0), |sum, s| {
        (
            sum.0 + s.correct,
            sum.1 + s.scored,
            sum.2 + s.promotions,
            sum.3 + s.drifts,
        )
    });
    telemetry.emit(
        "lab_result",
        &[
            ("passed", V::Flag(passed)),
            ("correct", V::Count(correct)),
            ("scored", V::Count(scored)),
            ("errors", V::Count(scored - correct)),
            ("promotions", V::Count(promotions)),
            ("drifts", V::Count(drifts)),
            (
                "elapsedNs",
                V::Count(started.elapsed().as_nanos().min(u64::MAX as u128) as u64),
            ),
        ],
    )?;
    if !passed {
        return Err("continuous adaptation or recurrence control failed; inspect outputs".into());
    }
    Ok(())
}
