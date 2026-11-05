use axon_uic::{behavior_lab::Rng, numeric::*};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    hint::black_box,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy)]
enum Family {
    Calibration,
    Affine,
    Product,
    Tax,
    Energy,
    Reciprocal,
    Response,
    Distance2,
    Distance3,
}
impl Family {
    fn name(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Affine => "affine_parameters",
            Self::Product => "line_values",
            Self::Tax => "tax_values",
            Self::Energy => "kinetic_energy",
            Self::Reciprocal => "reciprocal",
            Self::Response => "nonlinear_response",
            Self::Distance2 => "distance2",
            Self::Distance3 => "distance3_transfer",
        }
    }
    fn arity(self) -> usize {
        match self {
            Self::Calibration | Self::Affine | Self::Reciprocal | Self::Response => 1,
            Self::Product | Self::Energy | Self::Distance2 => 2,
            _ => 3,
        }
    }
    fn truth(self, x: &[f64]) -> f64 {
        match self {
            Self::Calibration => 2.0 * x[0] + 1.0,
            Self::Affine => 1.75 * x[0] - 4.25,
            Self::Product => x[0] * x[1],
            Self::Tax => x[0] * x[1] * (1.0 + x[2]),
            Self::Energy => 0.5 * x[0] * x[1].powi(2),
            Self::Reciprocal => x[0].recip(),
            Self::Response => x[0] / (1.0 + x[0].powi(2)),
            Self::Distance2 => x[0].hypot(x[1]),
            Self::Distance3 => x[0].hypot(x[1]).hypot(x[2]),
        }
    }
    fn training(self) -> Vec<NumericExample> {
        let inputs: Vec<Vec<f64>> = match self {
            Self::Calibration | Self::Affine => [-7.0, 0.0, 0.25, 3.0, 16.0]
                .into_iter()
                .map(|x| vec![x])
                .collect(),
            Self::Reciprocal => [-4.0, -2.0, 0.5, 1.0, 2.0, 4.0, 7.0]
                .into_iter()
                .map(|x| vec![x])
                .collect(),
            Self::Response => [-4.0, -1.3, 0.0, 0.5, 1.0, 2.0, 7.0]
                .into_iter()
                .map(|x| vec![x])
                .collect(),
            Self::Product => vec![
                vec![2.0, 3.0],
                vec![5.0, 7.0],
                vec![11.0, 2.0],
                vec![0.0, 9.0],
                vec![-3.0, 4.0],
                vec![1.25, 8.0],
            ],
            Self::Tax => vec![
                vec![20.0, 3.0, 0.1],
                vec![15.0, 2.0, 0.2],
                vec![8.0, 7.0, 0.05],
                vec![0.0, 9.0, 0.3],
                vec![13.0, 4.0, 0.0],
                vec![1.25, 8.0, 0.07],
                vec![3.0, 1.0, 0.5],
                vec![11.0, 5.0, 0.12],
            ],
            Self::Energy => vec![
                vec![2.0, 3.0],
                vec![5.0, 7.0],
                vec![11.0, -2.0],
                vec![0.0, 9.0],
                vec![3.0, -4.0],
                vec![1.25, 8.0],
                vec![7.0, 0.0],
            ],
            Self::Distance2 => vec![
                vec![3.0, 4.0],
                vec![5.0, 12.0],
                vec![8.0, 15.0],
                vec![7.0, 24.0],
                vec![0.0, 9.0],
                vec![-3.0, 4.0],
                vec![4.0, -3.0],
                vec![1.0, 1.0],
            ],
            Self::Distance3 => vec![
                vec![2.0, 3.0, 6.0],
                vec![1.0, 2.0, 2.0],
                vec![3.0, 4.0, 12.0],
                vec![8.0, 9.0, 12.0],
                vec![0.0, 0.0, 5.0],
                vec![-2.0, 3.0, 6.0],
                vec![3.0, -4.0, -12.0],
                vec![1.0, 1.0, 1.0],
            ],
        };
        inputs
            .into_iter()
            .map(|x| {
                let y = self.truth(&x);
                NumericExample::new(x, y)
            })
            .collect()
    }
    fn heldout(self, seed: u64, rows: usize) -> Vec<NumericExample> {
        let mut rng = Rng::new(seed);
        (0..rows)
            .map(|i| {
                let mut x: Vec<f64> = (0..self.arity())
                    .map(|_| (rng.pick(200_001) as f64 - 100_000.0) / 37.0)
                    .collect();
                if matches!(self, Self::Tax) {
                    x[0] = x[0].abs();
                    x[1] = (1 + rng.pick(100)) as f64;
                    x[2] = rng.pick(1001) as f64 / 1000.0;
                }
                if matches!(self, Self::Product | Self::Energy) {
                    x[0] = x[0].abs();
                }
                if matches!(self, Self::Reciprocal) && x[0] == 0.0 {
                    x[0] = 0.01;
                }
                if i % 13 == 0 && !matches!(self, Self::Reciprocal) {
                    x.fill(0.0);
                }
                let output = self.truth(&x);
                NumericExample::new(x, output)
            })
            .collect()
    }
}
fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut f = OpenOptions::new().create_new(true).write(true).open(path)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(())
}
fn tsv(rows: &[NumericExample]) -> String {
    let mut text = (0..rows[0].input.len())
        .map(|i| format!("x{i}"))
        .collect::<Vec<_>>()
        .join("\t");
    text.push_str("\ty\n");
    for row in rows {
        for x in &row.input {
            text.push_str(&format!("{x:.17}\t"));
        }
        text.push_str(&format!("{:.17}\n", row.output));
    }
    text
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 {
        return Err(
            "usage: axon-uic-program-lab <new-output-directory> [seed] [rows-per-task 100..100000]"
                .into(),
        );
    }
    let output = PathBuf::from(&args[0]);
    let seed = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(62917531_u64);
    let rows = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(20_000_usize);
    if !(100..=100_000).contains(&rows) {
        return Err("row limit".into());
    }
    fs::create_dir(&output)?;
    let all_started = Instant::now();
    let mut report = String::from(
        "family,status,examples,candidates,scalar_work,search_terms,search_cost,tool_expansions,instructions,model_bytes,retained_skill_bytes,search_ns,save_reload_ns,inference_ns,validation_ns,correct,holdout,max_absolute_error,expression\n",
    );
    let mut geometric: Option<NumericSkill> = None;
    let mut successes = 0;
    let mut correct_total = 0;
    let mut capsule_bytes = 0;
    let families = [
        Family::Calibration,
        Family::Affine,
        Family::Product,
        Family::Tax,
        Family::Energy,
        Family::Reciprocal,
        Family::Response,
        Family::Distance2,
        Family::Distance3,
    ];
    for (index, family) in families.into_iter().enumerate() {
        let context = format!("program-lab/{}/v1", family.name());
        let training = family.training();
        write_new(
            &output.join(format!("{}.training.tsv", family.name())),
            tsv(&training).as_bytes(),
        )?;
        let budget = NumericBudget {
            max_cost: if matches!(family, Family::Distance3) {
                5
            } else {
                8
            },
            ..NumericBudget::default()
        };
        let library = if matches!(family, Family::Distance3) {
            geometric.iter().collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        if matches!(family, Family::Distance3) {
            let baseline = NumericSkill::learn(
                "no_library",
                &context,
                training.clone(),
                &[],
                budget,
                NumericTolerance::default(),
            );
            let text = match baseline {
                Ok(s) => format!("FOUND\n{}\n{:?}\n", s.program().expression(), s.search()),
                Err(e) => format!("{e}\n"),
            };
            write_new(&output.join("distance3.no-library.txt"), text.as_bytes())?;
        }
        let learned = NumericSkill::learn(
            family.name(),
            &context,
            training.clone(),
            &library,
            budget,
            NumericTolerance::default(),
        );
        let skill = match learned {
            Ok(s) => s,
            Err(e) => {
                println!("{}: REFUSED / {e}", family.name());
                let s = match &e {
                    NumericError::Budget(s) | NumericError::NoProgram(s) => Some(*s),
                    _ => None,
                };
                let mut fields = vec![String::new(); 19];
                fields[0] = family.name().into();
                fields[1] = "REFUSED".into();
                fields[2] = training.len().to_string();
                fields[16] = rows.to_string();
                if let Some(s) = s {
                    fields[3] = s.candidates.to_string();
                    fields[4] = s.scalar_work.to_string();
                    fields[5] = s.retained_terms.to_string();
                    fields[11] = s.elapsed_ns.to_string();
                }
                fields[18] = format!("\"{}\"", e.to_string().replace('"', "\"\""));
                report.push_str(&fields.join(","));
                report.push('\n');
                continue;
            }
        };
        let storage = Instant::now();
        let path = output.join(format!("{}.axnum", family.name()));
        skill.save_new(&path)?;
        let reopened = NumericSkill::open(&path, &context)?;
        if reopened.fingerprint() != skill.fingerprint() {
            return Err("identity changed on reload".into());
        }
        let storage_ns = storage.elapsed().as_nanos();
        // Generate evaluation only after synthesis/storage. It is never passed back to the learner.
        let evaluation = family.heldout(seed.wrapping_add(index as u64 * 104729), rows);
        write_new(
            &output.join(format!("{}.heldout.tsv", family.name())),
            tsv(&evaluation).as_bytes(),
        )?;
        let inferred = Instant::now();
        let mut aggregate = 0.0;
        for row in &evaluation {
            aggregate += black_box(reopened.predict(&context, black_box(&row.input))?);
        }
        black_box(aggregate);
        let inference_ns = inferred.elapsed().as_nanos();
        let validate = Instant::now();
        let mut correct = 0;
        let mut error = 0.0_f64;
        for row in &evaluation {
            let prediction = reopened.predict(&context, &row.input)?;
            error = error.max((prediction - row.output).abs());
            if reopened.tolerance().agrees(prediction, row.output) {
                correct += 1;
            }
        }
        let validation_ns = validate.elapsed().as_nanos();
        let stats = reopened.search();
        let size = reopened.to_bytes().len();
        capsule_bytes += size;
        correct_total += correct;
        let status = if correct == rows {
            successes += 1;
            "PASS"
        } else {
            "REFUTED"
        };
        println!(
            "{}: {status} {correct}/{rows}; {} bytes; search {:.3} ms; {}",
            family.name(),
            size,
            stats.elapsed_ns as f64 / 1e6,
            reopened.program().expression()
        );
        report.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.17},\"{}\"\n",
            family.name(),
            status,
            training.len(),
            stats.candidates,
            stats.scalar_work,
            stats.retained_terms,
            stats.search_cost,
            stats.library_expansions,
            reopened.program().instructions(),
            size,
            reopened.retained_bytes(),
            stats.elapsed_ns,
            storage_ns,
            inference_ns,
            validation_ns,
            correct,
            rows,
            error,
            reopened.program().expression()
        ));
        if matches!(family, Family::Distance2) {
            geometric = Some(reopened);
        }
    }
    write_new(&output.join("results.csv"), report.as_bytes())?;
    let unsupported = [0.2_f64, 0.7, 1.3, 2.1, 3.2]
        .into_iter()
        .map(|x| NumericExample::new(vec![x], x.sin()))
        .collect();
    let sine = NumericSkill::learn(
        "sine",
        "negative/sine",
        unsupported,
        &[],
        NumericBudget {
            max_cost: 5,
            max_terms: 4000,
            max_work: 100_000,
        },
        NumericTolerance::default(),
    );
    let sine_refused = sine.is_err();
    let sine_detail = match sine {
        Ok(s) => format!("UNEXPECTED FIT: {}", s.program().expression()),
        Err(e) => e.to_string(),
    };
    let contradictory = NumericSkill::learn(
        "conflict",
        "negative/conflict",
        vec![
            NumericExample::new(vec![1.0], 1.0),
            NumericExample::new(vec![1.0], 2.0),
        ],
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    );
    let conflict_refused = matches!(contradictory, Err(NumericError::Contradiction));
    let old = NumericSkill::learn(
        "calibration",
        "sensor/revision",
        Family::Calibration.training(),
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )?;
    let mut monitor = NumericMonitor::new(&old);
    monitor.observe(&[10.0], 21.0)?;
    let drift = matches!(
        monitor.observe(&[10.0], 31.0),
        Err(NumericError::Drift { .. })
    );
    let suspended = monitor.predict(&[10.0]).is_err();
    let revision = old.revise(
        "sensor/revision",
        [-7.0, 0.0, 0.25, 3.0, 16.0]
            .into_iter()
            .map(|x| NumericExample::new(vec![x], 3.0 * x + 1.0))
            .collect(),
        &[],
        NumericBudget::default(),
    )?;
    old.save_new(&output.join("sensor-old.axnum"))?;
    revision.save_new(&output.join("sensor-revision.axnum"))?;
    let fresh = NumericSkill::open(&output.join("sensor-revision.axnum"), "sensor/revision")?;
    let recovery = fresh.predict("sensor/revision", &[10.0])? == 31.0
        && old.predict("sensor/revision", &[10.0])? == 21.0
        && fresh.parent() == Some(old.fingerprint());
    let ambiguity = NumericSkill::learn(
        "ambiguous",
        "negative/ambiguity",
        vec![
            NumericExample::new(vec![0.0], 0.0),
            NumericExample::new(vec![1.0], 1.0),
        ],
        &[],
        NumericBudget::default(),
        NumericTolerance::default(),
    )?;
    let ambiguous_wrong = ambiguity.predict("negative/ambiguity", &[3.0])? != 9.0;
    let controls = format!(
        "sine_refused={sine_refused}\nsine_detail={sine_detail}\ncontradiction_refused={conflict_refused}\ndrift_detected={drift}\nmonitor_suspended={suspended}\nrevision_recovered={recovery}\nunderconstrained_data_extrapolated_wrong={ambiguous_wrong}\n"
    );
    write_new(&output.join("controls.txt"), controls.as_bytes())?;
    let summary = format!(
        "seed={seed}\nrows_per_task={rows}\nfamilies_passed={successes}/9\ncorrect_unseen={correct_total}\nlearned_capsules_bytes={capsule_bytes}\nprocess_lifecycle_ns={}\nheldout_is_internal_not_external=true\nprimitive_root_is_supplied=true\nconstant_proposals_include_affine_fit=true\nAGI=false\n",
        all_started.elapsed().as_nanos()
    );
    write_new(&output.join("summary.txt"), summary.as_bytes())?;
    print!("{controls}{summary}");
    if !sine_refused || !conflict_refused || !drift || !suspended || !recovery || !ambiguous_wrong {
        return Err("negative/feedback control failed".into());
    }
    Ok(())
}
