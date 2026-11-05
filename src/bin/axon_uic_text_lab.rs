use axon_uic::{
    TextBudget, TextExample, TextForgeError, TextPrediction, TextSkill, TextSkillError,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::Read,
    path::PathBuf,
    time::Instant,
};

struct Task {
    train: Vec<TextExample>,
    tests: Vec<(TextExample, String)>,
}
fn main() {
    if let Err(e) = run() {
        eprintln!("text lab: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut fixture = None;
    let mut report = None;
    let mut skills = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = PathBuf::from(args.next().ok_or("missing option value")?);
        match flag.as_str() {
            "--fixture" if fixture.is_none() => fixture = Some(value),
            "--report" if report.is_none() => report = Some(value),
            "--skills" if skills.is_none() => skills = Some(value),
            _ => return Err("unknown or duplicate option".into()),
        }
    }
    let fixture = fixture.ok_or("--fixture is required")?;
    let report = report.ok_or("--report is required")?;
    let skills = skills.ok_or("--skills is required")?;
    if skills.exists() || report.exists() {
        return Err("existing outputs are preserved".into());
    }
    let file = File::open(fixture)?;
    if file.metadata()?.len() > 1024 * 1024 {
        return Err("fixture byte limit".into());
    }
    let mut bytes = Vec::new();
    file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("fixture byte limit".into());
    }
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(bytes.as_slice());
    if reader.headers()?
        != &csv::StringRecord::from(vec!["task", "split", "input", "output", "expected_status"])
    {
        return Err("invalid fixture headers".into());
    }
    let mut tasks = BTreeMap::<String, Task>::new();
    for row in reader.records() {
        let row = row?;
        if row[0].is_empty()
            || row[0].len() > 64
            || !row[0]
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            return Err("invalid task name".into());
        }
        let task = tasks.entry(row[0].into()).or_insert_with(|| Task {
            train: Vec::new(),
            tests: Vec::new(),
        });
        let example = TextExample::new(&row[2], &row[3]);
        match &row[1] {
            "train" => task.train.push(example),
            "test"
                if ["agreed", "needs_example", "out_of_domain", "unsupported"]
                    .contains(&&row[4]) =>
            {
                task.tests.push((example, row[4].into()))
            }
            _ => return Err("invalid split/status".into()),
        }
    }
    if tasks.len() > 128 {
        return Err("task count limit".into());
    }
    fs::create_dir_all(&skills)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&report)?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record([
        "task",
        "input",
        "expected_status",
        "expected_output",
        "status",
        "observed_output",
        "correct",
        "fit_ns",
        "load_ns",
        "predict_ns",
        "artifact_bytes",
        "pieces",
        "search_work",
    ])?;
    let mut correct = 0;
    let mut cases = 0;
    let mut wrong_agreed = 0;
    for (name, task) in &tasks {
        let inputs = task.train.iter().map(|e| &e.input).collect::<BTreeSet<_>>();
        if task.train.is_empty()
            || task.tests.is_empty()
            || task.tests.iter().any(|(e, _)| inputs.contains(&e.input))
        {
            return Err("empty task or leaked test input".into());
        }
        // Test targets are never passed to synthesis, refinement or selection.
        let start = Instant::now();
        let fitted = TextSkill::learn(
            name,
            format!("text-lab/{name}"),
            task.train.clone(),
            TextBudget::default(),
        );
        let fit_ns = start.elapsed().as_nanos();
        let (skill, failure, artifact_bytes, load_ns) = match fitted {
            Ok(skill) => {
                let path = skills.join(format!("{name}.axtext"));
                let artifact_bytes = skill.save_new(&path)?;
                let start = Instant::now();
                let loaded = TextSkill::load(&path)?;
                let load_ns = start.elapsed().as_nanos();
                assert_eq!(skill, loaded);
                (Some(loaded), "", artifact_bytes, load_ns)
            }
            Err(TextSkillError::Forge(TextForgeError::Unsupported)) => (None, "unsupported", 0, 0),
            Err(TextSkillError::Forge(TextForgeError::Limit(_))) => {
                (None, "budget_exhausted", 0, 0)
            }
            Err(e) => return Err(e.into()),
        };
        for (example, expected_status) in &task.tests {
            let start = Instant::now();
            let (status, observed) = if let Some(skill) = &skill {
                let prediction = skill.predict(skill.context(), &example.input)?;
                match prediction {
                    TextPrediction::Agreed(w) => ("agreed", w.output),
                    TextPrediction::Ambiguous { first, second } => (
                        "needs_example",
                        format!("{:?} | {:?}", first.output, second.output),
                    ),
                    TextPrediction::OutOfDomain => ("out_of_domain", String::new()),
                }
            } else {
                (failure, String::new())
            };
            let predict_ns = start.elapsed().as_nanos();
            let passed =
                status == expected_status && (status != "agreed" || observed == example.output);
            cases += 1;
            correct += usize::from(passed);
            wrong_agreed += usize::from(status == "agreed" && !passed);
            writer.write_record([
                name.clone(),
                example.input.clone(),
                expected_status.clone(),
                example.output.clone(),
                status.into(),
                observed,
                passed.to_string(),
                fit_ns.to_string(),
                load_ns.to_string(),
                predict_ns.to_string(),
                artifact_bytes.to_string(),
                skill
                    .as_ref()
                    .map_or(0, |s| s.program().pieces())
                    .to_string(),
                skill
                    .as_ref()
                    .map_or(0, |s| s.program().search_work())
                    .to_string(),
            ])?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    println!(
        "tasks={}\ncases={cases}\ncorrect={correct}\nwrong_agreed={wrong_agreed}\nreport={}",
        tasks.len(),
        report.display()
    );
    Ok(())
}
