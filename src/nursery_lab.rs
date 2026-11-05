//! Durable adaptive search using the existing relational decoder and benchmark judge.
use crate::{
    demand::genome::bounded_path,
    live_store::checksum,
    numeric::NumericBudget,
    nursery::{Assessment, Nursery, SearchConfig, SearchStop},
    semantic::{Rule, SemanticSeed},
    semantic_cli::{load_lessons, pack},
    semantic_lab::{case_table, generated, independent, score},
    telemetry::{LabTelemetry, Value as V},
    text_skill::{publish_new, read_bounded},
};
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
fn frozen(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    bounded_path(path)?;
    if path.exists() {
        if read_bounded(path, 1_048_576)? != bytes {
            return Err("registry scope hash collision or modified frozen input".into());
        }
    } else {
        publish_new(path, bytes)?;
    }
    Ok(())
}
fn import_history(
    base: &Path,
    pool: &SemanticSeed,
    search: &mut Nursery,
) -> Result<(), Box<dyn Error>> {
    let history = base.parent().unwrap_or(base).join("runs");
    if !history.exists() {
        return Ok(());
    }
    bounded_path(&history)?;
    let signature = pool.structural_bytes()?;
    for (count, entry) in fs::read_dir(history)?.enumerate() {
        if count >= 1024 {
            return Err("history import directory quota".into());
        }
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let dir = entry.path().join("artifacts");
        if !dir.join("result.txt").is_file() {
            continue;
        }
        let source = ["acquired-library.axsem", "library.axsem"]
            .iter()
            .map(|p| dir.join(p))
            .find(|p| p.is_file());
        let Some(source) = source else {
            continue;
        };
        bounded_path(&source)?;
        if SemanticSeed::open(&source)?.structural_bytes()? != signature {
            continue;
        }
        let file = dir.join("population.tsv");
        if !file.is_file() {
            continue;
        }
        bounded_path(&file)?;
        let bytes = read_bounded(&file, 16_777_216)?;
        let mut csv = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_reader(bytes.as_slice());
        let Some(column) = csv
            .headers()?
            .iter()
            .position(|h| h == "bits" || h == "dna")
        else {
            continue;
        };
        let origin = entry.file_name().to_string_lossy().into_owned();
        for row in csv.records() {
            let row = row?;
            search.import_seen(row[column].parse()?, &origin)?;
        }
        // Consolidation can produce a structure not present in the old population table.
        for name in [
            "winner-0",
            "winner-1",
            "winner-2",
            "winner-3",
            "best-candidate",
        ] {
            let file = dir.join(name).join("relational.axsem");
            if file.is_file() {
                bounded_path(&file)?;
                let child = SemanticSeed::open(&file)?;
                search.import_seen(pool.mask_for(&child)?, &origin)?;
            }
        }
    }
    Ok(())
}
pub fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(1..=3).contains(&args.len()) {
        return Err(
            "usage: axon-uic-nursery-lab <new-output> [rng] [new-candidates 100..10000]".into(),
        );
    }
    let output = PathBuf::from(&args[0]);
    bounded_path(&output)?;
    fs::create_dir(&output)?;
    let rng = args
        .get(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(61423917u64);
    let limit = args
        .get(2)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1000usize);
    if !(100..=10000).contains(&limit) {
        return Err("nursery batch limit must be 100..10000".into());
    }
    let project = Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = std::env::var_os("AXON_NURSERY_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| project.join(".axon-memory/nursery"));
    bounded_path(&base)?;
    fs::create_dir_all(&base)?;
    let mut t = LabTelemetry::from_environment("nursery-v1");
    let started = Instant::now();
    t.emit(
        "lab_start",
        &[
            ("method", V::Text("Nursery::open")),
            ("population", V::Count(64)),
            ("rows", V::Count(limit as u64)),
        ],
    )?;
    // Once acquired, the library is reused; its exact structural identity scopes the registry.
    let path = base.join("relational-library.axsem");
    let pool = if path.exists() {
        SemanticSeed::open(&path)?
    } else {
        let ds = load_lessons(&project.join("evals/fixtures/relational-v1-lessons.tsv"))?;
        let mut rules = Vec::new();
        for (i, (name, examples)) in ds.iter().enumerate() {
            let clock = Instant::now();
            let r = Rule::learn(examples, NumericBudget::default())?;
            t.emit(
                "skill",
                &[
                    ("method", V::Text("Rule::learn")),
                    ("case", V::Count(i as u64)),
                    ("name", V::Text(name)),
                    ("expression", V::Text(&r.expression())),
                    ("elapsed_ns", V::Count(clock.elapsed().as_nanos() as u64)),
                ],
            )?;
            rules.push(r);
        }
        let pool = SemanticSeed::new(rules)?;
        pool.save_new(&path)?;
        pool
    };
    if pool.rules().is_empty() || pool.rules().len() > 19 {
        return Err("this decoder supports 1..19 inherited rule loci".into());
    }
    pool.save_new(&output.join("library.axsem"))?;
    // These formerly reserved probes are now explicitly development challenges, including failures.
    let mut cases = generated(0x519a37, 48);
    cases.extend(independent(
        &project.join("evals/fixtures/relational-v2-independent.tsv"),
    )?);
    case_table(&output.join("selection.tsv"), &cases)?;
    let pool_bytes = pool.structural_bytes()?;
    let suite = fs::read(output.join("selection.tsv"))?;
    let scope = format!(
        "semantic-subset-v1;judge-v1;pool={:016X};suite={:016X}",
        checksum(&pool_bytes),
        checksum(&suite)
    );
    let directory = base.join(format!(
        "registry-{:016X}-{:016X}",
        checksum(&pool_bytes),
        checksum(&suite)
    ));
    bounded_path(&directory)?;
    fs::create_dir_all(&directory)?;
    frozen(&directory.join("decoder.signature"), &pool_bytes)?;
    frozen(&directory.join("selection.tsv"), &suite)?;
    let config = SearchConfig {
        maximum: (1 << pool.rules().len()) - 1,
        ..SearchConfig::default()
    };
    let journal = directory.join("search.axep");
    let mut search = Nursery::open(&journal, &scope, config, rng)?;
    import_history(&base, &pool, &mut search)?;
    let previous = search.completed();
    let mut done = 0;
    let mut width = search.digits();
    let mut batch = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("new-candidates.tsv"))?,
    );
    batch.write_record([
        "dna",
        "parent",
        "digits",
        "correct",
        "checked",
        "wrong",
        "bytes",
        "elapsed_ns",
    ])?;
    frozen(&output.join("protocol.txt"),format!("protocol=nursery-v1\nscope={scope}\nchallenge_count={}\nmax_interpretable_dna={}\nmax_global_candidates={}\nbatch_new_candidates={limit}\ninitial_digits=1\ngrowth=one decimal place on capability plateau or width exhaustion\nplateau=192 new evaluations without capability improvement\nidentity=canonical decimal integer decoded to rule-inclusion loci, not RNG or lineage\nuniqueness=per exact decoder and fixed challenge suite, across restarts and jobs\nlegacy_pilot=compatible canonical structures imported as exclusion, never as invented current fitness\nselection=correctness, resource eligibility, memory/time Pareto parent portfolio\nlimits=131072 encoded bytes; 1000000000 evaluation ns; one worker; no energy measurement\nreserve=these v2 probes are consumed development, not a new blind reserve\nrepresentation=acquired relational rules only; more digits cannot invent missing grammar\nregistry={}\n",cases.len(),config.maximum,config.max_candidates,directory.display()).as_bytes())?;
    t.emit(
        "nursery_progress",
        &[
            ("method", V::Text("Nursery::resume")),
            ("candidates", V::Count(previous as u64)),
            ("unique", V::Count(search.seen() as u64)),
            ("genes", V::Count(width as u64)),
            ("unfinished", V::Count(search.unfinished() as u64)),
            ("imported", V::Count(search.imported() as u64)),
        ],
    )?;
    while done < limit {
        let Some(p) = search.propose()? else {
            break;
        };
        if search.digits() != width {
            width = search.digits();
            t.emit(
                "dna_growth",
                &[
                    ("method", V::Text("Nursery::grow")),
                    ("genes", V::Count(width as u64)),
                    ("candidates", V::Count(search.completed() as u64)),
                ],
            )?;
        }
        let brain = pool.subset(p.dna)?;
        let bytes = brain.to_bytes()?.len() as u64;
        let clock = Instant::now();
        let s = score(&brain, &cases);
        let ns = clock.elapsed().as_nanos() as u64;
        let a = Assessment {
            checked: s.checked,
            correct: s.correct,
            wrong: s.wrong,
            encoded_bytes: bytes,
            elapsed_ns: ns,
        };
        search.commit(p, a)?;
        done += 1;
        batch.write_record([
            p.dna.to_string(),
            p.parent
                .map(|n| n.to_string())
                .unwrap_or_else(|| "_".into()),
            p.digits.to_string(),
            s.correct.to_string(),
            s.checked.to_string(),
            s.wrong.to_string(),
            bytes.to_string(),
            ns.to_string(),
        ])?;
        t.emit(
            "nursery_candidate",
            &[
                ("method", V::Text("Nursery::commit")),
                ("genotype", V::Text(&p.dna.to_string())),
                (
                    "parent_a",
                    V::Text(
                        &p.parent
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "_".into()),
                    ),
                ),
                ("genes", V::Count(width as u64)),
                ("correct", V::Count(s.correct)),
                ("checked", V::Count(s.checked)),
                ("wrong", V::Count(s.wrong)),
                ("encoded_bytes", V::Count(bytes)),
                ("elapsed_ns", V::Count(ns)),
            ],
        )?;
        if done % 64 == 0 {
            batch.flush()?;
            let best = search.best().expect("evaluated");
            t.emit(
                "nursery_progress",
                &[
                    ("method", V::Text("Nursery::selection")),
                    ("candidates", V::Count(search.completed() as u64)),
                    ("unique", V::Count(search.seen() as u64)),
                    ("genes", V::Count(width as u64)),
                    ("correct", V::Count(best.assessment.correct)),
                    ("checked", V::Count(best.assessment.checked)),
                    ("wrong", V::Count(best.assessment.wrong)),
                    ("genotype", V::Text(&best.dna.to_string())),
                ],
            )?;
        }
    }
    batch.flush()?;
    let stop = search.stop().unwrap_or(SearchStop::BatchLimit);
    if let Some(best) = search.best() {
        let g = pack(&pool.subset(best.dna)?, &output.join("best-candidate"))?;
        t.emit(
            "genome",
            &[
                ("method", V::Text("Genome::capture")),
                ("name", V::Text("best-candidate; admission is separate")),
                ("encoded_bytes", V::Count(g.encoded_bytes() as u64)),
                ("seed_digits", V::Count(g.number().len() as u64)),
                ("artifact", V::Text("best-candidate/seed.axnum")),
            ],
        )?;
    }
    let mut all = csv::WriterBuilder::new().delimiter(b'\t').from_writer(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("population.tsv"))?,
    );
    all.write_record([
        "dna",
        "parent",
        "correct",
        "checked",
        "wrong",
        "bytes",
        "elapsed_ns",
    ])?;
    for i in search.entries() {
        all.write_record([
            i.dna.to_string(),
            i.parent
                .map(|p| p.to_string())
                .unwrap_or_else(|| "_".into()),
            i.assessment.correct.to_string(),
            i.assessment.checked.to_string(),
            i.assessment.wrong.to_string(),
            i.assessment.encoded_bytes.to_string(),
            i.assessment.elapsed_ns.to_string(),
        ])?;
    }
    all.flush()?;
    search.snapshot_new(&output.join("registry-snapshot.axep"))?;
    let best = search.best();
    let correct = best.map(|i| i.assessment.correct).unwrap_or(0);
    let checked = best.map(|i| i.assessment.checked).unwrap_or(0);
    frozen(&output.join("result.txt"),format!("outcome={}\nnew_candidates={done}\nprevious_candidates={previous}\ncompleted={}\nunique={}\ndna_digits={}\ngrowths={}\nunfinished={}\ncorrect={correct}\nchecked={checked}\nall_tests_passed={}\nelapsed_ns={}\n",stop.name(),search.completed(),search.seen(),search.digits(),search.growths(),search.unfinished(),stop==SearchStop::Winner,started.elapsed().as_nanos()).as_bytes())?;
    t.emit(
        "lab_result",
        &[
            ("method", V::Text("Nursery::result")),
            ("status", V::Text(stop.name())),
            ("new_candidates", V::Count(done as u64)),
            ("previous_candidates", V::Count(previous as u64)),
            ("candidates", V::Count(search.completed() as u64)),
            ("unique", V::Count(search.seen() as u64)),
            ("genes", V::Count(search.digits() as u64)),
            ("growths", V::Count(search.growths() as u64)),
            ("unfinished", V::Count(search.unfinished() as u64)),
            ("imported", V::Count(search.imported() as u64)),
            ("correct", V::Count(correct)),
            ("checked", V::Count(checked)),
            ("admitted", V::Flag(stop == SearchStop::Winner)),
            ("elapsed_ns", V::Count(started.elapsed().as_nanos() as u64)),
            ("artifact", V::Text("result.txt")),
        ],
    )?;
    println!(
        "nursery {}: {done} new, {} unique, {correct}/{checked}; {} digits; no blanket AGI claim",
        stop.name(),
        search.seen(),
        search.digits()
    );
    Ok(())
}
