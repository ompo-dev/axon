use crate::numeric::library::{DevelopmentBudget, MiningBudget, NumericLibrary};
use crate::{
    cli::CliError,
    numeric::*,
    text_skill::{TextSkillError, publish_new, read_bounded},
};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::Path,
};

pub(crate) const USAGE: &str = concat!(
    "\n  axon numeric-fit <examples.tsv> --name <name> --context <context> --skill <new.axnum> [--library <file.axnum> | --mined-library <file.axlib> --library-context <context>] [--max-cost <1..12>] [--max-terms <1..200000>]",
    "\n  axon numeric-develop <examples.tsv> --name <name> --context <context> --skill <new.axnum> --mined-library <file.axlib> --library-context <context> [--max-cost <1..12>] [--max-terms <1..200000>] [--max-attempts <1..16>] [--total-work <1..100000000>]",
    "\n  axon numeric-revise <old.axnum> <new-observations.tsv> --context <context> --skill <new.axnum> [--library <file.axnum> | --mined-library <file.axlib> --library-context <context>] [--max-cost <1..12>] [--max-terms <1..200000>]",
    "\n  axon numeric-inspect <skill.axnum> --context <context>",
    "\n  axon numeric-predict <skill.axnum> --context <context> --values <x0,x1,...> [--backend stack|shared]",
    "\n  axon numeric-eval <skill.axnum> <unseen.tsv> --context <context> [--backend stack|shared]",
    "\n  axon numeric-mine <sources.tsv> --context <context> --library <new.axlib>",
    "\n  axon numeric-library-inspect <file.axlib> --context <context>",
    "\n  axon numeric-evolve <training.tsv> <admission.tsv> --name <name> --context <context> --skill <new.axevo> [--ancestors <paths.txt>] [--seed <integer>] [--population <4..128>] [--generations <1..10000>] [--genes <2..64>] [--max-work <1..200000000>] [--refinement hybrid|mutation-only|balanced]",
    "\n  axon evolved-inspect <skill.axevo> --context <context>",
    "\n  axon evolved-predict <skill.axevo> --context <context> --values <x0,x1,...> [--backend stack|shared]",
    "\n  axon circuit-predict <circuit-number.txt> <x0,x1,...>",
    "\n  axon circuit-export <skill.axnum> --context <source-context> --output <new-directory>",
    "\n  axon dna-create <capsules.txt> --context <scope> --dna <new.axdna>",
    "\n  axon dna-inspect <file.axdna> --context <scope>",
    "\n  axon dna-mutate <file.axdna> --context <scope> --dna <new.axdna> --seed <integer>",
    "\n  axon dna-extend <file.axdna> <skill.axevo> --context <scope> --dna <new.axdna>",
    "\n  axon dna-predict <file.axdna> --context <scope> --module <hex-id> --values <x0,x1,...>",
    "\n  numeric-evolve also accepts --genome <file.axdna> --genome-context <scope>; other search traits/ancestors cannot override that genotype.",
);

fn options(
    args: &[String],
    positions: usize,
    allowed: &[&str],
) -> Result<BTreeMap<String, String>, CliError> {
    if args.len() < positions + 1 || args[1..=positions].iter().any(|s| s.starts_with("--")) {
        return Err(CliError::Usage);
    }
    let mut map = BTreeMap::new();
    let mut i = positions + 1;
    while i < args.len() {
        let value = args.get(i + 1).ok_or(CliError::Usage)?;
        if value.starts_with("--")
            || !allowed.contains(&args[i].as_str())
            || map.insert(args[i].clone(), value.clone()).is_some()
        {
            return Err(CliError::Usage);
        }
        i += 2;
    }
    Ok(map)
}
fn required<'a>(options: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, CliError> {
    options.get(key).map(String::as_str).ok_or(CliError::Usage)
}
pub(crate) fn examples(path: &Path, maximum: usize) -> Result<Vec<NumericExample>, NumericError> {
    let bytes = read_bounded(path, 2_097_152)?;
    let text = std::str::from_utf8(&bytes).map_err(|_| NumericError::Invalid("TSV UTF-8"))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or(NumericError::Invalid("empty TSV"))?;
    let arity = header_arity(header)?;
    let mut result = Vec::new();
    for line in lines {
        if result.len() >= maximum {
            return Err(NumericError::Invalid("TSV row limit"));
        }
        let (values, output) = row_values(line, arity)?;
        result.push(NumericExample::new(values[..arity].to_vec(), output));
    }
    if result.is_empty() {
        return Err(NumericError::Invalid("no TSV records"));
    }
    Ok(result)
}
fn header_arity(header: &str) -> Result<usize, NumericError> {
    let names: Vec<&str> = header.split('\t').collect();
    if names.len() < 2
        || names.len() > 5
        || names.last() != Some(&"y")
        || names[..names.len() - 1]
            .iter()
            .enumerate()
            .any(|(i, name)| *name != format!("x{i}"))
    {
        return Err(NumericError::Invalid(
            "TSV header must be x0, x1, ... and y, separated by tabs",
        ));
    }
    Ok(names.len() - 1)
}
fn row_values(line: &str, arity: usize) -> Result<([f64; 4], f64), NumericError> {
    let mut fields = line.split('\t');
    let mut values = [0.0; 4];
    let mut number = || -> Result<f64, NumericError> {
        let n = fields
            .next()
            .ok_or(NumericError::Invalid("TSV dimensions"))?
            .parse::<f64>()
            .map_err(|_| NumericError::Invalid("TSV number"))?;
        if !n.is_finite() {
            return Err(NumericError::Invalid("TSV finite numbers"));
        }
        Ok(n)
    };
    for value in &mut values[..arity] {
        *value = number()?;
    }
    let output = number()?;
    if fields.next().is_some() {
        return Err(NumericError::Invalid("TSV dimensions"));
    }
    Ok((values, output))
}
fn evaluation(
    path: &Path,
    skill: &NumericSkill,
    execution: Option<&NumericExecution>,
) -> Result<(usize, usize, usize, f64), NumericError> {
    const MAX_BYTES: u64 = 16_777_216;
    let file = File::open(path).map_err(TextSkillError::from)?;
    let mut reader = BufReader::new(file.take(MAX_BYTES + 1));
    let mut line = String::new();
    let mut bytes = 0_u64;
    let mut read = |line: &mut String| -> Result<usize, NumericError> {
        line.clear();
        // Limit each line before allocation can follow an unbounded external record.
        let n = reader
            .by_ref()
            .take(4097)
            .read_line(line)
            .map_err(TextSkillError::from)?;
        bytes += n as u64;
        if n > 4096 || bytes > MAX_BYTES {
            return Err(NumericError::Invalid("evaluation byte or line limit"));
        }
        Ok(n)
    };
    if read(&mut line)? == 0 {
        return Err(NumericError::Invalid("empty TSV"));
    }
    let arity = header_arity(line.trim_end_matches(['\r', '\n']))?;
    let mut total = 0;
    let mut correct = 0;
    let mut refused = 0;
    let mut maximum_error = 0.0_f64;
    while read(&mut line)? > 0 {
        if total >= 100_000 {
            return Err(NumericError::Invalid("TSV row limit"));
        }
        let (input, expected) = row_values(line.trim_end_matches(['\r', '\n']), arity)?;
        total += 1;
        let result = match execution {
            Some(e) => e.evaluate(&input[..arity]),
            None => skill.program().evaluate(&input[..arity]),
        };
        match result {
            Ok(value) => {
                maximum_error = maximum_error.max((value - expected).abs());
                if skill.tolerance().agrees(value, expected) {
                    correct += 1;
                }
            }
            Err(_) => refused += 1,
        }
    }
    if total == 0 {
        return Err(NumericError::Invalid("no TSV records"));
    }
    Ok((correct, total, refused, maximum_error))
}
fn inspect(skill: &NumericSkill) -> String {
    let s = skill.search();
    let mut text = format!(
        "NUMERICAL CAPABILITY / EMPIRICAL, NOT A GENERAL PROOF\nname={}\ncontext={}\nrevision={}\nparent={:?}\nidentity={:016x}\nexpression={}\ninputs={}\ninstructions={}\nexamples={}\nsearch_cost={}\ncompile_time_tool_expansions={}\ncandidates={}\nscalar_work={}\nretained_search_terms={}\nsearch_ns={}\nencoded_bytes={}\nretained_skill_bytes={}\nabsolute_tolerance={}\nrelative_tolerance={}\ndependencies={:?}\n",
        skill.name(),
        skill.context(),
        skill.revision(),
        skill.parent(),
        skill.fingerprint(),
        skill.program().expression(),
        skill.program().arity(),
        skill.program().instructions(),
        skill.examples().len(),
        s.search_cost,
        s.library_expansions,
        s.candidates,
        s.scalar_work,
        s.retained_terms,
        s.elapsed_ns,
        skill.to_bytes().len(),
        skill.retained_bytes(),
        skill.tolerance().absolute,
        skill.tolerance().relative,
        skill.dependencies()
    );
    let execution = skill.program().compile_execution();
    text.push_str(&format!("shared_execution_instructions={}\neliminated_duplicate_instructions={}\nshared_execution_scratch_bytes={}\nshared_execution_retained_bytes={}\n", execution.instructions(), execution.eliminated_instructions(), execution.scratch_bytes(), execution.retained_bytes()));
    text
}
fn inspect_library(library: &NumericLibrary) -> String {
    let s = library.statistics();
    let mut text = format!(
        "MINED NUMERICAL LIBRARY / STRUCTURAL COMPRESSION, NOT A GENERAL PROOF\ncontext={}\nidentity={:016x}\nsupplied_sources={}\nunique_structural_sources={}\nmodules={}\npatterns={}\ntree_visits={}\noriginal_tree_nodes={}\nrewritten_tree_nodes={}\ndefinition_nodes={}\nnet_saved_tree_nodes={}\nmining_ns={}\nencoded_bytes={}\nretained_library_bytes={}\n",
        library.context(),
        library.fingerprint(),
        s.supplied_sources,
        s.unique_sources,
        library.modules().len(),
        s.patterns,
        s.tree_visits,
        s.original_nodes,
        s.rewritten_nodes,
        s.definition_nodes,
        s.net_saved_nodes,
        s.elapsed_ns,
        library.to_bytes().len(),
        library.retained_bytes()
    );
    for (i, module) in library.modules().iter().enumerate() {
        text.push_str(&format!(
            "module_{i}={:016x}; inputs={}; net_saved_nodes={}; support={:?}; expression={}\n",
            module.fingerprint(),
            module.program().arity(),
            module.net_saved_nodes(),
            module.sources(),
            module.program().expression()
        ));
    }
    text
}
fn mine_manifest(path: &Path, context: &str) -> Result<NumericLibrary, NumericError> {
    let data = read_bounded(path, 32_768)?;
    let text = std::str::from_utf8(&data).map_err(|_| NumericError::Invalid("manifest UTF-8"))?;
    let mut lines = text.lines();
    if lines.next() != Some("skill\tcontext") {
        return Err(NumericError::Invalid(
            "manifest header must be skill and context separated by a tab",
        ));
    }
    let mut skills = Vec::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 2 || fields[0].is_empty() || skills.len() >= 64 {
            return Err(NumericError::Invalid("manifest fields or count"));
        }
        let file = path.parent().unwrap_or(Path::new(".")).join(fields[0]);
        skills.push(NumericSkill::open(&file, fields[1])?);
    }
    NumericLibrary::mine(
        context,
        &skills.iter().collect::<Vec<_>>(),
        MiningBudget::default(),
    )
}
fn evolved_sources(path: &Path) -> Result<Vec<evolution::EvolvedSkill>, NumericError> {
    let bytes = read_bounded(path, 65536)?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| NumericError::Invalid("ancestor manifest UTF-8"))?;
    let mut ancestors = Vec::new();
    for line in text.lines() {
        if ancestors.len() >= 8 || line.trim().is_empty() {
            return Err(NumericError::Invalid("ancestor manifest rows"));
        }
        let file = path.parent().unwrap_or(Path::new(".")).join(line.trim());
        ancestors.push(evolution::EvolvedSkill::from_bytes(&read_bounded(
            &file, 65536,
        )?)?);
    }
    Ok(ancestors)
}
pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let cmd = args[0].as_str();
    if cmd == "circuit-export" {
        let opts = options(args, 1, &["--context", "--output"])?;
        let skill = NumericSkill::open(Path::new(&args[1]), required(&opts, "--context")?)?;
        let dna = evolution::constructive::CircuitDna::from_program(skill.program())?;
        let bytes = dna.to_bytes();
        let number = dna.number();
        let output = Path::new(required(&opts, "--output")?);
        std::fs::create_dir(output)
            .map_err(TextSkillError::from)
            .map_err(NumericError::from)?;
        publish_new(&output.join("circuit.axcg"), &bytes).map_err(NumericError::from)?;
        publish_new(&output.join("circuit-number.txt"), number.as_bytes())
            .map_err(NumericError::from)?;
        return Ok(format!(
            "status=circuit_exported\nsource_examples_omitted={}\ncircuit_bytes={}\ndecimal_digits={}\nprogram_instructions={}\nruntime_context={}\nscope=compiled arithmetic only; source semantic context, units and privacy are not certified by this export",
            skill.examples().len(),
            bytes.len(),
            number.len(),
            skill.program().instructions(),
            evolution::constructive::CONTEXT
        ));
    }
    if cmd == "circuit-predict" {
        if args.len() != 3 {
            return Err(CliError::Usage);
        }
        let bytes = read_bounded(Path::new(&args[1]), 196609).map_err(NumericError::from)?;
        let number =
            std::str::from_utf8(&bytes).map_err(|_| NumericError::Invalid("circuit UTF-8"))?;
        let dna = evolution::constructive::CircuitDna::from_number(number)?;
        let input = args[2]
            .split(',')
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|_| NumericError::Invalid("circuit input"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let program = dna.program()?;
        let value = program.evaluate(&input)?;
        return Ok(format!(
            "status=calculated\nvalue={value}\nexpression={}\ngenes={}\ndecimal_digits={}\npersonal_episodes=0\ntraining_required=false",
            program.expression(),
            dna.genes(),
            number.len()
        ));
    }
    let positions = if matches!(
        cmd,
        "numeric-revise" | "numeric-eval" | "numeric-evolve" | "dna-extend"
    ) {
        2
    } else {
        1
    };
    let allowed: &[&str] = match cmd {
        "numeric-evolve" => &[
            "--name",
            "--context",
            "--skill",
            "--ancestors",
            "--seed",
            "--population",
            "--generations",
            "--genes",
            "--max-work",
            "--refinement",
            "--genome",
            "--genome-context",
        ],
        "dna-create" | "dna-extend" => &["--context", "--dna"],
        "dna-inspect" => &["--context"],
        "dna-mutate" => &["--context", "--dna", "--seed"],
        "dna-predict" => &["--context", "--module", "--values"],
        "evolved-inspect" => &["--context"],
        "evolved-predict" => &["--context", "--values", "--backend"],
        "numeric-fit" => &[
            "--max-terms",
            "--name",
            "--context",
            "--skill",
            "--library",
            "--max-cost",
            "--mined-library",
            "--library-context",
        ],
        "numeric-revise" => &[
            "--max-terms",
            "--context",
            "--skill",
            "--library",
            "--max-cost",
            "--mined-library",
            "--library-context",
        ],
        "numeric-develop" => &[
            "--max-terms",
            "--name",
            "--context",
            "--skill",
            "--mined-library",
            "--library-context",
            "--max-cost",
            "--max-attempts",
            "--total-work",
        ],
        "numeric-inspect" | "numeric-library-inspect" => &["--context"],
        "numeric-eval" => &["--context", "--backend"],
        "numeric-mine" => &["--context", "--library"],
        "numeric-predict" => &["--context", "--values", "--backend"],
        _ => return Err(CliError::Usage),
    };
    let options = options(args, positions, allowed)?;
    let context = required(&options, "--context")?;
    if cmd.starts_with("dna-") {
        let seed = if cmd == "dna-create" {
            let mut seed = seed::DevelopmentSeed::new(
                context,
                evolution::EvolutionBudget::balanced(),
                65_536,
            )?;
            for skill in evolved_sources(Path::new(&args[1]))? {
                seed = seed.extend(context, skill)?;
            }
            seed
        } else {
            seed::DevelopmentSeed::open(Path::new(&args[1]), context)?
        };
        let seed = match cmd {
            "dna-mutate" => seed.mutate(
                context,
                required(&options, "--seed")?
                    .parse()
                    .map_err(|_| CliError::Usage)?,
            )?,
            "dna-extend" => seed.extend(
                context,
                evolution::EvolvedSkill::from_bytes(
                    &read_bounded(Path::new(&args[2]), 65536).map_err(NumericError::from)?,
                )?,
            )?,
            "dna-predict" => {
                let identity = u64::from_str_radix(required(&options, "--module")?, 16)
                    .map_err(|_| CliError::Usage)?;
                let module = seed
                    .skills()
                    .iter()
                    .find(|s| s.fingerprint() == identity)
                    .ok_or(NumericError::Invalid("unknown DNA module"))?;
                let input: Vec<f64> = required(&options, "--values")?
                    .split(',')
                    .map(|v| v.parse().map_err(|_| CliError::Usage))
                    .collect::<Result<_, _>>()?;
                return Ok(format!(
                    "value={:.15}\ngenotype={:016x}\nmodule={identity:016x}",
                    module.predict(module.context(), &input)?,
                    seed.fingerprint()
                ));
            }
            _ => seed,
        };
        if cmd != "dna-inspect" {
            seed.save_new(Path::new(required(&options, "--dna")?))?;
        }
        let p = seed.policy();
        let modules = seed
            .skills()
            .iter()
            .map(|s| {
                format!(
                    "{:016x}:{}:{}",
                    s.fingerprint(),
                    s.context(),
                    s.program().expression()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(format!(
            "DEVELOPMENT DNA\ncontext={}\ngenotype={:016x}\nrevision={:016x}\ngeneration={}\nparents={:016x},{:016x}\nbytes={}\nbyte_cap={}\npopulation={}\ngenes={}\nmutations={}\nmodules={}\n{modules}",
            seed.context(),
            seed.fingerprint(),
            seed.revision_id(),
            seed.generation(),
            seed.parents()[0],
            seed.parents()[1],
            seed.to_bytes().len(),
            seed.byte_cap(),
            p.population,
            p.genes,
            p.mutations,
            seed.skills().len()
        ));
    }
    if cmd == "numeric-evolve" {
        let mut budget = evolution::EvolutionBudget::default();
        for (key, slot) in [
            ("--population", &mut budget.population),
            ("--generations", &mut budget.generations),
            ("--genes", &mut budget.genes),
        ] {
            if let Some(value) = options.get(key) {
                *slot = value.parse().map_err(|_| CliError::Usage)?;
            }
        }
        if let Some(value) = options.get("--max-work") {
            budget.max_work = value.parse().map_err(|_| CliError::Usage)?;
        }
        budget.local_refinement = match options
            .get("--refinement")
            .map(String::as_str)
            .unwrap_or("hybrid")
        {
            "hybrid" => true,
            "mutation-only" => false,
            "balanced" => {
                let policy = evolution::EvolutionBudget::balanced();
                budget.refinement_slice_work = policy.refinement_slice_work;
                budget.refinement_share_percent = policy.refinement_share_percent;
                budget.refine_inherited_calls = policy.refine_inherited_calls;
                true
            }
            _ => return Err(CliError::Usage),
        };
        let mut ancestors = Vec::new();
        if let Some(manifest) = options.get("--ancestors") {
            ancestors = evolved_sources(Path::new(manifest))?;
        }
        if let Some(file) = options.get("--genome") {
            if [
                "--ancestors",
                "--population",
                "--generations",
                "--genes",
                "--refinement",
            ]
            .iter()
            .any(|key| options.contains_key(*key))
            {
                return Err(CliError::Usage);
            }
            let genome = seed::DevelopmentSeed::open(
                Path::new(file),
                required(&options, "--genome-context")?,
            )?;
            budget = genome.budget(
                options
                    .get("--max-work")
                    .map(|v| v.parse().map_err(|_| CliError::Usage))
                    .transpose()?
                    .unwrap_or(genome.policy().max_work),
            )?;
            ancestors = genome.skills().to_vec();
        } else if options.contains_key("--genome-context") {
            return Err(CliError::Usage);
        }
        let refs: Vec<_> = ancestors.iter().collect();
        let seed = options
            .get("--seed")
            .map(|s| s.parse::<u64>().map_err(|_| CliError::Usage))
            .transpose()?
            .unwrap_or(82641973);
        let run = evolution::EvolvedSkill::evolve(
            required(&options, "--name")?,
            context,
            examples(Path::new(&args[1]), 32)?,
            examples(Path::new(&args[2]), 32)?,
            &refs,
            seed,
            budget,
            NumericTolerance::default(),
        )?;
        let s = run.statistics;
        let Some(skill) = run.skill else {
            return Err(NumericError::Evolution {
                reason: match run.stop {
                    evolution::EvolutionStop::Refuted => "separate evidence refuted candidate",
                    evolution::EvolutionStop::WorkLimit => "work limit",
                    evolution::EvolutionStop::GenerationLimit => "generation limit",
                    _ => "missing model",
                },
                evaluations: s.evaluations,
                work: s.work,
                loss: s.best_loss,
            }
            .into());
        };
        skill.save_new(Path::new(required(&options, "--skill")?))?;
        return Ok(evolved_inspect(&skill));
    }
    if matches!(cmd, "evolved-inspect" | "evolved-predict") {
        let skill = evolution::EvolvedSkill::open(Path::new(&args[1]), context)?;
        if cmd == "evolved-inspect" {
            return Ok(evolved_inspect(&skill));
        }
        let input = required(&options, "--values")?
            .split(',')
            .map(|s| s.parse::<f64>().map_err(|_| CliError::Usage))
            .collect::<Result<Vec<_>, _>>()?;
        let value = match options
            .get("--backend")
            .map(String::as_str)
            .unwrap_or("stack")
        {
            "stack" => skill.predict(context, &input)?,
            "shared" => skill.program().compile_execution().evaluate(&input)?,
            _ => return Err(CliError::Usage),
        };
        return Ok(format!(
            "EMPIRICAL EVOLVED PREDICTION / f64\nvalue={value:.15}\nexpression={}\nidentity={:016x}\n",
            skill.program().expression(),
            skill.fingerprint()
        ));
    }
    if cmd == "numeric-mine" {
        let library = mine_manifest(Path::new(&args[1]), context)?;
        library.save_new(Path::new(required(&options, "--library")?))?;
        return Ok(inspect_library(&library));
    }
    if cmd == "numeric-library-inspect" {
        return Ok(inspect_library(&NumericLibrary::open(
            Path::new(&args[1]),
            context,
        )?));
    }
    if matches!(cmd, "numeric-fit" | "numeric-revise" | "numeric-develop") {
        if options.contains_key("--library") && options.contains_key("--mined-library")
            || options.contains_key("--mined-library") != options.contains_key("--library-context")
        {
            return Err(CliError::Usage);
        }
        let mut budget = NumericBudget::default();
        if let Some(value) = options.get("--max-cost") {
            budget.max_cost = value.parse().map_err(|_| CliError::Usage)?;
        }
        if let Some(value) = options.get("--max-terms") {
            budget.max_terms = value.parse().map_err(|_| CliError::Usage)?;
        }
        let library = options
            .get("--library")
            .map(|file| {
                let bytes = read_bounded(Path::new(file), 32_768).map_err(NumericError::from)?;
                NumericSkill::from_bytes(&bytes)
            })
            .transpose()?;
        let refs: Vec<&NumericSkill> = library.iter().collect();
        let mined = options
            .get("--mined-library")
            .map(|file| {
                NumericLibrary::open(Path::new(file), options["--library-context"].as_str())
            })
            .transpose()?;
        let training = examples(Path::new(&args[positions]), 32)?;
        if cmd == "numeric-develop" {
            let mut development = DevelopmentBudget::default();
            if let Some(value) = options.get("--max-attempts") {
                development.max_attempts = value.parse().map_err(|_| CliError::Usage)?;
            }
            if let Some(value) = options.get("--total-work") {
                development.max_work = value.parse().map_err(|_| CliError::Usage)?;
            }
            let (skill, trace) = mined.ok_or(CliError::Usage)?.develop(
                required(&options, "--name")?,
                context,
                training,
                budget,
                development,
                NumericTolerance::default(),
            )?;
            skill.save_new(Path::new(required(&options, "--skill")?))?;
            let mut output = inspect(&skill);
            output.push_str(&format!("development_searches={}\ndevelopment_scalar_work={}\ndevelopment_proposal_patterns={}\ndevelopment_proposal_visits={}\ndevelopment_ns={}\ndemand_admitted_tool={:?}\n",trace.attempts.len(),trace.scalar_work,trace.proposal_patterns,trace.proposal_visits,trace.elapsed_ns,trace.admitted.as_ref().map(|m|m.fingerprint())));
            for (i, attempt) in trace.attempts.iter().enumerate() {
                output.push_str(&format!("development_attempt_{i}: tool={:?}; compatible={}; candidates={}; scalar_work={}; search_ns={}\n",attempt.proposed_tool,attempt.compatible,attempt.search.candidates,attempt.search.scalar_work,attempt.search.elapsed_ns));
            }
            return Ok(output);
        }
        let skill = if cmd == "numeric-fit" {
            if let Some(mined) = &mined {
                NumericSkill::learn_from_library(
                    required(&options, "--name")?,
                    context,
                    training,
                    mined,
                    budget,
                    NumericTolerance::default(),
                )?
            } else {
                NumericSkill::learn(
                    required(&options, "--name")?,
                    context,
                    training,
                    &refs,
                    budget,
                    NumericTolerance::default(),
                )?
            }
        } else {
            let old = NumericSkill::open(Path::new(&args[1]), context)?;
            if let Some(mined) = &mined {
                old.revise_from_library(context, training, mined, budget)?
            } else {
                old.revise(context, training, &refs, budget)?
            }
        };
        skill.save_new(Path::new(required(&options, "--skill")?))?;
        return Ok(inspect(&skill));
    }
    let skill = NumericSkill::open(Path::new(&args[1]), context)?;
    if cmd == "numeric-inspect" {
        return Ok(inspect(&skill));
    }
    let shared = match options
        .get("--backend")
        .map(String::as_str)
        .unwrap_or("stack")
    {
        "stack" => false,
        "shared" => true,
        _ => return Err(CliError::Usage),
    };
    let execution = shared.then(|| skill.program().compile_execution());
    if cmd == "numeric-predict" {
        let input = required(&options, "--values")?
            .split(',')
            .map(|s| s.parse::<f64>().map_err(|_| CliError::Usage))
            .collect::<Result<Vec<_>, _>>()?;
        let value = match &execution {
            Some(execution) => execution.evaluate(&input)?,
            None => skill.predict(context, &input)?,
        };
        return Ok(format!(
            "EMPIRICAL PREDICTION / f64\nvalue={value:.15}\nexpression={}\nidentity={:016x}\n",
            skill.program().expression(),
            skill.fingerprint()
        ));
    }
    let (correct, total, refused, maximum_error) =
        evaluation(Path::new(&args[2]), &skill, execution.as_ref())?;
    if correct != total {
        return Err(NumericError::Evaluation {
            correct,
            total,
            refused,
            maximum_error,
        }
        .into());
    }
    Ok(format!(
        "{}\ncorrect={correct}\ntotal={}\nrefused={refused}\nmax_absolute_error={maximum_error:.15}\nidentity={:016x}\n",
        "EMPIRICAL EVALUATION PASSED",
        total,
        skill.fingerprint()
    ))
}

fn evolved_inspect(skill: &evolution::EvolvedSkill) -> String {
    let s = skill.statistics();
    let complete = skill.has_complete_search_statistics();
    let counter = |n: u64| {
        if complete {
            n.to_string()
        } else {
            "unavailable".into()
        }
    };
    format!(
        "ADMITTED EVOLVED PROGRAM / empirical f64\nname={}\ncontext={}\nidentity={:016x}\ngenome={:016x}\ngeneration={}\nevaluated_genomes={}\nwork={}\nmutations={}\ncrossovers={}\nlocal_proposals={}\nactive_genes={}\ninstructions={}\ndependencies={:?}\ncapsule_bytes={}\nexpression={}\nsearch_statistics={}\n",
        skill.name(),
        skill.context(),
        skill.fingerprint(),
        skill.genome_identity(),
        counter(s.generation as u64),
        s.evaluations,
        s.work,
        counter(s.mutations),
        counter(s.crossovers),
        counter(s.local_proposals),
        skill.active_genes(),
        skill.program().instructions(),
        skill.dependencies(),
        skill.to_bytes().len(),
        skill.program().expression(),
        if complete { "complete" } else { "partial" }
    )
}
