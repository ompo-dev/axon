use crate::{
    Authority, Effect,
    cli::CliError,
    demand::{controller::*, genome::*, *},
};
use std::path::Path;

pub(crate) const USAGE: &str = "\n  axon genome-pack <catalog.axseed> <new.axnum> [procedure.axgoal ...]\n  axon genome-inspect <seed.axnum>\n  axon genome-birth <seed.axnum> <new-directory>\n  axon genome-number <seed.axnum> <new-directory> <module-id> <context> <x0,x1,...>\n  axon genome-mutate <seed.axnum> <new.axnum> <rng-u64>\n  axon genome-apply <seed.axnum> <new-directory> <events.tsv> <new-results.tsv> (text-workflows)";

pub(crate) fn command(args: &[String]) -> Result<String, CliError> {
    let verb = args.first().map(String::as_str).ok_or(CliError::Usage)?;
    let auth = Authority::from([Effect::ReadData, Effect::WriteData]);
    #[cfg(feature = "text-workflows")]
    if verb == "genome-apply" {
        return apply(args, &auth);
    }
    if verb == "genome-pack" {
        if !(3..=11).contains(&args.len()) {
            return Err(CliError::Usage);
        }
        let mut runtime = DemandRuntime::new(
            Catalog::load(&args[1], &auth)?,
            Seed {
                active_bytes: 262_144,
                active_modules: 4,
                ..Seed::default()
            },
        )?;
        let mut procedures = Vec::new();
        for path in &args[3..] {
            let bytes = crate::text_skill::read_bounded(Path::new(path), 65536)
                .map_err(|e| DemandError::Module(e.to_string()))?;
            procedures.push(Procedure::from_bytes(&bytes, runtime.catalog())?);
        }
        let genome = Genome::capture(
            "axon/genome-v1",
            &mut runtime,
            CompositionBudget::default(),
            &procedures,
            MAX_BYTES,
            &auth,
        )?;
        genome.save_number_new(Path::new(&args[2]), &auth)?;
        return Ok(report(&genome, "packed"));
    }
    let len = match verb {
        "genome-inspect" => 2,
        "genome-birth" => 3,
        "genome-number" => 6,
        "genome-mutate" => 4,
        _ => return Err(CliError::Usage),
    };
    if args.len() != len {
        return Err(CliError::Usage);
    }
    let genome = Genome::open_number(Path::new(&args[1]), &auth)?;
    match verb {
        "genome-inspect" => Ok(report(&genome, "decoded")),
        "genome-number" => {
            let values = args[5]
                .split(',')
                .take(5)
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|_| DemandError::Invalid("numeric input"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !(1..=4).contains(&values.len()) || values.iter().any(|n| !n.is_finite()) {
                return Err(DemandError::Invalid("finite numeric inputs").into());
            }
            let mut born = genome.birth(Path::new(&args[2]), BirthLimits::default(), &auth)?;
            let runtime = born.controller.runtime_mut();
            let value = runtime.predict_numeric(&args[3], &args[4], &values, &auth)?;
            let stats = runtime.stats();
            Ok(format!(
                "status=calculated\nvalue={value}\nmodule_loads={}\nactive_modules={}\nactive_payload_bytes={}\nteacher_queries=0\ntraining_required=false\nscope=numeric program result, not a semantic or action certificate",
                stats.loads, stats.active_modules, stats.active_bytes
            ))
        }
        "genome-birth" => {
            let birth = genome.birth(Path::new(&args[2]), BirthLimits::default(), &auth)?;
            Ok(format!(
                "{}\nactive_payload_bytes={}\nmodule_loads={}\nreceipt={}/birth.receipt",
                report(&genome, "born"),
                birth.controller.runtime().stats().active_bytes,
                birth.controller.runtime().stats().loads,
                args[2]
            ))
        }
        "genome-mutate" => {
            let seed = args[3].parse().map_err(|_| CliError::InvalidNumber {
                option: "rng-u64",
                value: args[3].clone(),
            })?;
            let child = genome.offspring(seed)?;
            child.save_number_new(Path::new(&args[2]), &auth)?;
            Ok(report(&child, "offspring"))
        }
        _ => Err(CliError::Usage),
    }
}
#[cfg(feature = "text-workflows")]
fn apply(args: &[String], auth: &Authority) -> Result<String, CliError> {
    if args.len() != 5 {
        return Err(CliError::Usage);
    }
    let genome = Genome::open_number(Path::new(&args[1]), auth)?;
    for p in [&args[3], &args[4]] {
        bounded_path(Path::new(p))?;
    }
    if Path::new(&args[4])
        .try_exists()
        .map_err(DemandError::from)?
    {
        return Err(DemandError::Invalid("output already exists").into());
    }
    let bytes =
        crate::text_skill::read_bounded(Path::new(&args[3]), genome.residency().batch_bytes as u64)
            .map_err(|e| DemandError::Module(e.to_string()))?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_reader(bytes.as_slice());
    if reader
        .headers()
        .map_err(|e| DemandError::Module(e.to_string()))?
        .iter()
        .collect::<Vec<_>>()
        != ["goal", "input"]
    {
        return Err(DemandError::Invalid("events headers: goal input").into());
    }
    let mut events = Vec::new();
    let procedures = genome.procedures()?;
    for row in reader.records() {
        let row = row.map_err(|e| DemandError::Module(e.to_string()))?;
        if events.len() >= genome.residency().max_rows
            || row[1].len() > genome.residency().max_value_bytes
            || !procedures.iter().any(|p| p.context() == &row[0])
        {
            return Err(DemandError::Invalid("events goal/input/rows").into());
        }
        events.push(row);
    }
    let mut birth = genome.birth(Path::new(&args[2]), BirthLimits::default(), auth)?;
    let mut cursors = procedures.iter().map(Procedure::cursor).collect::<Vec<_>>();
    let mut output = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_writer(Vec::new());
    output
        .write_record(["goal", "input", "prediction"])
        .map_err(|e| DemandError::Module(e.to_string()))?;
    for row in &events {
        let i = procedures
            .iter()
            .position(|p| p.context() == &row[0])
            .expect("preflight");
        let value = procedures[i].predict(
            &row[0],
            birth.controller.runtime_mut(),
            &mut cursors[i],
            &row[1],
            auth,
        )?;
        output
            .write_record([&row[0], &row[1], &value.to_string()])
            .map_err(|e| DemandError::Module(e.to_string()))?;
    }
    let output = output
        .into_inner()
        .map_err(|e| DemandError::Module(e.to_string()))?;
    if output.len() > genome.residency().batch_bytes {
        return Err(DemandError::Budget("output batch bytes").into());
    }
    crate::text_skill::publish_new(Path::new(&args[4]), &output)
        .map_err(|e| DemandError::Module(e.to_string()))?;
    Ok(format!(
        "{}\nrows={}\nvalidation=unscored\nexternal_actions=0\noutput={}",
        report(&genome, "applied"),
        events.len(),
        args[4]
    ))
}
fn report(g: &Genome, status: &str) -> String {
    format!(
        "status={status}\ngenome={:016X}\ncontext={}\ngeneration={}\nparent={:016X}\nmodules={}\nprocedures={}\nencoded_bytes={}\ndecimal_digits={}\nmax_steps={}\nactive_module_limit={}\npermissions=host_only",
        g.fingerprint(),
        g.context(),
        g.generation(),
        g.parent(),
        g.module_count(),
        g.procedure_count(),
        g.encoded_bytes(),
        g.number().len(),
        g.composition().max_steps,
        g.residency().active_modules
    )
}
