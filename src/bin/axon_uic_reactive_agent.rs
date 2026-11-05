use axon_uic::reactive::{agent::*, *};
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, Read, Write},
    path::Path,
};

fn json(s: &str) -> String {
    let mut b = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => b.push_str("\\\""),
            '\\' => b.push_str("\\\\"),
            '\n' => b.push_str("\\n"),
            '\r' => b.push_str("\\r"),
            '\t' => b.push_str("\\t"),
            c if c.is_control() => b.push_str(&format!("\\u{:04x}", c as u32)),
            c => b.push(c),
        }
    }
    b.push('"');
    b
}
fn emit(s: &str) -> io::Result<()> {
    let mut out = io::stdout().lock();
    writeln!(out, "{s}")?;
    out.flush()
}
fn status(a: &ReactiveAgent, kind: &str) -> String {
    let s = a.statistics();
    let (correct, distinct) = a.validation_progress();
    format!(
        "{{\"kind\":{},\"phase\":{},\"observations\":{},\"labels\":{},\"scored\":{},\"correct\":{},\"searches\":{},\"failed_searches\":{},\"rejected_candidates\":{},\"promotions\":{},\"drifts\":{},\"search_work\":{},\"validation_correct\":{correct},\"validation_distinct\":{distinct},\"retained_bytes\":{},\"program\":{}}}",
        json(kind),
        json(a.phase().as_str()),
        s.observations,
        s.labels,
        s.scored,
        s.correct,
        s.searches,
        s.failed_searches,
        s.rejected_candidates,
        s.promotions,
        s.drifts,
        s.search_work,
        a.retained_bytes(),
        a.skill()
            .or_else(|| a.candidate())
            .map(|s| json(&s.description()))
            .unwrap_or_else(|| "null".into())
    )
}
fn main() -> Result<(), Box<dyn Error>> {
    let usage = "axon-uic-reactive-agent --context <context> --arity <1..2> [--tool <acquired.axreact>] [--restore <active.axra>]\nTSV: O<TAB>id<TAB>integers ; F<TAB>id<TAB>actual ; P [<TAB>lo:hi[,lo:hi]] (proposal only) ; S ; C<TAB>new.axra ; E<TAB>new.axreact ; Q. One pending prediction. Integer values/IDs are JSON strings. Search is synchronous and bounded; EOF never promotes.";
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("{usage}");
        return Ok(());
    }
    let mut context = None;
    let mut arity = None;
    let mut restore = None;
    let mut tools = Vec::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 {
            return Err(usage.into());
        }
        match pair[0].as_str() {
            "--context" if context.is_none() => context = Some(pair[1].clone()),
            "--arity" if arity.is_none() => arity = Some(pair[1].parse::<usize>()?),
            "--restore" if restore.is_none() => restore = Some(pair[1].clone()),
            "--tool" if tools.len() < 4 => {
                let file = File::open(&pair[1])?;
                let mut bytes = Vec::new();
                file.take(1_048_577).read_to_end(&mut bytes)?;
                tools.push(ReactiveSkill::from_bytes(&bytes)?);
            }
            _ => return Err(usage.into()),
        }
    }
    let context = context.ok_or(usage)?;
    let arity = arity.ok_or(usage)?;
    if restore.is_some() && !tools.is_empty() {
        return Err("restore cannot add tools".into());
    }
    let mut agent = if let Some(path) = restore {
        let a = ReactiveAgent::resume(&context, Path::new(&path))?;
        if a.skill().is_none_or(|s| s.arity() != arity) {
            return Err("restored arity mismatch".into());
        }
        a
    } else {
        ReactiveAgent::new(&context, arity, tools, ReactiveBudget::default())?
    };
    emit(&status(&agent, "ready"))?;
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    loop {
        let mut line = String::new();
        let n = reader.by_ref().take(4097).read_line(&mut line)?;
        if n == 0 {
            break;
        }
        if n > 4096 {
            emit("{\"kind\":\"error\",\"message\":\"line limit\"}")?;
            break;
        }
        let fields: Vec<_> = line.trim_end_matches(['\r', '\n']).split('\t').collect();
        let result = (|| -> Result<bool, Box<dyn Error>> {
            match fields.as_slice() {
                ["O", id, input] => {
                    let values = input
                        .split(',')
                        .map(str::parse::<i64>)
                        .collect::<Result<Vec<_>, _>>()?;
                    let p = agent.observe(id.parse()?, &values)?;
                    emit(&format!(
                        "{{\"kind\":\"prediction\",\"id\":{},\"value\":{}}}",
                        json(&p.id.to_string()),
                        p.value
                            .map(|v| json(&v.to_string()))
                            .unwrap_or_else(|| "null".into())
                    ))?;
                }
                ["F", id, y] => {
                    let result = agent.feedback(id.parse()?, y.parse()?)?;
                    emit(&format!(
                        "{{\"kind\":\"feedback\",\"status\":{},\"phase\":{}}}",
                        json(result.as_str()),
                        json(agent.phase().as_str())
                    ))?;
                }
                ["P"] | ["P", _] => {
                    let proposal = if let [_, ranges] = fields.as_slice() {
                        let bounds = ranges
                            .split(',')
                            .map(|range| {
                                let (lo, hi) =
                                    range.split_once(':').ok_or("probe lo:hi expected")?;
                                Ok((lo.parse::<i64>()?, hi.parse::<i64>()?))
                            })
                            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                        agent.suggest_probe_with_bounds(&bounds)?
                    } else {
                        agent.suggest_probe()
                    };
                    let values = proposal
                        .map(|v| {
                            format!(
                                "[{}]",
                                v.iter()
                                    .map(|n| json(&n.to_string()))
                                    .collect::<Vec<_>>()
                                    .join(",")
                            )
                        })
                        .unwrap_or_else(|| "null".into());
                    emit(&format!("{{\"kind\":\"probe\",\"values\":{values}}}"))?;
                }
                ["S"] => emit(&status(&agent, "status"))?,
                ["C", path] => {
                    agent.checkpoint_new(Path::new(path))?;
                    emit("{\"kind\":\"checkpoint\"}")?;
                }
                ["E", path] => {
                    agent
                        .skill()
                        .ok_or("no admitted model")?
                        .save_new(Path::new(path))?;
                    emit("{\"kind\":\"export\"}")?;
                }
                ["Q"] => return Ok(false),
                _ => return Err("invalid TSV event".into()),
            }
            Ok(true)
        })();
        match result {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => emit(&format!(
                "{{\"kind\":\"error\",\"message\":{}}}",
                json(&e.to_string())
            ))?,
        }
    }
    emit(&status(&agent, "stopped"))?;
    Ok(())
}
