//! Medição física local de Auto-LIFT certificado no corte AXON-Λ².
//!
//! O workload materializa um milhão de Sources exchangeable alimentando um
//! Factor `max` comutativo. O baseline sempre materializa o estado inteiro; o
//! caminho lifted usa somente a classe certificada mais o membro em UNLIFT.

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use axon::core_lambda::{CertifiedAutoLift, GeneralFactor, GeneralGraph, GraphDelta};
use axon::system_info::detect_total_ram_bytes;

const DEFAULT_FACTORS: usize = 1_000_000;
const MAX_RUNS: usize = 10;
const WARMUPS: usize = 2;
const SAMPLES: usize = 15;
const FULL_ITERATIONS: usize = 3;
// Mantém cada amostra lifted acima da granularidade prática do relógio do host.
const LIFTED_ITERATIONS: usize = 100_000;

fn main() {
    let options = Options::parse();
    ensure_safe_allocation(options.factors);
    let reports = (0..options.runs)
        .map(|round| run_round(options.factors, round))
        .collect::<Vec<_>>();
    print_report(&reports, options);
}

#[derive(Clone, Copy)]
struct Options {
    runs: usize,
    factors: usize,
}

impl Options {
    fn parse() -> Self {
        let mut result = Self {
            runs: 3,
            factors: DEFAULT_FACTORS,
        };
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            match arguments[index].as_str() {
                "--runs" if index + 1 < arguments.len() => {
                    result.runs = parse_positive(&arguments[index + 1]);
                    if result.runs > MAX_RUNS {
                        usage();
                    }
                    index += 2;
                }
                "--factors" if index + 1 < arguments.len() => {
                    result.factors = parse_positive(&arguments[index + 1]);
                    index += 2;
                }
                _ => usage(),
            }
        }
        if result.factors < 2 {
            usage();
        }
        result
    }
}

fn parse_positive(value: &str) -> usize {
    value
        .parse::<usize>()
        .ok()
        .filter(|parsed| *parsed > 0)
        .unwrap_or_else(|| usage())
}

fn usage() -> ! {
    eprintln!("uso: axon_lambda_squared_physical_sweep [--runs 1..10] [--factors N >= 2]");
    std::process::exit(2);
}

fn ensure_safe_allocation(factors: usize) {
    // Limite deliberadamente conservador para Factors, SCC metadata, arestas,
    // índice de cores e buffers de avaliação. Não é medição de RSS.
    let estimated = u64::try_from(factors)
        .ok()
        .and_then(|value| value.checked_mul(256))
        .unwrap_or_else(|| usage());
    let cap = detect_total_ram_bytes()
        .map(|bytes| bytes / 4)
        .unwrap_or(512 * 1024 * 1024);
    if estimated > cap {
        eprintln!(
            "sweep estimates {:.1} MiB, above safe cap {:.1} MiB",
            estimated as f64 / 1024_f64.powi(2),
            cap as f64 / 1024_f64.powi(2),
        );
        usage();
    }
}

struct RoundReport {
    graph_build_ns: u128,
    lift_discovery_ns: u128,
    full: Measurement,
    lifted: Measurement,
    class_count: usize,
    class_members: usize,
    certificate_valid: bool,
    parity: bool,
    full_checksum: u64,
    lifted_checksum: u64,
}

fn run_round(factors: usize, round: usize) -> RoundReport {
    let built = Instant::now();
    let (graph, goal) = exchangeable_graph(factors);
    let graph_build_ns = built.elapsed().as_nanos();

    let discovered = Instant::now();
    let lift = CertifiedAutoLift::discover(&graph).expect("exchangeable graph must certify");
    let lift_discovery_ns = discovered.elapsed().as_nanos();
    let updates = updates(factors, round);
    let (full_checksum, lifted_checksum, parity) =
        parity_checksums(&graph, &lift, goal, &updates[..8]);

    let full_work = || {
        measure(FULL_ITERATIONS, |index| {
            graph
                .full_value_after(goal, black_box(updates[index % updates.len()]))
                .expect("validated full query")
        })
    };
    let lifted_work = || {
        measure(LIFTED_ITERATIONS, |index| {
            let update = black_box(updates[index % updates.len()]);
            lift.unlift(update.factor, update.replacement_value)
                .expect("all updates are lifted")
                .lifted_max(&lift, &graph, goal)
                .expect("certificate remains valid")
        })
    };
    let (full, lifted) = if round.is_multiple_of(2) {
        (full_work(), lifted_work())
    } else {
        let lifted = lifted_work();
        let full = full_work();
        (full, lifted)
    };

    RoundReport {
        graph_build_ns,
        lift_discovery_ns,
        full,
        lifted,
        class_count: lift.classes().len(),
        class_members: lift.classes()[0].members.len(),
        certificate_valid: lift.verify(&graph),
        parity,
        full_checksum,
        lifted_checksum,
    }
}

fn exchangeable_graph(factors: usize) -> (GeneralGraph, usize) {
    let mut definitions = Vec::with_capacity(factors + 1);
    let mut inputs = Vec::with_capacity(factors);
    for factor in 0..factors {
        definitions.push(GeneralFactor::source(7));
        inputs.push(factor);
    }
    definitions.push(GeneralFactor::max(inputs, i64::MIN));
    let goal = factors;
    (
        GeneralGraph::new(definitions).expect("fixed exchangeable graph"),
        goal,
    )
}

fn updates(factors: usize, round: usize) -> Vec<GraphDelta> {
    (0..32)
        .map(|index| {
            let factor = (index * 7_919 + round * 101) % factors;
            GraphDelta::replace_source(factor, 100 + index as i64)
        })
        .collect()
}

fn parity_checksums(
    graph: &GeneralGraph,
    lift: &CertifiedAutoLift,
    goal: usize,
    updates: &[GraphDelta],
) -> (u64, u64, bool) {
    updates.iter().enumerate().fold(
        (0xA80C_1A0B_7EED_u64, 0xA80C_1A0B_7EED_u64, true),
        |(full_checksum, lifted_checksum, equal), (index, update)| {
            let full = graph
                .full_value_after(goal, *update)
                .expect("valid full value");
            let lifted = lift
                .unlift(update.factor, update.replacement_value)
                .expect("valid local UNLIFT")
                .lifted_max(lift, graph, goal)
                .expect("valid lifted value");
            (
                mix_checksum(full_checksum, full as u64, index),
                mix_checksum(lifted_checksum, lifted as u64, index),
                equal && full == lifted,
            )
        },
    )
}

#[derive(Clone)]
struct Measurement {
    nanoseconds_per_call: Vec<u128>,
}

impl Measurement {
    fn p50(&self) -> u128 {
        self.nanoseconds_per_call[self.nanoseconds_per_call.len() / 2]
    }

    fn p95(&self) -> u128 {
        self.nanoseconds_per_call[(self.nanoseconds_per_call.len() * 95).div_ceil(100) - 1]
    }
}

fn measure(iterations: usize, mut work: impl FnMut(usize) -> i64) -> Measurement {
    let mut checksum = 0xA80C_1A0B_7EED_u64;
    for warmup in 0..WARMUPS {
        for index in 0..iterations {
            checksum = mix_checksum(checksum, work(warmup * iterations + index) as u64, index);
        }
    }
    let mut nanoseconds_per_call = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let started = Instant::now();
        for index in 0..iterations {
            checksum = mix_checksum(
                checksum,
                work(sample * iterations + index) as u64,
                sample * iterations + index,
            );
        }
        nanoseconds_per_call.push(started.elapsed().as_nanos() / iterations as u128);
    }
    nanoseconds_per_call.sort_unstable();
    black_box(checksum);
    Measurement {
        nanoseconds_per_call,
    }
}

fn print_report(reports: &[RoundReport], options: Options) {
    let full = median_measurement(reports.iter().map(|report| &report.full));
    let lifted = median_measurement(reports.iter().map(|report| &report.lifted));
    let discovery = median(reports.iter().map(|report| report.lift_discovery_ns));
    let speedup = full.0 as f64 / lifted.0.max(1) as f64;
    let break_even = if full.0 > lifted.0 {
        (discovery as f64 / (full.0 - lifted.0) as f64).ceil()
    } else {
        f64::INFINITY
    };
    let eliminated_ratio = 1.0 - 3.0 / (options.factors + 1) as f64;

    println!("# AXON-Λ² — Certified Auto-LIFT Physical Sweep\n");
    println!("- Host: {}", host_metadata());
    println!(
        "- Protocolo: {} rodada(s), {} aquecimentos, {} amostras; ordem full/lifted alternada.",
        reports.len(),
        WARMUPS,
        SAMPLES
    );
    println!(
        "- Topologia materializada: {} Sources idênticos → um Factor `max` comutativo. Cada consulta altera um Source e executa UNLIFT local.\n",
        options.factors
    );

    println!("## Integridade semântica e custo de descoberta");
    println!(
        "| build graph p50 | descobrir+certificar p50 | classes | membros na classe | certificado | full=lifted |"
    );
    println!("|---:|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {} | {} | {} | {} |",
        format_duration(median(reports.iter().map(|report| report.graph_build_ns))),
        format_duration(discovery),
        reports[0].class_count,
        reports[0].class_members,
        reports.iter().all(|report| report.certificate_valid),
        reports.iter().all(|report| report.parity),
    );
    println!(
        "Checksums sobre as mesmas 8 mutações: {:016X}/{:016X}",
        combined_checksum(reports.iter().map(|report| report.full_checksum)),
        combined_checksum(reports.iter().map(|report| report.lifted_checksum)),
    );

    println!("\n## Consulta após UNLIFT local");
    println!(
        "| full p50/p95 | lifted+unlift p50/p95 | razão observada | break-even | SER lógico de leituras |"
    );
    println!("|---:|---:|---:|---:|---:|");
    println!(
        "| {}/{} | {}/{} | {:.2}x | {} consultas | {:.6}% |",
        format_duration(full.0),
        format_duration(full.1),
        format_duration(lifted.0),
        format_duration(lifted.1),
        speedup,
        if break_even.is_finite() {
            format!("{break_even:.0}")
        } else {
            "n/a".to_owned()
        },
        eliminated_ratio * 100.0,
    );
    println!(
        "\nLimites: este é um Auto-LIFT exato, mas restrito a Sources de mesmo valor e grau unitário alimentando `max` comutativo. O certificado é verificado na descoberta; a query verifica o digest imutável em O(1). Não mede energia, não descobre automorfismos gerais e não autoriza approximate LIFT."
    );
}

fn median_measurement<'a>(measurements: impl Iterator<Item = &'a Measurement>) -> (u128, u128) {
    let measurements = measurements.collect::<Vec<_>>();
    (
        median(measurements.iter().map(|measurement| measurement.p50())),
        median(measurements.iter().map(|measurement| measurement.p95())),
    )
}

fn median(values: impl Iterator<Item = u128>) -> u128 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

fn mix_checksum(state: u64, value: u64, index: usize) -> u64 {
    state.rotate_left(11) ^ value.rotate_left((index % 64) as u32) ^ index as u64
}

fn combined_checksum(checksums: impl Iterator<Item = u64>) -> u64 {
    checksums
        .enumerate()
        .fold(0xA80C_1A0B_7EED_u64, |state, (index, checksum)| {
            mix_checksum(state, checksum, index)
        })
}

fn format_duration(nanoseconds: u128) -> String {
    if nanoseconds >= 1_000_000 {
        format!("{:.3} ms", nanoseconds as f64 / 1_000_000.0)
    } else if nanoseconds >= 1_000 {
        format!("{:.3} µs", nanoseconds as f64 / 1_000.0)
    } else {
        format!("{nanoseconds} ns")
    }
}

fn host_metadata() -> String {
    let rustc = command_output("rustc", &["-V"]).unwrap_or_else(|| "rustc indisponível".into());
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    #[cfg(windows)]
    {
        let cpu = command_output(
            "powershell",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name).Trim()",
            ],
        )
        .unwrap_or_else(|| "CPU indisponível".into());
        let ram = detect_total_ram_bytes()
            .map(|bytes| format!("{:.2} GB", bytes as f64 / 1024_f64.powi(3)))
            .unwrap_or_else(|| "RAM indisponível".into());
        format!(
            "OS={} arch={} CPU={} RAM={} threads={} compilador={}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            cpu,
            ram,
            threads,
            rustc,
        )
    }
    #[cfg(not(windows))]
    {
        format!(
            "OS={} arch={} threads={} compilador={}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            threads,
            rustc,
        )
    }
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new(program).args(arguments).output().ok()?;
    if !output.status.success() || !output.stderr.is_empty() {
        return None;
    }
    let output = String::from_utf8(output.stdout).ok()?;
    let compact = output.split_whitespace().collect::<Vec<_>>().join(" ");
    (!compact.is_empty()).then_some(compact)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn certified_auto_lift_matches_full_after_a_local_specialization() {
        let (graph, goal) = exchangeable_graph(10_000);
        let lift = CertifiedAutoLift::discover(&graph).unwrap();
        let delta = GraphDelta::replace_source(19, 99);
        let lifted = lift.unlift(delta.factor, delta.replacement_value).unwrap();
        assert!(lift.verify(&graph));
        assert_eq!(
            graph.full_value_after(goal, delta).unwrap(),
            lifted.lifted_max(&lift, &graph, goal).unwrap(),
        );
    }

    #[test]
    fn checksum_mixer_preserves_repeated_samples() {
        assert_ne!(combined_checksum([42, 42].into_iter()), 0);
    }
}
