//! Sweep físico do corte AXON-Λ.
//!
//! Mede tempo de CPU local para uma Factor Fabric afim materializada. Custos do
//! kernel (energia, bytes, risco) continuam declarados e não são inferidos aqui.

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use axon::core_lambda::{
    AdaptiveMode, ChainFabric, CostWeights, Demand, EvidenceDelta, LiftedPopulation,
};
use axon::system_info::detect_total_ram_bytes;

const DEFAULT_FACTORS: usize = 1_000_000;
const DEFAULT_CHAIN_LEN: usize = 1_000;
const MAX_RUNS: usize = 30;
const WARMUPS: usize = 2;
const SAMPLES: usize = 15;
const FULL_ITERATIONS: usize = 3;
const DELTA_ITERATIONS: usize = 1_000;
const LIFT_DIRECT_ITERATIONS: usize = 8;
const LIFTED_ITERATIONS: usize = 5_000;

fn main() {
    let options = Options::parse();
    ensure_safe_allocation(options.factors);
    let reports = (0..options.runs)
        .map(|round| run_round(options, round))
        .collect::<Vec<_>>();
    print_report(&reports, options);
}

#[derive(Clone, Copy)]
struct Options {
    runs: usize,
    factors: usize,
    chain_len: usize,
}

impl Options {
    fn parse() -> Self {
        let mut result = Self {
            runs: 3,
            factors: DEFAULT_FACTORS,
            chain_len: DEFAULT_CHAIN_LEN,
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
                "--chain-len" if index + 1 < arguments.len() => {
                    result.chain_len = parse_positive(&arguments[index + 1]);
                    index += 2;
                }
                _ => usage(),
            }
        }
        if !result.factors.is_multiple_of(result.chain_len) || result.chain_len < 2 {
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
    eprintln!(
        "uso: axon_lambda_physical_sweep [--runs 1..30] [--factors N] [--chain-len N]; factors deve ser multiplo de chain-len >= 2"
    );
    std::process::exit(2);
}

fn ensure_safe_allocation(factors: usize) {
    let fabric = ChainFabric::estimated_storage_bytes(factors).unwrap_or_else(|| usage());
    // Dois fabrics, vetor full temporário e população usada no teste LIFT.
    let requested = fabric
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(factors as u64 * 32))
        .unwrap_or_else(|| usage());
    let safe_cap = detect_total_ram_bytes()
        .map(|bytes| bytes / 4)
        .unwrap_or(512 * 1024 * 1024);
    if requested > safe_cap {
        eprintln!(
            "sweep needs an estimated {:.1} MiB, above safe cap {:.1} MiB",
            requested as f64 / 1024_f64.powi(2),
            safe_cap as f64 / 1024_f64.powi(2),
        );
        usage();
    }
}

struct RoundReport {
    local_build_ns: u128,
    global_build_ns: u128,
    local_full: Measurement,
    local_delta: Measurement,
    global_adaptive: Measurement,
    local_full_mode: AdaptiveMode,
    local_delta_mode: AdaptiveMode,
    global_mode: AdaptiveMode,
    local_b: usize,
    local_f: usize,
    local_a: usize,
    global_a: usize,
    local_results_equal: bool,
    global_results_equal: bool,
    local_full_parity_checksum: u64,
    local_delta_parity_checksum: u64,
    global_full_parity_checksum: u64,
    global_delta_parity_checksum: u64,
    local_overlay_equal: bool,
    global_overlay_equal: bool,
    lift_build_ns: u128,
    lift_direct: Measurement,
    lift_quotient: Measurement,
    lift_results_equal: bool,
    lift_classes: usize,
}

fn run_round(options: Options, round: usize) -> RoundReport {
    let built = Instant::now();
    let local = ChainFabric::new(options.factors, options.chain_len).expect("validated shape");
    let local_build_ns = built.elapsed().as_nanos();

    let built = Instant::now();
    let global = ChainFabric::new(options.factors, options.factors).expect("validated shape");
    let global_build_ns = built.elapsed().as_nanos();

    let local_goal = options.chain_len - 1;
    let local_source = options.chain_len / 2;
    let local_updates = updates(&local, local_source, 64);
    let global_updates = updates(&global, 0, 64);
    let local_demand = Demand::exact(local_goal);
    let global_demand = Demand::exact(options.factors - 1);

    let first_local = local
        .query(local_demand, local_updates[0], CostWeights::latency_only())
        .expect("fixed local query");
    let first_global = global
        .query(
            global_demand,
            global_updates[0],
            CostWeights::latency_only(),
        )
        .expect("fixed global query");
    let (local_full_parity_checksum, local_delta_parity_checksum, local_results_equal) =
        parity_checksums(&local, local_demand, &local_updates[..8]);
    let (global_full_parity_checksum, global_delta_parity_checksum, global_results_equal) =
        parity_checksums(&global, global_demand, &global_updates[..2]);

    let full_work = || {
        measure(FULL_ITERATIONS, |index| {
            local
                .full_query(
                    local_demand,
                    black_box(local_updates[index % local_updates.len()]),
                )
                .unwrap()
                .value
        })
    };
    let delta_work = || {
        measure(DELTA_ITERATIONS, |index| {
            local
                .query(
                    local_demand,
                    black_box(local_updates[index % local_updates.len()]),
                    CostWeights::latency_only(),
                )
                .unwrap()
                .value
        })
    };
    let (local_full, local_delta) = if round.is_multiple_of(2) {
        (full_work(), delta_work())
    } else {
        let delta = delta_work();
        let full = full_work();
        (full, delta)
    };
    let global_adaptive = measure(FULL_ITERATIONS, |index| {
        global
            .query(
                global_demand,
                black_box(global_updates[index % global_updates.len()]),
                CostWeights::latency_only(),
            )
            .unwrap()
            .value
    });

    let values = (0..options.factors)
        .map(|index| (index as u64 % 256).wrapping_mul(97).wrapping_add(11))
        .collect::<Vec<_>>();
    let built = Instant::now();
    let lifted = LiftedPopulation::from_values(&values);
    let lift_build_ns = built.elapsed().as_nanos();
    let lift_results_equal = direct_sum(&values) == lifted.lifted_sum();
    let lift_direct = measure(LIFT_DIRECT_ITERATIONS, |_| direct_sum(black_box(&values)));
    let lift_quotient = measure(LIFTED_ITERATIONS, |_| lifted.lifted_sum());

    RoundReport {
        local_build_ns,
        global_build_ns,
        local_full,
        local_delta,
        global_adaptive,
        local_full_mode: local
            .full_query(local_demand, local_updates[0])
            .unwrap()
            .mode,
        local_delta_mode: first_local.mode,
        global_mode: first_global.mode,
        local_b: first_local.slice.demanded_factors,
        local_f: first_local.slice.changed_factors,
        local_a: first_local.slice.active_factors,
        global_a: first_global.slice.active_factors,
        local_results_equal,
        global_results_equal,
        local_full_parity_checksum,
        local_delta_parity_checksum,
        global_full_parity_checksum,
        global_delta_parity_checksum,
        local_overlay_equal: local.delta_overlay_matches_full(local_updates[0]).unwrap(),
        global_overlay_equal: global
            .delta_overlay_matches_full(global_updates[0])
            .unwrap(),
        lift_build_ns,
        lift_direct,
        lift_quotient,
        lift_results_equal,
        lift_classes: lifted.classes().len(),
    }
}

fn parity_checksums(
    fabric: &ChainFabric,
    demand: Demand,
    updates: &[EvidenceDelta],
) -> (u64, u64, bool) {
    updates.iter().enumerate().fold(
        (0xA80C_1A0B_7EED_u64, 0xA80C_1A0B_7EED_u64, true),
        |(full_checksum, delta_checksum, equal), (index, update)| {
            let full = fabric.full_query(demand, *update).unwrap().value;
            let delta = fabric
                .query(demand, *update, CostWeights::latency_only())
                .unwrap()
                .value;
            (
                mix_checksum(full_checksum, full, index),
                mix_checksum(delta_checksum, delta, index),
                equal && full == delta,
            )
        },
    )
}

fn updates(fabric: &ChainFabric, factor: usize, count: usize) -> Vec<EvidenceDelta> {
    let base = fabric.base_value(factor).expect("fixed factor");
    (0..count)
        .map(|index| {
            EvidenceDelta::new(
                factor,
                base.rotate_left((index % 63) as u32) ^ (index as u64 + 1).wrapping_mul(0x9E37),
            )
        })
        .collect()
}

fn direct_sum(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0_u64, |sum, value| sum.wrapping_add(*value))
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

fn measure(iterations: usize, mut work: impl FnMut(usize) -> u64) -> Measurement {
    let mut checksum = 0xA80C_1A0B_7EED_u64;
    for warmup in 0..WARMUPS {
        for index in 0..iterations {
            checksum = mix_checksum(checksum, work(warmup * iterations + index), index);
        }
    }
    let mut nanoseconds_per_call = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        let started = Instant::now();
        for index in 0..iterations {
            checksum = mix_checksum(
                checksum,
                work(sample * iterations + index),
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

fn mix_checksum(state: u64, value: u64, index: usize) -> u64 {
    state.rotate_left(11) ^ value.rotate_left((index % 64) as u32) ^ index as u64
}

fn print_report(reports: &[RoundReport], options: Options) {
    let local_full = median_measurement(reports.iter().map(|report| &report.local_full));
    let local_delta = median_measurement(reports.iter().map(|report| &report.local_delta));
    let global = median_measurement(reports.iter().map(|report| &report.global_adaptive));
    let direct = median_measurement(reports.iter().map(|report| &report.lift_direct));
    let lifted = median_measurement(reports.iter().map(|report| &report.lift_quotient));
    let speedup = local_full.0 as f64 / local_delta.0.max(1) as f64;
    let lift_speedup = direct.0 as f64 / lifted.0.max(1) as f64;
    let lift_build = median(reports.iter().map(|report| report.lift_build_ns));
    let break_even = if direct.0 > lifted.0 {
        (lift_build as f64 / (direct.0 - lifted.0) as f64).ceil()
    } else {
        f64::INFINITY
    };

    println!("# AXON-Λ — Demand × Delta Physical Sweep\n");
    println!("- Host: {}", host_metadata());
    println!(
        "- Protocolo: {} rodada(s), {} aquecimentos, {} amostras; ordem full/delta alternada por rodada.",
        reports.len(),
        WARMUPS,
        SAMPLES
    );
    println!(
        "- Topologia materializada: {} Factors por fabric; cenário local = {} cadeias de {}, cenário adversarial = uma cadeia de {}.\n",
        options.factors,
        options.factors / options.chain_len,
        options.chain_len,
        options.factors,
    );

    println!("## 1. Build e paridade semântica");
    println!(
        "| build local p50 | build global p50 | local=full | global=full | base+overlay local/global |"
    );
    println!("|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {} | {} | {}/{} |",
        format_duration(median(reports.iter().map(|report| report.local_build_ns))),
        format_duration(median(reports.iter().map(|report| report.global_build_ns))),
        reports.iter().all(|report| report.local_results_equal),
        reports.iter().all(|report| report.global_results_equal),
        reports.iter().all(|report| report.local_overlay_equal),
        reports.iter().all(|report| report.global_overlay_equal),
    );

    println!("\n## 2. Demand × Delta local");
    println!(
        "| B (demanda) | F (mudança) | A=B∩F | full p50/p95 | delta p50/p95 | razão | modo full/delta |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {} | {}/{} | {}/{} | {:.2}x | {:?}/{:?} |",
        reports[0].local_b,
        reports[0].local_f,
        reports[0].local_a,
        format_duration(local_full.0),
        format_duration(local_full.1),
        format_duration(local_delta.0),
        format_duration(local_delta.1),
        speedup,
        reports[0].local_full_mode,
        reports[0].local_delta_mode,
    );
    println!(
        "Checksums de paridade full/delta (mesmas 8 mudanças): {:016X}/{:016X}",
        combined_checksum(
            reports
                .iter()
                .map(|report| report.local_full_parity_checksum),
        ),
        combined_checksum(
            reports
                .iter()
                .map(|report| report.local_delta_parity_checksum),
        ),
    );

    println!("\n## 3. Cascata adversarial");
    println!("| A=B∩F | adaptive p50/p95 | modo selecionado | paridade |");
    println!("|---:|---:|---:|---:|");
    println!(
        "| {} | {}/{} | {:?} | {} |",
        reports[0].global_a,
        format_duration(global.0),
        format_duration(global.1),
        reports[0].global_mode,
        reports.iter().all(|report| report.global_results_equal),
    );
    println!(
        "Checksums de paridade full/delta (mesmas 2 mudanças): {:016X}/{:016X}",
        combined_checksum(
            reports
                .iter()
                .map(|report| report.global_full_parity_checksum),
        ),
        combined_checksum(
            reports
                .iter()
                .map(|report| report.global_delta_parity_checksum),
        ),
    );

    println!("\n## 4. LIFT / quotient exato");
    println!(
        "| população | classes | construir índice p50 | direto p50/p95 | lifted p50/p95 | razão | break-even estimado | exato |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {} | {}/{} | {}/{} | {:.2}x | {} consultas | {} |",
        options.factors,
        reports[0].lift_classes,
        format_duration(lift_build),
        format_duration(direct.0),
        format_duration(direct.1),
        format_duration(lifted.0),
        format_duration(lifted.1),
        lift_speedup,
        if break_even.is_finite() {
            format!("{break_even:.0}")
        } else {
            "n/a".to_owned()
        },
        reports.iter().all(|report| report.lift_results_equal),
    );

    println!(
        "\nLimites: os tempos são observações de CPU deste host e a regra é afim/linear; custos do semiring são declarados. O resultado não demonstra uma Factor Fabric geral, descoberta automática de simetria ou eficiência energética."
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
    fn local_and_global_workloads_preserve_the_same_decision() {
        let fabric = ChainFabric::new(10_000, 100).unwrap();
        let local = EvidenceDelta::new(50, 777);
        assert_eq!(
            fabric.full_query(Demand::exact(99), local).unwrap().value,
            fabric
                .query(Demand::exact(99), local, CostWeights::latency_only())
                .unwrap()
                .value,
        );
        assert!(fabric.delta_overlay_matches_full(local).unwrap());
    }

    #[test]
    fn checksum_mixer_cannot_cancel_two_equal_samples() {
        assert_ne!(combined_checksum([42, 42].into_iter()), 0);
    }

    #[test]
    fn lift_preserves_the_direct_aggregate() {
        let values = (0..10_000)
            .map(|value| value as u64 % 16)
            .collect::<Vec<_>>();
        let lifted = LiftedPopulation::from_values(&values);
        assert_eq!(direct_sum(&values), lifted.lifted_sum());
        assert_eq!(lifted.classes().len(), 16);
    }
}
