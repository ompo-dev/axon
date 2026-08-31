//! Sweep físico V7-X: mede execução CPU, cópias de mundos e repetição compilável.
//!
//! Os orçamentos semânticos são lógicos. O benchmark não materializa 4 GiB de
//! corpus; ele mede a construção e as consultas do plano no hardware local.

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use axon::core_v7x::{
    CapitalDispatch, CognitiveCapitalRuntime, MIB, SemanticResolution, SemanticVirtualMemory,
    VersionedWorld, WorldBase,
};
use axon::system_info::detect_total_ram_bytes;

const KIB: u64 = 1024;
const FACTOR_BYTES: usize = 8 * 1024 * 1024;
const SEMANTIC_BUDGETS_MIB: [u64; 4] = [64, 128, 512, 4_096];
const WORLD_COUNTS: [usize; 4] = [10, 100, 1_000, 10_000];
const CAPITAL_CALLS: usize = 10_000;
const CAPITAL_INPUT: u64 = 4_096;
const WARMUPS: usize = 1;
const SAMPLES: usize = 5;

fn main() {
    let options = Options::parse();
    let world_bytes = checked_world_bytes(options.world_kib);
    ensure_safe_full_copy(world_bytes);
    let payload = factor_payload();
    let reports = (0..options.runs)
        .map(|_| run_round(&payload, world_bytes))
        .collect::<Vec<_>>();
    print_report(&reports, &host_metadata(), options.world_kib);
}

#[derive(Clone, Copy)]
struct Options {
    runs: usize,
    world_kib: u64,
}

impl Options {
    fn parse() -> Self {
        let mut runs = 3;
        let mut world_kib = 32;
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--runs" if index + 1 < args.len() => {
                    runs = parse_positive(&args[index + 1]);
                    index += 2;
                }
                "--world-kib" if index + 1 < args.len() => {
                    world_kib = parse_positive(&args[index + 1]) as u64;
                    index += 2;
                }
                _ => usage(),
            }
        }
        Self { runs, world_kib }
    }
}

fn parse_positive(raw: &str) -> usize {
    raw.parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| usage())
}

fn usage() -> ! {
    eprintln!("uso: axon_v7x_physical_sweep [--runs N] [--world-kib N]");
    std::process::exit(2);
}

fn checked_world_bytes(world_kib: u64) -> usize {
    world_kib
        .checked_mul(KIB)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .filter(|bytes| *bytes >= std::mem::size_of::<u64>())
        .unwrap_or_else(|| usage())
}

fn ensure_safe_full_copy(world_bytes: usize) {
    let requested = (world_bytes as u64)
        .checked_mul(WORLD_COUNTS.last().copied().unwrap_or_default() as u64)
        .unwrap_or_else(|| usage());
    let safe_cap = detect_total_ram_bytes()
        .map(|bytes| bytes / 4)
        .unwrap_or(512 * MIB);
    if requested > safe_cap {
        eprintln!(
            "full-copy workload needs {:.1} MiB, above the safe cap of {:.1} MiB",
            requested as f64 / MIB as f64,
            safe_cap as f64 / MIB as f64,
        );
        usage();
    }
}

fn factor_payload() -> Vec<u8> {
    (0..FACTOR_BYTES)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(17))
        .collect()
}

struct RoundReport {
    factor_exact: Measurement,
    factor_compiled: Measurement,
    factor_approximate: Measurement,
    factor_equivalent: bool,
    semantic: Vec<SemanticPoint>,
    worlds: Vec<WorldPoint>,
    capital_baseline: Measurement,
    capital_reused: Measurement,
    capital_equivalent: bool,
    capital_compiled_hits: u64,
}

struct SemanticPoint {
    budget_mib: u64,
    construct: Measurement,
    query: Measurement,
    protected_recall: f64,
    exact_recall: f64,
}

struct WorldPoint {
    count: usize,
    full_copy: OneShot,
    cow: OneShot,
    full_copy_bytes: u64,
    shared_bytes: u64,
    values_preserved: bool,
}

fn run_round(payload: &[u8], world_bytes: usize) -> RoundReport {
    let factor_exact = measure(SAMPLES, || exact_factor(payload));
    let factor_compiled = measure(SAMPLES, || compiled_factor(payload));
    let factor_approximate = measure(SAMPLES, || approximate_factor(payload));
    let factor_equivalent = exact_factor(payload) == compiled_factor(payload);

    let semantic = SEMANTIC_BUDGETS_MIB
        .into_iter()
        .map(|budget_mib| {
            let construct = measure(SAMPLES, || {
                let memory = SemanticVirtualMemory::synthetic(1_024, budget_mib * MIB).unwrap();
                black_box(
                    memory.morphology().active_bytes ^ memory.morphology().archived_detail_bytes,
                )
            });
            let memory = SemanticVirtualMemory::synthetic(1_024, budget_mib * MIB).unwrap();
            let query = measure(SAMPLES, || semantic_query_checksum(&memory));
            SemanticPoint {
                budget_mib,
                construct,
                query,
                protected_recall: memory.protected_recall_fraction(),
                exact_recall: memory.recall_fraction(SemanticResolution::Exact),
            }
        })
        .collect();

    let worlds = WORLD_COUNTS
        .into_iter()
        .map(|count| benchmark_worlds(world_bytes, count))
        .collect();

    let capital_inputs = capital_inputs(CAPITAL_CALLS, CAPITAL_INPUT);
    let capital_baseline_measurement = measure(SAMPLES, || capital_baseline(&capital_inputs));
    let capital_reused_measurement = measure(SAMPLES, || capital_reused(&capital_inputs).0);
    let (capital_checksum, capital_compiled_hits) = capital_reused(&capital_inputs);
    let capital_equivalent = capital_checksum == capital_baseline(&capital_inputs);

    RoundReport {
        factor_exact,
        factor_compiled,
        factor_approximate,
        factor_equivalent,
        semantic,
        worlds,
        capital_baseline: capital_baseline_measurement,
        capital_reused: capital_reused_measurement,
        capital_equivalent,
        capital_compiled_hits,
    }
}

fn exact_factor(payload: &[u8]) -> u64 {
    payload.iter().fold(0_u64, |sum, byte| {
        sum.rotate_left(5).wrapping_add(u64::from(*byte))
    })
}

fn compiled_factor(payload: &[u8]) -> u64 {
    let mut sum = 0_u64;
    let chunks = payload.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[0]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[1]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[2]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[3]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[4]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[5]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[6]));
        sum = sum.rotate_left(5).wrapping_add(u64::from(chunk[7]));
    }
    remainder.iter().fold(sum, |state, byte| {
        state.rotate_left(5).wrapping_add(u64::from(*byte))
    })
}

fn approximate_factor(payload: &[u8]) -> u64 {
    payload.iter().step_by(32).fold(0_u64, |sum, byte| {
        sum.rotate_left(5).wrapping_add(u64::from(*byte))
    })
}

fn semantic_query_checksum(memory: &SemanticVirtualMemory) -> u64 {
    (0..1_024_u32).fold(0_u64, |checksum, handle| {
        let value = memory
            .recall(handle, SemanticResolution::Summary)
            .unwrap_or_default();
        checksum.rotate_left(9) ^ value
    })
}

fn benchmark_worlds(world_bytes: usize, count: usize) -> WorldPoint {
    let base = (0..world_bytes / std::mem::size_of::<u64>())
        .map(|index| index as u64)
        .collect::<Vec<_>>();
    let (full_copy, _, full_values_preserved) = timed_world(|| full_copy_worlds(&base, count));
    let (cow, shared_bytes, values_preserved) = timed_world(|| cow_worlds(&base, count));
    let full_copy_bytes = (base.len() as u64)
        .saturating_mul(std::mem::size_of::<u64>() as u64)
        .saturating_mul(count as u64);
    WorldPoint {
        count,
        full_copy,
        cow,
        full_copy_bytes,
        shared_bytes,
        values_preserved: values_preserved && full_values_preserved,
    }
}

fn full_copy_worlds(base: &[u64], count: usize) -> (u64, u64, bool) {
    let mut worlds = Vec::with_capacity(count);
    for index in 0..count {
        let mut world = base.to_vec();
        let changed = index % world.len();
        world[changed] = (index as u64).wrapping_mul(17);
        worlds.push(world);
    }
    let values_preserved = worlds
        .iter()
        .enumerate()
        .all(|(index, world)| world[index % world.len()] == (index as u64).wrapping_mul(17));
    let checksum = worlds
        .iter()
        .enumerate()
        .fold(0_u64, |state, (index, world)| {
            state.rotate_left(9) ^ world[index % world.len()]
        });
    black_box(&worlds);
    (checksum, 0, values_preserved)
}

fn cow_worlds(base: &[u64], count: usize) -> (u64, u64, bool) {
    let root = VersionedWorld::from_base(WorldBase::new(base.to_vec()));
    let worlds = (0..count)
        .map(|index| {
            root.fork(&[(index % base.len(), (index as u64).wrapping_mul(17))])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let values_preserved = worlds.iter().enumerate().all(|(index, world)| {
        world.value(index % base.len()) == Some((index as u64).wrapping_mul(17))
    });
    let checksum = worlds.iter().fold(0_u64, |state, world| {
        state.rotate_left(9) ^ world.checksum()
    });
    let shared_bytes = VersionedWorld::shared_footprint_bytes(&worlds);
    black_box(&worlds);
    (checksum, shared_bytes, values_preserved)
}

fn capital_inputs(calls: usize, base_input: u64) -> Vec<u64> {
    (0..calls)
        .map(|index| base_input.saturating_add((index as u64).wrapping_mul(977) % 4_096))
        .collect()
}

fn capital_baseline(inputs: &[u64]) -> u64 {
    inputs.iter().fold(0_u64, |checksum, input| {
        checksum.rotate_left(7) ^ interpreted_triangular(black_box(*input))
    })
}

fn capital_reused(inputs: &[u64]) -> (u64, u64) {
    let mut runtime = CognitiveCapitalRuntime::new(3).unwrap();
    let mut checksum = 0_u64;
    let mut compiled_hits = 0_u64;
    for input in inputs {
        let (next, outcome) = runtime.solve("triangular", black_box(*input)).unwrap();
        compiled_hits += u64::from(outcome.dispatch == CapitalDispatch::Compiled);
        checksum = checksum.rotate_left(7) ^ outcome.answer;
        runtime = next;
    }
    (checksum, compiled_hits)
}

fn interpreted_triangular(input: u64) -> u64 {
    let mut sum = 0_u64;
    for value in 1..=input {
        sum = sum.wrapping_add(black_box(value));
    }
    black_box(sum)
}

#[derive(Clone)]
struct Measurement {
    nanoseconds: Vec<u128>,
    checksum: u64,
}

impl Measurement {
    fn p50(&self) -> u128 {
        self.nanoseconds[self.nanoseconds.len() / 2]
    }

    fn p95(&self) -> u128 {
        self.nanoseconds[(self.nanoseconds.len() * 95).div_ceil(100) - 1]
    }
}

#[derive(Clone, Copy)]
struct OneShot {
    nanoseconds: u128,
    checksum: u64,
}

fn measure(samples: usize, mut work: impl FnMut() -> u64) -> Measurement {
    let mut checksum = 0xA70D_0008_u64;
    let mut invocation = 0_usize;
    for _ in 0..WARMUPS {
        checksum = mix_checksum(checksum, work(), invocation);
        invocation = invocation.saturating_add(1);
    }
    let mut nanoseconds = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        checksum = mix_checksum(checksum, work(), invocation);
        invocation = invocation.saturating_add(1);
        nanoseconds.push(started.elapsed().as_nanos());
    }
    nanoseconds.sort_unstable();
    black_box(checksum);
    Measurement {
        nanoseconds,
        checksum,
    }
}

fn timed_world(work: impl FnOnce() -> (u64, u64, bool)) -> (OneShot, u64, bool) {
    let started = Instant::now();
    let (checksum, shared_bytes, values_preserved) = work();
    (
        OneShot {
            nanoseconds: started.elapsed().as_nanos(),
            checksum,
        },
        shared_bytes,
        values_preserved,
    )
}

fn print_report(reports: &[RoundReport], host: &str, world_kib: u64) {
    println!("# AXON V7-X — Contractive Physical Sweep\n");
    println!("- Host: {host}");
    println!(
        "- Protocolo: {} rodada(s); factor/semantic/capital com {} aquecimento e {} amostras; COW é uma materialização por rodada.",
        reports.len(),
        WARMUPS,
        SAMPLES
    );
    println!(
        "- Mundo base físico: {world_kib} KiB; o maior full-copy materializa {:.1} MiB de payload por rodada.\n",
        world_kib as f64 * WORLD_COUNTS.last().copied().unwrap_or_default() as f64 / 1024.0,
    );

    println!("## 1. Contracted Factor (8 MiB reais)");
    println!(
        "| exact p50/p95 | compiled p50/p95 | approximate p50/p95 | exact=compiled | checksum exact/compiled |"
    );
    println!("|---:|---:|---:|---:|---:|");
    let exact = median_measurement(reports.iter().map(|report| &report.factor_exact));
    let compiled = median_measurement(reports.iter().map(|report| &report.factor_compiled));
    let approximate = median_measurement(reports.iter().map(|report| &report.factor_approximate));
    println!(
        "| {}/{} | {}/{} | {}/{} | {} | {:016X}/{:016X} |",
        format_ns(exact.0),
        format_ns(exact.1),
        format_ns(compiled.0),
        format_ns(compiled.1),
        format_ns(approximate.0),
        format_ns(approximate.1),
        reports.iter().all(|report| report.factor_equivalent),
        combined_checksum(reports.iter().map(|report| report.factor_exact.checksum)),
        combined_checksum(reports.iter().map(|report| report.factor_compiled.checksum)),
    );

    println!("\n## 2. Semantic Virtual Memory (plano lógico executado no CPU)");
    println!("| Budget | construir p50 | consultar p50 | recall protegido | recall exato |");
    println!("|---:|---:|---:|---:|---:|");
    for index in 0..SEMANTIC_BUDGETS_MIB.len() {
        let points = reports
            .iter()
            .map(|report| &report.semantic[index])
            .collect::<Vec<_>>();
        println!(
            "| {} MiB | {} | {} | {:.1}% | {:.1}% |",
            points[0].budget_mib,
            format_ns(median(points.iter().map(|point| point.construct.p50()))),
            format_ns(median(points.iter().map(|point| point.query.p50()))),
            points[0].protected_recall * 100.0,
            points[0].exact_recall * 100.0,
        );
    }

    println!("\n## 3. Versioned World COW (payload real materializado)");
    println!(
        "| worlds | full-copy p50 | COW p50 | payload full-copy | full/shared lógico | branches válidos | checksum |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for index in 0..WORLD_COUNTS.len() {
        let points = reports
            .iter()
            .map(|report| &report.worlds[index])
            .collect::<Vec<_>>();
        let full_ns = median(points.iter().map(|point| point.full_copy.nanoseconds));
        let cow_ns = median(points.iter().map(|point| point.cow.nanoseconds));
        let logical_ratio = points[0].full_copy_bytes as f64 / points[0].shared_bytes.max(1) as f64;
        println!(
            "| {} | {} | {} | {:.1} MiB | {:.1}x | {} | {:016X} |",
            points[0].count,
            format_ms(full_ns),
            format_ms(cow_ns),
            points[0].full_copy_bytes as f64 / MIB as f64,
            logical_ratio,
            points.iter().all(|point| point.values_preserved),
            combined_checksum(
                points
                    .iter()
                    .map(|point| point.full_copy.checksum ^ point.cow.checksum),
            ),
        );
    }

    println!("\n## 4. Cognitive Capital (10.000 tarefas exatas)");
    let baseline = median_measurement(reports.iter().map(|report| &report.capital_baseline));
    let reused = median_measurement(reports.iter().map(|report| &report.capital_reused));
    let speedup = baseline.0 as f64 / reused.0.max(1) as f64;
    println!(
        "| baseline p50 | após compilação p50 | razão observada | resultados exatos | compiled hits |"
    );
    println!("|---:|---:|---:|---:|---:|");
    println!(
        "| {} | {} | {:.2}x | {} | {} |",
        format_ms(baseline.0),
        format_ms(reused.0),
        speedup,
        reports.iter().all(|report| report.capital_equivalent),
        median(
            reports
                .iter()
                .map(|report| u128::from(report.capital_compiled_hits)),
        ),
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
        .fold(0xA70D_7E58_11C0_5EED, |state, (index, checksum)| {
            mix_checksum(state, checksum, index)
        })
}

fn mix_checksum(state: u64, value: u64, index: usize) -> u64 {
    state.rotate_left(9) ^ value.rotate_left((index % 64) as u32) ^ index as u64
}

fn format_ns(nanoseconds: u128) -> String {
    format!("{nanoseconds} ns")
}

fn format_ms(nanoseconds: u128) -> String {
    format!("{:.3} ms", nanoseconds as f64 / 1_000_000.0)
}

fn host_metadata() -> String {
    let rustc = command_output("rustc", &["-V"]).unwrap_or_else(|| "rustc indisponivel".into());
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
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
        .unwrap_or_else(|| "CPU indisponivel".into());
        let ram = detect_total_ram_bytes()
            .map(|bytes| format!("{:.2} GB", bytes as f64 / 1024_f64.powi(3)))
            .unwrap_or_else(|| "RAM indisponivel".into());
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
    fn compiled_factor_preserves_the_exact_payload_result() {
        let payload = factor_payload();
        assert_eq!(exact_factor(&payload), compiled_factor(&payload));
        assert_ne!(exact_factor(&payload), approximate_factor(&payload));
    }

    #[test]
    fn checksum_mixer_does_not_cancel_a_repeated_value() {
        assert_ne!(combined_checksum([0x1234, 0x1234].into_iter()), 0);
    }

    #[test]
    fn world_builders_preserve_the_same_branch_mutations() {
        let base = (0..64_u64).collect::<Vec<_>>();
        let (_, _, full_valid) = full_copy_worlds(&base, 10);
        let (_, shared_bytes, cow_valid) = cow_worlds(&base, 10);

        assert!(full_valid);
        assert!(cow_valid);
        assert!(shared_bytes < 10 * base.len() as u64 * 8);
    }

    #[test]
    fn cognitive_capital_keeps_the_baseline_answer_exact() {
        let inputs = capital_inputs(20, 128);
        assert_eq!(capital_baseline(&inputs), capital_reused(&inputs).0);
        assert_eq!(capital_reused(&inputs).1, 17);
    }
}
