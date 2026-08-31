//! Sweep real da V7: compila morfologias e toca um working set limitado.
//!
//! O plano avalia orcamentos logicos ate 16 GiB. A materializacao fisica e
//! limitada por `--touch-cap-mib` para evitar consumir toda a RAM do host.

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use axon::core_v7::{
    CognitiveBodyPlan, CognitiveRegion, MorphogenicCompiler, ResourceBudget, WorkloadProfile,
};
use axon::system_info::detect_total_ram_bytes;

const MIB: u64 = 1024 * 1024;
const BUDGETS_MIB: [u64; 7] = [64, 128, 256, 512, 1_024, 4_096, 16_384];
const WARMUPS: usize = 2;
const SAMPLES: usize = 7;

fn main() {
    let options = Options::parse();
    let host = host_metadata();
    let touch_cap_bytes = physical_touch_cap_bytes(options.touch_cap_mib);
    let reports = (0..options.runs)
        .map(|_| run_round(touch_cap_bytes))
        .collect::<Vec<_>>();
    print_report(&reports, &host, options.touch_cap_mib);
}

#[derive(Clone, Copy)]
struct Options {
    runs: usize,
    touch_cap_mib: u64,
}

impl Options {
    fn parse() -> Self {
        let mut runs = 1;
        let mut touch_cap_mib = 64;
        let args = std::env::args().skip(1).collect::<Vec<_>>();
        let mut index = 0;
        while index < args.len() {
            match args[index].as_str() {
                "--runs" if index + 1 < args.len() => {
                    runs = parse_positive(&args[index + 1]);
                    index += 2;
                }
                "--touch-cap-mib" if index + 1 < args.len() => {
                    touch_cap_mib = parse_positive(&args[index + 1]) as u64;
                    index += 2;
                }
                _ => usage(),
            }
        }
        Self {
            runs,
            touch_cap_mib,
        }
    }
}

fn parse_positive(raw: &str) -> usize {
    raw.parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| usage())
}

fn usage() -> ! {
    eprintln!("uso: axon_v7_morphogenic_sweep [--runs N] [--touch-cap-mib N]");
    std::process::exit(2);
}

fn run_round(touch_cap_bytes: u64) -> RoundReport {
    let compiler = MorphogenicCompiler::default();
    let workload = WorkloadProfile::research();
    let points = BUDGETS_MIB
        .into_iter()
        .map(|budget_mib| {
            let budget = ResourceBudget::memory_only(budget_mib * MIB);
            let plan = compiler.compile(budget, workload).unwrap();
            let compile = measure(SAMPLES, || {
                let plan = compiler.compile(budget, workload).unwrap();
                black_box(plan_checksum(&plan))
            });
            let touched_bytes = plan.active_bytes.min(touch_cap_bytes);
            let touch = measure(SAMPLES, || touch_memory(&plan, touched_bytes));
            BudgetPoint {
                budget_mib,
                quality: plan.quality,
                shadow_memory_price: plan.shadow_memory_price,
                active_mib: plan.active_bytes / MIB,
                archived_mib: plan.archived_bytes / MIB,
                top_regions: top_regions(&plan),
                compile,
                touch,
                touched_mib: touched_bytes / MIB,
            }
        })
        .collect::<Vec<_>>();
    RoundReport { points }
}

fn measure(samples: usize, mut work: impl FnMut() -> u64) -> Measurement {
    let mut checksum = 0xA70D_0007_u64;
    for _ in 0..WARMUPS {
        checksum ^= work();
    }
    let mut nanoseconds = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        checksum ^= work();
        nanoseconds.push(started.elapsed().as_nanos());
    }
    nanoseconds.sort_unstable();
    black_box(checksum);
    Measurement {
        nanoseconds,
        checksum,
    }
}

fn touch_memory(plan: &CognitiveBodyPlan, cap_bytes: u64) -> u64 {
    let mut regions = Vec::new();
    let mut checksum = plan_checksum(plan);
    for (region, bytes) in materialized_regions(plan, cap_bytes) {
        let mut data = vec![0_u8; bytes];
        let salt = region as u8;
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(31).wrapping_add(salt);
        }
        checksum = data.iter().step_by(64).fold(checksum, |state, byte| {
            state.rotate_left(5) ^ u64::from(*byte)
        });
        regions.push(data);
    }
    black_box(&regions);
    black_box(checksum)
}

fn plan_checksum(plan: &CognitiveBodyPlan) -> u64 {
    plan.allocations.iter().fold(
        plan.quality.to_bits() ^ plan.active_bytes ^ plan.archived_bytes,
        |checksum, allocation| {
            checksum.rotate_left(7)
                ^ allocation.bytes
                ^ allocation.minimum_bytes.rotate_left(11)
                ^ allocation.desired_bytes.rotate_left(23)
                ^ ((allocation.region as u64) << 48)
                ^ ((allocation.tier as u64) << 56)
        },
    )
}

fn materialized_regions(plan: &CognitiveBodyPlan, cap_bytes: u64) -> Vec<(CognitiveRegion, usize)> {
    let active = plan
        .allocations
        .iter()
        .filter(|allocation| allocation.bytes > 0)
        .collect::<Vec<_>>();
    let total_active = active
        .iter()
        .map(|allocation| allocation.bytes)
        .sum::<u64>()
        .max(1);
    let mut remaining = cap_bytes;
    active
        .iter()
        .enumerate()
        .map(|(index, allocation)| {
            let share = if index + 1 == active.len() {
                remaining
            } else {
                cap_bytes
                    .saturating_mul(allocation.bytes)
                    .checked_div(total_active)
                    .unwrap_or(0)
                    .min(remaining)
            };
            remaining = remaining.saturating_sub(share);
            (
                allocation.region,
                usize::try_from(share).expect("physical touch cap fits usize"),
            )
        })
        .filter(|(_, bytes)| *bytes > 0)
        .collect()
}

fn physical_touch_cap_bytes(touch_cap_mib: u64) -> u64 {
    let requested = touch_cap_mib.checked_mul(MIB).unwrap_or_else(|| usage());
    let safe_host_cap = detect_total_ram_bytes()
        .map(|bytes| bytes / 4)
        .unwrap_or(512 * MIB);
    if requested > safe_host_cap || usize::try_from(requested).is_err() {
        eprintln!(
            "touch cap must fit the process and stay at or below one quarter of detected RAM ({:.0} MiB)",
            safe_host_cap as f64 / MIB as f64,
        );
        usage();
    }
    requested
}

fn top_regions(plan: &CognitiveBodyPlan) -> String {
    let mut allocations = plan.allocations.clone();
    allocations.sort_by(|left, right| right.bytes.cmp(&left.bytes));
    allocations
        .into_iter()
        .filter(|allocation| allocation.region != CognitiveRegion::Kernel)
        .take(3)
        .map(|allocation| format!("{}:{}MiB", allocation.region.name(), allocation.bytes / MIB))
        .collect::<Vec<_>>()
        .join(", ")
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

struct BudgetPoint {
    budget_mib: u64,
    quality: f64,
    shadow_memory_price: f64,
    active_mib: u64,
    archived_mib: u64,
    top_regions: String,
    compile: Measurement,
    touch: Measurement,
    touched_mib: u64,
}

struct RoundReport {
    points: Vec<BudgetPoint>,
}

fn print_report(reports: &[RoundReport], host: &str, touch_cap_mib: u64) {
    println!("# AXON V7 — Morphogenic Resource Sweep\n");
    println!("- Host: {host}");
    println!(
        "- Protocolo: {} rodada(s), {} aquecimentos, {} amostras por medicao.",
        reports.len(),
        WARMUPS,
        SAMPLES,
    );
    println!(
        "- Orcamento ate 16 GiB e planejado logicamente; toque fisico de RAM limitado a {touch_cap_mib} MiB por medicao.\n"
    );
    println!(
        "| Orcamento | Q(M) | preco sombra | ativo | arquivado | topologia dominante | compile p50/p95 | touch p50/p95 | RAM tocada | checksum |"
    );
    println!("|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|");
    for index in 0..BUDGETS_MIB.len() {
        let points = reports
            .iter()
            .map(|report| &report.points[index])
            .collect::<Vec<_>>();
        let compile_p50 = median(points.iter().map(|point| point.compile.p50()));
        let compile_p95 = median(points.iter().map(|point| point.compile.p95()));
        let touch_p50 = median(points.iter().map(|point| point.touch.p50()));
        let touch_p95 = median(points.iter().map(|point| point.touch.p95()));
        let checksum = combined_checksum(points.iter().map(|point| point.touch.checksum));
        println!(
            "| {} MiB | {:.3} | {:.3e} | {} MiB | {} MiB | {} | {}/{} | {}/{} | {} MiB | {:016X} |",
            points[0].budget_mib,
            points[0].quality,
            points[0].shadow_memory_price,
            points[0].active_mib,
            points[0].archived_mib,
            points[0].top_regions,
            format_ns(compile_p50),
            format_ns(compile_p95),
            format_ms(touch_p50),
            format_ms(touch_p95),
            points[0].touched_mib,
            checksum,
        );
    }
}

fn combined_checksum(checksums: impl Iterator<Item = u64>) -> u64 {
    checksums
        .enumerate()
        .fold(0xA70D_7E57_11C0_5EED, |state, (round, value)| {
            state.rotate_left(9) ^ value.rotate_left((round % 64) as u32) ^ round as u64
        })
}

fn median(values: impl Iterator<Item = u128>) -> u128 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
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
    fn physical_touch_is_partitioned_by_the_compiled_body_plan() {
        let plan = MorphogenicCompiler::default()
            .compile(
                ResourceBudget::memory_only(128 * MIB),
                WorkloadProfile::research(),
            )
            .unwrap();
        let regions = materialized_regions(&plan, MIB);
        assert_eq!(
            regions.iter().map(|(_, bytes)| *bytes).sum::<usize>(),
            MIB as usize
        );
        assert!(regions.len() > 1);
        assert_ne!(regions[0].1, regions[1].1);
    }

    #[test]
    fn physical_touch_checksum_is_deterministic_for_a_plan() {
        let plan = MorphogenicCompiler::default()
            .compile(
                ResourceBudget::memory_only(64 * MIB),
                WorkloadProfile::balanced(),
            )
            .unwrap();
        assert_eq!(
            touch_memory(&plan, 64 * 1024),
            touch_memory(&plan, 64 * 1024)
        );
    }

    #[test]
    fn combined_checksums_do_not_cancel_on_even_runs() {
        assert_ne!(combined_checksum([0x1234, 0x1234].into_iter()), 0);
    }
}
