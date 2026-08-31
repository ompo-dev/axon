//! Rodada de medição física V6-X executada em CPU/RAM local.
//!
//! Mede kernels reais da CPU. Não representa hardware ausente como se fosse
//! medido: p-bit, fotônica, quantum, energia e temperatura ficam fora daqui.

use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use axon::core_v5::CostOrigin;
use axon::core_v6::{
    PhysicalBackend, PhysicalCompiler, PhysicalCost, PhysicalCostUnit, PhysicalOperation,
    PhysicalOperationKind, PhysicalProfile, PhysicalStateKind, PrecisionRequirement,
    ReversibleScratch,
};
use axon::system_info::detect_total_ram_bytes;

const SIZES: [usize; 12] = [
    64, 256, 1_024, 4_096, 16_384, 65_536, 262_144, 524_288, 1_048_576, 4_194_304, 16_777_216,
    67_108_864,
];
const FOCUS_SIZE: usize = 524_288;
const LOCALITY_BYTES: usize = 32 * 1024 * 1024;
const EVICTION_BYTES: usize = 64 * 1024 * 1024;
const WARMUPS: usize = 3;
const SAMPLES: usize = 15;
const TARGET_BYTES_PER_SAMPLE: usize = 4 * 1024 * 1024;

fn main() {
    let rounds = parse_rounds();
    let host = host_metadata();
    let reports = (0..rounds).map(|_| run_round()).collect::<Vec<_>>();
    print_report(&reports, &host);
}

fn parse_rounds() -> usize {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [] => 1,
        [flag, count] if flag == "--runs" => count
            .parse::<usize>()
            .ok()
            .filter(|count| *count > 0)
            .unwrap_or_else(|| usage()),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("uso: axon_v6x_physical_sweep [--runs N]");
    std::process::exit(2);
}

fn run_round() -> RoundReport {
    RoundReport {
        breakdown: boundary_breakdown(FOCUS_SIZE),
        curve: SIZES.into_iter().map(curve_point).collect(),
        hot_cold: hot_cold_measurement(FOCUS_SIZE),
        locality: locality_measurement(),
        handles: handle_measurement(FOCUS_SIZE),
        verification: verification_measurement(FOCUS_SIZE),
        disposal: disposal_measurement(4_096),
        realize_scale: realize_scale_measurement(),
    }
}

fn boundary_breakdown(size: usize) -> Vec<Measurement> {
    let value = signs(0xB0A0_0001, size);
    let key = signs(0xB0A0_0002, size);
    let mut pipeline = BoundaryPipeline::new(value.clone(), key.clone());
    let mut fused = vec![0; size];
    let iterations = iterations(size, 7);
    let direct = measure("fused bind: A+B→C", size * 3, iterations, || {
        fused_bind(&value, &key, &mut fused);
        observed_slice(&fused)
    });
    let encode = measure("encode A,B", size * 2, iterations, || pipeline.encode());
    let copy = measure("copy A',B'", size * 2, iterations, || pipeline.copy());
    let xor = measure("XOR A',B'", size * 3, iterations, || pipeline.xor());
    let decode = measure("decode C'", size * 2, iterations, || pipeline.decode());
    let full = measure("boundary completa", size * 9, iterations, || {
        pipeline.full()
    });
    let allocation = measure("alocar e tocar 8 buffers", size * 8, iterations, || {
        let mut allocated = BoundaryPipeline::new(value.clone(), key.clone());
        allocated.touch_all_buffers()
    });
    pipeline.full();
    fused_bind(&value, &key, &mut fused);
    assert_eq!(pipeline.output(), fused);

    vec![direct, encode, copy, xor, decode, full, allocation]
}

fn curve_point(size: usize) -> CurvePoint {
    let value = signs(0xC0DE_1001, size);
    let key = signs(0xC0DE_1002, size);
    let mut fused = vec![0; size];
    let mut pipeline = BoundaryPipeline::new(value.clone(), key.clone());
    fused_bind(&value, &key, &mut fused);
    let expected = fused.clone();
    let verifier = SampleVerifier::new(64);
    let profile = measured_profile(size as u64);
    let compiler = PhysicalCompiler;
    let iterations = iterations(size, 8);
    let direct = measure("direct", size * 3, iterations, || {
        fused_bind(&value, &key, &mut fused);
        observed_slice(&fused)
    });
    let boundary = measure("boundary", size * 9, iterations, || pipeline.full());
    let verify = measure("verify-exato", size * 2, iterations, || {
        u64::from(black_box(fused.as_slice()) == black_box(expected.as_slice()))
    });
    let sampled = measure("verify-amostrado-64", 128, iterations, || {
        u64::from(verifier.verify(&fused, &expected))
    });
    let uncompute = measure("uncompute", size * 4, iterations, || {
        uncompute_roundtrip(size)
    });
    let realize = measure("realize", 0, 100_000, || {
        let plan = compiler
            .realize(exact_operation(), std::slice::from_ref(&profile))
            .expect("measured CPU profile must be eligible");
        u64::from(matches!(plan.backend, PhysicalBackend::CpuExact))
    });
    pipeline.full();
    assert_eq!(pipeline.output(), fused);
    assert!(verifier.verify(&fused, &expected));
    let checksum = full_checksum(&fused);
    assert_eq!(full_checksum(pipeline.output()), checksum);

    CurvePoint {
        size,
        direct,
        boundary,
        verify,
        sampled,
        uncompute,
        realize,
        checksum,
    }
}

fn hot_cold_measurement(size: usize) -> HotColdReport {
    let value = signs(0xA107_0001, size);
    let key = signs(0xA107_0002, size);
    let mut output = vec![0; size];
    let hot = measure("cache quente", size * 3, iterations(size, 3), || {
        fused_bind(&value, &key, &mut output);
        observed_slice(&output)
    });
    let mut eviction = vec![0_u8; EVICTION_BYTES];
    let evicted = measure_evicted("após sweep de 64 MiB", size * 3, || {
        evict_cache(&mut eviction);
        let started = Instant::now();
        fused_bind(&value, &key, &mut output);
        (started.elapsed(), observed_slice(&output))
    });
    HotColdReport { hot, evicted }
}

fn locality_measurement() -> LocalityReport {
    const BLOCK: usize = 4_096;
    let blocks = LOCALITY_BYTES / BLOCK;
    let data = signs(0x10CA_11A1, LOCALITY_BYTES);
    let clustered = (0..blocks).collect::<Vec<_>>();
    let scattered = shuffled_indices(blocks, 0x10CA_11A2);
    let grouped = measure("fatores agrupados", LOCALITY_BYTES, 8, || {
        scan_factor_blocks(&data, &clustered, BLOCK)
    });
    let dispersed = measure("fatores dispersos", LOCALITY_BYTES, 8, || {
        scan_factor_blocks(&data, &scattered, BLOCK)
    });
    LocalityReport { grouped, dispersed }
}

fn handle_measurement(size: usize) -> HandleReport {
    let factor = signs(0xA11D_1001, size);
    let key = signs(0xA11D_1002, size);
    let mut output = vec![0; size];
    let mut materialized = vec![0; size];
    let iterations = iterations(size, 4);
    let handle = measure("FactorHandle: referência", size * 3, iterations, || {
        fused_bind(&factor, &key, &mut output);
        observed_slice(&output)
    });
    let copy = measure("FactorCopy: materializar", size * 4, iterations, || {
        materialized.copy_from_slice(&factor);
        fused_bind(&materialized, &key, &mut output);
        observed_slice(&output)
    });
    HandleReport { handle, copy }
}

fn verification_measurement(size: usize) -> VerificationReport {
    let expected = signs(0x0E71_0001, size);
    let observed = expected.clone();
    let verifier = SampleVerifier::new(64);
    let iterations = iterations(size, 2);
    let l0 = measure("L0: dimensão", 0, 200_000, || {
        u64::from(black_box(observed.len()) == black_box(expected.len()))
    });
    let l1 = measure("L1: 64 amostras", 128, iterations, || {
        u64::from(verifier.verify(&observed, &expected))
    });
    let l3 = measure("L3: igualdade exata", size * 2, iterations, || {
        u64::from(black_box(observed.as_slice()) == black_box(expected.as_slice()))
    });
    VerificationReport { l0, l1, l3 }
}

fn disposal_measurement(size: usize) -> DisposalReport {
    let source = vec![0x3C_u8; size];
    let key = vec![0xA5_u8; size];
    let checkpoint = source.clone();
    let mut scratch = source.clone();
    let mut output = vec![0; size];
    let iterations = iterations(size, 4);
    let zero = measure("ZERO", size, iterations, || {
        scratch.fill(0);
        black_box(scratch.as_slice());
        observed_bytes(&scratch)
    });
    let restore = measure("CHECKPOINT+RESTORE", size, iterations, || {
        scratch.copy_from_slice(&checkpoint);
        black_box(scratch.as_slice());
        observed_bytes(&scratch)
    });
    let recompute = measure("RECOMPUTE", size * 3, iterations, || {
        byte_xor(&source, &key, &mut output);
        observed_bytes(&output)
    });
    let uncompute = measure("UNCOMPUTE lógico", size * 4, iterations, || {
        uncompute_roundtrip(size)
    });
    DisposalReport {
        zero,
        restore,
        recompute,
        uncompute,
    }
}

fn realize_scale_measurement() -> Vec<ScalePoint> {
    [1_usize, 10, 100, 1_000, 10_000, 1_000_000]
        .into_iter()
        .map(|decisions| {
            let cycles = if decisions >= 1_000_000 { 3 } else { 100 };
            ScalePoint {
                decisions,
                measurement: measure("REALIZE por região", 0, cycles, || {
                    realize_region(decisions)
                }),
            }
        })
        .collect()
}

fn measure(
    name: &'static str,
    payload_bytes: usize,
    iterations: usize,
    mut work: impl FnMut() -> u64,
) -> Measurement {
    let mut checksum = 0xC0DE_CAFE_D15E_A5E5_u64;
    for _ in 0..WARMUPS {
        checksum = run_batch(iterations, checksum, &mut work);
    }
    let mut picoseconds = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        checksum = run_batch(iterations, checksum, &mut work);
        picoseconds.push(started.elapsed().as_nanos().saturating_mul(1_000) / iterations as u128);
    }
    picoseconds.sort_unstable();
    black_box(checksum);
    Measurement {
        name,
        payload_bytes,
        picoseconds,
    }
}

fn measure_evicted(
    name: &'static str,
    payload_bytes: usize,
    mut work: impl FnMut() -> (Duration, u64),
) -> Measurement {
    let mut checksum = 0_u64;
    for _ in 0..WARMUPS {
        checksum ^= work().1;
    }
    let mut picoseconds = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let (elapsed, result) = work();
        checksum ^= result;
        picoseconds.push(elapsed.as_nanos().saturating_mul(1_000));
    }
    picoseconds.sort_unstable();
    black_box(checksum);
    Measurement {
        name,
        payload_bytes,
        picoseconds,
    }
}

fn run_batch(iterations: usize, checksum: u64, work: &mut impl FnMut() -> u64) -> u64 {
    (0..iterations).fold(checksum, |state, _| {
        state.rotate_left(7) ^ black_box(work())
    })
}

fn iterations(size: usize, passes: usize) -> usize {
    (TARGET_BYTES_PER_SAMPLE / size.saturating_mul(passes).max(1)).clamp(1, 65_536)
}

#[derive(Clone)]
struct Measurement {
    name: &'static str,
    payload_bytes: usize,
    picoseconds: Vec<u128>,
}

impl Measurement {
    fn p50(&self) -> u128 {
        self.picoseconds[self.picoseconds.len() / 2]
    }

    fn p95(&self) -> u128 {
        self.picoseconds[(self.picoseconds.len() * 95).div_ceil(100) - 1]
    }
}

struct CurvePoint {
    size: usize,
    direct: Measurement,
    boundary: Measurement,
    verify: Measurement,
    sampled: Measurement,
    uncompute: Measurement,
    realize: Measurement,
    checksum: u64,
}

struct HotColdReport {
    hot: Measurement,
    evicted: Measurement,
}

struct LocalityReport {
    grouped: Measurement,
    dispersed: Measurement,
}

struct HandleReport {
    handle: Measurement,
    copy: Measurement,
}

struct VerificationReport {
    l0: Measurement,
    l1: Measurement,
    l3: Measurement,
}

struct DisposalReport {
    zero: Measurement,
    restore: Measurement,
    recompute: Measurement,
    uncompute: Measurement,
}

struct ScalePoint {
    decisions: usize,
    measurement: Measurement,
}

struct RoundReport {
    breakdown: Vec<Measurement>,
    curve: Vec<CurvePoint>,
    hot_cold: HotColdReport,
    locality: LocalityReport,
    handles: HandleReport,
    verification: VerificationReport,
    disposal: DisposalReport,
    realize_scale: Vec<ScalePoint>,
}

fn print_report(reports: &[RoundReport], host: &str) {
    println!("# AXON V6-X — Physical Boundary Sweep\n");
    println!("- Host: {host}");
    println!(
        "- Protocolo: {} rodada(s), {} aquecimentos e {} amostras por medição.",
        reports.len(),
        WARMUPS,
        SAMPLES,
    );
    println!("- Tempo de parede; não mede energia, temperatura ou hardware ausente.\n");
    print_table(
        "## 1. Decomposição da fronteira — 512 KiB",
        reports,
        |report| report.breakdown.iter().collect(),
    );
    print_curve(reports);
    print_table(
        "## 3. Quente versus após sweep de 64 MiB",
        reports,
        |report| vec![&report.hot_cold.hot, &report.hot_cold.evicted],
    );
    print_table(
        "## 4. Localidade: agrupado versus disperso",
        reports,
        |report| vec![&report.locality.grouped, &report.locality.dispersed],
    );
    print_table("## 5. FactorHandle versus cópia", reports, |report| {
        vec![&report.handles.handle, &report.handles.copy]
    });
    print_table(
        "## 6. Níveis de verificação — 512 KiB",
        reports,
        |report| {
            vec![
                &report.verification.l0,
                &report.verification.l1,
                &report.verification.l3,
            ]
        },
    );
    print_table("## 7. State Disposal — 4 KiB", reports, |report| {
        vec![
            &report.disposal.zero,
            &report.disposal.restore,
            &report.disposal.recompute,
            &report.disposal.uncompute,
        ]
    });
    print_realize_scale(reports);
}

fn print_table<'a>(
    title: &str,
    reports: &'a [RoundReport],
    get: impl Fn(&'a RoundReport) -> Vec<&'a Measurement>,
) {
    println!("{title}");
    println!("| Trabalho | p50 mediano ns/op | p95 mediano ns/op | faixa p50 | payload |");
    println!("|---|---:|---:|---:|---:|");
    for index in 0..get(&reports[0]).len() {
        let values = reports
            .iter()
            .map(|report| get(report)[index])
            .collect::<Vec<_>>();
        println!("{}", render_aggregate(&values));
    }
    println!();
}

fn print_curve(reports: &[RoundReport]) {
    println!("## 2. Curva por tamanho");
    println!(
        "| Tamanho | direct p50 µs | boundary p50 µs | razão | verify exato µs | verify 64 µs | uncompute µs | REALIZE ns | checksum integral |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for index in 0..SIZES.len() {
        let points = reports
            .iter()
            .map(|report| &report.curve[index])
            .collect::<Vec<_>>();
        let direct = median(points.iter().map(|point| point.direct.p50()));
        let boundary = median(points.iter().map(|point| point.boundary.p50()));
        let verify = median(points.iter().map(|point| point.verify.p50()));
        let sampled = median(points.iter().map(|point| point.sampled.p50()));
        let uncompute = median(points.iter().map(|point| point.uncompute.p50()));
        let realize = median(points.iter().map(|point| point.realize.p50()));
        assert!(
            points
                .iter()
                .all(|point| point.checksum == points[0].checksum)
        );
        println!(
            "| {} | {} | {} | {:.2}x | {} | {} | {} | {} | {:016X} |",
            format_bytes(points[0].size),
            format_microseconds(direct),
            format_microseconds(boundary),
            boundary as f64 / direct.max(1) as f64,
            format_microseconds(verify),
            format_microseconds(sampled),
            format_microseconds(uncompute),
            format_nanoseconds(realize),
            points[0].checksum,
        );
    }
    println!();
}

fn print_realize_scale(reports: &[RoundReport]) {
    println!("## 8. REALIZE por região");
    println!("| Decisões por região | p50 total µs | p50 ns/decisão | faixa p50 total |");
    println!("|---:|---:|---:|---:|");
    for index in 0..reports[0].realize_scale.len() {
        let points = reports
            .iter()
            .map(|report| &report.realize_scale[index])
            .collect::<Vec<_>>();
        let total = median(points.iter().map(|point| point.measurement.p50()));
        let min = points
            .iter()
            .map(|point| point.measurement.p50())
            .min()
            .expect("points");
        let max = points
            .iter()
            .map(|point| point.measurement.p50())
            .max()
            .expect("points");
        println!(
            "| {} | {} | {} | {}–{} |",
            points[0].decisions,
            format_microseconds(total),
            format_nanoseconds(total / points[0].decisions as u128),
            format_microseconds(min),
            format_microseconds(max),
        );
    }
    println!();
}

fn render_aggregate(values: &[&Measurement]) -> String {
    let p50 = median(values.iter().map(|measurement| measurement.p50()));
    let p95 = median(values.iter().map(|measurement| measurement.p95()));
    let min = values
        .iter()
        .map(|measurement| measurement.p50())
        .min()
        .expect("values");
    let max = values
        .iter()
        .map(|measurement| measurement.p50())
        .max()
        .expect("values");
    format!(
        "| {} | {} | {} | {}–{} | {} |",
        values[0].name,
        format_nanoseconds(p50),
        format_nanoseconds(p95),
        format_nanoseconds(min),
        format_nanoseconds(max),
        format_bytes(values[0].payload_bytes),
    )
}

fn median(values: impl Iterator<Item = u128>) -> u128 {
    let mut values = values.collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

fn format_nanoseconds(picoseconds: u128) -> String {
    format!("{:.3}", picoseconds as f64 / 1_000.0)
}

fn format_microseconds(picoseconds: u128) -> String {
    format!("{:.3}", picoseconds as f64 / 1_000_000.0)
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024 * 1024) as f64)
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024_f64)
    } else {
        format!("{bytes} B")
    }
}

struct BoundaryPipeline {
    value: Vec<i8>,
    key: Vec<i8>,
    encoded_value: Vec<u8>,
    encoded_key: Vec<u8>,
    moved_value: Vec<u8>,
    moved_key: Vec<u8>,
    encoded_result: Vec<u8>,
    output: Vec<i8>,
}

impl BoundaryPipeline {
    fn new(value: Vec<i8>, key: Vec<i8>) -> Self {
        assert_eq!(value.len(), key.len());
        let size = value.len();
        Self {
            value,
            key,
            encoded_value: vec![0; size],
            encoded_key: vec![0; size],
            moved_value: vec![0; size],
            moved_key: vec![0; size],
            encoded_result: vec![0; size],
            output: vec![0; size],
        }
    }

    fn encode(&mut self) -> u64 {
        encode_signs(&self.value, &mut self.encoded_value);
        encode_signs(&self.key, &mut self.encoded_key);
        observed_bytes(&self.encoded_value) ^ observed_bytes(&self.encoded_key)
    }

    fn copy(&mut self) -> u64 {
        self.moved_value.copy_from_slice(&self.encoded_value);
        self.moved_key.copy_from_slice(&self.encoded_key);
        black_box(&self.moved_value);
        black_box(&self.moved_key);
        observed_bytes(&self.moved_value) ^ observed_bytes(&self.moved_key)
    }

    fn xor(&mut self) -> u64 {
        for ((result, value), key) in self
            .encoded_result
            .iter_mut()
            .zip(&self.moved_value)
            .zip(&self.moved_key)
        {
            *result = value ^ key;
        }
        black_box(&self.encoded_result);
        observed_bytes(&self.encoded_result)
    }

    fn decode(&mut self) -> u64 {
        decode_signs(&self.encoded_result, &mut self.output);
        observed_slice(&self.output)
    }

    fn full(&mut self) -> u64 {
        self.encode();
        self.copy();
        self.xor();
        self.decode()
    }

    fn output(&self) -> &[i8] {
        &self.output
    }

    fn touch_all_buffers(&mut self) -> u64 {
        let checksum = touch_signs(&mut self.value, 0x11)
            ^ touch_signs(&mut self.key, 0x22)
            ^ touch_bytes(&mut self.encoded_value, 0x33)
            ^ touch_bytes(&mut self.encoded_key, 0x44)
            ^ touch_bytes(&mut self.moved_value, 0x55)
            ^ touch_bytes(&mut self.moved_key, 0x66)
            ^ touch_bytes(&mut self.encoded_result, 0x77)
            ^ touch_signs(&mut self.output, 0x88);
        black_box(checksum)
    }
}

fn touch_signs(values: &mut [i8], salt: u8) -> u64 {
    let checksum =
        values
            .iter_mut()
            .enumerate()
            .fold(u64::from(salt), |checksum, (index, value)| {
                *value = if (index + usize::from(salt)) % 2 == 0 {
                    1
                } else {
                    -1
                };
                checksum.rotate_left(5) ^ u64::from(*value as u8)
            });
    black_box(checksum)
}

fn touch_bytes(values: &mut [u8], salt: u8) -> u64 {
    let checksum =
        values
            .iter_mut()
            .enumerate()
            .fold(u64::from(salt), |checksum, (index, value)| {
                *value = (index as u8).wrapping_add(salt);
                checksum.rotate_left(5) ^ u64::from(*value)
            });
    black_box(checksum)
}

fn fused_bind(value: &[i8], key: &[i8], output: &mut [i8]) {
    for ((output, value), key) in output.iter_mut().zip(value).zip(key) {
        *output = value * key;
    }
    black_box(output);
}

fn encode_signs(input: &[i8], output: &mut [u8]) {
    for (encoded, value) in output.iter_mut().zip(input) {
        *encoded = u8::from(*value < 0);
    }
    black_box(output);
}

fn decode_signs(input: &[u8], output: &mut [i8]) {
    for (decoded, value) in output.iter_mut().zip(input) {
        *decoded = if *value == 0 { 1 } else { -1 };
    }
    black_box(output);
}

fn observed_slice(values: &[i8]) -> u64 {
    let value = u64::from(values[0] as u8)
        ^ u64::from(values[values.len() / 2] as u8).rotate_left(11)
        ^ u64::from(values[values.len() - 1] as u8).rotate_left(23);
    black_box(value)
}

fn observed_bytes(values: &[u8]) -> u64 {
    let value = u64::from(values[0])
        ^ u64::from(values[values.len() / 2]).rotate_left(11)
        ^ u64::from(values[values.len() - 1]).rotate_left(23);
    black_box(value)
}

fn full_checksum(values: &[i8]) -> u64 {
    values
        .iter()
        .enumerate()
        .fold(values.len() as u64, |checksum, (index, value)| {
            checksum.rotate_left(5) ^ ((index as u64).rotate_left(17)) ^ u64::from(*value as u8)
        })
}

struct SampleVerifier {
    samples: usize,
}

impl SampleVerifier {
    fn new(samples: usize) -> Self {
        Self { samples }
    }

    fn verify(&self, observed: &[i8], expected: &[i8]) -> bool {
        if observed.len() != expected.len() {
            return false;
        }
        if observed.is_empty() {
            return self.samples == 0;
        }
        (0..self.samples).all(|index| {
            let position = self.position(index, observed.len());
            observed[position] == expected[position]
        })
    }

    fn position(&self, index: usize, length: usize) -> usize {
        index.wrapping_mul(0x9E37_79B9) % length
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
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
        .unwrap_or_else(|| "CPU indisponível".into());
        let ram = detect_total_ram_bytes()
            .map(|bytes| format!("{:.2} GB", bytes as f64 / 1024_f64.powi(3)))
            .unwrap_or_else(|| "RAM indisponível".into());
        let power = command_output("powercfg", &["/getactivescheme"])
            .unwrap_or_else(|| "plano de energia indisponível".into());
        format!(
            "OS={} arch={} CPU={} RAM={} threads={} energia={} compilador={}",
            std::env::consts::OS,
            std::env::consts::ARCH,
            cpu,
            ram,
            threads,
            power,
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

fn signs(seed: u64, size: usize) -> Vec<i8> {
    let mut state = seed;
    (0..size)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if state >> 63 == 0 { 1 } else { -1 }
        })
        .collect()
}

fn evict_cache(buffer: &mut [u8]) {
    for byte in buffer.iter_mut().step_by(64) {
        *byte = byte.wrapping_add(1);
    }
    black_box(buffer);
}

fn shuffled_indices(size: usize, seed: u64) -> Vec<usize> {
    let mut indices = (0..size).collect::<Vec<_>>();
    let mut state = seed;
    for index in (1..size).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        indices.swap(index, (state as usize) % (index + 1));
    }
    indices
}

fn scan_factor_blocks(data: &[i8], indices: &[usize], block: usize) -> u64 {
    let sum = indices.iter().fold(0_i64, |sum, index| {
        let start = index * block;
        let block_sum = data[start..start + block]
            .iter()
            .step_by(64)
            .map(|value| i64::from(*value))
            .sum::<i64>();
        sum.wrapping_add(block_sum)
    });
    black_box(sum as u64)
}

fn byte_xor(left: &[u8], right: &[u8], output: &mut [u8]) {
    for ((output, left), right) in output.iter_mut().zip(left).zip(right) {
        *output = left ^ right;
    }
    black_box(output);
}

fn uncompute_roundtrip(size: usize) -> u64 {
    let initial = vec![0x3C_u8; size];
    let temporary = initial.iter().map(|byte| byte ^ 0xFF).collect::<Vec<_>>();
    let result = temporary.iter().rev().copied().collect::<Vec<_>>();
    let restored = ReversibleScratch::new(initial)
        .compute("derive temporary witness", temporary)
        .commit_result(result)
        .uncompute()
        .expect("committed result allows uncompute");
    assert_eq!(restored.pending_steps(), 0);
    black_box(restored.working());
    observed_bytes(restored.committed_result().expect("result stays committed"))
}

fn exact_operation() -> PhysicalOperation {
    PhysicalOperation {
        kind: PhysicalOperationKind::ExactVerification,
        precision: PrecisionRequirement::Exact,
        latency_target_ns: None,
    }
}

fn realize_region(decisions: usize) -> u64 {
    let profile = measured_profile(1);
    let compiler = PhysicalCompiler;
    (0..decisions).fold(0_u64, |sum, _| {
        let plan = compiler
            .realize(exact_operation(), std::slice::from_ref(&profile))
            .expect("CPU profile is eligible");
        sum + u64::from(matches!(plan.backend, PhysicalBackend::CpuExact))
    })
}

fn measured_profile(latency_ns: u64) -> PhysicalProfile {
    PhysicalProfile {
        backend: PhysicalBackend::CpuExact,
        state: PhysicalStateKind::Digital,
        supports_exact: true,
        operations: vec![PhysicalOperationKind::ExactVerification],
        cost: PhysicalCost {
            encode_units: 0,
            move_units: 0,
            compute_units: 0,
            decode_units: 0,
            verify_units: 0,
            cooling_units: 0,
            calibration_units: 0,
            wear_units: 0,
            latency_ns,
            error_milliunits: 0,
            origin: CostOrigin::Measured,
            unit: PhysicalCostUnit::AbstractScore,
            source_id: 0x5357_4545_502D_4350,
            calibration_id: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{BoundaryPipeline, SampleVerifier, full_checksum, fused_bind, signs};

    #[test]
    fn fused_bind_matches_the_materialized_boundary_pipeline() {
        let value = vec![1_i8, -1, 1, -1];
        let key = vec![-1_i8, -1, 1, 1];
        let mut pipeline = BoundaryPipeline::new(value.clone(), key.clone());
        let mut fused = vec![0_i8; value.len()];
        pipeline.full();
        fused_bind(&value, &key, &mut fused);
        assert_eq!(pipeline.output(), fused);
    }

    #[test]
    fn sampled_verifier_rejects_a_changed_sampled_dimension() {
        let expected = vec![1_i8; 64];
        let mut observed = expected.clone();
        observed[0] = -1;
        assert!(!SampleVerifier::new(8).verify(&observed, &expected));
    }

    #[test]
    fn sampled_verifier_handles_empty_vectors_without_panicking() {
        assert!(SampleVerifier::new(0).verify(&[], &[]));
        assert!(!SampleVerifier::new(1).verify(&[], &[]));
    }

    #[test]
    fn sampled_verifier_is_not_presented_as_full_verification() {
        let verifier = SampleVerifier::new(8);
        let expected = vec![1_i8; 128];
        let untouched = (0..expected.len())
            .find(|candidate| {
                !(0..8).any(|sample| verifier.position(sample, expected.len()) == *candidate)
            })
            .expect("eight samples leave an unchecked dimension");
        let mut observed = expected.clone();
        observed[untouched] = -1;
        assert!(verifier.verify(&observed, &expected));
        assert_ne!(observed, expected);
    }

    #[test]
    fn large_boundary_output_has_exact_match_and_checksum() {
        let value = signs(0xC0DE_1001, 524_288);
        let key = signs(0xC0DE_1002, 524_288);
        let mut pipeline = BoundaryPipeline::new(value.clone(), key.clone());
        let mut fused = vec![0_i8; value.len()];
        pipeline.full();
        fused_bind(&value, &key, &mut fused);
        assert_eq!(pipeline.output(), fused);
        assert_eq!(full_checksum(pipeline.output()), full_checksum(&fused));
        assert_eq!(full_checksum(&fused), 0x18C5_38C9_0805_2B0C);
    }
}
