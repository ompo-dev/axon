//! Benchmark executado na CPU local para a implementação V6-X.
//!
//! Mede tempo de parede, não energia. Os únicos backends executados são CPU e
//! RAM deste computador; nomes como P-bit, fotônico e quantum não participam
//! deste binário porque não há esses dispositivos conectados ao processo.

use std::hint::black_box;
use std::time::Instant;

use axon::core_v5::CostOrigin;
use axon::core_v6::{
    PhysicalBackend, PhysicalCompiler, PhysicalCost, PhysicalCostUnit, PhysicalOperation,
    PhysicalOperationKind, PhysicalProfile, PhysicalStateKind, PrecisionRequirement,
    ReversibleScratch,
};

const DIMENSION: usize = 262_144;
const WARMUP_SAMPLES: usize = 5;
const MEASURED_SAMPLES: usize = 25;
const VECTOR_ITERATIONS: usize = 24;
const POLICY_ITERATIONS: usize = 100_000;
const SCRATCH_BYTES: usize = 4_096;
const SCRATCH_ITERATIONS: usize = 2_000;

fn main() {
    let rounds = requested_rounds();
    let results = (0..rounds).map(|_| run_round()).collect::<Vec<_>>();

    print_report(&results);
}

fn run_round() -> RoundResult {
    let value = signed_vector(0xA601_0001, DIMENSION);
    let key = signed_vector(0xA601_0002, DIMENSION);
    let mut direct = CpuDirectBind::new(value.clone(), key.clone());
    let mut boundary = CpuBoundaryPipeline::new(value, key);

    let direct_result = benchmark("bind direto CPU", VECTOR_ITERATIONS, DIMENSION * 2, || {
        direct.run()
    });
    let boundary_result = benchmark(
        "bind CPU com encode/copia/xor/decode",
        VECTOR_ITERATIONS,
        DIMENSION * 2,
        || boundary.run(),
    );
    assert_eq!(direct.output(), boundary.output());

    let expected = direct.output().to_vec();
    let verifier = CpuExactVerification::new(expected.clone(), expected);
    assert!(verifier.verify());
    let verification_result = benchmark(
        "verificação exata CPU",
        VECTOR_ITERATIONS,
        DIMENSION * 2,
        || u64::from(verifier.verify()),
    );

    let compiler = PhysicalCompiler;
    let operation = exact_operation();
    let measured_cpu_profile = cpu_profile(verification_result.p50_ns_ceil().max(1) as u64);
    let realized = compiler
        .realize(operation, std::slice::from_ref(&measured_cpu_profile))
        .expect("the measured CPU profile must satisfy exact verification");
    assert_eq!(realized.backend, PhysicalBackend::CpuExact);
    assert_eq!(
        realized.cost.latency_ns,
        verification_result.p50_ns_ceil() as u64
    );
    let policy_result = benchmark("REALIZE exato na CPU", POLICY_ITERATIONS, 0, || {
        let plan = compiler
            .realize(operation, std::slice::from_ref(&measured_cpu_profile))
            .expect("the measured CPU profile must remain eligible");
        backend_checksum(plan.backend)
    });
    let direct_policy_result = benchmark("baseline: CPU direta", POLICY_ITERATIONS, 0, || {
        backend_checksum(black_box(PhysicalBackend::CpuExact))
    });
    let uncompute_result = benchmark(
        "UNCOMPUTE lógico com 4 KiB scratch",
        SCRATCH_ITERATIONS,
        SCRATCH_BYTES,
        uncompute_roundtrip,
    );

    RoundResult {
        direct: direct_result,
        boundary: boundary_result,
        verification: verification_result,
        realize: policy_result,
        direct_policy: direct_policy_result,
        uncompute: uncompute_result,
        direct_checksum: checksum(direct.output()),
        boundary_checksum: checksum(boundary.output()),
    }
}

fn print_report(rounds: &[RoundResult]) {
    assert!(
        !rounds.is_empty(),
        "at least one benchmark round is required"
    );
    let verification_latency_ns = median_p50(rounds, |round| &round.verification).div_ceil(1_000);
    println!("# Axon V6-X — benchmark real da CPU local\n");
    println!(
        "- Hardware usado: {} threads lógicas disponíveis; dimensão de vetor: {}; RAM de scratch: {} KiB.",
        std::thread::available_parallelism().map_or(1, usize::from),
        DIMENSION,
        SCRATCH_BYTES / 1024,
    );
    println!(
        "- Protocolo: {} rodada(s), cada uma com {} aquecimentos + {} amostras; medições em ns por operação.",
        rounds.len(),
        WARMUP_SAMPLES,
        MEASURED_SAMPLES,
    );
    println!(
        "- Saída verificada fora da região cronometrada: bind direto e pipeline de fronteira produziram o mesmo vetor; checksum direto={:016X}, fronteira={:016X}.",
        rounds[0].direct_checksum, rounds[0].boundary_checksum,
    );
    println!(
        "- Perfil medido registrado no Ψ-IR: {:?}, latência p50={} ns/op, origem={:?}.",
        PhysicalBackend::CpuExact,
        verification_latency_ns,
        CostOrigin::Measured,
    );
    print_results_table(rounds);
    let direct = median_p50(rounds, |round| &round.direct);
    let boundary = median_p50(rounds, |round| &round.boundary);
    let realize = median_p50(rounds, |round| &round.realize);
    let direct_policy = median_p50(rounds, |round| &round.direct_policy);
    println!(
        "\n- Overhead observado da pipeline CPU com fronteiras: {:.2}x sobre bind direto (p50).",
        boundary as f64 / direct.max(1) as f64,
    );
    println!(
        "- Overhead observado da decisão REALIZE: {:.2}x sobre escolher CPU sem validação (p50).",
        realize as f64 / direct_policy.max(1) as f64,
    );
    println!(
        "- Escopo: estes números são medições de CPU/RAM deste PC. Não medem energia e não executam P-bit, analógico, fotônico, reversível físico ou quantum."
    );
}

fn requested_rounds() -> usize {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [] => 1,
        [flag, count] if flag == "--runs" => count
            .parse::<usize>()
            .ok()
            .filter(|count| *count > 0)
            .unwrap_or_else(|| {
                eprintln!("uso: axon_v6x_cpu_bench [--runs N], com N maior que zero");
                std::process::exit(2);
            }),
        _ => {
            eprintln!("uso: axon_v6x_cpu_bench [--runs N]");
            std::process::exit(2);
        }
    }
}

fn print_results_table(rounds: &[RoundResult]) {
    if rounds.len() == 1 {
        println!(
            "\n| Trabalho executado nesta CPU | p50 ns/op | p95 ns/op | mín. | máx. | Payload efetivo |"
        );
        println!("|---|---:|---:|---:|---:|---:|");
        for result in [
            &rounds[0].direct,
            &rounds[0].boundary,
            &rounds[0].verification,
            &rounds[0].realize,
            &rounds[0].direct_policy,
            &rounds[0].uncompute,
        ] {
            println!("{}", result.table_row());
        }
        return;
    }

    println!(
        "\n| Trabalho executado nesta CPU | mediana dos p50 ns/op | faixa dos p50 | Payload efetivo |"
    );
    println!("|---|---:|---:|---:|");
    println!(
        "{}",
        aggregate_table_row("bind direto CPU", DIMENSION * 2, rounds, |round| &round
            .direct)
    );
    println!(
        "{}",
        aggregate_table_row(
            "bind CPU com encode/copia/xor/decode",
            DIMENSION * 2,
            rounds,
            |round| &round.boundary
        )
    );
    println!(
        "{}",
        aggregate_table_row("verificação exata CPU", DIMENSION * 2, rounds, |round| {
            &round.verification
        })
    );
    println!(
        "{}",
        aggregate_table_row("REALIZE exato na CPU", 0, rounds, |round| &round.realize)
    );
    println!(
        "{}",
        aggregate_table_row("baseline: CPU direta", 0, rounds, |round| &round
            .direct_policy)
    );
    println!(
        "{}",
        aggregate_table_row(
            "UNCOMPUTE lógico com 4 KiB scratch",
            SCRATCH_BYTES,
            rounds,
            |round| &round.uncompute
        )
    );
}

fn aggregate_table_row(
    name: &str,
    payload_bytes: usize,
    rounds: &[RoundResult],
    get: impl Fn(&RoundResult) -> &BenchmarkResult,
) -> String {
    let mut values = rounds
        .iter()
        .map(|round| get(round).p50_picoseconds())
        .collect::<Vec<_>>();
    values.sort_unstable();
    let payload = if payload_bytes == 0 {
        "decisão".to_string()
    } else {
        format!("{} KiB", payload_bytes / 1024)
    };
    format!(
        "| {name} | {} | {}–{} | {payload} |",
        format_nanoseconds(values[values.len() / 2]),
        format_nanoseconds(values[0]),
        format_nanoseconds(*values.last().expect("rounds are nonempty")),
    )
}

fn median_p50(rounds: &[RoundResult], get: impl Fn(&RoundResult) -> &BenchmarkResult) -> u128 {
    let mut values = rounds
        .iter()
        .map(|round| get(round).p50_picoseconds())
        .collect::<Vec<_>>();
    values.sort_unstable();
    values[values.len() / 2]
}

#[derive(Debug)]
struct BenchmarkResult {
    name: &'static str,
    payload_bytes: usize,
    samples_picoseconds_per_operation: Vec<u128>,
}

struct RoundResult {
    direct: BenchmarkResult,
    boundary: BenchmarkResult,
    verification: BenchmarkResult,
    realize: BenchmarkResult,
    direct_policy: BenchmarkResult,
    uncompute: BenchmarkResult,
    direct_checksum: u64,
    boundary_checksum: u64,
}

impl BenchmarkResult {
    fn p50_picoseconds(&self) -> u128 {
        self.samples_picoseconds_per_operation[self.samples_picoseconds_per_operation.len() / 2]
    }

    fn p50_ns_ceil(&self) -> u128 {
        self.p50_picoseconds().div_ceil(1_000)
    }

    fn p95_picoseconds(&self) -> u128 {
        let index = (self.samples_picoseconds_per_operation.len() * 95).div_ceil(100) - 1;
        self.samples_picoseconds_per_operation[index]
    }

    fn table_row(&self) -> String {
        let min = self.samples_picoseconds_per_operation[0];
        let max = *self
            .samples_picoseconds_per_operation
            .last()
            .expect("benchmark always has samples");
        let payload = if self.payload_bytes == 0 {
            "decisão".to_string()
        } else {
            format!("{} KiB", self.payload_bytes / 1024)
        };
        format!(
            "| {} | {} | {} | {} | {} | {} |",
            self.name,
            format_nanoseconds(self.p50_picoseconds()),
            format_nanoseconds(self.p95_picoseconds()),
            format_nanoseconds(min),
            format_nanoseconds(max),
            payload,
        )
    }
}

fn benchmark(
    name: &'static str,
    iterations_per_sample: usize,
    payload_bytes: usize,
    mut work: impl FnMut() -> u64,
) -> BenchmarkResult {
    let mut checksum = 0xC0DE_CAFE_D15E_A5E5_u64;
    for _ in 0..WARMUP_SAMPLES {
        checksum = run_batch(iterations_per_sample, checksum, &mut work);
    }

    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
    for _ in 0..MEASURED_SAMPLES {
        let started = Instant::now();
        checksum = run_batch(iterations_per_sample, checksum, &mut work);
        samples.push(
            started.elapsed().as_nanos().saturating_mul(1_000) / iterations_per_sample as u128,
        );
    }
    samples.sort_unstable();
    black_box(checksum);
    BenchmarkResult {
        name,
        payload_bytes,
        samples_picoseconds_per_operation: samples,
    }
}

fn format_nanoseconds(picoseconds: u128) -> String {
    format!("{:.3}", picoseconds as f64 / 1_000.0)
}

fn run_batch(iterations: usize, checksum: u64, work: &mut impl FnMut() -> u64) -> u64 {
    (0..iterations).fold(checksum, |accumulator, _| {
        accumulator.rotate_left(7) ^ black_box(work())
    })
}

struct CpuDirectBind {
    value: Vec<i8>,
    key: Vec<i8>,
    output: Vec<i8>,
}

impl CpuDirectBind {
    fn new(value: Vec<i8>, key: Vec<i8>) -> Self {
        assert_eq!(value.len(), key.len());
        Self {
            output: vec![0; value.len()],
            value,
            key,
        }
    }

    fn run(&mut self) -> u64 {
        for ((output, value), key) in self.output.iter_mut().zip(&self.value).zip(&self.key) {
            *output = value * key;
        }
        black_box(self.output.as_slice());
        self.output.len() as u64
    }

    fn output(&self) -> &[i8] {
        &self.output
    }
}

struct CpuExactVerification {
    observed: Vec<i8>,
    expected: Vec<i8>,
}

impl CpuExactVerification {
    fn new(observed: Vec<i8>, expected: Vec<i8>) -> Self {
        Self { observed, expected }
    }

    fn verify(&self) -> bool {
        black_box(self.observed.as_slice()) == black_box(self.expected.as_slice())
    }
}

/// Executa na CPU a forma completa de uma fronteira: codificar, copiar,
/// computar em outra representação e decodificar. Não simula desempenho de
/// um acelerador externo; mede justamente o trabalho de fronteira local.
struct CpuBoundaryPipeline {
    value: Vec<i8>,
    key: Vec<i8>,
    encoded_value: Vec<u8>,
    encoded_key: Vec<u8>,
    moved_value: Vec<u8>,
    moved_key: Vec<u8>,
    encoded_result: Vec<u8>,
    output: Vec<i8>,
}

impl CpuBoundaryPipeline {
    fn new(value: Vec<i8>, key: Vec<i8>) -> Self {
        assert_eq!(value.len(), key.len());
        let dimension = value.len();
        Self {
            value,
            key,
            encoded_value: vec![0; dimension],
            encoded_key: vec![0; dimension],
            moved_value: vec![0; dimension],
            moved_key: vec![0; dimension],
            encoded_result: vec![0; dimension],
            output: vec![0; dimension],
        }
    }

    fn run(&mut self) -> u64 {
        encode_signs(&self.value, &mut self.encoded_value);
        encode_signs(&self.key, &mut self.encoded_key);
        self.moved_value.copy_from_slice(&self.encoded_value);
        self.moved_key.copy_from_slice(&self.encoded_key);
        for ((result, value), key) in self
            .encoded_result
            .iter_mut()
            .zip(&self.moved_value)
            .zip(&self.moved_key)
        {
            *result = value ^ key;
        }
        decode_signs(&self.encoded_result, &mut self.output);
        black_box(self.output.as_slice());
        self.output.len() as u64
    }

    fn output(&self) -> &[i8] {
        &self.output
    }
}

fn encode_signs(input: &[i8], output: &mut [u8]) {
    for (encoded, value) in output.iter_mut().zip(input) {
        *encoded = u8::from(*value < 0);
    }
}

fn decode_signs(input: &[u8], output: &mut [i8]) {
    for (decoded, value) in output.iter_mut().zip(input) {
        *decoded = if *value == 0 { 1 } else { -1 };
    }
}

fn checksum(values: &[i8]) -> u64 {
    values
        .iter()
        .fold(0x9E37_79B9_7F4A_7C15_u64, |state, value| {
            state.rotate_left(5) ^ (*value as i64 as u64).wrapping_mul(0xA24B_AED4_963E_E407)
        })
}

fn signed_vector(seed: u64, dimension: usize) -> Vec<i8> {
    let mut state = seed;
    (0..dimension)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if state >> 63 == 0 { 1 } else { -1 }
        })
        .collect()
}

fn exact_operation() -> PhysicalOperation {
    PhysicalOperation {
        kind: PhysicalOperationKind::ExactVerification,
        precision: PrecisionRequirement::Exact,
        latency_target_ns: None,
    }
}

fn cpu_profile(latency_ns: u64) -> PhysicalProfile {
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
            source_id: 0x4350_552D_4245_4E43,
            calibration_id: 1,
        },
    }
}

fn backend_checksum(backend: PhysicalBackend) -> u64 {
    match backend {
        PhysicalBackend::CpuExact => 0x0043_5055,
        _ => 0,
    }
}

fn uncompute_roundtrip() -> u64 {
    let initial = vec![0x3C_u8; SCRATCH_BYTES];
    let temporary = initial.iter().map(|byte| byte ^ 0xFF).collect::<Vec<_>>();
    let result = temporary.iter().rev().copied().collect::<Vec<_>>();
    let restored = ReversibleScratch::new(initial)
        .compute("derive temporary witness", temporary)
        .commit_result(result)
        .uncompute()
        .expect("a committed result permits logical uncompute");
    assert_eq!(restored.pending_steps(), 0);
    assert_eq!(restored.working(), &vec![0x3C_u8; SCRATCH_BYTES]);
    byte_checksum(
        restored
            .committed_result()
            .expect("result remains committed"),
    )
}

fn byte_checksum(values: &[u8]) -> u64 {
    values
        .iter()
        .fold(0xD6E8_FEB8_6659_FD93_u64, |state, value| {
            state.rotate_left(5) ^ u64::from(*value).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        })
}

#[cfg(test)]
mod tests {
    use super::{CpuBoundaryPipeline, CpuDirectBind, CpuExactVerification, ReversibleScratch};

    #[test]
    fn boundary_pipeline_preserves_the_same_bind_result_as_direct_cpu_execution() {
        let value = vec![1, -1, 1, -1, 1, 1, -1, -1];
        let key = vec![-1, -1, 1, 1, -1, 1, -1, 1];
        let mut direct = CpuDirectBind::new(value.clone(), key.clone());
        let mut pipeline = CpuBoundaryPipeline::new(value, key);

        assert_eq!(direct.run(), pipeline.run());
        assert_eq!(direct.output(), pipeline.output());
    }

    #[test]
    fn logical_uncompute_keeps_the_committed_result_and_restores_scratch() {
        let scratch = ReversibleScratch::new(vec![1_u8, 2, 3])
            .compute("temporary", vec![9_u8, 9, 9])
            .commit_result(vec![7_u8]);
        let restored = scratch.uncompute().expect("result was committed");

        assert_eq!(restored.working(), &vec![1, 2, 3]);
        assert_eq!(restored.committed_result(), Some(&vec![7]));
        assert_eq!(restored.pending_steps(), 0);
    }

    #[test]
    fn exact_verification_rejects_a_single_changed_dimension() {
        let verifier = CpuExactVerification::new(vec![1_i8, -1, 1], vec![1_i8, 1, 1]);

        assert!(!verifier.verify());
    }
}
