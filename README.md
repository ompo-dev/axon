# AXON-UIC

Rebuild limpo da AXON como **Universal Intelligence Calculus**: um núcleo Rust sem dependências que só aceita otimizações quando preservam contrato e sempre mantém fallback exato.

## O que executa hoje

| Lei | Primitiva | Garantia atual |
|---|---|---|
| Refinar só até decidir | `RefinementSet`, `DecisionCertificate` | resultado só contrai; ação para quando limites não se sobrepõem |
| Calcular só `needed ∩ changed` | `ExecutionSlice` | cone backward do objetivo intersectado com delta forward |
| Escolher Delta ou Full por contrato e custo | `IncrementalizabilityAnalyzer`, `CostEstimate` | `SUM`/`COUNT`/`XOR` locais são exatos; `SORT` usa Full |
| Reusar decisão física já medida | `StrategyEvidence`, `MetaJit` | estratégia só vira reflexo com intervalos não sobrepostos no mesmo domínio |
| Comparar custo sem esconder fronteira | `BenchContract` | `HOT = execution`; `LIFECYCLE` soma todas as fases medidas da mesma rodada |
| Separar verdade de execução | `SemanticArtifact`, `PhysicalRealization` | selo semântico imutável não depende do backend físico atual |
| Criar e reusar capacidade persistente | `ArtifactStore`, `ArtifactLifetime` | criação, reload, invalidação e break-even têm custo explícito |
| Comprimir rajadas de escrita | `coalesce_adjacent_last_writes` | última escrita adjacente preserva estado final |
| Medir adaptação local | `axon-uic-hybrid-sweep` | compara Hybrid fim a fim, Oracle pré-compilado e Change Fabric com ingestão contabilizada |
| Derivar delta declarado e checá-lo | `DeltaForge`, `DerivedSumPlan` | `fold(Add mod u64)` deriva cache/regra; `MIN` é recusado |
| Derivar média exata | `DerivedAveragePlan`, `ExactAverage` | `AVG` deriva `sum:u128 + count:usize`; não recebe cache do chamador |
| Executar tarefa declarada | `axon solve` | parser tipado, guard barato, hash semântico e resultado exato |
| Quocientar estrutura equivalente | `LiftCertificate` | LIFT exato para classe de fontes idênticas; fora dela usa fallback |
| Preservar comportamento | `AbstractionContract`, `SemanticContract` | limite de erro explícito; remorph exige contrato igual |
| Preferir descrição física barata | `PhysicalCost`, `CostPrices` | refinamentos disputam custo de latência, bytes e energia |
| Segurança por capability | `CapabilityGate` | impossibilidade ou efeito sem autoridade bloqueia execução |
| Remorph só amortizado | `Morphology`, `RemorphPolicy` | base protegida, histerese e Migration Tax |
| Otimização nunca quebra correção | `run_checked` | erro ou divergência retorna resultado exato |

Não é AGI, percepção, robótica nem descoberta científica geral. É fundação executável e testável dessas regras.

## Rodar

```powershell
cargo test --all-targets
cargo run --bin axon-uic
cargo run --bin axon -- solve .\task.axon
cargo run --release --bin axon-uic-bench -- --mib 64 --queries 20 --runs 5
cargo run --release --bin axon-uic-delta-sweep -- --mib 64 --runs 5 --max-updates 8000000
cargo run --release --bin axon-uic-hybrid-sweep -- --mib 64 --runs 30 --hardware-id <cpu-ram-profile>
cargo run --release --bin axon-uic-deltaforge-sum -- --mib 64 --runs 15
cargo run --release --bin axon-uic-deltaforge-avg -- --mib 64 --runs 15
```

Resultados e protocolo: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/RESULTS.md](docs/RESULTS.md). Contratos e limites: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/BENCH_CONTRACT.md](docs/BENCH_CONTRACT.md), [docs/DELTA_ALGEBRA.md](docs/DELTA_ALGEBRA.md), [docs/DELTAFORGE.md](docs/DELTAFORGE.md), [docs/HYBRID_RECOMPUTE.md](docs/HYBRID_RECOMPUTE.md).

Um arquivo de tarefa é declarativo:

```text
task average_stream {
  data numbers: Vec<u64> = [2, 4, 9]
  goal derive IncrementalArtifact<AverageExactU64>
}
```

`axon solve` grava artifacts em `.axon-artifacts/` ao lado da tarefa (ou em `--artifact-dir`). Esse diretório é estado local e está no `.gitignore`.
