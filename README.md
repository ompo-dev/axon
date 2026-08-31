# AXON-UIC

Rebuild limpo da AXON como **Universal Intelligence Calculus**: um núcleo Rust sem dependências que só aceita otimizações quando preservam contrato e sempre mantém fallback exato.

## O que executa hoje

| Lei | Primitiva | Garantia atual |
|---|---|---|
| Refinar só até decidir | `RefinementSet`, `DecisionCertificate` | resultado só contrai; ação para quando limites não se sobrepõem |
| Calcular só `needed ∩ changed` | `ExecutionSlice` | cone backward do objetivo intersectado com delta forward |
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
cargo run --release --bin axon-uic-bench -- --mib 64 --queries 20 --runs 5
```

Resultados e protocolo: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md), [docs/RESULTS.md](docs/RESULTS.md). Contratos e limites: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
