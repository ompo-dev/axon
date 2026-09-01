# AXON-UIC results

Run date: 2026-09-01. Host: Windows x86_64, Intel Core i7-13650HX (14 cores / 20 logical processors), 15.73 GiB RAM.

Command:

```powershell
cargo run --release --bin axon-uic-bench -- --mib 64 --queries 20 --runs 5
```

| Metric | Result |
|---|---:|
| Physical vector per engine | 64 MiB |
| Full batch p50 / p95 | 57.115 / 58.701 ms |
| Delta batch p50 / p95 | 0.000030 / 0.000031 ms |
| Full per query p50, derived from batch | 2,855,755.000 ns |
| Delta per query p50, derived from batch | 1.500 ns |
| Observed batch speedup p50 | 1,903,836.67× |
| Observed per-query speedup p50 | 1,903,836.67× |
| Logical reads per run, full / delta | 1,342,177,280 / 320 bytes |
| First-batch exact trace parity | true |
| Exact final accumulator validation after all 10,000 batches | true |
| Final checksum after all delta batches | `2E5C00BF16482DC0` |

Interpretação: Full e Delta acima são ambos um batch de 20 updates; a normalização Delta divide somente pelos 10.000 batches de calibração. Valores por query são derivados dos mesmos batches, por isso a razão é idêntica. Este PC confirmou que, neste workload exato de soma sob update pontual, recomputar estado inteiro é muito mais caro que aplicar Delta local. Razão é específica deste benchmark, não é medida de inteligência geral, energia, GPU ou AGI.

## Delta Algebra response curve

Command:

```powershell
cargo run --release --bin axon-uic-delta-sweep -- --mib 64 --runs 5 --max-updates 8000000
```

Mesmo host, vetor físico de 64 MiB, cinco rodadas por ponto. Cada chave recebe quatro escritas adjacentes. Todos os pontos passaram paridade exata; checksum final do maior ponto: `3A8AA3E313AB7EAC`.

| Eventos | Escritas finais | Suporte | Melhor Full p50 ms | Melhor Delta p50 ms | Escolha |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.000012% | 3.510 | 0.000001 | Delta |
| 16 | 4 | 0.000048% | 3.656 | 0.000007 | Delta |
| 256 | 64 | 0.000763% | 3.341 | 0.000121 | Delta |
| 4,096 | 1,024 | 0.012207% | 3.327 | 0.002304 | Delta |
| 65,536 | 16,384 | 0.195312% | 3.635 | 0.115863 | Delta |
| 262,144 | 65,536 | 0.781250% | 4.101 | 0.763575 | Delta |
| 1,000,000 | 250,000 | 2.980232% | 5.565 | 3.360300 | Delta |
| 2,000,000 | 500,000 | 5.960464% | 7.625 | 6.733400 | Delta |
| 4,000,000 | 1,000,000 | 11.920929% | 11.597 | 13.814800 | Full |
| 8,000,000 | 2,000,000 | 23.841858% | 19.755 | 28.117900 | Full |

Resultado: neste workload, crossover observado fica entre 5,96% e 11,92% de suporte final. A seleção não assume “Delta sempre vence”: escolheu `Full` nos dois maiores pontos.

Coalescência não é vitória automática. Em 65.536 eventos, Delta sem coalescer foi 0.115863 ms e Delta+coalesce 0.150687 ms; em 1.000.000, foi 4.141100 ms e 3.360300 ms. Custo de construir fluxo coalescido importa.

No maior ponto, redução Full lê logicamente 67.108.864 bytes; Delta coalescido lê 32.000.000 bytes para `old`/`new`: eliminação estrutural de 52.31628418%. Isso não inclui leitura do stream, alocação, tráfego de DRAM nem energia.

Limite: prova apenas `SUM` modular de `u64` sob escrita pontual local e última escrita adjacente. Não prova compilação geral de deltas, causalidade, representação nova, capacidade de “jump” ou AGI.

## Hybrid Recompute por shard

Command:

```powershell
cargo run --release --bin axon-uic-hybrid-sweep -- --mib 64 --runs 30 --hardware-id i7-13650HX-16GiB
```

Workload: 64 shards de 1 MiB; oito shards densos usam `FULL_LOCAL`, um shard usa Delta coalescido, um usa Delta bruto e 54 usam `SKIP`. Todas as 30 rodadas tiveram paridade exata e checksum `8D35C7539911D6CB`.

| Caminho | p50 ms |
|---|---:|
| Full global | 4.240 |
| Delta bruto global | 1.331 |
| Delta coalescido global | 4.772 |
| Hybrid fim a fim | 3.875 |
| Hybrid Oracle, executor pré-compilado | 1.393 |
| Compiler Hybrid | 2.642 |
| `validate + index` | 1.618 |
| `classify + materialize` | 1.031 |
| Change Fabric, ingestão + query | 9.858 |
| Change Fabric, ingestão | 8.284 |
| Change Fabric, query | 1.560 |

Oracle perde para Delta bruto no p50 (`0.96×`), mas os 30 pares se cruzam. `StrategyEvidence` classificou Hybrid como `Inconclusive`, com headroom de `-471 bp`; não há Meta-JIT nem refutação formal neste domínio. Hybrid fim a fim perde para Delta porque `Adaptation Tax` é `2.14×` e `Oracle Gap` é `2.78×`.

Change Fabric continua dominado neste regime: custo de ingestão gera `Adaptation Tax` de `5.31×`, e lifecycle ficou `0.13×` do Delta bruto. Esse resultado é útil: mover planejamento para ingestão não cria ganho se a manutenção custa mais que a compilação que elimina. Verificação exata foi medida fora dos timers: Hybrid `3.085 ms`, Oracle `3.127 ms`, Fabric `3.179 ms`.

## DeltaForge-SUM

Command:

```powershell
cargo run --release --bin axon-uic-deltaforge-sum -- --mib 64 --runs 15 --hardware-id i7-13650HX-16GiB
```

Mesmo host, vetor de 64 MiB. `DeltaForge` recebeu apenas `FoldSpec::AddModU64`, derivou `CommutativeGroup`, `ModularTotal` e `SubtractOldThenAddNew`; referência medida em `0.002100 ms`. As 45 execuções tiveram paridade exata; maior checksum: `6C45776C2F16B041`.

| Escritas finais | Full HOT / LIFECYCLE p50 ms | Raw HOT / LIFECYCLE p50 ms | Forge HOT / LIFECYCLE p50 ms | Raw×Forge HOT |
|---:|---:|---:|---:|---|
| 1,024 | 17.537 / 34.085 | 12.554 / 38.454 | 11.430 / 63.165 | inconclusiva, +895 bp |
| 1,000,000 | 20.812 / 45.213 | 17.426 / 50.878 | 17.933 / 82.546 | inconclusiva, -290 bp |
| 4,000,000 | 34.499 / 90.093 | 39.147 / 103.818 | 37.749 / 157.986 | inconclusiva, +357 bp |

`HOT` mede somente execução. `LIFECYCLE` soma as fases registradas da mesma rodada: geração do `ReplaceDelta`, reserva, inicialização, síntese, checker, execução, validação e teardown. `ingestion` e `planning` são zero neste batch e aparecem explicitamente. Em 4.000.000 escritas, o checker do Forge foi `56.495 ms` p50, explicando o lifecycle maior. Embora o p50 HOT do Forge seja menor em dois pontos, os pares se cruzam; não há promoção de estratégia. O resultado demonstra derivação restrita e correção sob contrato, não descoberta geral, aprendizado ou prova formal.
