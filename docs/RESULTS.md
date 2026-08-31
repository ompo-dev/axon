# AXON-UIC results

Run date: 2026-08-31. Host: Windows x86_64, Intel Core i7-13650HX (14 cores / 20 logical processors), 15.73 GiB RAM.

Command:

```powershell
cargo run --release --bin axon-uic-bench -- --mib 64 --queries 20 --runs 5
```

| Metric | Result |
|---|---:|
| Physical vector per engine | 64 MiB |
| Full p50 / p95 | 65.738 / 70.487 ms |
| Delta p50 / p95, normalized | 0.000027 / 0.000029 ms |
| Observed p50 ratio, normalized | 2,434,748.15× |
| Logical reads per run, full / delta | 1,342,177,280 / 320 bytes |
| First-batch exact trace parity | true |
| Exact final accumulator validation after all 10,000 batches | true |
| Final checksum after all delta batches | `2E5C00BF16482DC0` |

Interpretação: este PC confirmou que, neste workload exato de soma sob update pontual, recomputar estado inteiro é muito mais caro que aplicar delta local. Razão é específica deste benchmark, não é medida de inteligência geral, energia, GPU ou AGI.

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
