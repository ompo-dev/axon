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
