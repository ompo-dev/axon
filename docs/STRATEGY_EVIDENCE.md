# Strategy Evidence

`StrategyEvidence` torna resultado negativo reutilizável sem generalizá-lo além do domínio medido.

Cada registro contém assinatura de workload, baseline, execução-oráculo do candidato, intervalos fechados de latência e `OracleHeadroom` em pontos-base. Para medições reais, `from_paired_samples` só cria reflexo se todas as rodadas pareadas concordam; intervalos isolados podem permanecer inconclusivos. A decisão por intervalos segue:

| Condição | Estado | Meta-JIT |
|---|---|---|
| `candidate.upper < baseline.lower` | `Supported` | guarda candidato |
| `candidate.lower > baseline.upper` | `LatencyDominated` | guarda baseline |
| intervalos sobrepostos | `Inconclusive` | não cria reflexo |

No último caso não há promoção nem refutação. É necessário medir novamente em shadow.

Para o Hybrid/Change Fabric anterior, a conclusão continua limitada a `SUM mod u64`, 64 MiB, stream canônico por shard e `FinalStateOnly`. `WorkloadSignature` também guarda shard count, layout do stream, razão query/ingestão, versão do protocolo, métrica e identificador de hardware. O identificador deve descrever CPU e hierarquia de memória medida. Qualquer diferença reabre a decisão.

`MetaJit` não executa estratégia nem substitui benchmark. Ele só guarda uma decisão já suportada por evidência e retorna `None` quando a assinatura não é exatamente a mesma.
