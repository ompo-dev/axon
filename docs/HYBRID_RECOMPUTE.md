# Hybrid Recompute

`axon-uic-hybrid-sweep` testa uma mudança de decisão global para decisão local por shard.

| Estratégia local | Condição do experimento |
|---|---|
| `SKIP` | shard sem eventos |
| `RAW_DELTA` | eventos esparsos sem duplicata adjacente |
| `COALESCED_DELTA` | duplicatas adjacentes e suporte local abaixo de 50% |
| `FULL_LOCAL` | suporte final local de pelo menos 50% |

Entrada é `ChangeSet` indexado por shard: eventos precisam chegar ordenados por shard. Compilador valida índices, identifica runs, coalesce somente última escrita adjacente e classifica cada shard. Ele referencia slices para shards sem duplicatas; só aloca versão coalescida onde há ganho possível.

Correção: `hybrid_total = Σ shard_total`, com soma modular `u64`. Cada round exige igualdade contra Full global, Delta bruto e Delta coalescido; checksum final do vetor verifica o acumulador híbrido.

`Hybrid Oracle` pré-compila o mesmo plano antes do timer e mede somente o executor. É um limite diagnóstico, não resultado fim a fim: o custo de sua pré-compilação é publicado separadamente. `ChangeFabric` tenta manter esse plano durante ingestão; mede sempre `ingest + query`, nunca apenas query.

Em 64 MiB / 30 rodadas, Oracle perdeu para Delta bruto no p50 (`1.593` contra `1.495 ms`), logo política local ainda não é uma vitória robusta. Hybrid fim a fim ficou em `4.308 ms`: `Adaptation Tax = 2.08×` e `Oracle Gap = 2.70×`. Change Fabric ficou em `9.284 ms`; sua ingestão de `7.594 ms` produz `Adaptation Tax = 4.51×`. Ela foi rejeitada para este workload, embora preserve paridade.

Limite: classificador usa regra morfológica determinística, não curva online aprendida; `ChangeSet` deve vir ordenado por shard; coalescência exige `FinalStateOnly`; não há árvore hierárquica, `DeltaForge`, síntese de estado auxiliar nem prova para outros operadores.
