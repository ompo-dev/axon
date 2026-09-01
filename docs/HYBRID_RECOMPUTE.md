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

Resultado atual: Hybrid supera Full global, mas perde para Delta bruto porque Delta já é barato para as escritas contíguas deste workload e compilação custa 2.028 ms p50. Isso é o resultado esperado de uma arquitetura que mede rent: não promover Hybrid sem vantagem medida.

Limite: classificador usa regra morfológica determinística, não curva online aprendida; `ChangeSet` deve vir ordenado por shard; não há `DeltaForge`, síntese de estado auxiliar nem prova para outros operadores.
