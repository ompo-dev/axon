# Delta Algebra

`IncrementalizabilityAnalyzer` é contrato explícito, não detector mágico de programas.

| Operador | Classe | Delta exato | Coalescência | Decisão segura |
|---|---|---:|---:|---|
| `SUM` | `Constant` | sim | sim | compara custo Delta contra Full |
| `COUNT` | `Constant` | sim | sim | compara custo Delta contra Full |
| `XOR` | `Constant` | sim | sim | compara custo Delta contra Full |
| `SORT` | `Global` | não declarado | não | `Full` obrigatório |

`CostEstimate` calcula `fixed + per_change × support + validation`. Delta só é escolhido se for estritamente menor que Full. Empate escolhe Full.

`SumState` é imutável: `apply_delta`, `apply_coalesced` e `full_after` retornam novo estado. `apply_coalesced` exige `ObservationFrontier::FinalStateOnly`. Testes exigem igualdade exata entre os três caminhos e rejeitam suporte, índice ou fronteira inválida.

`coalesce_adjacent_last_writes` remove somente escritas consecutivas para mesma chave. Isso preserva estado final; não é uma permissão para reordenar eventos com efeitos intermediários observáveis. `coalesce_adjacent_at_frontier(..., ObservationFrontier::FinalStateOnly)` torna essa condição explícita e recusa `IntermediateObserved`.

Limite deliberado: não há compilador de `F + ΔF`, prova automática de incrementalizabilidade, inferência causal ou benchmark de inteligência geral. Novo operador só entra depois de contrato, teste de paridade e sweep físico.
