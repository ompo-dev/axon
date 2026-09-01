# Delta Algebra

`IncrementalizabilityAnalyzer` é contrato explícito, não detector mágico de programas.

| Operador | Classe | Delta exato | Coalescência | Decisão segura |
|---|---|---:|---:|---|
| `SUM` | `Constant` | sim | sim | compara custo Delta contra Full |
| `COUNT` | `Constant` | sim | sim | compara custo Delta contra Full |
| `XOR` | `Constant` | sim | sim | compara custo Delta contra Full |
| `SORT` | `Global` | não declarado | não | `Full` obrigatório |

`CostEstimate` calcula `fixed + per_change × support + validation`. Delta só é escolhido se for estritamente menor que Full. Empate escolhe Full.

## Change Structures

`ChangeStructure` separa valor, delta, `diff` e `apply`. A primeira realização é deliberadamente pequena:

| Estrutura | Valor | Delta | Lei executável |
|---|---|---|---|
| `ModularU64` | `u64` | `u64` | `apply(x, diff(y, x)) == y` com aritmética modular |
| `VectorU64` | `Vec<u64>` | `ReplaceDelta` canônico | cada replacement carrega `index`, `old`, `new` |

`ReplaceDelta` canônico só garante índices estritamente crescentes; ele não conhece tamanho do vetor. `VectorU64::apply` valida shape e valores antigos antes de criar novo vetor. Stream incompatível retorna `ChangeError`; valor anterior permanece intacto. `SumFold` é primeiro `IncrementalOp`: cache é total modular e seu delta é `Σ(new - old)`.

Isso ainda não infere operadores arbitrários, nem é prova formal. As leis são verificadas por testes determinísticos de overflow, paridade e aplicação transacional.

`SumState` é imutável: `apply_delta`, `apply_coalesced` e `full_after` retornam novo estado. `apply_coalesced` exige `ObservationFrontier::FinalStateOnly`. Testes exigem igualdade exata entre os três caminhos e rejeitam suporte, índice ou fronteira inválida.

`coalesce_adjacent_last_writes` remove somente escritas consecutivas para mesma chave. Isso preserva estado final; não é uma permissão para reordenar eventos com efeitos intermediários observáveis. `coalesce_adjacent_at_frontier(..., ObservationFrontier::FinalStateOnly)` torna essa condição explícita e recusa `IntermediateObserved`.

Limite deliberado: não há compilador de `F + ΔF`, prova automática de incrementalizabilidade, inferência causal ou benchmark de inteligência geral. Novo operador só entra depois de contrato, teste de paridade e sweep físico.
