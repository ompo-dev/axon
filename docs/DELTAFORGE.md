# DeltaForge-SUM

`DeltaForge` recebe `FoldSpec::AddModU64`, não callback de updater. Ele reconhece grupo comutativo modular e produz:

| Artefato derivado | Valor |
|---|---|
| álgebra | `CommutativeGroup` |
| cache | `ModularTotal` |
| regra | `SubtractOldThenAddNew` |

`DerivedSumPlan::check` valida, para vetor e `ReplaceDelta` concretos, que aplicar delta de saída ao total antigo produz mesmo valor que fold completo depois da mudança. `MIN` é recusado: não há inversa nem estado auxiliar derivado.

O escopo é uma gramática declarada de um operador. Não é descoberta cega de programas, theorem prover, nem aprendizado. O checker é executável, não prova formal.

## Resultado físico inicial

Em 64 MiB e 15 rodadas por ponto, todas as 45 comparações Full/Raw/Forge tiveram paridade exata. Cada janela temporizada inclui `VectorU64::apply` imutável, portanto compara custo total de materializar mudança mais fold, não atualização isolada de cache. Forge ficou `~2%` atrás do updater manual em todos os p50; portanto a derivação está semanticamente correta, mas não merece promoção como otimização física nesta realização. Em 4.000.000 writes, Full venceu os dois deltas porque materializar `ReplaceDelta` e atualizar cada entrada passou a custar mais que recomputar a soma.

O checker completo custa muito mais que execução e fica fora do timer. Ele é gate de shadow/promoção, nunca verificação por update em hot path.
