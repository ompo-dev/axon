# DeltaForge-SUM

`DeltaForge` recebe `FoldSpec::AddModU64`, não callback de updater. Ele reconhece grupo comutativo modular e produz:

| Artefato derivado | Valor |
|---|---|
| álgebra | `CommutativeGroup` |
| cache | `ModularTotal` |
| regra | `SubtractOldThenAddNew` |

`DerivedSumPlan::check` valida, para vetor e `ReplaceDelta` concretos, que aplicar delta de saída ao total antigo produz mesmo valor que fold completo depois da mudança. `MIN` é recusado: não há inversa nem estado auxiliar derivado.

O escopo é uma gramática declarada de um operador. Não é descoberta cega de programas, theorem prover, nem aprendizado. O checker é executável, não prova formal.

## Resultado físico com BenchContract

Em 64 MiB e 15 rodadas por ponto, todas as 45 comparações Full/Raw/Forge tiveram paridade exata. O protocolo `2` separa `HOT` de `LIFECYCLE`; por isso estes valores não devem ser comparados em valor absoluto ao protocolo anterior.

| Escritas finais | Raw HOT p50 ms | Forge HOT p50 ms | Raw LIFECYCLE p50 ms | Forge LIFECYCLE p50 ms | Evidência pareada HOT |
|---:|---:|---:|---:|---:|---|
| 1.024 | 12.554 | 11.430 | 38.454 | 63.165 | inconclusiva |
| 1.000.000 | 17.426 | 17.933 | 50.878 | 82.546 | inconclusiva |
| 4.000.000 | 39.147 | 37.749 | 103.818 | 157.986 | inconclusiva |

O Forge tem p50 HOT menor em dois pontos, mas as 15 rodadas pareadas se cruzam em todos eles. Logo `StrategyEvidence` não promove `ForgedDelta`; não existe Meta-JIT nem alegação de ganho físico. Em lifecycle ele perde claramente, pois o checker concreto custa no maior ponto p50 `56.495 ms`; ele é gate de shadow/promoção, nunca verificação por update no hot path.

## DeltaForge-AVG

`FoldSpec::AverageExactU64` deriva `DerivedAveragePlan`. O cache é criado pelo artifact como `(sum: u128, count: usize)`; o chamador não pode fornecer um `sum` ou `count` arbitrário. A saída é `ExactAverage`, uma fração sem arredondamento. A regra é `SubtractOldThenAddNewPreserveCount`.

O contrato numérico é `ExactU128`: `AVG` não usa `SUM mod 2^64`. O teste inclui `[u64::MAX, u64::MAX]` e preserva o numerador largo, portanto overflow modular não altera silenciosamente a média. Se o acumulador `u128` não comportasse a entrada, o plano retorna erro de overflow; não degrada a semântica.

`SemanticArtifact` carrega capability, certificado, guards, versões do kernel/semântica e um selo de conteúdo estável. Na criação, o kernel de regras confere a composição declarada; no reload, confere versão, certificado, guards e selo em tempo constante. Este selo detecta corrupção acidental e drift de versão, mas não é assinatura criptográfica. `PhysicalRealization` é separado e hoje é apenas `Interpreter` com evidência `Unmeasured`.

`DerivedAveragePlan::check` permanece como auditoria concreta de benchmark/teste. Ele não roda no reuso de produção: cada execução aplica apenas `RuntimeGuards` de custo constante, como input não vazio e semântica `ReplaceFinalState`.

`MIN` continua recusado. Não é correto reutilizar a regra de AVG/SUM para ele: o mínimo exige estado auxiliar diferente (por exemplo, contagem/estrutura dos candidatos), que ainda não foi derivado nem certificado.
