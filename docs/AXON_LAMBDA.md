# AXON-Λ — Kernel matemático executável (corte inicial)

## Escopo

AXON-Λ define uma fronteira semântica abaixo das versões Rust da AXON. Rust
continua sendo uma realização; este corte não formaliza Lean, não implementa uma
Factor Fabric geral, não descobre simetrias automaticamente e não integra o
runtime `.axon` estável.

O domínio executável é deliberadamente estreito: Factors afins em cadeias
lineares, sobre aritmética modular `u64`. Essa restrição permite testar as
invariantes sem afirmar equivalência de programas arbitrários.

```text
ContractedMorphism  -> substituição segura de implementação
DEMAND × DELTA      -> cone cognitivo ativo
Pareto cost algebra -> seleção por política explícita
LIFT / UNLIFT       -> quotient exato de valores exchangeable
```

## Contrato e refinamento

`ContractedMorphism` contém identificador semântico, assinatura de entrada e
saída, digest da regra, revisão, precondições, garantias, erro máximo e força de
verificação. Uma realização só pode substituir outra quando:

```text
semantic_id, assinatura e digest da regra são iguais
precondições novas ⊆ precondições exigidas
garantias novas ⊇ garantias exigidas
erro novo ≤ erro exigido
verificação nova ≥ verificação exigida
```

Assim, revisão crescente sozinha nunca autoriza uma troca. `REALIZE` filtra
primeiro por refinamento e depois pelo `DecisionCertificate`; uma variante
aproximada pode satisfazer um contrato amplo, mas é recusada se seu envelope de
erro puder inverter a decisão.

## Demand × Delta

Para uma decisão `g` e uma evidência alterada `Δ`:

```text
B_g = cone de dependências pedido por DEMAND
F_Δ = cone de consequências da alteração
A   = B_g ∩ F_Δ
```

O baseline `full_query` materializa os valores de todos os Factors. O caminho
incremental aplica a derivada exata da regra afim, `DF(x, Δx) = 3Δx`, somente no
cone ativo. A escolha é adaptativa: se `A` é global, o custo de latência
declarado seleciona `FullRecompute`; incrementalidade não é tratada como regra
universal.

Para impedir uma aceleração que corrompa o estado, `delta_overlay_matches_full`
compara todos os valores de `base + overlay` contra a recomputação total fora da
janela de tempo.

## Custo e quotient

`CostVector` preserva cinco dimensões inteiras: energia abstrata, bytes movidos,
latência, memória e risco. Os custos deste corte são **declarados**, não watts,
joules nem telemetria. `ParetoFrontier` remove somente opções dominadas; os
pesos do hardware ou do objetivo escolhem no fim.

`LiftedPopulation` agrupa valores iguais em `LiftedClass { representative,
multiplicity }`. `lifted_sum` é exatamente igual ao somatório individual e
`unlift_value` altera apenas a contribuição solicitada. O índice é construído a
partir de equivalência explícita de valores; isso ainda não é detecção automática
de simetria estrutural.

## Conformance Rust ↔ Python

O micro-journal canônico é implementado por Rust e Python sem dependências:

```powershell
cargo test --lib core_lambda::tests::python_and_rust_implementations_emit_the_same_canonical_journal
cargo run --bin axon_lambda_conformance
python tools/axon_lambda_conformance.py
```

O teste compara o journal normalizando apenas a quebra de linha do Windows. Ele
testa esse micro-domínio, não prova conformidade universal de implementações
futuras.

## Medição física real — 31-08-2026

Host: Intel Core i7-13650HX, 15,73 GB de RAM detectada, 20 threads lógicas,
Windows x86_64, Rust 1.94.1. Executado em `--release`, com três rodadas, dois
aquecimentos, quinze amostras por rodada e ordem full/delta alternada.

```powershell
cargo test --lib core_lambda
cargo test --bin axon_lambda_physical_sweep
cargo run --bin axon_lambda_lab
cargo run --release --bin axon_lambda_physical_sweep -- --runs 3 --factors 1000000 --chain-len 1000
```

Foram alocados 1.000.000 `FactorNode`s reais por Fabric. O cenário local usa
1.000 cadeias de 1.000 Factors; o adversarial usa uma única cadeia de um milhão.
Os tempos por query são derivados de lotes para reduzir ruído: full executa três
queries por amostra; delta e LIFT executam mais repetições e são normalizados por
query. As verificações de estado e checksums de paridade ficam fora do tempo.

| Medida | Resultado |
|---|---:|
| Construir Fabric local / global | 11,317 ms / 11,537 ms p50 |
| Local: `B` / `F` / `A` | 1.000 / 500 / 500 |
| Local full / delta | 4,018 ms / 3,416 µs p50 |
| Razão observada local | 1.176,37× |
| Paridade de decisão local | 8/8 mudanças; checksum idêntico `B9F6F5F0B722713C` |
| Estado local `base + overlay` | igual ao full |
| Adversarial: `A` | 1.000.000 |
| Adversarial adaptativo | `FullRecompute`, 22,023 ms p50; 2/2 mudanças com checksum idêntico |
| LIFT: 1.000.000 valores / 256 classes | índice 7,556 ms; direto 126,925 µs; lifted 54 ns p50 |
| LIFT: razão / break-even | 2.350,46× por consulta depois do índice; ~60 consultas |

Leitura correta: a interseção pequena eliminou trabalho em uma topologia
controlada; a cascata global mostrou explicitamente o caso em que o runtime não
deve escolher delta. O resultado não transfere automaticamente para grafos
arbitrários, linguagem natural, custo energético, hardware especializado ou AGI.

## Próximas falsificações

1. Trocar cadeias por DAGs materializados com múltiplas dependências e marcação
   por época, preservando paridade de todos os nós.
2. Calibrar perfis de custo por host e backend antes de usar tempo medido no
   seletor `REALIZE`.
3. Generalizar `LIFT` de valor idêntico para equivalência declarada e, depois,
   tentar falsificar detectores automáticos de simetria em dados cegos.
4. Ampliar a conformance para journal, revision DAG e rollback antes de começar
   um specification kernel em Lean.
