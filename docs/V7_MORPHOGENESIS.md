# AXON V7 — Morphogenic Cognitive Fabric

## Corte implementado

V7 começa com um contrato falsificável, não com uma alegação de AGI: o mesmo
`MorphogenicCompiler` recebe um seed cognitivo, um orçamento de recursos e um
perfil de workload; devolve um `CognitiveBodyPlan` com regiões ativas e
arquivadas.

Este primeiro corte declara apenas `memory_bytes`: latência, compute e largura
de banda ainda não são aceitos como se já orientassem a alocação. Eles entram
somente quando houver medição física e uma política correspondente.

```text
seed cognitivo
    ↓
resource budget + workload
    ↓
marginal-utility allocation / shadow price
    ↓
cognitive body plan
    ↓
metabolism: remorph somente se benefício futuro > migração
ou se o plano atual deixar de caber no orçamento
```

As regiões atuais são kernel, working state, semantic codes, retrieval index,
program cache, memória episódica e candidate worlds. Cada uma possui mínimo,
capacidade desejada, curva de utilidade saturante e tier (`Critical`,
`Compressed`, `Balanced`, `Expanded`). A memória que não cabe fica declarada
como arquivada: não existe afirmação de que informação incompressível seja
armazenada magicamente na RAM ativa.

`CognitiveMetabolism` é imutável: ele devolve um novo estado quando a utilidade
esperada do novo workload ao longo dos ciclos futuros é maior que o custo
explícito de migrar os bytes ativos, ou obrigatoriamente quando o plano atual
não cabe no novo orçamento. O modelo de preço ainda é lógico; ele não infere
watts ou joules.

## O que este corte prova e o que não prova

| Hipótese | Resultado atual |
|---|---|
| Uma seed pode gerar planos diferentes sob 64 MiB e 16 GiB | Sim, deterministicamente. |
| Qualidade do modelo cresce sem ultrapassar o orçamento | Sim, no mundo sintético V7. |
| Conhecimento grande pode ficar fora da mente ativa | Representado como estado arquivado; não é compressão impossível. |
| Remorph tem custo e pode ser recusado | Sim, por teste de benefício futuro versus migração; redução de orçamento força um plano que caiba. |
| Corpo, linguagem e ferramentas compilam sozinhos | Não implementado; dependem de sensores, dados e contratos externos. |
| A qualidade `Q(M)` mede inteligência geral | Não. É a utilidade da política sintética declarada neste experimento. |

## Como reproduzir

```powershell
cargo test --lib core_v7
cargo run --bin axon_v7_lab
cargo test --bin axon_v7_morphogenic_sweep
cargo run --release --bin axon_v7_morphogenic_sweep -- --runs 3 --touch-cap-mib 64
```

O sweep avalia logicamente os orçamentos de 64 MiB a 16 GiB, mas materializa e
varre **64 MiB reais** por ponto, divididos proporcionalmente entre as regiões
selecionadas pelo plano. Isso evita ocupar 16 GiB num PC de 16 GB e mantém
honesta a separação entre planejamento lógico e teste físico. O parâmetro
`--touch-cap-mib` é limitado a um quarto da RAM detectada.

## Resultado real — 30-08-2026

Host: Intel Core i7-13650HX, 15,73 GB RAM detectada, 20 threads lógicas,
Windows x86_64, Rust 1.94.1, `--release`. Três rodadas, dois aquecimentos e
sete amostras por ponto.

| Orçamento lógico | Q(M) sintético | RAM ativa no plano | Arquivada | Regiões dominantes | Compile p50 | Materializar/varrer 64 MiB p50 |
|---:|---:|---:|---:|---|---:|---:|
| 64 MiB | 0,091 | 64 MiB | 15.904 MiB | working 16 MiB; semantic 8 MiB | 300 ns | 15,034 ms |
| 128 MiB | 0,182 | 128 MiB | 15.840 MiB | working 16 MiB; semantic 8 MiB | 9,0 µs | 15,014 ms |
| 256 MiB | 0,239 | 256 MiB | 15.712 MiB | working 118 MiB; retrieval 22 MiB | 24,9 µs | 13,894 ms |
| 512 MiB | 0,310 | 512 MiB | 15.456 MiB | working 183 MiB; semantic/retrieval 87 MiB | 58,8 µs | 14,368 ms |
| 1 GiB | 0,415 | 1 GiB | 14.944 MiB | working 296 MiB; semantic 226 MiB | 125,7 µs | 14,850 ms |
| 4 GiB | 0,738 | 4 GiB | 11.872 MiB | semantic 1.029 MiB; programs 893 MiB | 476,8 µs | 14,816 ms |
| 16 GiB | 1,000 | 15.968 MiB | 0 MiB | episodic 8.192 MiB; semantic/programs 2.048 MiB | 935,2 µs | 15,666 ms |

Os checksums de cada ponto foram impressos pelo executável, e os testes cobrem
determinismo, orçamento mínimo, monotonicidade de `Q(M)`, degradação com
archive, expansão sob memória abundante, metabolismo e particionamento da RAM
real tocada.

## Leitura correta

O dado físico novo não demonstra que mais RAM produz AGI. Demonstra que o
planejador consegue gerar e executar, neste hardware, corpos cognitivos de
tamanhos diferentes sem exceder o orçamento lógico e sem materializar o estado
inteiro de 16 GiB. A curva `Q(M)` é uma hipótese agora mensurável: a próxima
rodada deve substituir sua utilidade declarada por tarefas de recuperação,
programas e controle com métricas observadas.

## Próximas falsificações

1. Trocar `Q(M)` por accuracy/latência em conjuntos cegos de retrieval,
   planejamento e programas compilados.
2. Medir o custo de remorph por região e validar a previsão de migração contra
   bytes e tempo observados.
3. Integrar o plano ao `CognitiveRuntime`, preservando kernel, ledger e
   verificador como regiões críticas.
4. Só então adicionar `Embodiment Compiler`, `Language Compiler` e
   `Interface Compiler` atrás de contratos e benchmarks próprios.
