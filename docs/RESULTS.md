# AXON-UIC results

Run date: 2026-09-01. Host: Windows x86_64, Intel Core i7-13650HX (14 cores / 20 logical processors), 15.73 GiB RAM.

Command:

```powershell
cargo run --release --bin axon-uic-bench -- --mib 64 --queries 20 --runs 5
```

| Metric | Result |
|---|---:|
| Physical vector per engine | 64 MiB |
| Full batch p50 / p95 | 57.115 / 58.701 ms |
| Delta batch p50 / p95 | 0.000030 / 0.000031 ms |
| Full per query p50, derived from batch | 2,855,755.000 ns |
| Delta per query p50, derived from batch | 1.500 ns |
| Observed batch speedup p50 | 1,903,836.67× |
| Observed per-query speedup p50 | 1,903,836.67× |
| Logical reads per run, full / delta | 1,342,177,280 / 320 bytes |
| First-batch exact trace parity | true |
| Exact final accumulator validation after all 10,000 batches | true |
| Final checksum after all delta batches | `2E5C00BF16482DC0` |

Interpretação: Full e Delta acima são ambos um batch de 20 updates; a normalização Delta divide somente pelos 10.000 batches de calibração. Valores por query são derivados dos mesmos batches, por isso a razão é idêntica. Este PC confirmou que, neste workload exato de soma sob update pontual, recomputar estado inteiro é muito mais caro que aplicar Delta local. Razão é específica deste benchmark, não é medida de inteligência geral, energia, GPU ou AGI.

## Delta Algebra response curve

Command:

```powershell
cargo run --release --bin axon-uic-delta-sweep -- --mib 64 --runs 5 --max-updates 8000000
```

Mesmo host, vetor físico de 64 MiB, cinco rodadas por ponto. Cada chave recebe quatro escritas adjacentes. Todos os pontos passaram paridade exata; checksum final do maior ponto: `3A8AA3E313AB7EAC`.

| Eventos | Escritas finais | Suporte | Melhor Full p50 ms | Melhor Delta p50 ms | Escolha |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.000012% | 3.510 | 0.000001 | Delta |
| 16 | 4 | 0.000048% | 3.656 | 0.000007 | Delta |
| 256 | 64 | 0.000763% | 3.341 | 0.000121 | Delta |
| 4,096 | 1,024 | 0.012207% | 3.327 | 0.002304 | Delta |
| 65,536 | 16,384 | 0.195312% | 3.635 | 0.115863 | Delta |
| 262,144 | 65,536 | 0.781250% | 4.101 | 0.763575 | Delta |
| 1,000,000 | 250,000 | 2.980232% | 5.565 | 3.360300 | Delta |
| 2,000,000 | 500,000 | 5.960464% | 7.625 | 6.733400 | Delta |
| 4,000,000 | 1,000,000 | 11.920929% | 11.597 | 13.814800 | Full |
| 8,000,000 | 2,000,000 | 23.841858% | 19.755 | 28.117900 | Full |

Resultado: neste workload, crossover observado fica entre 5,96% e 11,92% de suporte final. A seleção não assume “Delta sempre vence”: escolheu `Full` nos dois maiores pontos.

Coalescência não é vitória automática. Em 65.536 eventos, Delta sem coalescer foi 0.115863 ms e Delta+coalesce 0.150687 ms; em 1.000.000, foi 4.141100 ms e 3.360300 ms. Custo de construir fluxo coalescido importa.

No maior ponto, redução Full lê logicamente 67.108.864 bytes; Delta coalescido lê 32.000.000 bytes para `old`/`new`: eliminação estrutural de 52.31628418%. Isso não inclui leitura do stream, alocação, tráfego de DRAM nem energia.

Limite: prova apenas `SUM` modular de `u64` sob escrita pontual local e última escrita adjacente. Não prova compilação geral de deltas, causalidade, representação nova, capacidade de “jump” ou AGI.

## Hybrid Recompute por shard

Command:

```powershell
cargo run --release --bin axon-uic-hybrid-sweep -- --mib 64 --runs 30 --hardware-id i7-13650HX-16GiB
```

Workload: 64 shards de 1 MiB; oito shards densos usam `FULL_LOCAL`, um shard usa Delta coalescido, um usa Delta bruto e 54 usam `SKIP`. Todas as 30 rodadas tiveram paridade exata e checksum `8D35C7539911D6CB`.

| Caminho | p50 ms |
|---|---:|
| Full global | 4.240 |
| Delta bruto global | 1.331 |
| Delta coalescido global | 4.772 |
| Hybrid fim a fim | 3.875 |
| Hybrid Oracle, executor pré-compilado | 1.393 |
| Compiler Hybrid | 2.642 |
| `validate + index` | 1.618 |
| `classify + materialize` | 1.031 |
| Change Fabric, ingestão + query | 9.858 |
| Change Fabric, ingestão | 8.284 |
| Change Fabric, query | 1.560 |

Oracle perde para Delta bruto no p50 (`0.96×`), mas os 30 pares se cruzam. `StrategyEvidence` classificou Hybrid como `Inconclusive`, com headroom de `-471 bp`; não há Meta-JIT nem refutação formal neste domínio. Hybrid fim a fim perde para Delta porque `Adaptation Tax` é `2.14×` e `Oracle Gap` é `2.78×`.

Change Fabric continua dominado neste regime: custo de ingestão gera `Adaptation Tax` de `5.31×`, e lifecycle ficou `0.13×` do Delta bruto. Esse resultado é útil: mover planejamento para ingestão não cria ganho se a manutenção custa mais que a compilação que elimina. Verificação exata foi medida fora dos timers: Hybrid `3.085 ms`, Oracle `3.127 ms`, Fabric `3.179 ms`.

## DeltaForge-SUM

Command:

```powershell
cargo run --release --bin axon-uic-deltaforge-sum -- --mib 64 --runs 15 --hardware-id i7-13650HX-16GiB
```

Mesmo host, vetor de 64 MiB. `DeltaForge` recebeu apenas `FoldSpec::AddModU64`, derivou `CommutativeGroup`, `ModularTotal` e `SubtractOldThenAddNew`; referência medida em `0.002100 ms`. As 45 execuções tiveram paridade exata; maior checksum: `6C45776C2F16B041`.

| Escritas finais | Full HOT / LIFECYCLE p50 ms | Raw HOT / LIFECYCLE p50 ms | Forge HOT / LIFECYCLE p50 ms | Raw×Forge HOT |
|---:|---:|---:|---:|---|
| 1,024 | 17.537 / 34.085 | 12.554 / 38.454 | 11.430 / 63.165 | inconclusiva, +895 bp |
| 1,000,000 | 20.812 / 45.213 | 17.426 / 50.878 | 17.933 / 82.546 | inconclusiva, -290 bp |
| 4,000,000 | 34.499 / 90.093 | 39.147 / 103.818 | 37.749 / 157.986 | inconclusiva, +357 bp |

`HOT` mede somente execução. `LIFECYCLE` soma as fases registradas da mesma rodada: geração do `ReplaceDelta`, reserva, inicialização, síntese, checker, execução, validação e teardown. `ingestion` e `planning` são zero neste batch e aparecem explicitamente. Em 4.000.000 escritas, o checker do Forge foi `56.495 ms` p50, explicando o lifecycle maior. Embora o p50 HOT do Forge seja menor em dois pontos, os pares se cruzam; não há promoção de estratégia. O resultado demonstra derivação restrita e correção sob contrato, não descoberta geral, aprendizado ou prova formal.

## DeltaForge-AVG com artifact persistente

Command executado neste host:

```powershell
cargo run --release --bin axon-uic-deltaforge-avg -- --mib 64 --runs 15
```

### Resultado anterior — checker por dataset

O resultado abaixo foi produzido antes da separação entre `SemanticArtifact` e `PhysicalRealization`. Ele mantém valor histórico, mas não é o protocolo atual: o checker concreto era repetido a cada dataset.

| Escritas finais | Full HOT / LIFECYCLE p50 ms | AVG reutilizado HOT / LIFECYCLE p50 ms | Persistir artifact p50 ms | Break-even medido |
|---:|---:|---:|---:|---|
| 1.024 | 14.962 / 25.588 | 10.551 / 65.434 | 1.490 | não atingido em 15 usos |
| 1.000.000 | 17.898 / 33.823 | 15.743 / 81.401 | 1.542 | não atingido em 15 usos |
| 4.000.000 | 26.827 / 58.035 | 28.096 / 121.153 | 1.469 | não atingido em 15 usos |

No maior ponto, o reload do artifact foi `0.818 ms` p50; a verificação concreta custou `51.989 ms` p50. Esse custo motivou o corte semântico abaixo.

### Resultado atual — certificado semântico selado

Nota posterior (13/09/2026): as 15 rodadas abaixo são ciclos frios independentes. A coluna histórica “não atingido em 15 usos” não demonstra ausência de break-even de uma instância viva. Os resultados de AVG-LIVE ao final deste documento medem essa outra fronteira.

Mesmo comando, mesmo host e 15 rodadas por ponto. Cada rodada cria, sincroniza e recarrega um artifact semântico. O reload verifica somente selo, versões e guards; `DerivedAveragePlan::check` não roda por dataset. As 45 comparações Full/AVG continuaram com paridade exata.

| Escritas finais | Full HOT / LIFECYCLE p50 ms | AVG reutilizado HOT / LIFECYCLE p50 ms | Persistir artifact p50 ms | Break-even medido |
|---:|---:|---:|---:|---|
| 1.024 | 14.601 / 25.079 | 10.097 / 30.269 | 1.454 | não atingido em 15 usos |
| 1.000.000 | 18.220 / 34.144 | 15.416 / 40.569 | 1.505 | não atingido em 15 usos |
| 4.000.000 | 26.766 / 58.405 | 27.982 / 69.066 | 1.612 | não atingido em 15 usos |

No maior ponto, `verification` por reuso foi `0 ms`; `artifact_load` foi `0.922 ms` p50 e validação independente foi `4.622 ms` p50. O lifecycle ainda perde porque este benchmark reconstrói input, cache e validação em toda rodada. Conclusão: **não promover** realização física ainda; porém o erro anterior — pagar checker linear de ~52 ms por dataset — foi removido. O resultado não mede aprendizagem, descoberta ou generalização.

## LearnBench inicial

Command:

```powershell
cargo run --bin axon-uic-learn-bench -- --epochs 8
```

Este é o primeiro loop mensurável de aprendizado no projeto. Ele não treina linguagem, percepção ou autonomia; treina um `StructurePrior` linear e pequeno para ordenar candidatos de derivação em tarefas sintéticas.

| Métrica | Resultado |
|---|---:|
| Tarefas de avaliação | 4 |
| Tentativas antes da experiência | 16 |
| Tentativas depois da experiência | 4 |
| Redução de busca | 75,00% |

Depois do treino, o primeiro candidato escolhido foi o aceito para `SUM`, `AVG`, `VARIANCE` e `MIN` sintéticos. Treino e avaliação usam as mesmas quatro estruturas e respostas predefinidas. A redução mede memorização/ordenação neste currículo, não generalização nem implementação de `VARIANCE` ou `MIN`. A etapa seguinte executada foi AVG-LIVE e aprendizado de custos reais, descritos abaixo.

## Estado contínuo e treino por execução (13/09/2026)

Relatório completo, fontes de pesquisa, protocolo e arquivos: [LIVE_LEARNING_2026-09-13.md](LIVE_LEARNING_2026-09-13.md).

AVG-LIVE completou três rodadas em cinco tamanhos, com 10.000 batches de 1.024 alterações por tamanho: 150.000 batches ao todo, todos com auditoria final exata. A mediana por batch em 64 MiB foi 2,3–2,5 microssegundos; em 1 GiB, 8,6–8,7 microssegundos. O custo físico não ficou constante, embora o caminho incremental não percorra o vetor inteiro.

O experimento durável de 1 MiB aplicou 256 batches sincronizados e passou por 23 reaberturas/replays exatos. A mediana durável foi 0,3426 ms por batch. Testes adicionais mataram processos em três pontos e verificaram recuperação e continuação, além de truncar cada posição possível do último registro.

Treinamos três políticas de custo independentes, cada uma com 1.024 observações de execução e 256 casos de avaliação separados. A política ocupa 520 bytes em memória e 536 bytes no checkpoint. Nas avaliações com alterações espalhadas, a execução da política reduziu tempo em 8,68–11,38% ante sempre Full e 11,42–12,74% ante sempre Delta. Houve 596 escolhas corretas em 602 comparações com diferença de pelo menos 10%; outros 166 casos ficaram fora dessa métrica. Todos os 768 casos mantiveram paridade exata. Nenhuma realização foi promovida automaticamente por esses números.

Treinar custou 1,12–1,29 segundo por rodada. Esse custo não se paga nos 256 casos de avaliação; o arquivo aprendido permite reaproveitamento posterior. Uma nova execução carregou a política e avaliou 256 casos adicionais sem retreinar, preservando correção. Estes resultados demonstram manutenção de estado e seleção adaptativa restrita a AVG, não linguagem, compreensão do mundo ou AGI.

### Continuação: compactação e aprendizado por ação

Adicionamos compactação periódica com snapshot sincronizado, publicação por renomeação e lock lateral estável. Em três pares de 1.024 batches duráveis, o arquivo final caiu de 26.271.832 para 1.048.664 bytes (96,0% menor), e o tempo acumulado de recuperação caiu 65,1–68,9%. Compactar a cada 64 batches custou 56,9–60,8 ms adicionais por rodada. Paridade permaneceu exata, inclusive nos testes de processo interrompido nas fronteiras da compactação.

`execute_and_learn` permite aprender executando somente uma estratégia por evento. Há exploração periódica para observar alternativas; eventos inválidos não treinam. Retomamos uma política de 1.024 observações e chegamos a 3.072 mantendo o checkpoint em 536 bytes. Na avaliação separada de 256 casos, a política executada economizou 10,86% ante sempre Full e 11,75% ante sempre Delta, com resultados exatos. O experimento de continuação custou 4,503 s; não provou vantagem sobre o modelo anterior. Detalhes e arquivos estão no relatório vinculado acima.

## Robustez e comparação mais exigente

Na rodada seguinte do mesmo dia, 118 testes passaram em debug e release. O teste diferencial concluiu 1.048.576 passos, 56.584.156 substituições e 1.834.899 comparações exatas, sem divergência. Novos testes verificaram 352 corrupções byte a byte, 353 comprimentos de arquivo, registros semanticamente inválidos com checksum recalculado e interrupções durante a gravação.

A recuperação de uma fonte de 4 MiB solicitou 4.195.162 bytes ao alocador Rust, apenas 858 além da fonte. A execução de 1.000 pares de atualizações, incluindo aprendizado, não fez novas alocações. Isso não mede RSS nem inclui a construção prévia dos deltas.

Cinco novos modelos receberam 8.192 observações cada, seguidas de 8.192 eventos de avaliação congelada por modelo, com oito regimes de carga. A política economizou 7,56–8,24% ante sempre Delta e 22,38–24,30% ante sempre Full. Porém, frente a uma regra fixa simples de densidade, a economia foi somente 0,39–1,50% na avaliação. Cobrando também a execução durante o aprendizado, ficou 0,11–1,41% mais lenta que essa regra nas cinco rodadas. Não houve amortização nesse horizonte nem promoção automática.

Esse protocolo executa cada evento uma vez por candidato e equilibra posições e predecessores; não deve ser misturado numericamente com os probes do protocolo anterior. Detalhes, comandos, dados brutos e limites: [ROBUSTNESS_2026-09-13.md](ROBUSTNESS_2026-09-13.md).
