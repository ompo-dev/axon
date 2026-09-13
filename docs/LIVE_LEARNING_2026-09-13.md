# AXON: pesquisa, estado continuo e treinamento

Data: 13/09/2026. Host local Windows, Rust 1.98.1, compilacao release, sem dependencias externas. Medicoes executadas em sequencia, sem benchmarks concorrentes. Os dados de entrada sao sinteticos e deterministas; os tempos usados no aprendizado sao observados na execucao real.

## O que mudou

`LiveAverage` possui a fonte e seu acumulador exato, verifica a identidade e a sequencia dos eventos e mantem os dois juntos. Um batch invalido nao modifica o estado. A materializacao custa O(N) uma vez; as atualizacoes incrementais fazem O(delta) operacoes; a consulta retorna a fracao exata em O(1). O objeto tem 144 bytes neste build, alem do vetor fonte. Isto nao comprime o vetor.

`LiveAverageStore` grava um snapshot inicial e eventos em um unico arquivo. Cada evento tem tamanho, complemento do tamanho e checksum, seguido por dados canonicos. A sincronizacao acontece antes de alterar a copia em memoria e antes de confirmar ao chamador. Um segundo escritor e recusado pelo lock do sistema operacional. Um erro de I/O deixa a instancia bloqueada ate a recuperacao, pois o evento pode ter sido gravado parcialmente ou confirmado sem resposta.

Na abertura, a fonte e hidratada e validada, e os eventos completos sao reaplicados uma vez, verificando fonte, versao, sequencia e valores anteriores. Um ultimo registro incompleto e descartado; corrupcao em registro completo provoca erro. Uma interrupcao depois da sincronizacao e antes da resposta pode confirmar o evento: o cliente deve consultar a versao recuperada antes de repetir. A continuacao abaixo adicionou compactacao periodica. Ainda nao ha recovery em O(1), concorrencia distribuida ou garantia validada contra queda fisica de energia. Checksum detecta corrupcao acidental; nao autentica arquivos adversariais.

`OnlineAveragePolicy` armazena custos em 16 contextos pequenos: quatro faixas de tamanho e quatro de densidade de mudancas. Aprende a preferir um dos dois algoritmos AVG ja implementados, Full e Incremental. A media dos custos passa a usar atualizacao exponencial de peso 1/8 apos o aquecimento. Novos contextos usam Full ate existirem quatro observacoes por estrategia. A politica nao aceita ou rejeita a corretude matematica; os contratos do executor continuam obrigatorios.

O checkpoint tem 536 bytes, e a estrutura ocupa 520 bytes em memoria. Ele inclui versao do formato, identificador do contexto e checksum. CPU/contexto, layout ou perfil debug/release diferentes invalidam o carregamento. O identificador padrao vem de PROCESSOR_IDENTIFIER; use `--context` explicito para distinguir maquinas/configuracoes com o mesmo identificador de CPU. A API permite observacoes posteriores; a calibracao deste experimento mede ambas as estrategias em cada caso. Nao e descoberta autonoma de algoritmos nem aprendizado de linguagem.

## Pesquisa aplicada

1. [DBSP, Budiu et al.](https://arxiv.org/abs/2203.16684): calculo de manutencao incremental de consultas sobre fluxos. Aplicacao local: manter o acumulador da instancia entre eventos. A analogia orienta a implementacao; nao reproduzimos todo o sistema nem sua teoria.
2. [Enzyme, versao de abril de 2026, secao 4.5](https://arxiv.org/html/2603.27775v2#S4.SS5): escolha entre atualizacao incremental e recomputacao com custos informados por execucoes historicas. Aplicacao local: memoria limitada de custos reais, com referencias Full e Delta na avaliacao.
3. [River: progressive validation](https://riverml.xyz/dev/api/evaluate/progressive-val-score/): avaliar a previsao antes de revelar a resposta. Aplicacao local: escolher antes de observar os tempos; congelar o aprendizado durante a avaliacao separada.
4. [SQLite: Write-Ahead Logging](https://sqlite.org/wal.html): registrar mudancas duraveis antes de incorpora-las ao estado principal. Aplicacao local: append, sync e replay com sequencia. Nosso pequeno journal nao implementa SQLite nem herda todas as suas garantias.

Essas fontes justificam as escolhas de engenharia. Nenhuma delas demonstra que esta implementacao tenha compreensao geral, experiencias subjetivas ou que evoluira inevitavelmente para AGI.

## AVG continuo

Comando da primeira rodada:

```powershell
cargo run --release --bin axon-uic-live-avg -- --mib 1,16,64,256,1024 --batches 10000 --updates 1024 --paired-batches 128 --seed 20260913 --output target/live-repeat.json --durable target/live-repeat.journal
```

Repeticoes com seeds 20260914 e 20260915 usaram 32 batches pareados iniciais. Os arquivos duraveis devem ser novos: a criacao recusa sobrescrever um estado anterior.

Cada tamanho recebe 10.000 batches na mesma instancia. Foram 150.000 batches e 153.600.000 substituicoes nos tres experimentos de escala. Todas as comparacoes do prefixo e auditorias finais foram exatas. O benchmark mede processamento local com deltas prontos; geracao dos dados e auditoria sao contabilizadas separadamente.

| Fonte | Materializacao, primeira rodada | Atualizar + consultar, p50 nas 3 rodadas | Full p50, primeira rodada |
|---|---:|---:|---:|
| 1 MiB | 0,065 ms | 2,1-2,3 us | 45,6 us |
| 16 MiB | 1,202 ms | 2,1-2,4 us | 910,0 us |
| 64 MiB | 4,361 ms | 2,3-2,5 us | 4.426,3 us |
| 256 MiB | 18,659 ms | 2,6-7,9 us | 20.518,4 us |
| 1 GiB | 141,008 ms | 8,6-8,7 us | 123.345,9 us |

A fonte cresceu 1024 vezes e o tempo incremental aumentou cerca de quatro vezes nos extremos. Isso apoia a ausencia de varredura global no caminho incremental, mas refuta uma promessa literal de latencia independente de N. O salto e a variacao em 256 MiB pedem perfil de cache/TLB, alocacao e escalonamento antes de atribuir uma causa precisa. O break-even observado no prefixo foi no primeiro ou segundo batch, cobrando materializacao/sintese inicial ao candidato. Nao e uma estimativa estatistica de desempenho universal.

O valor anterior de aproximadamente 10 ms vinha de outra fronteira, que reconstruia estado. Nao se deve anunciar uma aceleracao de 10 ms para 2,3 us como se fosse o mesmo trabalho medido.

Com a fonte fixa em 64 MiB, uma verificacao complementar de 2.000 batches mostrou:

| Alteracoes por batch | Atualizar + consultar p50 |
|---:|---:|
| 64 | 0,1 us |
| 1.024 | 2,3-2,5 us (rodadas principais) |
| 4.096 | 29,3 us |
| 65.536 | 640,1 us |

O custo cresce com a mudanca, com efeitos fisicos nao lineares. O resultado de 0,1 us esta proximo da granularidade do relogio e nao serve como medida precisa por operacao.

Dados completos: [seed 1](../.ecc/benchmarks/live-20260913-spread.json), [seed 2](../.ecc/benchmarks/live-20260914-spread.json), [seed 3](../.ecc/benchmarks/live-20260915-spread.json), [delta 64](../.ecc/benchmarks/live-delta-64.json), [delta 4096](../.ecc/benchmarks/live-delta-4096.json), [delta 65536](../.ecc/benchmarks/live-delta-65536.json).

## Durabilidade

O experimento separado usou 1 MiB, 256 batches de 1.024 alteracoes, `sync_all` por batch e 23 reaberturas em pontos determinados pelo gerador. Todos os estados recuperados foram exatamente iguais aos esperados.

| Medida | Resultado |
|---|---:|
| Snapshot inicial | 3,5933 ms |
| Batch duravel p50 | 0,3426 ms |
| Recuperacao acumulada nas 23 aberturas | 436,7932 ms |

Os testes de integracao tambem encerraram a forca processos depois de 1, 13 e 47 eventos confirmados; todos recuperaram e continuaram corretamente. Outro teste remove cada sufixo possivel do ultimo registro e confere replay, truncamento e nova escrita. Foram verificados erros de origem, versao, sequencia, valor anterior, indice, arquivo corrompido e escritor simultaneo.

O custo duravel e muito maior que o custo em RAM. Snapshot/replay leem a fonte e o journal; esta primeira rodada nao usava compactacao. O efeito dos checkpoints foi medido na continuacao abaixo. Nao houve teste de falha fisica do disco, perda de energia, rede ou comportamento do OneDrive durante sincronizacao.

## Treinamento medido

Tres treinamentos independentes usaram 32 epocas, 16 contextos por epoca e duas observacoes de custo por contexto: 1.024 observacoes por politica. O programa gera valores, densidades e posicoes, executa as estrategias e observa qual custou menos. Os rótulos de velocidade nao sao respostas predefinidas.

Treino: vetores com 256, 4.096, 65.536 e 262.144 elementos. Avaliacao: 384, 6.144, 98.304 e 393.216 elementos, seeds/valores/posicoes separados, 256 casos por rodada. As densidades sao 1/128, 1/8, 1/2 e 1. Sao novos casos dentro da mesma familia AVG e das mesmas faixas de caracteristicas; nao um teste de novos operadores.

A previsao vem antes de cada par de medicoes no treino. A politica fica congelada na avaliacao. O caminho escolhido tambem executa em uma terceira instancia; seu timer inclui decisao, validacao, atualizacao e consulta. Todos os caminhos avancam a fonte e sao conferidos contra uma soma independente.

| Seed | Full fixo | Delta fixo | Politica executada | Economia vs Full | Economia vs Delta |
|---:|---:|---:|---:|---:|---:|
| 20260913 | 37,773 ms | 38,359 ms | 33,473 ms | 11,38% | 12,74% |
| 20260914 | 37,421 ms | 37,816 ms | 33,360 ms | 10,85% | 11,78% |
| 20260915 | 36,984 ms | 38,127 ms | 33,774 ms | 8,68% | 11,42% |

Esses tempos sao somas dos custos medios por batch, medidos com repeticoes locais; nao o tempo total do programa de avaliacao. Os tres pares de referencia sustentam vantagem experimental neste workload. Nao se mediu superioridade sobre uma heuristica de densidade ajustada, que pode fazer escolhas semelhantes.

Foram 596 escolhas corretas em 602 comparacoes onde os tempos diferiram pelo menos 10%: 99,0%. Os outros 166 casos nao entram na taxa, mas seus tempos continuam nos totais. Esse limiar descritivo nao e intervalo de confianca nem criterio de promocao de `StrategyEvidence`. A paridade exata vale para todos os 768 casos.

Cada treino custou 1,12-1,29 segundo, inclusive geracao, probes e auditorias. A economia de aproximadamente 3,2-4,3 ms por bloco de 256 casos nao paga esse treino durante a avaliacao. Extrapolando a mesma mistura e custos, seriam aproximadamente 67-90 mil batches para amortizar a calibracao frente a Full. Isso e uma estimativa, nao um break-even medido. A persistencia torna possivel amortizar em execucoes futuras.

Comando reproduzivel (use um caminho novo para cada checkpoint):

```powershell
cargo run --release --bin axon-uic-online-train -- --epochs 32 --seed 20260913 --checkpoint target/avg-trained.bin --output target/avg-trained.json
cargo run --release --bin axon-uic-online-train -- --epochs 0 --seed 20261001 --resume target/avg-trained.bin --output target/avg-resumed.json
```

Na execucao de recuperacao realizada, a politica carregada manteve 1.024 observacoes, sem aprender novas durante a avaliacao. Escolheu corretamente em 201/203 casos decisivos e executou os 256 casos em 36,387 ms de custo acumulado por batch, frente a 40,407 ms Full e 42,665 ms Delta.

Um controle com alteracoes contiguas tambem foi treinado e avaliado separadamente: 191/206 escolhas decisivas corretas (92,7%), politica 34,982 ms, Full 40,320 ms e Delta 38,545 ms. Essa taxa menor e mantida no relatorio; nao foi usada para reajustar o modelo depois de ver a avaliacao.

Resultados e modelos: [treino 1](../.ecc/benchmarks/online-v2-20260913.json), [modelo 1](../.ecc/benchmarks/online-v2-20260913.policy), [treino 2](../.ecc/benchmarks/online-v2-20260914.json), [treino 3](../.ecc/benchmarks/online-v2-20260915.json), [retomada](../.ecc/benchmarks/online-v2-resumed.json), [controle contiguo](../.ecc/benchmarks/online-v2-contiguous.json). Os arquivos `online-20260913-seed1.*` sao a rodada exploratoria v1, que estimava o custo escolhido a partir dos pares; nao entram nos resultados v2 acima.

## Proximas capacidades e limites

Validacao inicial: 95 testes passaram, incluindo recuperacao e processos encerrados a forca. A continuacao adicionou os testes e experimentos abaixo.

Agora existe uma base para manter estatisticas exatas continuamente, aprender escolhas de execucao e reaproveitar essa experiencia depois de reiniciar. O caminho conservador e avaliar drift/custos de exploracao em fluxos mais variados e integrar manutencao de novas estatisticas com os mesmos contratos.

`VARIANCE`, `MIN`, modelos do mundo, linguagem e planejamento geral continuam pendentes. A lista de candidatos do antigo LearnBench nao implementa essas capacidades. Nao houve treinamento de um modelo de linguagem, descoberta de algoritmos novos ou evidencia de AGI. A leveza medida pertence a este mecanismo restrito; nao demonstra armazenamento de conhecimento humano geral em poucos bytes.

## Continuacao: compactacao e aprendizado pela acao executada

`LiveAverageStore::compact()` grava um snapshot completo e sincronizado em um temporario vizinho, depois publica esse arquivo por renomeacao. O arquivo lateral `.lock` permanece bloqueado durante a troca do arquivo de dados, impedindo outro escritor de entrar pela troca do identificador de arquivo. Esse lock nao deve ser removido enquanto houver acesso ao store. A API [Rust rename](https://doc.rust-lang.org/std/fs/fn.rename.html) substitui o destino no mesmo sistema de arquivos; os testes desta maquina cobrem as fronteiras dessa publicacao.

O snapshot e escrito com um buffer de 64 KiB, sem alocar outra copia completa da fonte para serializacao. A recuperacao ainda hidrata a fonte inteira. A compactacao e O(N) e precisa de espaco para o novo snapshot enquanto o antigo permanece valido. Uma interrupcao antes da publicacao pode deixar um temporario orfao, ignorado pela abertura do estado principal. Nao implementamos limpeza automatica desses arquivos de falha.

Os testes encerram o processo depois da criacao do temporario, depois do sync e depois da publicacao. Em todos os casos, os eventos ja confirmados permanecem disponiveis e novas alteracoes podem continuar. Uma falha forcada de publicacao bloqueia o uso da instancia ate reabrir; o arquivo anterior continua recuperavel. A versao e a sequencia nao reiniciam ao compactar.

Medimos tres pares de experimentos com 1 MiB de fonte, 1.024 batches de 1.024 mudancas sincronizadas e os mesmos pontos de reabertura dentro de cada par. O candidato compacta a cada 64 batches. A ordem dos candidatos foi invertida no segundo par.

| Medida | Sem compactar | Compactar a cada 64 batches |
|---|---:|---:|
| Arquivo final de dados | 26.271.832 bytes | 1.048.664 bytes |
| Maior arquivo ativo | 26.271.832 bytes | 2.625.112 bytes |
| Checkpoints por rodada | 0 | 16 |
| Custo total dos checkpoints | 0 | 56,9-60,8 ms |
| Recuperacao acumulada, seed 20260913 (75 aberturas) | 1.891,0 ms | 660,1 ms |
| Recuperacao acumulada, seed 20260914 (69 aberturas) | 1.677,2 ms | 572,8 ms |
| Recuperacao acumulada, seed 20260915 (68 aberturas) | 1.808,4 ms | 562,3 ms |

O arquivo final ficou 96,0% menor. A recuperacao acumulada caiu 65,1-68,9%. Contando criacao do snapshot, gravacoes duraveis, compactacoes e recuperacoes medidas, os totais cairam de 2.142,7/1.898,3/2.031,5 ms para 959,2/855,4/840,4 ms. Esse experimento tem muitos reinicios; nao representa um servico que fica ligado sem reiniciar. Sem o ganho de recuperacao, os checkpoints acrescentam cerca de 55-59 us amortizados por batch e reduzem crescimento em disco. O pico de espaco total tambem inclui o snapshot temporario, que nao entra na coluna de arquivo ativo.

Todas as paridades foram exatas. Dados: [par 1 referencia](../.ecc/benchmarks/compaction-baseline.json), [par 1 compactado](../.ecc/benchmarks/compaction-every64.json), [par 2 referencia](../.ecc/benchmarks/compaction-baseline-r2.json), [par 2 compactado](../.ecc/benchmarks/compaction-compact-r2.json), [par 3 referencia](../.ecc/benchmarks/compaction-baseline-r3.json), [par 3 compactado](../.ecc/benchmarks/compaction-compact-r3.json).

```powershell
cargo run --release --bin axon-uic-live-avg -- --mib 1 --batches 256 --updates 1024 --paired-batches 32 --durable-batches 1024 --checkpoint-every 64 --durable target/compact-repeat.journal --output target/compact-repeat.json
```

### Aprender somente com a estrategia escolhida

`OnlineAveragePolicy::execute_and_learn` agora executa um evento uma vez, observa o tempo dessa acao e atualiza o modelo. Eventos rejeitados nao alteram a fonte nem o aprendizado. O aquecimento coleta ambas as alternativas de forma alternada; depois a politica experimenta a alternativa a cada 16 observacoes daquele contexto. Isso evita abandonar uma estrategia para sempre caso as condicoes mudem. `choose()` continua sem exploracao ou aprendizado para avaliacoes congeladas.

Os testes verificaram paridade ao aprender com 256 eventos reais e adaptacao a uma inversao de custos controlada. A inversao usa custos sinteticos e verifica o mecanismo; nao demonstra adaptacao a drift fisico arbitrario.

Retomamos o modelo salvo da primeira rodada com 128 epocas e feedback apenas da acao escolhida. O benchmark ainda executa os dois caminhos para obter referencias cientificas, mas o aprendizado recebe somente o custo escolhido; a API `execute_and_learn` dispensa essa segunda execucao.

| Medida da continuacao | Resultado |
|---|---:|
| Observacoes antes/depois | 1.024 / 3.072 |
| Checkpoint depois de continuar | 536 bytes |
| Tempo do experimento de treino | 4,503 s |
| Avaliacao separada | 256 casos |
| Escolhas corretas com diferenca de pelo menos 10% | 200 / 201 |
| Custo da politica executada na avaliacao | 33,803 ms |
| Economia ante sempre Full / sempre Delta | 10,86% / 11,75% |
| Paridade exata | todos os casos |

Os outros 55 casos continuam nos totais de tempo. O treino adicional tem custo e nao demonstrou superar o modelo anterior; demonstrou continuar aprendendo sem aumentar o arquivo e manter desempenho util neste conjunto. Resultado: [treino continuado](../.ecc/benchmarks/online-selected-continued.json), [modelo continuado](../.ecc/benchmarks/online-selected-continued.policy).

```powershell
cargo run --release --bin axon-uic-online-train -- --epochs 128 --seed 20261002 --feedback selected --resume .ecc/benchmarks/online-v2-20260913.policy --checkpoint target/continued-policy.bin --output target/continued-policy.json
```

Validacao final da continuacao: 102 testes passaram em `cargo test --all-targets`; Clippy com avisos tratados como erros, verificacao de formatacao e `git diff --check` passaram. Os 19 relatorios JSON desta sessao foram validados por parser. Os modelos e as amostras permanecem salvos no projeto; nenhuma estrategia foi promovida automaticamente e nenhum commit foi criado.
