# AXON: robustez e avaliacao mais exigente

Rodada de 13/09/2026. Host: Intel Core i7-13650HX, 14 nucleos/20 processadores logicos, Windows x86_64 MSVC, Rust 1.98.1. Os numeros usados nos nomes dos experimentos sao sementes, nao datas de execucao. Todos os cinco treinos abaixo ocorreram nesta rodada.

## Conclusao

A manutencao de AVG exato ficou mais protegida e a recuperacao deixou de alocar uma segunda copia completa da fonte. A suite cresceu de 102 para 118 testes, passando em debug e release.

O novo aprendizado medido funciona, mas nao venceu uma regra fixa simples quando cobramos tanto o periodo de aprendizado quanto a avaliacao. Essa conclusao e mais restritiva que comparar apenas com sempre Full ou sempre Delta. Nao promovemos automaticamente o modelo nem alteramos suas regras depois de observar os resultados desta avaliacao.

Nada nesta rodada demonstra AGI, experiencia subjetiva, compreensao de linguagem ou descoberta geral de algoritmos. O aprendizado real continua restrito a escolher entre duas implementacoes existentes de media exata.

## Correcoes implementadas

- O resumo de `StructurePrior` agora mostra regressao como economia negativa. Antes, a subtracao saturada escondia pioras como zero.
- A caracteristica `numeric_exact` passou a participar do vetor de aprendizado; antes existia no contrato, mas era ignorada pelo ranker.
- A pontuacao usa `i64`, evitando overflow ao somar pesos `i32` saturados.
- O ranker atualiza pesos somente quando a resposta nao estava em primeiro lugar, com reforco e penalizacao de mesma magnitude. Repetir 10.000 vezes um acerto nao altera os pesos.
- Uma atualizacao vazia nao provoca varredura completa da fonte, mesmo se Full tiver sido solicitado. Versao e sequencia continuam avancando normalmente.
- O snapshot e recuperado em blocos de 64 KiB para um unico vetor de valores. Reservas de memoria relevantes usam `try_reserve_exact` e retornam erro se falharem.
- `StoreLimits`, `create_with_limits` e `open_with_limits` limitam valores da fonte, alteracoes por evento e quantidade de eventos no journal. Uma recusa nao confirma nem altera o estado.

O ranker sintetico ainda faz 16 -> 4 tentativas nas quatro estruturas do seu proprio curriculo depois de oito epocas. Isso nao e generalizacao nem implementacao de MIN/VARIANCE.

## Limites de armazenamento

| Limite padrao | Valor |
|---|---:|
| Valores da fonte | 134.217.728 u64, equivalente a 1 GiB |
| Alteracoes por evento | 1.048.576 |
| Eventos desde o snapshot | 1.000.000 |

Os limites sao configuracao da abertura, nao parte persistida do formato. Ao reabrir, o chamador deve fornecer limites suficientes para o arquivo. Compactar renova o limite de eventos sem zerar a versao. O formato de dados permanece `AXLIVE01`.

Os limites de fonte sao checados antes da alocacao durante a recuperacao. O limite de alteracoes e checado antes da escrita do evento. Na criacao, a fonte ja pertence ao chamador; nao evitamos o custo que ele ja teve para materializa-la. O limite de eventos tambem limita trabalho de replay. Nenhum deles representa uma cota global de RAM ou de disco do processo.

## Testes executados

### Historias longas e oraculo independente

`axon-uic-stress` usa tres estados separados: Delta, Full e politica que aprende pela acao executada. Um quarto vetor de referencia e atualizado independentemente; cada resultado e comparado com uma soma completa em `u128`, a fonte integral e a versao esperada.

| Medida | Resultado |
|---|---:|
| Sementes | 128 |
| Passos por semente | 8.192 |
| Passos totais | 1.048.576 |
| Eventos validos | 611.633 |
| Eventos invalidos rejeitados | 436.943 |
| Substituicoes em eventos validos | 56.584.156 |
| Comparacoes independentes exatas | 1.834.899 |
| Divergencias | 0 |
| Tempo do experimento release | 1,537 s |

Os vetores variam de 1 a 4.096 elementos. As sequencias incluem zero, `u64::MAX`, fracoes, lotes vazios, substituicoes sem mudanca, substituicao integral, valor anterior incorreto, fonte/versao/sequencia incorretas e indice fora da fonte. Rejeicoes verificam ausencia de mutacao parcial e, quando aplicavel, ausencia de aprendizado. Testes adicionais verificam comutatividade de alteracoes disjuntas e restauracao por operacao inversa.

Este teste privilegia quantidade de historias e casos extremos. Nao e benchmark de fontes de 1 GiB; os testes de tamanho grande pertencem ao [relatorio anterior](LIVE_LEARNING_2026-09-13.md).

Dados: [campanha de um milhao de passos](../.ecc/benchmarks/stress-20260913-million.json).

### Arquivos danificados e falhas de processo

`storage_adversarial.rs` cria um snapshot e tres eventos, totalizando 352 bytes. Foram executados:

- 352 casos de corrupcao, invertendo um bit em cada byte do arquivo completo. Todos foram rejeitados sem modificar o arquivo danificado.
- 353 comprimentos possiveis, de zero bytes ao arquivo completo. Snapshot incompleto e recusado; uma cauda incompleta do journal e descartada e o ultimo prefixo valido pode continuar recebendo eventos.
- 14 corrupcoes semanticas com checksum recalculado: artefato, agregado, contagem, fonte, versao, sequencia, valor anterior e indices. Nenhuma passou pela validacao.
- Checkpoint valido na versao maxima de `u64`: consulta permitida, novo evento recusado sem wraparound ou escrita.
- Limites de recursos na criacao, abertura e aplicacao; recusa sem mutacao e renovacao do limite de eventos ao compactar.

Os testes internos encerram um processo depois do cabecalho, do payload, da sincronizacao e da aplicacao em RAM. Somente o cabecalho recupera o estado anterior; depois de um registro completo, a reabertura reconheceu o novo evento nesta maquina, inclusive quando ainda nao havia confirmacao ao chamador. Reenviar a mesma versao/evento foi recusado. Isso exige que o chamador consulte a versao recuperada quando o resultado de uma gravacao for incerto.

Uma falha real de escrita em handle somente leitura bloqueou consultas, novas escritas e compactacao ate a reabertura, preservando o estado anterior. Os testes anteriores de interrupcao nos tres pontos de compactacao tambem passaram novamente.

Esses testes nao simulam queda fisica de energia, falha de controlador, disco cheio ou todos os sistemas de arquivos. Checksum FNV nao e autenticacao criptografica. Os temporarios de checkpoint abandonados por um processo interrompido ainda nao sao limpos automaticamente.

### Memoria

O teste isolado instrumenta o alocador Rust e contabiliza bytes solicitados durante as janelas medidas.

| Janela | Bytes alocados |
|---|---:|
| 1.000 pares de atualizacoes, uma direta e uma com aprendizado | 0 |
| Fonte de 4 MiB recuperada | 4.195.162 |
| Dados da fonte | 4.194.304 |
| Diferenca | 858 |

Isso nao e RSS, memoria da pilha ou cache do sistema operacional. Os dados e deltas ja estao materializados antes da janela de atualizacao. A recuperacao continua O(N), pois precisa ler a fonte e verificar o agregado. A melhoria e remover a duplicacao do vetor, nao tornar a recuperacao constante.

## Novo treino e avaliacao

O novo binario `axon-uic-online-stream` aplica cada evento uma vez por candidato em quatro estados independentes:

1. Sempre Delta.
2. Sempre Full.
3. Regra fixa: Delta quando `k <= floor(n/4)`, Full nos demais casos.
4. Politica aprendida: observa somente a estrategia que executou durante o treino; permanece congelada na avaliacao.

O protocolo v2 equilibra a posicao de execucao e os pares de predecessores imediatos usando as ordens 0-1-3-2, 1-2-0-3, 2-3-1-0 e 3-0-2-1. Todos recebem o mesmo delta. O timer da politica inclui escolha, exploracao, execucao e atualizacao do modelo no treino. As referencias nunca fornecem seus tempos ao aprendiz.

Oito fases mudam densidade e localidade: esparsa espalhada, densa espalhada, esparsa contigua, densa contigua, vazia, mistura aleatoria, fronteiras de densidade e indices quentes. Sao 256 eventos por tamanho/fase.

Treino: fontes de 384, 6.144, 98.304 e 393.216 valores. Avaliacao congelada: 511, 8.191, 131.071 e 524.287, com outra semente de dados. Sao novos tamanhos dentro das mesmas quatro faixas do modelo, nao transferencia para outra tarefa matematica.

Cada uma das cinco politicas recebeu 8.192 observacoes e foi avaliada em outros 8.192 eventos. Total: 40.960 observacoes de treino, 40.960 eventos de avaliacao e 327.680 verificacoes exatas dos quatro candidatos. Cada politica continua ocupando 520 bytes em RAM e 536 no checkpoint.

### Resultados sem excluir os casos desfavoraveis

Tempos acumulados de execucao da avaliacao, em milissegundos. Valores positivos na ultima coluna significam economia; negativos seriam piora.

| Semente | Delta | Full | Regra fixa | Aprendido | Economia ante regra |
|---|---:|---:|---:|---:|---:|
| 20260914 | 1.115,897 | 1.356,652 | 1.035,810 | 1.027,022 | 0,85% |
| 20260915 | 1.101,908 | 1.329,890 | 1.028,502 | 1.013,025 | 1,50% |
| 20260916 | 1.135,579 | 1.352,321 | 1.054,860 | 1.049,711 | 0,49% |
| 20260917 | 1.112,233 | 1.333,965 | 1.033,692 | 1.023,288 | 1,01% |
| 20260918 | 1.140,592 | 1.365,684 | 1.050,652 | 1.046,569 | 0,39% |

Na avaliacao, a politica economizou 7,56-8,24% ante sempre Delta e 22,38-24,30% ante sempre Full. Mas a regra simples explica quase todo esse ganho. Durante o aprendizado, a politica custou 1,67-4,63% a mais que essa regra. Somando treino e avaliacao dos candidatos, a politica ficou 0,11-1,41% mais lenta em todas as cinco repeticoes: o aprendizado nao se amortizou nesse horizonte.

O caso vazio deixa visivel o custo de decisao: somando as cinco avaliacoes, foram aproximadamente 0,22 ms para a regra e 0,33 ms para a politica. O aumento percentual e grande, mas o custo absoluto e pequeno. A politica tambem perdeu nas fases densa espalhada, esparsa contigua e indices quentes quando agregadas entre sementes. Nao houve exclusao desses tempos dos totais.

Cada experimento completo levou 19,23-19,69 s, incluindo geracao, quatro candidatos, construcao do relatorio e auditoria independente. Esses tempos de parede nao sao o custo de um servico com apenas um candidato. A serializacao final do JSON e a persistencia do modelo ficam fora desses timers. Os ganhos pequenos ante a regra nao sao certificado estatistico nem promessa entre maquinas: ha resolucao do relogio, cache, escalonamento e correlacao temporal entre eventos. Nao ajustamos o limiar da regra nem o aprendiz a partir desta avaliacao.

O primeiro ensaio [stream v1](../.ecc/benchmarks/online-stream-20260913.json) fica preservado como exploratorio. Ele equilibrava posicoes, mas nao predecessores. Nao entra na tabela v2. O protocolo anterior com probes de ida/volta tambem permanece historico e nao deve ser combinado numericamente com este.

Dados: [resumo recalculado](../.ecc/benchmarks/online-stream-summary.json), [rodada 14](../.ecc/benchmarks/online-stream-v2-20260914.json), [rodada 15](../.ecc/benchmarks/online-stream-v2-20260915.json), [rodada 16](../.ecc/benchmarks/online-stream-v2-20260916.json), [rodada 17](../.ecc/benchmarks/online-stream-v2-20260917.json), [rodada 18](../.ecc/benchmarks/online-stream-v2-20260918.json). Os checkpoints correspondentes usam o mesmo nome com extensao `.policy`.

## Reproduzir

Use um caminho novo para checkpoints, pois eles nao substituem modelos anteriores.

```powershell
cargo test --all-targets
cargo test --release --all-targets
cargo clippy --all-targets -- -D warnings
cargo run --release --bin axon-uic-stress -- --seeds 128 --steps 8192 --seed 20260913 --output target/stress-repeat.json
cargo run --release --bin axon-uic-online-stream -- --steps 256 --seed 20260914 --output target/online-stream-v2-20260914.json --checkpoint target/online-stream-v2-20260914.policy
.\scripts\validate-online-stream.ps1 -Directory target -Output target/online-stream-summary.json
cargo test --release --test memory_contracts -- --nocapture
```

O validador usa o parser JSON do PowerShell e recalcula todos os totais a partir das amostras, verifica contagens, equilibrio da ordem e tamanho dos checkpoints. Os cinco relatorios e 81.920 registros passaram nessa validacao.

## Trabalho ainda necessario

Prioridade mensuravel: reduzir o custo de explorar/aprender e impedir que eventos sem sinal util distorcam as estimativas, depois avaliar com um novo conjunto reservado. Tambem faltam medidas sob concorrencia, pressao de memoria, outras maquinas e falhas de energia, alem de manutencao automatica dos temporarios.

Para avancar alem de AVG, sera preciso implementar novas capacidades reais, definir tarefas que exijam transferencia e memoria de experiencias, e testar aprendizado com dados externos. Mais epocas neste seletor nao produzem, por si so, compreensao, criatividade ou AGI. Nao existe um teste finito que estabeleca que nao ha mais nada a melhorar.
