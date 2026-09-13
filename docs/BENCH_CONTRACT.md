# BenchContract

`BenchContract` é o vocabulário obrigatório para tornar benchmarks AXON comparáveis. Ele não cria custo artificial: apenas classifica custo que o binário mediu.

| Fase | Significado |
|---|---|
| `input_generation` | geração determinística do input que entra na transação |
| `initialization` | cópia/estado inicial e cache necessário antes da execução |
| `synthesis` | derivação ou compilação da capacidade usada |
| `verification` | checker/certificado semântico |
| `allocation` | reservas de memória da transação |
| `ingestion` | manutenção ao receber eventos |
| `planning` | seleção/materialização de plano |
| `artifact_load` | leitura e reconstrução de capability persistida |
| `artifact_persist` | criação e publicação sincronizada do artifact persistente |
| `execution` | aplicação do plano ao estado preparado |
| `result_validation` | comparação independente com resultado exato |
| `teardown` | descarte do estado temporário da transação |

## Métricas

`HOT` é somente `execution`.

`LIFECYCLE` é a soma verificada de todas as fases acima na **mesma rodada**. O p50 de `LIFECYCLE` é calculado sobre essas somas por rodada; ele não é a soma de p50s de fases diferentes.

`Duration::ZERO` é emitido somente quando a fase não se aplica ou não foi medida naquele protocolo. Isso permanece explícito na tabela, em vez de mover custo para fora da saída.

`verification` de um `SemanticArtifact` é replay de certificado, versão e selo por conteúdo; deve escalar com artifact, não com vetor. `result_validation` continua sendo auditoria independente do experimento. Não se deve usar custo de auditoria para alegar custo de guard de produção, nem retirar a auditoria da tabela quando ela foi medida.

## Regras de comparação

1. Só compare HOT com HOT e LIFECYCLE com LIFECYCLE.
2. Publique o comando, hardware, sequência completa de rodadas, ordem permutada e paridade semântica.
3. Compare candidatos e baseline no mesmo workload/seed por rodada. P50 isolado não promove estratégia; `StrategyEvidence` precisa de pares e de domínio físico igual.
4. Uma fronteira de benchmark precisa declarar o que é entrada pronta, cache persistente, ingestão e teardown. Mudar a fronteira exige nova versão de protocolo.

## Implementação inicial

`axon-uic-deltaforge-sum` foi o primeiro binário convertido. `axon-uic-deltaforge-avg` também mede criação persistente e reload de artifact: cria um arquivo temporário por rodada, chama `sync_all` antes de publicar e mede o reload da mesma capability. `ingestion` e `planning` são zero porque estes são batches diretos com regra já declarada; não são custo oculto.

Os binários legados continuam com seus próprios protocolos até serem migrados. Seus números absolutos não são comparáveis aos números `BenchContract` do DeltaForge.

## Protocolo vivo v3

`LiveBenchContract` é aditivo: mantém os contratos legados e classifica `setup`, `steady`, `durability`, `recovery` e `audit`. `production()` soma as primeiras quatro fases com detecção de overflow. Auditoria é publicada separadamente.

`axon-uic-live-avg` materializa uma instância uma vez e aplica todos os batches sobre essa mesma instância. O caminho incremental valida índices, valores anteriores, fonte e sequência; não clona o vetor nem refaz a soma. O timer inclui validação, aplicação e consulta. A geração do vetor e dos deltas, que é trabalho do gerador experimental, tem campos próprios. Não representa uma medida de ingestão de sensores ou de um sistema externo.

Full é medido num prefixo pareado (128 batches na primeira rodada, 32 nas repetições), com a ordem Full/Delta alternada. O restante mede continuidade incremental; não se extrapola Full para alegar uma comparação pareada de todos os 10 mil batches. O break-even usa apenas os custos acumulados do prefixo, cobrando síntese e materialização inicial do candidato; não cobra cache inicial à referência Full, que pode começar na primeira consulta. Ele é um cruzamento de custos observado, não uma promoção estatística.

`production_ram_ns` é síntese + materialização + atualizações/consultas na memória. O modo opcional `--durable` é outro experimento: snapshot de 1 MiB, 256 eventos por padrão (configuráveis até 10.000 por `--durable-batches`), com `sync_all` e reaberturas determinísticas. O seu timer de batch inclui serialização, checksum, gravação, sincronização, atualização e consulta. Recuperação é informada à parte. Os tempos RAM não são promessa de latência durável.

O gerador acessa os valores anteriores antes do timer, aquecendo os endereços alterados. Consultas isoladas usam leituras agrupadas; números submicrossegundo estão sujeitos à resolução do relógio. Localidade, cache, pressão de memória e escalonamento podem alterar o tempo mesmo sem uma varredura proporcional ao vetor.

`axon-uic-online-train` mede pares de estratégias, faz a previsão antes de revelar os tempos e mantém avaliação separada sem aprendizado. Além do custo estimado a partir dos pares, executa a política em uma terceira instância e mede decisão + aplicação. As repetições alternam ida/volta para amortizar o relógio, depois avançam o estado uma vez e auditam a fonte modificada. O relatório registra amostras, custo de calibração e ambas as referências fixas. Economia por batch na avaliação não inclui amortização do treinamento.

`--checkpoint-every N` habilita compactação periódica no experimento durável. `checkpoint_total_ns` registra escrita, sync e publicação do snapshot. Esse custo não entra no p50 de aplicação do batch e deve ser somado ao avaliar o custo total. `peak_active_file_bytes` mede apenas o arquivo ativo, não seu temporário. `--feedback selected` no treino fornece ao aprendiz somente o custo da estratégia escolhida; os pares de referência continuam custando tempo no experimento e constam em `training_wall_ns`.

## Protocolo online-stream v2

`axon-uic-online-stream` mede quatro estados independentes sobre os mesmos eventos: Delta, Full, regra fixa `k <= n/4` e política. Cada evento é executado uma vez por candidato, sem probes de ida/volta. Posições e pares de predecessores imediatos são equilibrados em cada grupo de quatro eventos por tamanho/fase. O modelo recebe apenas o tempo da ação que executou no treino; referências não fornecem feedback. Na avaliação, o modelo fica congelado e as fontes têm novos tamanhos e dados.

Os timers incluem seleção e aplicação, mais exploração e atualização do modelo no treino. Geração, auditoria independente, demais candidatos e montagem dos registros entram no tempo de parede do experimento, mas não no custo operacional de cada candidato. Serialização final de JSON e persistência do checkpoint ficam fora desses timers. Nenhum deles mede ingestão externa ou durabilidade do estado vivo.

As oito fases incluem mudança de densidade/localidade, eventos vazios e fronteiras das faixas do seletor. Todos os casos entram nos totais. O JSON preserva amostras, p50, p95, contagens e tempos por fase. A comparação relevante inclui o custo de aprender; ganho somente na avaliação não estabelece amortização. A regra fixa é uma referência obrigatória para não atribuir ao aprendizado uma vantagem já obtida por uma decisão simples. Resultados pequenos entre candidatos não são promoção estatística.

`axon-uic-stress` mede correção em histórias aleatórias com oráculo independente, não velocidade de produção. O teste de alocação mede bytes solicitados ao alocador Rust em janelas isoladas; não mede RSS ou cache do sistema operacional. Protocolo, resultados positivos e negativos: [ROBUSTNESS_2026-09-13.md](ROBUSTNESS_2026-09-13.md).
