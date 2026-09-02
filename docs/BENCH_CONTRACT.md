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
