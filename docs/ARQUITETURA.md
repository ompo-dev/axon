# Arquitetura Axon v3.2

## Visao de desenho

Principio central:
- O estado cognitivo vive no `.axon`.
- O runtime Rust e orquestrador de execucao, IO e persistencia.

Separacao funcional:
- **Control lane**: comandos locais (`/help`, `/mode`, `/flush`, `/checkpoint`, `/quit`, `/corrigir`).
- **Cognitive lane**: texto conversacional que entra no grafo associativo.

## Runtime e concorrencia

Threads principais:
- `Input`: teclado/raw/fallback.
- `TickEngine`: loop de 100 Hz.
- `Persist`: flush/journal/checkpoint.
- `Render`: frames da TUI com invalidacao.
- `Telemetry`: contadores por segundo.

Cadencias configuradas (`src/config.rs`):
- Tick: `10 ms` (`100 Hz`).
- Flush journal: `250 ms` ou `64 KiB`.
- Checkpoint: `5 s` ou `128 MiB` de journal acumulado.

Fluxo por tick (alto nivel):
1. Drenar eventos de input.
2. Rotear comandos slash no control lane.
3. Enfileirar texto no cerebro quando nao e slash.
4. Rodar `brain.step(...)`.
5. Coletar mutacoes e atualizar buffer pendente.
6. Sinalizar render quando necessario.

## Modelo cognitivo unificado

Arquivo: `src/memory/mod.rs`.

Nao ha separacao real entre "memoria" e "conceito":
- Ambos sao nos do mesmo grafo.
- Aprendizado e ajuste de arestas/temperatura/fase.

Tipos de no:
- `Concept`
- `Episode`
- `Temporal`
- `Cue`

Tipos de aresta:
- `CoActivation`
- `TemporalBinding`
- `ContextBinding`
- `Contrast`
- `Correction`

Estado por no:
- `temperature`
- `amplitude`
- `phase`
- `omega`
- `frequency`
- `salience`

Estado por aresta:
- `strength`
- `temperature`
- `delay`
- `confidence`
- `kind`

Temperatura dinamica:
- recencia + frequencia + saliencia com decaimento exponencial continuo.

Propagacao associativa:
- expansao em largura limitada por profundidade.
- interferencia por fase (`cos(phase_delta)`).
- combinacao de excitacao/inibicao para ranquear hipoteses.

Correcao cognitiva:
- reforca trilha correta (`Correction`).
- enfraquece coativacoes concorrentes erradas (`LinkWeaken`).

## Geracao de resposta

Arquivo: `src/cortex/mod.rs`.

Pipeline:
1. Receber entrada em fluxo de caracteres.
2. Criar/atualizar estado local (`assemblies`, arestas delta, frequencias).
3. Chamar `memory.rank_hypotheses(...)`.
4. Selecionar hipotese dominante:
- `deterministic`: top score.
- `stochastic`: amostragem ponderada por score.
5. Emitir resposta incremental.

Fallback de baixa confianca:
- pergunta curta de desambiguacao.

## Persistencia `.axon`

Arquivos: `src/axon_format/mod.rs` e `src/storage/mod.rs`.

Formato fisico:
- superblock A/B com `generation` monotona e checksum.
- paginas 4 KiB com header + payload.
- `append` de journal por lotes.
- snapshot chunked em paginas `META`.

Durabilidade:
- `journal-first` para mutacoes.
- checkpoint periodico consolidando estado.
- leitor ignora paginas invalidas por checksum/header.

Mutacoes registradas em journal:
- `InputChar`
- `OutputChar`
- `EdgeUpdate`
- `Spawn`, `Merge`, `Prune`
- `TempUpdate`
- `LinkCreate`, `LinkStrengthen`, `LinkWeaken`
- `TemporalRebind`
- `CorrectionApplied`

Snapshot:
- versao atual escrita: `v3`.
- leitura compativel: `v1`, `v2`, `v3`.

## TUI sem flicker

Arquivo: `src/tui/mod.rs`.

Tecnica:
- double-buffer logico (`prev_lines` vs frame atual).
- patch incremental por diferenca de prefixo comum.
- operacoes pontuais de cursor + `ClearToEnd`.
- alt-screen e synchronized output quando suportado.

Editor de input:
- cursor esquerda/direita.
- backspace/delete.
- historico (`Up/Down`) com restauracao de rascunho.
- slash suggestions com selecao e autocomplete.

## GPU e backend numerico

Arquivo: `src/gpu/mod.rs`.

Estado atual:
- probe de disponibilidade CUDA Driver API por FFI (`cuInit`).
- fallback automatico para CPU quando indisponivel.
- kernels CUDA de computacao pesada ainda nao implementados.

## Compatibilidade e migracao

No `load_state`:
- tenta carregar snapshot mais recente.
- aplica replay do journal apos `last_lsn`.
- se snapshot antigo tiver semantica legada, converte para grafo unificado na primeira escrita.

## Limites conhecidos (v3.2)

- Linux ainda usa fallback de input por linha (sem raw mode completo).
- Diversos warnings de `dead_code` ainda existem.
- Path de GPU ainda e de deteccao, nao de execucao numerica acelerada completa.

## Núcleo cognitivo V3 experimental

Arquivos: `src/core_v3/`.

O núcleo V3 é um simulador isolado: não altera o loop atual, a TUI ou o formato
`.axon`. A integração só começa depois de benchmarks que comprovem seus
invariantes.

Substratos:

1. `SemanticMesh`: fatos composicionais (`sujeito`, `relação`, `objeto`) com
   assinatura hiperdimensional ternária.
2. `DynamicField`: células dendríticas com ramos semântico, temporal, causal,
   visual e linguístico. Só o ramo compatível com o evento muda.
3. `EpisodicStore`: log imutável por API de eventos importantes, recuperado por
   assinatura para preservar detalhes exatos fora do estado ativo.

Fluxo:

```text
entrada já codificada -> AdaptiveEventCodec -> SalienceGate
                                             -> DynamicField + EpisodicStore
                     -> silêncio, se valor cognitivo baixo
```

O `AdaptiveEventCodec` é uma fronteira tipada, não um modelo de linguagem ou
de visão completo: recebe assinatura semântica de um encoder externo e mantém
resíduos multimodais em escalas independentes (byte, fonema, morfema, palavra,
frase, conceito e intenção). Só resíduos acima de um limiar esparso são
roteados. A experiência episódica indexa a assinatura semântica, enquanto os
detalhes específicos da modalidade continuam separados.

`SalienceGate` usa erro de previsão, relevância para objetivo, incerteza,
ganho de informação, novidade e custo. Portanto um evento previsível, mas
importante, ainda pode acordar o núcleo.

O `DynamicField` preserva `eligibility traces` por ramo. Um `CreditPacket`
válido aplica recompensa e erro de previsão exclusivamente a ramos que ficaram
elegíveis — ou a uma célula endereçada — e ajusta sua plasticidade local; não
há taxa de aprendizado global nem backpropagação pelo núcleo inteiro.

`REFRAME` é separado de adaptação local. O motor abductivo gera ilhas de
hipóteses independentes — remover premissa, inverter causalidade, introduzir
mediador/causa latente, fundir/separar conceitos, criar relação/operador ou
expandir dimensionalidade — e as elimina por testes contrafactuais. Hipóteses
refutadas entram no `NegativeArchive` para não serem repetidas.

O `ThoughtCompiler` transforma uma trajetória repetidamente bem-sucedida em
`Circuit`, mas exige guards de contexto: condição inválida faz `Deoptimized`
e devolve a execução ao caminho deliberativo.

O `run_jump_benchmark()` atual é sintético e determinístico. Ele apenas prova
que intervenções podem falsificar `A -> B` e selecionar uma causa latente;
não demonstra descoberta científica geral nem superioridade sobre LLMs.

## NEXUS V4 experimental: plano de controle cognitivo

Arquivos: `src/core_v4/`.

A V4 preserva a V3 como referência e introduz o plano de controle da proposta
NEXUS, ainda sem integrar o loop, a TUI ou o formato `.axon` estáveis.

```text
request + sinais + capacidades disponíveis + candidatos
                         ↓
                 CognitiveScheduler
                         ↓
      estratégia + nível + orçamento máximo permitido
```

O scheduler não inventa uma ferramenta ou caminho de raciocínio: cada
`CandidateStrategy` declara capacidades necessárias, confiança esperada e
orçamento em eventos, bytes movidos e microjoules estimados. Candidatos sem
capacidade disponível ou acima do orçamento são rejeitados antes da escolha.
Ele privilegia uma resolução cognitiva baixa quando ela já é suficiente e só
autoriza `REFRAME` quando a evidência combina resíduo estruturado, persistência,
novidade e baixo ganho de adaptação.
Quando esse limiar é atingido, uma estratégia `REFRAME` viável tem precedência;
`Ask` só é fallback se não houver um `REFRAME` dentro das capacidades e do
orçamento declarados.

O quarto substrato, `ProceduralFabric`, compila apenas caminhos repetidamente
verificados e mantém guards. Um guard inválido devolve a execução ao caminho
geral por deoptimization.

`MemoryFirewall` e `ReversibleJournal` definem a proteção inicial para
aprendizado contínuo: conhecimento `Protected` recebe uma hipótese em ramo
separado, nunca alteração in-place; o journal preserva `before` e `after` para
rollback exato pela integração futura com os stores reais.

Os contadores da V4 ainda são **estimativas declaradas pelos candidatos**, não
medição de hardware. Portanto ela não demonstra economia de bytes, joules ou
tempo; isso exige instrumentação de runtime e benchmarks comparativos antes de
qualquer alegação de 10x, 100x ou 1000x. Codecs multimodais, handles zero-copy,
world model, homeostase, garbage collector e currículo permanecem fases
seguintes, guiadas por métricas e não por suposições.

## V5/Ω experimental: cognição multi-substrato orientada a informação

Arquivos: `src/core_v5/` e `src/experiments/v5_omega.rs`.

A V5/Ω mantém V3/V4 como referência e continua isolada do runtime estável. Ela
transforma a proposta de economia cognitiva em contratos testáveis: custos têm
proveniência (`Declared` ou `Measured`), o scheduler escolhe utilidade por custo
de compute/movimento/store/erase, ProgramCells só são promovidas após compressão
e holdout, e toda alteração reversível preserva `before`, `after` e provenance.

Também há uma população de modelos de mundo com diversidade estrutural e um
planejador de intervenções que separa previsões concorrentes sem receber a
classe-oráculo. O profiler compila trajetórias verificadas para macros
cognitivas e deotimiza sob guard inválido. O compilador físico é apenas uma
interface declarativa: operações exatas não podem ir para backend aproximado;
similaridade pode receber perfil HDC/analógico no futuro.

Nenhum dos custos V5 é telemetria física nesta fase. Resultados, limites e
comandos reproduzíveis estão em `docs/V5_OMEGA.md`.
