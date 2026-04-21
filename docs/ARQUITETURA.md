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
