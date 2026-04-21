# Axon (Rust, zero-crates)

Implementacao inicial do plano Axon v3: runtime associativo online, arquivo `.axon` paginado, TUI com chat/observatorio, journal + snapshot, fallback CPU quando CUDA nao estiver disponivel.

## Comandos

```bash
axon tui --brain <caminho.axon> --create-if-missing --dict <dict.txt> --mode deterministic|stochastic
axon ingest --brain <caminho.axon> --dict <dict.txt>
axon inspect --brain <caminho.axon>
axon dump-header --brain <caminho.axon>
axon dump-region --brain <caminho.axon> --region semantic|memory|cortex|journal|obs
axon verify --brain <caminho.axon>
axon compact --brain <caminho.axon>
```

No TUI:
- `/f1` modo chat
- `/f2` modo observatorio
- `/f5` force flush
- `/f6` force checkpoint
- `/quit` sair

## Formato `.axon`

- arquivo unico (single-file)
- little-endian
- pagina de 4096 bytes
- superblocos ping-pong nas paginas 0 e 1
- dados a partir da pagina 2
- checksum por pagina e por superbloco

Tipos de pagina suportados:
- `FREE`
- `META`
- `ASSEMBLY_NODE`
- `EDGE_CSR`
- `EDGE_DELTA`
- `EPISODE`
- `CONCEPT`
- `JOURNAL`
- `OBS_TILE`
- `ALLOC_MAP`

## Dicionario TXT estruturado

Exemplo:

```txt
abelha: inseto polinizador social que pode produzir mel.
antissocial: comportamento de oposicao a convivencia social.

algoritmo: sequencia finita de passos para resolver um problema.
```

Entradas sao separadas por linha em branco. A primeira linha pode usar `lema: definicao`.

## Arquitetura de codigo

- `src/main.rs`: bootstrap e dispatch CLI
- `src/cli`: parse manual de argumentos
- `src/axon_format`: superbloco/header/pagina/checksum
- `src/storage`: pager, append journal, snapshot, verify scan
- `src/cortex`: campo associativo, hebb+oja, spawn/merge/prune
- `src/memory`: episodica curta + decaimento/reforco
- `src/semantic`: ingestao de dicionario e conceitos
- `src/runtime`: threads (`Input`, `TickEngine`, `Persist`, `Render`, `Telemetry`)
- `src/tui`: input/render para chat e observatorio
- `src/gpu`: probe CUDA via FFI dinamico
- `src/inspect`: inspect/dump/verify
- `src/platform`: caps de recurso (RAM/CPU)

## Estado atual

- implementado: formato, runtime, comandos e persistencia basica
- implementado: replay por journal e snapshot consolidado
- implementado: compactacao por reescrita do estado vivo
- implementado: fallback CPU automatico
- pendente para proxima fase: kernels CUDA reais (`K1/K2/K3`), locking leitor-escritor mais forte, observatorio grafico mais rico, harness de crash automatizado
