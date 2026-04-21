# Axon v3.2 (Rust `zero-crates`)

Axon e um runtime local de IA associativa online com estado persistido em arquivo unico `.axon`.
O executavel em Rust funciona como gerenciador (TUI, IO, persistencia, limites de execucao). O estado cognitivo vive no proprio `.axon`.

## Objetivo do projeto

- IA com aprendizado em tempo de execucao.
- Sem arquitetura de next-token como objetivo central.
- Memoria, conceito e correlacao no mesmo grafo dinamico.
- Persistencia autocontida (`single-file`) com journal + snapshot.
- Operacao local, sem dependencia de framework externo.

## Estado atual (v3.2)

- Runtime com loop de ticks (100 Hz), persistencia assincrona e TUI.
- Grafo unificado em memoria (`UnifiedGraph`) com temperatura, amplitude e fase.
- Slash commands tratados no gerenciador e isolados do lane cognitivo.
- Snapshot v3 com compatibilidade de leitura v1/v2.
- Renderer diferencial na TUI para reduzir flicker perceptivel.

## Requisitos

- Rust toolchain estavel (`cargo`, `rustc`).
- Windows ou Linux.
- CUDA opcional.

Observacao sobre plataforma:
- No Windows, a TUI usa modo raw nativo por FFI.
- No Linux, o modo raw ainda nao esta implementado; a entrada cai em fallback por linha.

## Build e execucao

Build:

```bash
cargo build
```

Execucao do binario em desenvolvimento:

```bash
cargo run -- <comando> [args]
```

## Inicio rapido

1. Criar/abrir um cerebro e entrar na TUI:

```bash
cargo run -- tui --brain .\base.axon --create-if-missing --mode stochastic
```

2. Opcional: carregar dicionario na inicializacao:

```bash
cargo run -- tui --brain .\base.axon --create-if-missing --dict .\meu_dicionario.txt --mode deterministic
```

3. Importar dicionario fora da TUI:

```bash
cargo run -- ingest --brain .\base.axon --dict .\meu_dicionario.txt
```

4. Verificar integridade:

```bash
cargo run -- verify --brain .\base.axon
```

## Comandos CLI

```bash
axon tui --brain <path.axon> [--create-if-missing] [--dict <dict.txt>] [--mode deterministic|stochastic]
axon ingest --brain <path.axon> --dict <dict.txt>
axon inspect --brain <path.axon>
axon dump-header --brain <path.axon>
axon dump-region --brain <path.axon> --region <semantic|memory|cortex|journal|obs>
axon verify --brain <path.axon>
axon compact --brain <path.axon>
```

Semantica dos principais comandos:
- `tui`: conversa e evolucao online.
- `ingest`: injeta entradas de dicionario no grafo unificado.
- `inspect`: resumo de paginas e regioes.
- `verify`: varredura de checksums/paginas.
- `compact`: reescrita compactada do estado vivo.

## Uso da TUI

Atalhos de teclado:
- `F1`: modo `Chat`
- `F2`: modo `Observatory`
- `F5`: force flush de journal
- `F6`: force checkpoint
- `Tab`: autocomplete de slash
- `Up/Down`: historico ou navegacao no menu slash
- `Ctrl+C`: sair com shutdown seguro

Slash commands:
- `/f1`
- `/f2`
- `/f5`
- `/f6`
- `/mode deterministic`
- `/mode stochastic`
- `/corrigir errado|certo`
- `/help`
- `/quit`

Regra de firewall:
- Qualquer entrada iniciada por `/` e tratada localmente no gerenciador.
- Comandos slash nao entram no processamento cognitivo.

## Formato do arquivo `.axon`

Propriedades:
- Arquivo unico autocontido.
- Little-endian.
- Paginas de 4096 bytes.
- Superblock A/B (ping-pong) nas paginas 0 e 1.
- Dados a partir da pagina 2.
- Checksum por pagina e por superblock.

Tipos de pagina:
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

Snapshot:
- Gravacao atual em versao `v3`.
- Leitura compativel com `v1`, `v2`, `v3`.

## Dicionario TXT estruturado

Formato minimo:

```txt
abelha: inseto polinizador social que pode produzir mel.
antissocial: comportamento de oposicao a convivencia social.

algoritmo: sequencia finita de passos para resolver um problema.
```

Regras:
- Entradas separadas por linha em branco.
- Primeira linha pode estar em `lema: definicao`.
- Linhas adicionais entram como parte da definicao.

## Estrategia de uso com multiplos cerebros

Modelo recomendado:

1. Criar um cerebro base (`base.axon`).
2. Alimentar com conhecimento comum.
3. Copiar arquivo para especializacao manual:
- `copy base.axon coder.axon`
- `copy base.axon creative.axon`
4. Evoluir cada `.axon` de forma independente.

Cada arquivo preserva seu proprio historico, correlacoes e temperatura de arestas/nos.

## Arquitetura (resumo)

Modulos:
- `src/main.rs`: bootstrap e dispatch CLI.
- `src/cli`: parse de argumentos.
- `src/runtime`: orquestracao, ticks, threads e persistencia.
- `src/cortex`: dinamica de ativacao/plasticidade/geracao de resposta.
- `src/memory`: grafo associativo unificado (nos/arestas/temperatura/fase).
- `src/storage`: journal, snapshot, superblock e scan de paginas.
- `src/axon_format`: layout binario de superblock/pagina/checksum.
- `src/tui`: input editor, slash menu, renderer diferencial.
- `src/gpu`: probe CUDA via Driver API.
- `src/inspect`: comandos de diagnostico.

Detalhamento tecnico completo:
- [docs/ARQUITETURA.md](C:/Projects/Teste/axon/docs/ARQUITETURA.md)
- [docs/OPERACAO.md](C:/Projects/Teste/axon/docs/OPERACAO.md)

## Testes e verificacao

Checks recomendados:

```bash
cargo fmt
cargo check
cargo test
cargo run -- verify --brain .\base.axon
```

## Troubleshooting

Erro `os error 2` (arquivo nao encontrado):
- Verifique caminho de `--brain` e `--dict`.
- Se o cerebro nao existe, use `--create-if-missing`.

Erro de lock no binario (`os error 5` no Windows):
- Feche processos `axon.exe` ativos antes de rebuild.

Muitos warnings de `dead_code`:
- Nao bloqueiam execucao.
- Sao pontos pendentes de limpeza/refino.
