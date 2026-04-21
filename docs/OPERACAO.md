# Operacao e estrategias de uso

## 1) Bootstrap de ambiente

Pre-requisitos:
- Rust instalado e acessivel no `PATH`.
- Diretorio de trabalho com permissao de escrita.

Comandos iniciais:

```bash
cargo check
cargo test
```

## 2) Criar e iniciar um cerebro

Criar (se nao existir) e entrar na TUI:

```bash
cargo run -- tui --brain .\base.axon --create-if-missing --mode stochastic
```

Abrir cerebro existente:

```bash
cargo run -- tui --brain .\base.axon --mode deterministic
```

Quando usar cada modo:
- `deterministic`: reproducao e depuracao.
- `stochastic`: variacao controlada e exploracao.

## 3) Ingestao de conhecimento

Formato recomendado para dicionario:
- blocos com `lema: definicao`.
- linhas em branco separando entradas.

Ingestao offline:

```bash
cargo run -- ingest --brain .\base.axon --dict .\meu_dicionario.txt
```

Ingestao na inicializacao da TUI:

```bash
cargo run -- tui --brain .\base.axon --create-if-missing --dict .\meu_dicionario.txt --mode stochastic
```

## 4) Estrategia de multiplos `.axon`

Padrao pratico:
1. Evoluir `base.axon` com fundamento geral.
2. Copiar para especialidades.
3. Treinar cada copia com corpus e dialogo especifico.

Exemplo:
- `base.axon`
- `coder.axon`
- `creative.axon`

Cada arquivo fica independente apos a copia.

## 5) Operacao diaria (runbook)

Antes de sessao:
1. `cargo check`
2. `cargo run -- verify --brain <brain.axon>`

Durante sessao:
1. Conversar normalmente.
2. Usar `/corrigir errado|certo` para ajustes de vies.
3. Usar `F5` ou `/f5` se quiser forcar flush.
4. Usar `F6` ou `/f6` para checkpoint imediato.

Apos sessao:
1. Encerrar com `/quit` ou `Ctrl+C`.
2. Rodar `inspect` e `verify`.

Manutencao periodica:

```bash
cargo run -- compact --brain .\base.axon
```

## 6) Diagnostico e observabilidade

Comandos utilitarios:

```bash
cargo run -- inspect --brain .\base.axon
cargo run -- dump-header --brain .\base.axon
cargo run -- dump-region --brain .\base.axon --region journal
cargo run -- verify --brain .\base.axon
```

Interpretacao rapida:
- `inspect`: visao geral de geracao, paginas e regioes.
- `dump-header`: metadados e estado do superblock.
- `dump-region`: amostra de paginas por regiao.
- `verify`: valida checksums/header de paginas.

## 7) Estrategias cognitivas de treino

Para consolidar associacoes:
- repetir contexto com variacoes controladas.
- introduzir temporalidade explicita (`hoje`, `ontem`, `amanha`).
- corrigir erros de proximidade com `/corrigir`.

Para evitar ruido:
- manter janelas de treino com tema unico.
- usar arquivos `.axon` separados por dominio.
- nao misturar corpus de dominios incompatveis no mesmo cerebro quando o objetivo for especializacao.

Para maior reprodutibilidade:
- usar `--mode deterministic`.
- comparar comportamento antes/depois de ingestoes.
- manter snapshots/backup por data.

## 8) Backup e recuperacao

Backup manual simples:

```bash
copy .\base.axon .\backup\base_YYYYMMDD.axon
```

Recuperacao:
- se `verify` falhar, restaurar ultimo backup valido.
- rodar `inspect` no backup antes de voltar para producao.

## 9) Problemas comuns

`os error 2`:
- caminho de `--brain` ou `--dict` incorreto.

`Acesso negado (os error 5)` em build:
- `axon.exe` ainda em execucao.
- encerrar processo ativo antes de recompilar.

Linux sem comportamento de editor em tempo real:
- esperado na versao atual por fallback de input por linha.

## 10) Politica de qualidade recomendada

Checklist minimo para alterar o core:
1. `cargo fmt`
2. `cargo check`
3. `cargo test`
4. `cargo run -- verify --brain <arquivo_teste.axon>`
5. teste manual curto em `tui` (chat + slash + quit)
