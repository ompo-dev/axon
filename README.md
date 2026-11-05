# AXON

[![CI](https://github.com/ompo-dev/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/ompo-dev/axon/actions/workflows/ci.yml) [![Rust](https://img.shields.io/badge/core-Rust-000000?logo=rust)](https://www.rust-lang.org/) [![UI](https://img.shields.io/badge/UI-React%20%2B%20TypeScript-61dafb?logo=react&logoColor=111)](ui/) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Inteligência adaptativa compacta baseada em seed, estado plástico e experiência.**

AXON é um projeto de pesquisa. Investiga uma inteligência portátil, modular e
adaptativa que aprenda com experiências verificadas, construa procedimentos e
carregue capacidades sob demanda. O protótipo atual **não é uma LLM, não é uma
AGI e não demonstrou compreensão geral**. Resultados válidos são os reproduzíveis
pelo código, fixtures e testes.

Documento canônico: [docs/AXON.md](docs/AXON.md). Este README é a entrada
operacional e técnica; não substitui os contratos, resultados negativos ou
limites registrados no documento mestre.

## Hipótese e arquitetura

```text
seed/genoma → organização herdável e módulos
motor       → interpretador, runtime e limites
estado      → associações plásticas do indivíduo
experiência → episódios, evidências e resultados
programas   → procedimentos exatos e verificáveis
descendentes→ seleção por transferência, custo e retenção
```

Fluxo: ambiente → bytes/sinais → adaptador/proveniência → seed/genoma → Demand
Runtime → módulo ativo → estado plástico → hipótese/programa/plano → execução
com autoridade limitada → resultado externo → journal → validação/replay →
retenção individual ou descendente.

| Camada | Responsabilidade | Decisão |
| --- | --- | --- |
| Genoma/seed | Identidade e organização carregável | Separado de episódios privados |
| Demand Runtime | Ativação e limites | Carga sob demanda |
| Estado plástico | Associação e adaptação | Atualização transacional |
| Journal | Origem, resultado, custo e falha | Replay só de experiências aceitas |
| Programas | Cálculo/procedimento exato | Verificação e serialização estáveis |
| Evolução | Mutação, seleção e descendência | Holdout e unicidade obrigatórios |
| Autoridade | Efeitos permitidos | Capacidades explícitas |

### Decisões importantes

O núcleo não trata tokens como memória central. Bytes são entrada física; memória
durável é formada por associações, estruturas, programas, episódios e
proveniência. Tokens e palavras podem existir dentro de um adaptador linguístico.
Isso é uma hipótese de arquitetura, não uma prova de superioridade.

Matematicamente, o núcleo combina vetores limitados, projeções sinápticas
quantizadas, associações delta/Hebbian, tolerâncias explícitas, programas
numéricos e verificadores independentes. A matemática simbólica dá exatidão; o
estado plástico dá adaptação. Adaptadores continuam necessários: bytes sozinhos
não contêm semântica.

## Capacidades reais hoje

- cálculos e programas numéricos em famílias delimitadas;
- memória episódica com fatos, revisão, retração, conflitos e replay;
- associações vetoriais aprendidas durante o uso;
- rejeição de experiências sem mutar o estado plástico;
- persistência, reconstrução, genomas, nascimento e descendentes;
- módulos sob demanda, texto e arquivos estruturados em tarefas específicas;
- evolução sem repetição, autoridade, orçamento, custo e proveniência;
- ciclo de desenvolvimento com treino ativo e reserva independente;
- campanhas definidas em JSON, sem editar Rust para trocar desafios.

`src/development.rs` e `axon-uic-development-lab` separam treino de reserva,
ordenam replay pelo maior erro, aceitam apenas alvos externos validados, medem a
reserva sem treiná-la e reportam convergência ou `pass-limit`. Isso é aprendizado
plástico supervisionado e auditável; ainda não é autoaprendizado aberto.

### Resultados recentes

- Matemática composta: **7/7 famílias** e **7.000/7.000** casos de admissão.
  Gramática, operadores, busca e admissão foram fornecidos pelo laboratório.
- Relacional: **32.768 seeds únicas**, melhor **119/124**; mais DNA não criou a
  representação semântica que faltava.
- Contratos: ciclo cognitivo **6/6**, desenvolvimento **3/3**, sessão estrutural
  **11/11** e Clippy sem warnings na configuração validada.

Os recibos ficam em `.axon-memory/`, local e ignorado pelo Git. Escopos,
resultados históricos e negativos estão em [docs/AXON.md](docs/AXON.md).

## Treinamento e evolução

Protocolo: desafio → previsão → execução → resultado independente → erro →
journal → atualização validada → reserva nova → retenção/replay → descendente.
Populações podem mutar seeds e selecionar por correção, custo, latência, memória,
estabilidade e transferência. Aumentar dígitos não compensa representação
incapaz. O AXON nunca deve transformar a própria previsão em rótulo de treino.

Plano JSON: `evals/fixtures/development-scale-v1.json`. Execução: `cargo run
--release --features curriculum --bin axon-uic-development-lab -- <plan.json>
<journal.axep> [report.json]`.

## Comparação

GPT/Claude/Codex têm linguagem ampla e escala de pré-treino muito maior; Llama,
Gemma e MiniMax oferecem modelos locais pré-treinados; Mamba/SSM exploram estado
sequencial; JEPA/V-JEPA prevê representações; BDH é referência para plasticidade
esparsa. AXON combina genoma, plasticidade, episódios, programas e descendentes,
mas ainda tem linguagem aberta e generalização limitadas.

Comparar 262 KB de payload com bilhões de pesos seria enganoso: AXON é menor
porque sabe muito menos. Ainda não há benchmark justo contra esses sistemas que
meça a mesma tarefa, latência end-to-end, memória, energia, qualidade e custo de
adaptação.

## Estrutura e operação

`src/` contém o núcleo Rust; `tests/` contratos; `evals/fixtures/` corpora e
reservas; `docs/AXON.md` o documento mestre; `ui/` o observatório React,
TypeScript, Bun, Tailwind e Motion; `scripts/` auditorias; `Dockerfile` o
empacotamento local.

Com Rust stable: `cargo check --features curriculum --tests`, `cargo test
--features curriculum --tests` e `cargo clippy --features curriculum
--all-targets -- -D warnings`. Na UI: `cd ui; bun install --frozen; bun run
check; bun run build; bun run test`. Para observar localmente: `pwsh -File
scripts/start-observatory.ps1 -NoBuild`.

Docker: `docker build -t axon:dev .` e `docker run --rm -p 4317:4317 axon:dev`.
O [Dockerfile](Dockerfile) compila Rust, constrói a UI e executa o servidor Bun.
O workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) verifica core,
contratos, Clippy, UI e imagem Docker. Deploy não é automático por padrão.

O observatório só apresenta telemetria emitida pelo core; dados ausentes ficam
indisponíveis. Fixtures antigas não são atividade ao vivo.

## Princípios e tags

Não usar previsão como rótulo; não promover pela mesma reserva do treino; não
chamar fixtures de telemetria real; não afirmar watts ou equivalência a LLM sem
medição end-to-end; preservar resultados negativos e dados não relacionados.

Tags: `adaptive-ai`, `artificial-intelligence`, `cognitive-architecture`,
`continual-learning`, `online-learning`, `neuro-symbolic`,
`evolutionary-computation`, `episodic-memory`, `plasticity`, `program-synthesis`,
`rust`, `react`, `typescript`, `bun`, `tailwindcss`, `motion`, `research-prototype`.

## Licença

Distribuído sob [MIT](LICENSE). Para referências, resultados, limites e o que o
projeto ainda não demonstrou, leia [docs/AXON.md](docs/AXON.md).
