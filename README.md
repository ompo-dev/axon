# AXON

[![CI](https://github.com/ompo-dev/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/ompo-dev/axon/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/core-Rust-000000?logo=rust)](https://www.rust-lang.org/)
[![UI](https://img.shields.io/badge/UI-React%20%2B%20TypeScript-61dafb?logo=react&logoColor=111)](ui/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Inteligência adaptativa compacta baseada em seed, estado plástico e experiência

AXON é um projeto experimental de inteligência artificial geral portátil,
adaptativa e eficiente. A hipótese é combinar genoma herdável, estado plástico,
memória episódica, programas verificáveis e experiência para adquirir capacidades
transferíveis sem depender de um modelo monolítico com bilhões de parâmetros.

O objetivo de longo prazo é um sistema capaz de aprender, compreender, criar,
planejar, experimentar, adaptar-se a ambientes novos e operar ferramentas ou
corpos. O protótipo atual **não é uma LLM, não é uma AGI e não demonstrou
compreensão geral**. Essa distinção é parte do projeto.

> O documento canônico é [docs/AXON.md](docs/AXON.md). Ele contém histórico,
> contratos, resultados negativos, limites e o plano M0--M10. Este README é a
> apresentação técnica e o manual de entrada.

## 1. A ideia: seed como DNA, motor como organismo

```text
SEED / GENOMA = organização hereditária, módulos, priors e limites
MOTOR AXON     = interpretador, runtime, ambiente, aprendizado e evolução
```

A seed é análoga a um DNA computacional. Ela não precisa conter a resposta de
cada problema. Ela contém uma organização inicial capaz de produzir um
indivíduo, ativar módulos, receber experiências, mudar estado plástico e gerar
descendentes.

O motor executa bytes e sinais, coordena módulos, registra episódios, valida
consequências e decide quando uma mudança permanece no indivíduo ou pode ser
consolidada em uma descendência.

```text
seed → indivíduo → desafio → hipótese/ação → resultado externo
     → erro → experiência validada → estado plástico
     → descendente mutado → seleção por capacidade, custo e transferência
```

Mais bytes ou mais gerações não garantem inteligência. Uma representação que
não expressa uma relação precisa ser melhorada antes de ser ampliada.

## 2. Como o AXON funciona

```text
ambiente, arquivo, usuário ou sensor
  ↓
bytes/sinais + adaptador + formato + proveniência
  ↓
seed/genoma → Demand Runtime → módulo ativo
  ↓
estado plástico, memória de trabalho e contexto
  ↓
predição, programa, plano ou ação
  ↓
execução controlada e resultado independente
  ↓
journal episódico → validação → replay → retenção ou descendente
```

| Camada | Responsabilidade | Motivo |
| --- | --- | --- |
| Genoma/seed | Organização e capacidades herdáveis | Distribuição e evolução |
| Demand Runtime | Carga, ativação e orçamento | Estado ativo compacto |
| Estado plástico | Associação e adaptação individual | Aprendizado online |
| Journal | Eventos, origem, custo e falha | Experiência auditável |
| Programas | Procedimentos exatos | Matemática verificável |
| Controlador | Metas, rotas e feedback | Coordenação de módulos |
| Autoridade | Efeitos permitidos | Segurança e escopo |
| Evolução | Mutação, seleção e consolidação | Descendentes mensuráveis |

## 3. Bytes, tokens e representações

Texto, imagem, áudio, número e arquivo chegam ao computador como bytes ou sinais.
AXON não afirma que bytes sozinhos tenham significado. Adaptadores precisam
aprender ou definir formato, posição, regularidade, relação e contexto.

O projeto não usa tokens como unidade universal da memória central. Tokens,
palavras e subpalavras podem existir no módulo linguístico, mas a memória durável
é organizada como relações entre padrão, estado, consequência e evidência:

```text
entrada → padrão/relação → estado → consequência → experiência
```

Essa é uma hipótese de arquitetura, não uma prova de que AXON seja melhor que
Transformers. A vantagem só existirá se houver transferência, retenção e custo
menor em tarefas novas.

## 4. Ciclo cognitivo e plasticidade

O `CognitiveCycle` implementa observe--predict--verify--learn:

```text
necessidade → observação → recuperação → predição
→ execução → resultado externo → erro → journal
→ atualização plástica validada → replay, retenção ou descarte
```

A atualização é transacional:

```text
validada: estado atual → cópia → associação → journal persistido → promoção
rejeitada: estado atual → journal de rejeição → estado permanece intacto
```

Se o journal falha, o estado não é promovido. O AXON não deve usar a própria
previsão como rótulo; sem ambiente, oráculo ou critério externo, isso seria
apenas auto-confirmação.

## 5. Genoma, estado, memória e programas

| Elemento | Responsabilidade | Herança automática |
| --- | --- | --- |
| Genoma/seed | Organização e capacidades iniciais | Só após consolidação |
| Estado plástico | Adaptações do indivíduo | Não |
| Memória episódica | Eventos, fontes e evidências privadas | Não |
| Programa | Procedimento exato verificado | Só após auditoria |
| Checkpoint | Continuidade explícita | Conforme política |
| Descendente | Nova organização selecionada | Após admissão |

Experiências privadas não entram automaticamente na seed. Isso permite distribuir
a mesma seed para várias pessoas sem distribuir seus históricos pessoais.

Estado plástico favorece adaptação, mas pode esquecer. Journal favorece
proveniência, mas ocupa armazenamento. Programas dão exatidão, mas cobrem menos
casos. Carga sob demanda reduz estado ativo, mas não elimina I/O e inicialização.

## 6. Engenharia matemática

| Regime | Uso |
| --- | --- |
| Vetores limitados | Chaves, valores, associações e estado |
| Projeções quantizadas | Sinapses compactas |
| Regras delta/Hebbian | Atualização local validada |
| Programas numéricos | Soma, produto, raízes, distância e funções |
| Busca construtiva | Exploração de grafos aritméticos pequenos |
| Tolerâncias | Separação entre erro numérico e acerto |
| Verificador independente | Juiz separado do gerador |
| Seleção evolutiva | Comparação por mérito e custo |

O objetivo não é transformar toda inteligência em regras simbólicas. Matemática
exata é usada quando reduz erro; plasticidade é usada quando adaptação importa.
Operadores, gramáticas, formatos, limites e critérios ainda são fornecidos pela
engenharia em muitos experimentos. Isso limita a interpretação dos resultados.

## 7. Capacidades reais hoje

O repositório demonstra:

- cálculos exatos e programas numéricos delimitados;
- memória episódica com fatos, revisão, retração e conflitos;
- associações vetoriais aprendidas online;
- rejeição de experiências sem mutação plástica;
- persistência, replay e reconstrução após reinício;
- genomas, nascimento, descendentes e módulos sob demanda;
- texto e arquivos estruturados em tarefas específicas;
- evolução sem repetição de candidatos;
- autoridade, orçamento, proveniência e custos;
- campanhas de desenvolvimento editáveis em JSON.

Ainda não demonstrou conversação livre, linguagem geral, visão/áudio gerais,
descoberta matemática aberta, corpo físico geral, autoedição segura, desempenho
geral superior a LLMs ou AGI.

## 8. Treinamento e evolução

O protocolo é:

```text
desafio → previsão → execução → resultado independente
→ erro → journal → atualização validada → reserva nova
→ retenção/replay → possível descendente
```

Uma campanha pode criar seeds únicas, avaliar treino, manter uma reserva cega,
eliminar erros, selecionar por acerto/custo/latência/memória, gerar mutações e
repetir em desafios estruturalmente novos.

Mais gerações não criam uma representação ausente. A campanha relacional chegou
a 119/124 depois de exaurir 32.768 estruturas únicas. O próximo salto precisa
melhorar a representação composicional, não apenas aumentar o DNA.

### Desenvolvimento configurável em JSON

`src/development.rs` e `axon-uic-development-lab` oferecem tarefas de treino,
reserva independente, replay pelo maior erro, alvos externos validados, limites
de passes/trabalho/memória e parada `converged` ou `pass-limit`.

Plano atual: `evals/fixtures/development-scale-v1.json`.

## 9. Resultados experimentais

### Matemática composta

- 7/7 famílias admitidas;
- 7.000/7.000 casos de admissão corretos;
- famílias incluem quadrado, produto, hipotenusa, fórmula quadrática, raiz e
  distância 3D;
- gramática, operadores, busca e admissão foram fornecidos pelo laboratório.

Isso demonstra indução numérica delimitada, não descoberta matemática aberta.

### Treino plástico

Campanha `development-scale-v1`:

| Medida | Resultado |
| --- | ---: |
| Treino | 6 tarefas |
| Reserva inédita | 3 tarefas |
| Passes | 1 |
| Passos | 6 |
| Reserva inicial | 0/3 |
| Reserva final | 3/3 |
| MSE | 0,210000 → 0,000051 |

Isso comprova transferência de escala dentro da representação sináptica usada.
Não comprova linguagem, raciocínio geral ou aprendizado sem supervisão.

### Evolução relacional

- 32.768 seeds únicas;
- primeiro lote: 36/124;
- melhor final: 119/124;
- nenhum vencedor 124/124.

O negativo é vinculante: mais população não resolveu os cinco contraexemplos.

### Linguagem histórica

| Avaliação documentada | Respostas exatas | Intenções corretas | Cobertura |
| --- | ---: | ---: | ---: |
| Conjunto delimitado | 858/1.504 | 1.178/1.504 | 100% |
| Reserva independente | 3.333/6.491 | 4.739/6.491 | conforme protocolo |

Esses dados são tarefas estruturadas, não compreensão livre de português. Cobertura
significa saída ou recusa registrada, não acerto total.

### Integridade atual

- CognitiveCycle: 6/6;
- desenvolvimento: 4/4;
- bindings relacionais: 3/3;
- sessão estrutural: 11/11;
- UI: 23/23;
- Clippy com `-D warnings`: aprovado localmente.

## 10. Tamanho, velocidade e eficiência

Limites atuais de artefatos:

| Artefato | Limite |
| --- | ---: |
| Vetor de tarefa | 32 dimensões |
| Tarefas por conjunto | 8.192 |
| Modelo sináptico | 196.608 bytes |
| Seed semântica | 131.072 bytes |
| Genoma composto | 262.144 bytes |

Esses números não incluem processo, sistema operacional, UI, I/O, journal,
cache, memória de compilação ou auditoria.

FP16 usa cerca de 2 bytes por parâmetro e INT4 cerca de 0,5 byte antes de
overhead. Um modelo de 8B teria aproximadamente 16 GB em FP16 ou 4 GB em INT4
apenas nos pesos. AXON é muito menor porque sabe muito menos; isso não implica
capacidade equivalente.

Ainda não existe comparação end-to-end justa contra GPT, Claude, Codex, Llama,
Gemma, MiniMax, Mamba ou JEPA. Trabalho lógico não é tempo de CPU; watts e
joules ainda não têm medição confiável.

## 11. Comparação de arquiteturas

| Arquitetura | Força | Limitação | Relação com AXON |
| --- | --- | --- | --- |
| Transformer/LLM | Linguagem ampla e geração | Pesos e pré-treino grandes | Pipeline diferente |
| MoE | Especialização esparsa | Roteamento e parâmetros | Inspira modularidade |
| Mamba/SSM/RNN | Estado sequencial | Treino em escala | Inspira recorrência |
| JEPA/V-JEPA | Predição de representação | Não é agente geral sozinho | Inspira percepção/mundo |
| BDH | Plasticidade e estado esparso | Port local experimental | Inspira sinapses |
| Simbólica | Exatidão e verificação | Cobertura limitada | Base dos programas |
| AXON | Seed, estado, episódios e evolução | Generalização limitada | Arquitetura híbrida |

Não usar tokens como unidade fundamental é uma decisão arquitetural, não uma
prova contra Transformers. Imagens, áudio e ações também precisam de codificação,
decomposição e representação.

## 12. Critérios de promoção

Uma capacidade só pode ser promovida quando demonstrar:

1. transferência para entradas realmente novas;
2. reserva independente sem vazamento;
3. retenção após interferência;
4. replay fiel após reinício;
5. melhoria sobre estado congelado;
6. baseline apropriado;
7. custo, latência e memória reproduzíveis;
8. proveniência de fontes e rótulos;
9. nenhuma previsão usada como própria verdade;
10. herança sem transportar memória privada indevida.

## 13. Roadmap M0--M10

| Marco | Meta | Estado |
| --- | --- | --- |
| M0 | Seed, runtime, memória e integridade | Validado no escopo delimitado |
| M1 | Texto, matemática, composição e transferência | Aberto; 119/124 |
| M2 | Representações composicionais novas | Próximo após M1 |
| M3 | Transferência entre domínios | Planejado |
| M4 | Currículo adaptativo e aquisição ativa | Planejado |
| M5 | Ferramentas criadas sob verificação | Planejado |
| M6 | Mundo mutável, percepção e memória longa | Planejado |
| M7 | Corpo, sensores e affordances | Planejado |
| M8 | Evolução populacional em larga escala | Planejado |
| M9 | Portabilidade e seeds especializadas | Planejado |
| M10 | Avaliação geral independente | Meta de pesquisa |

O próximo grande salto é representar composicionalmente os cinco contraexemplos,
ligar isso ao ciclo plástico, repetir a reserva e só então avançar.

## 14. Estrutura do repositório

```text
src/              núcleo Rust, runtime, memória e laboratórios
tests/            contratos e testes independentes
evals/fixtures/   corpora, reservas, referências e schemas
docs/AXON.md      documento mestre e histórico completo
ui/               React + TypeScript + Bun + Tailwind + Motion
scripts/          auditorias e verificação de artefatos
examples/         exemplos operacionais
.github/workflows CI do core, UI e Docker
Dockerfile        imagem do observatório local
```

Relações principais: `Genome/Seed → DemandRuntime → módulos`; `CognitiveCycle
→ EpisodeStore → estado plástico`; `DevelopmentPlan → run_plan →
CognitiveCycle`; `Nursery → SemanticSeed → evolução`; `UI server → binários Rust
e relatórios allowlisted`.

## 15. Uso local

Requisitos: Rust stable, Bun 1.4+ e PowerShell no Windows ou shell equivalente.

```powershell
cargo check --features curriculum --tests
cargo test --features curriculum --tests
cargo clippy --features curriculum --all-targets -- -D warnings
```

Campanha JSON:

```powershell
cargo run --release --features curriculum --bin axon-uic-development-lab -- `
  evals/fixtures/development-scale-v1.json `
  .axon-memory/development-json/episodes.axep `
  .axon-memory/development-json/report.json
```

UI e observatório:

```powershell
cd ui
bun install --frozen-lockfile
bun run check
bun run build
bun run test
cd ..
```

Inicie o observatório com `pwsh -File scripts/start-observatory.ps1 -NoBuild`.
Ele exibe somente telemetria emitida pelo core. Dados ausentes ficam
indisponíveis; fixtures antigas não são atividade ao vivo.

## 16. Docker e CI/CD

```powershell
docker build -t axon:dev .
docker run --rm -p 4317:4317 axon:dev
```

O [Dockerfile](Dockerfile) compila Rust, constrói a UI e executa Bun. O
[`.dockerignore`](.dockerignore) exclui `target`, `node_modules`, `dist`,
`.axon-memory` e ZIPs históricos do contexto de build.

O workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) verifica core,
Clippy, contratos, UI e build Docker. Publicação não é automática: deploy exige
credenciais, ambiente e decisão explícita.

## 17. Desenvolvimento responsável

- Não usar previsão do AXON como rótulo do próprio treino.
- Não promover uma seed pelo mesmo conjunto que a treinou.
- Não transformar fixtures históricas em telemetria ao vivo.
- Não afirmar watts, latência ou equivalência a LLM sem medição end-to-end.
- Não remover resultados negativos.
- Não misturar memória privada com genoma sem auditoria.
- Não executar ferramentas fora da autoridade declarada.
- Preservar mudanças não relacionadas e preferir operações reversíveis.

## 18. Tags, licença e referências

Tags: `adaptive-ai`, `artificial-intelligence`, `cognitive-architecture`,
`general-intelligence-research`, `continual-learning`, `online-learning`,
`episodic-memory`, `plasticity`, `neuro-symbolic`, `program-synthesis`,
`evolutionary-computation`, `world-model`, `agent-architecture`, `rust`, `react`,
`typescript`, `bun`, `tailwindcss`, `motion`, `research-prototype`.

Distribuído sob [MIT](LICENSE). O documento mestre preserva referências a JEPA,
Mamba, BDH, programação por síntese, evolução, currículos e memória episódica.
Resultados de outros sistemas não são atribuídos ao AXON.

Para histórico completo, erros conhecidos, contratos e decisões de pesquisa,
leia [docs/AXON.md](docs/AXON.md).
