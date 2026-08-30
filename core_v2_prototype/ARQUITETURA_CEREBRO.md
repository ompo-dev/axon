# Plano máximo — cérebro digital ultra-leve (arquitetura honesta)

Objetivo: um sistema que vê/ouve, aprende, adapta, lembra, compõe novo do que tem, testa,
"pensa" e "imagina" — pesando quase nada em disco E pra rodar. Ancorado em números reais
(ver `efficiency_1bit.py`) e neurociência (VSA/Kanerva/Eliasmith, CLS/McClelland, predictive
coding/Friston, binding-por-gama).

## Princípio único (por que é leve E geral)
**TUDO é um hipervetor de 1 bit (D≈10000). TODA cognição é XOR / bundle / permute / popcount.**
Uma representação, uma álgebra — para percepção, memória, raciocínio, todas as modalidades.
Não há "camadas de pesos" gigantes; há vetores esparsos e operações de inteiros.

Por que o cérebro é eficiente (e o que copiamos):
| princípio do cérebro | nossa versão | ganho |
|---|---|---|
| esparsidade (~2% dispara) | hipervetores esparsos / SDM | energia ∝ ativos, não total |
| evento-dirigido (só no spike) | computa só na entrada/surpresa (predictive coding) | não gasta à toa |
| 1-bit ruidoso | bits ±1, XOR/popcount | 8–32× menor que float, sem GPU |
| aprendizado local (sem backprop) | bind/Hebbian 1-shot | sem guardar ativações, sem treino |
| memória = estrutura endereçável | cleanup por conteúdo | recupera do parcial |
| compõe de poucos primitivos | bind/bundle | infinito de finito |
| esquece de propósito | temperatura/decay (axon) + consolidação CLS | espaço constante |

## Orçamento real (do benchmark, não promessa)
- 1 conceito/fato/cena/percepção = **1.22 KB**.
- 1 milhão de conceitos = **1.2 GB** (RAM/disco comum). LLM 7B = 14 GB e não aprende no uso.
- bind = XOR (~800 ns em Python, ~ns em Rust/SIMD). Sem matmul, sem GPU.
- Gargalo: **cleanup** (busca do mais próximo) é O(N). Solução: SDM de Kanerva / LSH → sublinear.

## As 3 camadas (o desenho)
```
  PERCEPÇÃO            COGNIÇÃO (o núcleo)                 CONTROLE
  texto ─┐        ┌── VSA: bind/bundle/permute ──┐     oscilação/fase (axon)
  áudio ─┼─encoders┤   memória CLS (rápida+lenta) ├──   temperatura/atenção
  imagem─┘  →hiper │   predictive coding (surpresa)│     tick 100Hz (tempo real)
           vetor   └── cleanup (SDM indexado) ─────┘     .axon (persistência)
```
- **Percepção (CORRIGIDO — princípio do bebê):** o axon NÃO sabe o que é texto/áudio/vídeo e
  NÃO recebe features prontas (dar 'assento, pernas' seria semântica humana injetada = cheating).
  Ele recebe um STREAM CRU de valores (bytes/samples/pixels — não sabe qual), atribui um
  hipervetor aleatório a cada símbolo cru (sem significado), e DESCOBRE a estrutura pela
  ESTATÍSTICA (probabilidade de transição) — como bebês segmentam fala contínua em palavras
  (Saffran-Aslin-Newport 1996). Modality-agnostic: o MESMO código acha estrutura em qualquer
  stream. Feito em `percepcao_crua.py` (áudio-sintético: motifs recuperados 100%; texto:
  fragmentos certos, F1~0.34 com corpus minúsculo — melhora com mais exposição + chunking por
  merge). Estrutura EMERGE; nada é dado à mão.
  NOTA: `senso_comum.py` (protótipos de Rosch) usava features à mão — é ilustração da teoria de
  categorias graduadas, NÃO o caminho real. O real é: as features/protótipos EMERGEM da
  percepção crua acima, depois entram no núcleo cognitivo.
- **Cognição:** o que já construímos (`vsa_core.py` compõe/raciocina; `predictive_cls.py`
  aprende contínuo sem esquecer). Escala = SDM.
- **Controle:** a oscilação/fase do axon decide QUAL estrutura está ativa agora (binding por
  sincronia gama), a temperatura decide o que esquecer, o tick dá o "pensar em tempo real".

## Mapa capacidade → mecanismo (o que o usuário pediu)
| capacidade | mecanismo concreto | estado |
|---|---|---|
| **vê/ouve** | encoders modalidade→hipervetor (mesmo espaço) | a fazer (degrau 3) |
| **aprende** | bind 1-shot | ✅ feito |
| **adapta** | temperatura/decay + update-on-surprise | ✅ feito |
| **lembra** | cleanup endereçável por conteúdo + SDM | parcial (falta SDM) |
| **compõe novo do que tem** | bind/bundle/analogia (cria estruturas não vistas) | ✅ feito |
| **testa/experimenta** | laço predictive: compõe hipótese → prevê → compara → mantém se bate | a fazer (degrau 4) |
| **pensa** | sequência de ativações no grafo modulada por fase (trem de pensamento) | a fazer |
| **imagina** | compor hipervetores NÃO observados e "limpar" pra ver o que está perto (simulação mental) | a fazer, limitado |

## O gargalo e a solução (honesto)
Cleanup O(N) é o custo dominante. **Sparse Distributed Memory (Kanerva 1988):** endereços
esparsos + ativação por raio de Hamming → recuperação sublinear, tolerante a ruído, capacidade
enorme, e ainda mais leve (esparso). É o próximo item de engenharia sério.

## Escopo HONESTO (a linha anti-CHRONOS)
REAL e alcançável:
- Percepção multimodal unificada em 1 espaço leve; aprendizado contínuo 1-shot sem esquecer;
  composição/analogia; "imaginação" = recombinação composicional; tudo em MB, sem GPU.

NÃO alcançável (não prometer):
- **Consciência, cérebro humano completo, o "pulo" abdutivo** (LLMs can't jump — ninguém tem).
- Percepção profunda de vídeo no nível de uma CNN grande SEM nenhum front-end aprendido: a
  projeção aleatória captura similaridade, não features profundas. Perceber MUITO bem talvez
  exija um encoder pequeno aprendido (adiciona algum peso). O NÚCLEO cognitivo continua 1-bit
  minúsculo; o custo, se houver, fica no front-end de percepção.
- "Pensa/imagina" no sentido humano — temos recombinação e simulação LIMITADAS, não uma mente.

## Roadmap priorizado (cada passo testável, leve)
1. **SDM (Kanerva)** — cleanup sublinear + capacidade grande. Resolve o gargalo. (fundacional)
2. **Encoder de texto → hipervetor** — ingerir linguagem real (não fatos à mão).
3. **Laço predictive de teste/imaginação** — compor hipótese, prever, comparar, consolidar.
4. **Ponte fase-do-axon ↔ binding** — multiplexar várias estruturas ativas por fase (gama).
5. **Encoders áudio/imagem** — mesmo espaço D-bit → multimodal de verdade.
6. **Porta o núcleo estável pro Rust do axon** (SIMD → ns por op, persistência .axon).

## Veredito
Um cérebro digital "que faz tudo" no sentido humano = não existe e não vou fingir. Mas um
**sistema leve (MB, sem GPU) que percebe multimodal, aprende contínuo sem esquecer, compõe e
recombina, e roda em qualquer PC** é REAL, matematicamente ancorado, e cada degrau é testável.
Esse é o caminho honesto do "quase não pesa" — e já provamos os números.
