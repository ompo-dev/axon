# AXON core v2 — núcleo cognitivo composicional (VSA/HDC)

Protótipo do NOVO núcleo cognitivo do axon. Roda: `python vsa_core.py` (numpy só).

## Por que trocar o núcleo (diagnóstico honesto da v3.2)
A v3.2 (`src/memory/mod.rs`) é uma **rede semântica de ativação-espalhada** com ligação por
fase. Ela **associa e recupera** (X perto de Y), mas **não compõe, não consulta estrutura, não
raciocina por analogia**. É co-ocorrência de palavras — um paradigma de 1980. Sozinha não vira
raciocínio, muito menos "cérebro".

## O que é o v2: Arquitetura Vetorial Simbólica (VSA / Hyperdimensional Computing)
Conceitos = hipervetores de D=10000 dims (±1), quase-ortogonais. Álgebra:
- **bind** (⊗, mult): liga papel↔valor → `cor⊗vermelho`. Auto-inverso.
- **bundle** (+, maioria): superpõe fatos numa cena → `(cor⊗vermelho)+(forma⊗redonda)`.
- **permute** (ρ, rotação): codifica ordem/sequência.
- consultar = unbind; analogia = mapa de transformação entre estruturas.

Referências reais: Plate (HRR, 1995); Kanerva (Hyperdimensional, 2009); **Eliasmith SPAUN
(2012)** — um modelo de cérebro que funciona com isto.

## Resultados (o que a v3.2 NÃO faz)
| capacidade | resultado |
|---|---|
| composição + consulta | `cor de maçã → vermelho`, `classe de maçã → fruta` |
| sequência (ordem) | `posição 2 do alfabeto → c` |
| **analogia relacional** (1 exemplo) | `paris:frança :: ?:itália → roma`; `euro:frança :: ?:japão → iene` |
| robustez | consulta funciona com 20% dos bits corrompidos |

## Por que bate TODAS as suas restrições
- **Compõe e raciocina** (não só associa) — o buraco da v3.2.
- **Aprende em 1 exemplo, online, sem gradiente** — mantém a força "tempo real" do axon.
- **Ultra-leve** — só add/mult de vetores; roda em CPU/microcontrolador; sem treino pesado.
- **Cérebro-inspirado de verdade** (memória distribuída, robusta, Eliasmith/Kanerva).
- **Fresco** vs transformers (investimento em não-transformer +400% em 2 anos).

## A síntese com o axon existente (o roadmap)
O axon já tem o que falta na VSA pura: **dinâmica oscilatória, temperatura/atenção, loop de
tick 100Hz, persistência single-file**. Une-se assim:
- **VSA** = a REPRESENTAÇÃO composicional (o "conteúdo do pensamento").
- **Oscilação/temperatura do axon** = a ATENÇÃO/CONTROLE (o que ativar, ligar, esquecer, quando).
- **Tick loop** = o "pensar em tempo real"; **.axon** = a memória persistente.

## Limites honestos (sem virar CHRONOS)
- **NÃO é geração de linguagem, NÃO é AGI, NÃO é o "pulo" abdutivo** (LLMs can't jump —
  ninguém tem isso). É raciocínio COMPOSICIONAL, um degrau real, não o topo.
- Capacidade da VSA é limitada: bundle de coisas demais → ruído. Precisa de memória de limpeza
  + estrutura hierárquica (feito parcialmente; escalar é trabalho real).
- Falta: camada de linguagem, encoders de percepção (texto/áudio/vídeo → hipervetor), e um
  sinal que decida QUANDO ligar/agrupar (aprendizado da estrutura, não só dos símbolos).

## Degrau 2 (feito): aprendizado contínuo, na surpresa, sem esquecer
`predictive_cls.py` — fundamentado em neurociência REAL (pesquisada):
- **Predictive coding (Friston):** aprende só do ERRO de predição → liga na surpresa, não
  re-aprende o sabido.
- **Complementary Learning Systems (McClelland-McNaughton-O'Reilly 1995):** hipocampo (rápido,
  1-shot, episódios separados) + neocórtex (lento, consolidado). Dois armazéns = sem
  esquecimento catastrófico.
- **Binding por sincronia gama (fase):** a `bind` ⊗ é a versão algébrica; a fase do axon é o
  controle de qual estrutura está ativa (PLOS Comp Biol: oscilação = múltiplas memórias).

Resultados (coisas que LLM NÃO faz no uso):
| capacidade | resultado |
|---|---|
| aprender 1-shot em stream, **sem esquecer** | 100% até ~80 fatos; 89% em 160; degrada suave (capacidade, não esquecimento) |
| **aprender só na surpresa** (predictive coding) | 2ª passada de 30 fatos conhecidos: +0 aprendido, 30 pulados |
| **atualizar em tempo real** | `capital_franca: paris → lyon` na hora |

Limite honesto: capacidade ~100 fatos por bundle antes de saturar; CLS adia, não elimina.
Escalar = memória esparsa distribuída de Kanerva (SDM) + hierarquia. Trabalho real, não mágica.

## Fundamentação (neurociência + IA, pesquisada)
- Binding por sincronia gama 30–80Hz (temporal binding; Singer, von der Malsburg).
- VSA/HDC: Plate HRR 1995; Kanerva 2009; Eliasmith SPAUN 2012 (modelo de cérebro funcional).
- CLS: McClelland-McNaughton-O'Reilly 1995 (hipocampo/neocórtex, anti-esquecimento).
- Predictive coding: Friston (o cérebro prevê; aprende do erro).
- "LLMs can't jump" (Zahavy/DeepMind, ICML 2026): falta o pulo abdutivo — ninguém tem; isto
  é raciocínio composicional (degrau real), não o topo.

## Próximos degraus (cada um testável)
1. **Encoder de texto → hipervetor** (n-gramas/roles) para ingerir linguagem de verdade.
2. **Memória hierárquica** (cenas de cenas) + cleanup escalável (SDM de Kanerva).
3. **Controle oscilatório do axon** dirigindo bind/ativação (a ponte VSA↔axon).
4. **Multimodal**: mesmo espaço hiperdimensional para áudio/imagem (encoders → mesmos D dims).
5. Portar o núcleo estável pro Rust do axon.

## Veredito honesto
Isto é uma fundação cognitiva **real, leve, composicional e nova-o-bastante** — muito melhor
que a v3.2 pro seu objetivo. Não é o cérebro humano (ninguém sabe fazer). É o primeiro degrau
sólido, e cada degrau seguinte é testável. É ambição honesta, não promessa vazia.
