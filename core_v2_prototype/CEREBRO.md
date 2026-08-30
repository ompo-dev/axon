# AXON cérebro integrado — o conceito, funcionando ponta a ponta

`cerebro.py` — liga as peças num sistema só. Roda: `python cerebro.py` (numpy; selftest incluso).

## O que FUNCIONA (provado, rodando)
Um "bebê digital" que:
1. **recebe uma sequência de unidades** (percepção) e **aprende as transições** — 1-shot, online;
2. **gera**: dado um seed, continua com o que aprendeu (`obebe → pega`, `papai → fala com`);
3. **aprende novo em tempo real SEM esquecer**: ensinar "vovo conta historia" ao vivo não apaga
   o que já sabia, e o novo já funciona na hora;
4. **pesa KB**: 15 conceitos = 18 KB. Zero GPU, zero treino por gradiente.

Isto é o núcleo da "revolução leve": **vivo (aprende no uso), composicional, minúsculo** — o
oposto de um LLM (GBs de pesos congelados que não aprendem no uso).

## As peças (cada uma testada isolada + agora ligadas)
| módulo | papel | ciência |
|---|---|---|
| `percepcao_crua.py` | stream cru → estrutura emergente | Saffran 1996 (aprendizado estatístico infantil) |
| `vsa_core.py` | representação composicional (bind/bundle/analogia) | Plate/Kanerva/Eliasmith |
| `predictive_cls.py` | aprender contínuo, na surpresa, sem esquecer | Friston (predictive coding) + CLS (McClelland 1995) |
| `senso_comum.py` | protótipos graduados (categorias) | Rosch 1975 |
| `efficiency_1bit.py` | 1.25 KB/conceito, XOR/popcount, sem GPU | Kanerva HDC |
| `cerebro.py` | **o loop integrado: percebe → aprende → gera → contínuo** | — |

## Limites HONESTOS (a linha anti-CHRONOS)
- **Geração é 1ª ordem** (só palavra-atual → próxima) → fica repetitiva ("obebe pega obebe
  pega"). Falta CONTEXTO. Próximo passo real: contexto de ordem-n via bind posicional / fase
  (a ponte com a oscilação do axon — bind por sincronia gama segura várias unidades ativas).
- **Segmentação de stream cru GAPLESS** (texto sem espaços) é sub-problema aberto: meu
  heurístico acha estrutura em streams com gaps (áudio: motifs 100%) mas falha em char-corpus
  adversarial. Fronteira por entropia funciona quando há um separador claro.
- **NÃO é entendimento profundo, NÃO é raciocínio abdutivo** ("LLMs can't jump" — ninguém tem).
  É um aprendiz de sequências composicional, vivo e leve — uma fundação NOVA, não o topo.

## Roadmap (cada passo testável)
1. **Contexto ordem-n** (bind posicional + fase do axon) → geração coerente, não repetitiva.
2. **SDM (Kanerva)** → cleanup sublinear, capacidade de milhões (destrava escala).
3. **Segmentação robusta** de stream cru (chunking hierárquico + predição).
4. **Multimodal**: áudio/imagem no mesmo espaço 1-bit (encoders → mesmos D bits).
5. **Porta o núcleo estável pro Rust do axon** (SIMD: ns/op; persistência .axon; tick 100Hz).

## Veredito honesto
O CONCEITO está provado: um sistema que **aprende de um stream em tempo real, gera, não
esquece, e pesa KB** — algo que LLM não faz no uso. Não é o cérebro humano (ninguém sabe
fazer). É uma arquitetura nova, leve e viva, com cada peça ancorada em neurociência real e
rodando com selftest. O caminho da revolução leve é este — degrau por degrau, honesto.
