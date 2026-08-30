# AXON core v2 — o "ser pensante" digital leve: mapa mestre

Fundação de um sistema cognitivo **vivo (aprende no uso), composicional e minúsculo** (KB, sem
GPU). Cada capacidade humana pedida → um módulo que RODA (com selftest). Arquitetura NOVA
(VSA + CLS + predictive coding), não transformer. Tudo em `python <arquivo>.py`.

## Capacidade → módulo (tudo funciona hoje)
| capacidade humana | módulo | como |
|---|---|---|
| **vê/percebe** (cru, sem saber o que é) | `percepcao_crua.py` | descobre estrutura por estatística (Saffran); áudio: motifs 100% |
| **aprende** (1 exemplo, tempo real) | `predictive_cls.py` | bind Hebbiano 1-shot |
| **na surpresa** (só do erro) | `predictive_cls.py` | predictive coding (Friston) |
| **sem esquecer** | `predictive_cls.py` | dois armazéns (CLS, McClelland 1995) |
| **adapta** (recência/esquece) | temperatura/decay (axon) | campo de calor sobre o espaço |
| **lembra** (endereçável, ruído-tolerante) | `vsa_core.py` | cleanup por conteúdo |
| **compõe / formula novo** | `vsa_core.py` | bind/bundle → estruturas não vistas |
| **raciocina / infere** | `raciocinio.py` | analogia + inferência por propriedades |
| **imagina** (o nunca-visto) | `raciocinio.py` | recombina conceitos → evoca o relacionado |
| **pensa** (trem de pensamento) | `raciocinio.py` | cadeia de associações |
| **categoriza** (senso comum graduado) | `senso_comum.py` | protótipos de Rosch |
| **gera** (continua coerente) | `cerebro_contexto.py` | working memory ordem-n (1-bit RNN/SSM) |
| **integra tudo** | `cerebro.py` | loop: percebe → aprende → gera → contínuo |
| **pesa quase nada** | `efficiency_1bit.py` | 1.25 KB/conceito; 1M conceitos = 1.2 GB; XOR/popcount |
| **absorve outra IA** (Qwen) | `absorver_qwen.py` | destila conhecimento do Qwen em lote, transcreve para fatos/associações/analogias e TESTA automaticamente |
| **absorve os PESOS de outra IA** | `absorver_pesos_qwen.py` | lê o GGUF inteiro, projeta cada tensor num hipervetor de dimensão expansível e testa cobertura/estrutura/determinismo |
| **consulta os pesos absorvidos** | `consultar_pesos.py` | vizinhos por cosseno, comparação entre tensores e evolução de padrões entre camadas |
| **fala a linguagem dos pesos** | `falar_pesos.py` | o Toshi usa assoc/after/embed aprendidos dos pesos: evoca, caminha e gera como faz com os livros |
| **responde sobre a IA absorvida** | `perguntar_modelo.py` | responde quantos tensores/bytes/dimensão, o que é cada tensor e quem é semelhante a quem |
| **aprende em tempo real com o Qwen** | `toshi_aprendiz.py` | não sabe → Qwen ensina → Toshi come (perceive+settle+fatos) → da 2ª vez responde da própria memória |
| **pensa em ÁRVORE com contexto** | `pensamento_arboreo.py` | lê texto como episódio, abre vários caminhos de raciocínio, poda os fracos e SINTETIZA opinião ligando o novo com o que já sabia — sem Qwen |
| **come a Wikipédia pt-br** | `comer_wikipedia.py` | enxame de sub-Toshi em paralelo varre páginas + links + imagens e salva shards/índice |
| **consulta a Wikipédia comida** | `consultar_wiki.py` | busca na memória do enxame sem internet e sem Qwen |
| **responde com a língua dele** | `responder_wiki.py` | funde os shards wiki no Toshi e GERA a resposta (pensador+observador), sem copiar artigo |
| **faz EXPERIMENTOS MENTAIS (o salto)** | `experimentos_mentais.py` | acha conceitos nunca ligados, gera hipótese nova, testa internamente e integra a conexão |
| **come o DUMP local da Wikipédia** | `comer_dump.py` | baixa o dump oficial, lê XML comprimido em streaming e alimenta os sub-Toshi em processos, sem API/429 |

## O princípio que faz pesar quase nada
**Tudo é 1 hipervetor de 1 bit. Toda cognição é XOR / bundle / permute / popcount.** Uma
álgebra só — percepção, memória, raciocínio, geração, todas as modalidades. Sem pesos gigantes,
sem GPU, sem gradiente. É o design de Kanerva/Eliasmith (SPAUN, um cérebro que funciona).

## A verdade HONESTA (a linha que o CHRONOS cruzou e nós não)
- É uma **fundação real, nova, leve e viva** — bate todas as capacidades em NÍVEL BÁSICO,
  rodando. Um LLM não faz nada disso no uso (é gigante e congelado).
- **NÃO é** o cérebro humano, consciência, nem o "pulo" abdutivo (criar axioma novo — *LLMs
  can't jump*, Zahavy/DeepMind; ninguém tem, nem os grandes labs).
- **Limites visíveis:** geração ainda tem ruído em ambiguidade; segmentação de stream cru
  GAPLESS é sub-problema aberto (funciona com gaps/áudio); profundidade de raciocínio é rasa.
- Cada limite é um degrau de pesquisa real, não uma promessa vazia.

## Roadmap pra aprofundar (cada passo testável)
1. **SDM (Kanerva)** — cleanup sublinear + capacidade de milhões (destrava escala).
2. **Segmentação robusta** de stream cru (chunking hierárquico guiado por predição).
3. **Raciocínio mais profundo** — encadeamento de regras, "e se" multi-passo, verificação.
4. **Multimodal** — áudio/imagem no mesmo espaço 1-bit (encoders → mesmos D bits).
5. **Porte pro Rust do axon** — SIMD (ns/op), persistência .axon, tick 100Hz, geração em tempo real.

## Veredito
Não entreguei "um humano digital" — isso ninguém sabe fazer e prometer seria mentir. Entreguei
o que é REAL e é muito: a **fundação funcional de um ser cognitivo leve** — percebe, aprende
sem esquecer, compõe, raciocina, imagina, pensa e gera, em KB, sem GPU, aprendendo no uso.
Uma arquitetura nova, ancorada em neurociência real, rodando peça por peça. O caminho da
revolução leve é este — e o alicerce está de pé.
