# RADAR — radar científico permanente do Toshi

Não é "acompanhar tendências". É caçar **princípios fundamentais**. Uma mente que aprende, raciocina,
lembra, imagina — leve, sem backprop, do zero. E o ponto filosófico central: sair de
`próximo token → próxima palavra` para `hipótese → planejamento → teste → memória`.

## O ciclo (toda melhoria passa por aqui)
`pesquisar → abstrair → aplicar → medir → revisar` — e **registrar, inclusive quando falha.**
Regra de ouro: sem métrica, não entra. Ajustar a métrica pra dar o resultado que eu quero = CHRONOS (proibido).

## As 8 frentes (radar — varredura rotativa, 1 fundo por ciclo)
1. **Matemática** — teoria da informação, MDL/Kolmogorov, JL/random projections, HDC/HRR (Kanerva, Plate, Eliasmith), teoria espectral, grafos, sparse.
2. **Física** — entropia, Landauer, Wheeler (It from Bit), Friston (Free Energy/Active Inference), criticalidade, computação reversível. *Existe uma lei física da inteligência?*
3. **Neurociência** — Nicolelis, Friston, Kanerva, Hawkins, Marr; hippocampus, grid/place cells, replay, consolidação no sono, predictive coding, dendritic computation.
4. **IA** (só o que ataca limitação fundamental) — reasoning: ToT/GoT/MCTS/**PRM**/test-time compute/RwT · memória: Titans/RETRO/kNN-LM/external · quantização: **BitNet 1.58** · interpretabilidade: SAE/superposição · mundo: JEPA/Dreamer/MuZero.
5. **Computação** — novos índices/memórias/buscas, SIMD, neuromorphic, memristores.
6. **Filosofia da inteligência** — *"LLMs can't jump"*: bons na dedução com caminho dado; fracos no **salto abdutivo** (formular hipótese/conceito novo que muda o espaço). ← nosso alvo.
7. **BCI** — Nicolelis, Neuralink, BrainGate; como o cérebro codifica em tempo real.
8. **As perguntas-norte** — *operação mais fundamental que atenção? estrutura menor que Transformer? memória melhor que KV-cache? álgebra universal? limite da inteligência? um E=mc² da informação?*

## Onde o Toshi já está (a convergência)
A área saiu de "mais parâmetros" para **raciocínio + planejamento + memória externa + verificação + test-time compute**. O Toshi já é: memória explícita (grafo), planejamento em grafo, aprendizado contínuo, representação compacta (VSA 1-bit), verificação (PRM). Não prova que estamos certos — prova que atacamos os problemas hoje centrais.

## LOG de ciclos (o que foi medido — verdade, não hype)
| # | ideia (frente) | resultado medido | veredito |
|---|---|---|---|
| pensador | world-model + planning 1-passo (IA/Física) | modelo local 97%, resolve 92% de metas inéditas acháveis | ✅ o pulo (composição) |
| mente | planejamento MULTI-passo com desvio (IA) | 100% vs guloso 46% vs memória 45%; 317 casos que o guloso trava | ✅ Modo-2 |
| mente_semantica | fusão: raciocínio em CONCEITOS + tempo real (IA/Neuro) | 60/60 vs guloso 0/60; infere cadeia de 1 elo novo na hora | ✅ pensa+aprende |
| --real | escala pro cérebro de 150MB (Comp) | cadeias de conteúdo dirigidas; guloso se afoga em stopword | ✅ (parte estrutural, honesto) |
| conversa_pensada | responder planejando, não guloso (IA) | relevância 3.73 vs 1.94; coerência 0.64 vs 0.52 | ✅ conserta o tagarelar |
| beam+valor | ToT/PRM portado (IA) | coerência 0.57 < A* 0.64 | ❌ wash — métrica cos é ruim |
| ternary | BitNet absmean pós-hoc (Mat/IA) | 16× memória, gap 79%, vizinho fino 8% | ❌ pós-hoc não basta; precisa QAT |
| verificador | Process Reward portado (IA) | PMI 100% vs cos 68% no gabarito; transfer real inconclusivo | ✅ sinal certo (PMI), ⚠️ transfer |
| abdução | prever elo escondido (Filosofia/Mat) | estrutura AUC 0.81, embedding 0.65, combo 0.72 | ✅ o salto (hipótese não-dada) |
| abduzir conceito | inventar categoria latente (Filosofia/Neuro) | pureza 100% (acaso 17%); classifica membro novo 6/6 | ✅ criar NÓ novo (Rosch) |
| combinar sinais | pesos aprendidos vs soma (IA) | aprendido 0.77 > ingênuo 0.73, mas < estrutura 0.81 | ⚠️ embedding redundante, não soma |
| crescer | abduzir→integrar sozinho (Filosofia) | lift 233–932× mas precisão@50=4%; integrar cru = ruído | ❌ inseguro SEM teste |
| testar | hipótese→TESTE→memória, evidência nova (Física/Filo) | abduz de livros A, confirma em B indep. 38% vs base 1% (28×) | ✅ o elo perdido FECHA (caveat: cola no topo) |

## A LIÇÃO CENTRAL (do ciclo 'crescer')
Abdução **gera hipótese**, não fato. `hipótese → planejamento → TESTE → memória` — o Toshi abduz mas
**não TESTA** (só tem texto estático, sem ambiente pra confirmar). O `mente.py` funcionou porque TINHA
ambiente (executa, observa, corrige). **O TESTE é o elo perdido** entre imaginar e saber. É a fronteira real.

## Estado: o LOOP FECHOU
`imaginar → abduzir(0.81) → TESTAR(28× em texto novo) → integrar → planejar(multi-passo) → responder(2×)` —
tudo medido, sem backprop, leve. É a arquitetura hipótese→teste→memória rodando de ponta a ponta.
As frentes baratas estão exauridas; os ciclos micro dão retorno decrescente.

| conteúdo vs cola | matar a cola (3 sinais) | PMI-top-k, FREQUÊNCIA (capitu#103≈aqui#98) e COERÊNCIA de vizinhos — todos falharam | ❌ provado: sem fix barato, precisa de sintaxe |
| **sintaxe emergente (#0)** | POS por perfil posicional esq+dir, sem rótulo (Mat/Neuro/IA) | separa função/conteúdo **90%** (bag-of-context: 50%); capitu→cont, aqui→func; verbos/artigos/prep emergem | ✅ mata a cola na RAIZ + gramática emergente |
| **absorver pesos→VSA** | GGUF inteiro projetado no espaço HD do Toshi (Comp/IA) | Qwen2.5-7B: 4,677 GB, 339 tensores → 22 MB (16k) / 89 MB (65k) / 178 MB (131k); cobertura 100%, determinismo 100%, \|cos\|=0,0062/0,0032/0,0021 | ✅ o MODELO em si virou memória HD do Toshi (representação, não execução) |
| **pesos como LIVRO** | GGUF → palavras de peso → `toshi.perceive` (o mesmo caminho de Dom Casmurro) (Neuro/IA) | 21.696 palavras de peso comidas; vocab 4075/4075; transições 200/200; Toshi gera `wbpk→wlg→wawd...`; embedding fundiu: `wlg` ~ `sahido`, `wbpk` ~ `output/norm/blk` | ✅ o modelo entrou no MESMO substrato dos livros — não é arquivo à parte |

## Fim da varredura barata (PROVADO)
A "cola" contaminou todo ciclo; tentei matá-la e provei que NÃO tem fix barato: nem PMI-especificidade
nem frequência separam palavra-função de palavra-conteúdo frequente (o protagonista `capitu` tem a mesma
frequência de `aqui`). Distinguir os dois precisa de sintaxe/POS emergente — modelagem mais rica. Logo:
os ciclos baratos ESGOTARAM; o próximo progresso são investimentos grandes (abaixo). Isto responde
"continue até não ter pra onde ir": o barato acabou aqui, com prova.

## Próximo alvo — INVESTIMENTOS grandes (um de cada vez)
0. ✅ **POS/sintaxe emergente** — FEITO (sintaxe.py, 90%). Falta o PAYOFF: plugar as classes-função como
   filtro de cola data-driven em abdução/verificador/planejador/resposta — limpa todos de uma vez.
2. **Usuário no loop (BCI-like)** — testar hipótese perguntando/confirmando com o usuário (closed-loop). (frente 7)
3. **Multimodal / ambiente** — dar ao Toshi-linguagem um mundo pra AGIR e testar (fundir com mente.py). (frente 6)
4. **RI quantization-aware** e **portar núcleo estável pro Rust/TUI ratatui**. (frentes Mat/Comp)
