# Laboratório Experimental Axon

Este documento registra a primeira rodada de hipóteses falsificáveis para as
primitivas V3/V4. O laboratório é sintético, determinístico e isolado: não
altera o runtime, a TUI, a memória `.axon` nem pretende provar inteligência
geral ou economia de hardware.

## Como reproduzir

```powershell
cargo test --lib experiments::tests
cargo run --bin axon_lab
```

O binário executa quatro experimentos com um gerador determinístico interno.
Cinco execuções independentes da suíte produziram o mesmo relatório:

```text
SHA-256: 51376BA2E69E985F137880B8A2F6CF23F28915FFCDB64990C09ABFD6466CE72A
```

## Hipóteses e protocolo

| Hipótese | Mundo controlado | Baseline / ablação | Métrica |
|---|---|---|---|
| H1: a álgebra de HyperCells preserva composição | 96 ensaios, 257 dimensões, codebook de 32 itens e 3% de ruído no vetor ligado | chave bipolar densa versus chave ternária esparsa | recuperação exata, recuperação por busca, retenção de sinal |
| H2: atualização estritamente local generaliza além de pares vistos quando a representação é fatorizada | grade 4×4 de dois fatores; 8 combinações no treino e 8 em holdout | lookup de par exato versus regra local nos dois fatores ativos | acurácia no holdout |
| H3: `REFRAME` identifica a mudança estrutural que `ADAPT` não captura | 60 mundos: 20 `A→B`, 20 `B→A`, 20 causa latente comum | hipótese direta fixa; reframe com e sem contrafactuais | identificação estrutural correta |
| H4: o plano de controle reduz trabalho declarado sem corromper conhecimento protegido | lookup, rotina compilada, deliberação e anomalia estrutural | política sempre deliberativa | seleção correta, uso estimado, forks e rollbacks |

Em H3, o tipo do mundo é uma variável-oráculo apenas do gerador sintético: ele
produz as observações e contrafactuais, mas não é entregue ao modelo. A seleção
recebe somente a perda contrafactual do modelo atual e as hipóteses geradas.
Sem intervenção, ela mantém o modelo direto, em vez de receber a classe correta
de forma implícita.

## Resultados da primeira rodada

| Experimento | Resultado | Leitura honesta |
|---|---:|---|
| H1 — bind denso bipolar | 96/96 recuperação exata; 96/96 recuperação com 3% de ruído | A composição funciona no regime denso bipolar deste tamanho de codebook. |
| H1 — bind ternário esparso | 0/96 recuperação exata; 50,6% de retenção de sinal ativo | **Falsificado:** zero não pode ser simultaneamente “ausência” e elemento de uma operação multiplicativa auto-inversa. |
| H2 — lookup associativo | 4/8 holdouts (50%) | Memorizar o par não é suficiente para combinações inéditas. |
| H2 — regra local fatorizada | 6/8 holdouts (75%) | **Evidência parcial:** a representação separada melhora +25 p.p., mas ainda não é generalização frontier. |
| H3 — adaptação que mantém `A→B` | 20/60 (33,3%) | Correlação observacional não distingue direção e causa comum. |
| H3 — `REFRAME` + intervenções | 60/60 (100%) | Neste mundo, contrafactuais selecionam corretamente direto, reverso ou latente. Avalia 9 hipóteses por mundo que exigiu `REFRAME`. |
| H3 — `REFRAME` sem intervenção | 20/60 (33,3%) | **Falsificado:** observação correlacional sozinha continua insuficiente; não há “jump” identificável sem evidência que quebre a equivalência. |
| H4 — scheduler | 4/4 escolhas esperadas | Apenas prova comportamento contra candidatos e custos declarados pelo próprio cenário. |
| H4 — custo declarado | 9 eventos / 864 bytes / 156 µJ vs 16 / 1.920 / 360 | Redução de 43,8% em eventos, 55,0% em bytes e 56,7% em energia **estimada**, não física. |
| H4 — memória protegida | 12/12 forks; 12/12 rollbacks | O contrato reversível funciona no journal sintético. |

## Decisões que os dados já permitem

1. Não usar `0` no mesmo `bind` multiplicativo que precisa ser auto-inverso.
   Uma próxima hipótese deve separar máscara de esparsidade, usar binding de
   fase/permutação, ou guardar o suporte do vetor explicitamente.
2. Tratar fatorização como condição testável para generalização, não como
   garantia. A regra atual melhora o holdout, mas erra 2 dos 8 casos inéditos.
3. Definir `REFRAME` operacionalmente como **gerar modelos alternativos +
   coletar/usar intervenções + selecionar por perda contrafactual**. Sem o
   terceiro passo, o benchmark não autoriza dizer que houve descoberta.
4. Manter custos V4 rotulados como `estimated_use`. A passagem para hardware
   só começa após instrumentar bytes realmente movidos, tempo de CPU/GPU,
   consumo energético e taxa de deopt no runtime executável.

## Próxima rodada necessária

- Variar dimensão, esparsidade, ruído e tamanho de codebook para escolher a
  álgebra de HyperCells por curvas, não por um único ponto.
- Repetir H2 em várias sementes, com ruído, fatores de cardinalidade diferente,
  regras não lineares e holdouts fora da grade; reportar média e dispersão.
- Tornar H3 estocástico e cego ao gerador: separar treino, intervenções e
  teste; medir falsos reframes, precisão, recall e custo de explorar hipóteses.
- Integrar contadores reais ao runtime antes de comparar energia/latência com
  qualquer baseline de software ou hardware.

Enquanto essas rodadas não forem concluídas, os resultados são evidência de
design para protótipos, não leis computacionais estabelecidas.

## V5/Ω — rodada inicial

A V5/Ω possui laboratório separado (`cargo run --bin axon_v5_lab`) e mantém a
mesma restrição: perfis de custo e backends físicos são `Declared`, não
`Measured`. A rodada confirma contratos de ProgramCell, população de mundos,
intervenção ativa, rollback, localidade, Thought JIT e roteamento por precisão.

| Métrica | Resultado |
|---|---:|
| ProgramCell em holdout / compressão | 100% / 2,0× |
| Famílias de mundo retidas / intervenção discriminativa | 3/3 / sim |
| Rollback / rota lógica após coativação | exato / 8 → 0 |
| Thought JIT: compilação e deoptimization | sim / sim |
| Backend de verificação / similaridade | CPU exato / HDC aproximado |

O relatório completo, escopo e falsificações pendentes estão em
`docs/V5_OMEGA.md`.

## V6/Ω6 — primeiro corte de integração experimental

Execute:

```powershell
cargo test --lib core_v6
cargo run --bin axon_v6_lab
```

Esta rodada verifica 257 fatos sintéticos com retenção one-shot, um working set
de 1 Factor (ABR lógico de 0,0039), supersessão de duas revisões, escolha de
modelo por domínio/custo, supressão local de mensagem, JIT com deoptimization,
Learnability Gate, conhecimento negativo e limite de patch pelo kernel.

Cinco execuções independentes de `axon_v6_lab` tiveram saída idêntica:

```text
SHA-256: D247F632130A42A38E0B4A8DF826EF9DDC548AA66A1A0963EE86FDF8721F76FA
```

Ela prova apenas esses contratos determinísticos. `BTreeMap`, 64 bytes lógicos
por entrada e `CostVector::Declared` não são banco de dados, telemetria de RAM,
medição de energia ou benchmark de escala. O escopo completo e as próximas
falsificações estão em `docs/V6_OMEGA.md`.

## V6-X / Ψ-IR — Physics Compiler em simulação

Execute:

```powershell
cargo test
cargo run --bin axon_v6x_lab
```

O laboratório valida que operação exata permanece digital; custo de conversão
impede escolher um p-bit aparentemente barato; sampling aproximado pode usar
p-bit co-localizado; custos `Declared`/`Measured` mistos são recusados; e
`UNCOMPUTE` só vence quando é válido e mais barato que `ERASE`.

Cinco execuções produziram o mesmo relatório:

```text
SHA-256: 6CA5DF539F8A1E4BBFDF2CA7B882C7A399211D3A9173617AEC30DF53BD79A20F
```

São perfis e custos abstratos declarados, não medições de chip, energia,
quantum advantage ou hardware analógico. O escopo está em `docs/V6_X.md`.

## V7 e V7-X — morfogênese contratada

V7 mede a compilação de corpos cognitivos sob orçamento de memória; V7-X troca
`Q(M)` por um envelope de capacidade no corpus determinístico, contratos de
decisão, remorph transacional, worlds copy-on-write e custo de tarefa familiar.

Execute:

```powershell
cargo run --bin axon_v7_lab
cargo run --bin axon_v7x_lab
cargo run --release --bin axon_v7_morphogenic_sweep -- --runs 3 --touch-cap-mib 64
cargo run --release --bin axon_v7x_physical_sweep -- --runs 3 --world-kib 32
```

Os sweeps fazem medição real de CPU e RAM limitada. Planos de 4/16 GiB são
lógicos quando o executável declara explicitamente esse limite; eles não são
apresentados como alocação física integral. Resultados, checksums, limites e
falsificações pendentes: `docs/V7_MORPHOGENESIS.md` e
`docs/V7_X_CONTRACTIVE.md`.

## AXON-Λ — contratos e Demand × Delta

O corte AXON-Λ mantém Rust como uma realização de uma especificação semântica
restrita. Ele testa refinamento contratual, custo Pareto declarado, o cone
`B_g ∩ F_Δ`, fallback explícito de delta para recomputação global, quotient
exato (`LIFT`) e um micro-journal igual entre Rust e Python.

```powershell
cargo test --lib core_lambda
cargo test --bin axon_lambda_physical_sweep
cargo run --bin axon_lambda_lab
cargo run --bin axon_lambda_conformance
python tools/axon_lambda_conformance.py
cargo run --release --bin axon_lambda_physical_sweep -- --runs 3 --factors 1000000 --chain-len 1000
```

O sweep usa 1.000.000 Factors materializados e separa construção, paridade
semântica e execução cronometrada. Custos do semiring continuam declarados;
tempo de CPU local não é apresentado como energia. Escopo, resultados e limites:
`docs/AXON_LAMBDA.md`.

## AXON-Λ² — grafos gerais restritos e Auto-LIFT certificado

Este corte mede a primeira generalização verificável: DAGs e SCCs classificados
por estrutura, fingerprint de dependências, e um Auto-LIFT exato que descobre
Sources exchangeable de um `max` comutativo antes de comprimir. A cor é somente
candidata; o certificado é obrigatório.

```powershell
cargo test --lib core_lambda
cargo test --bin axon_lambda_squared_physical_sweep
cargo run --bin axon_lambda_squared_lab
cargo run --release --bin axon_lambda_squared_physical_sweep -- --runs 3 --factors 1000000
```

No host local, o sweep materializou 1.000.001 Factors, descobriu e certificou
uma classe de 1.000.000 Sources em 78,757 ms p50, preservou os checksums em
8/8 UNLIFTs, e mediu 9,351 ms p50 para full contra 4 ns p50 para o caminho
lifted já certificado. O custo da descoberta é separado e o escopo não inclui
automorfismos gerais, LIFT aproximado, energia ou hardware especializado.
Detalhes e limitações: `docs/AXON_LAMBDA_SQUARED.md`.
