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
SHA-256: 4B01141344094663239BB97F3DF863690654CAD6D31CE149A6A58516367813BE
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
SHA-256: 0EFBBF9F67E0251C0747426E19B5BA103FCC792DE929F6F3B938630D3B7A99AD
```

Ela prova apenas esses contratos determinísticos. `BTreeMap`, 64 bytes lógicos
por entrada e `CostVector::Declared` não são banco de dados, telemetria de RAM,
medição de energia ou benchmark de escala. O escopo completo e as próximas
falsificações estão em `docs/V6_OMEGA.md`.
