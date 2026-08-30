# V5/Ω — núcleo experimental de cognição orientada a informação

## Estado

V5/Ω existe como protótipo Rust isolado em `src/core_v5/`. Não altera o
runtime, a TUI, o formato `.axon`, nem encaminha trabalho para hardware
especializado. A meta desta fase é testar contratos arquiteturais, não alegar
AGI, economia física ou superioridade sobre Transformers.

## Tese operacional

Em vez de otimizar somente operações aritméticas, o plano V5 escolhe uma ação
por valor cognitivo dividido por custo de informação:

```text
compute + bytes moved + bytes stored + bytes erased
```

Toda estimativa possui proveniência obrigatória:

- `Declared`: cenário, orçamento ou perfil ainda não instrumentado.
- `Measured`: contador obtido na execução real.

Uma comparação entre as duas é rejeitada pelo tipo `CostVector`; portanto a
V5 não pode transformar um custo declarado em afirmação termodinâmica.

## Primitivas implementadas

| Primitiva V5 | Contrato atual | Limite deliberado |
|---|---|---|
| Thermodynamic Scheduler | escolhe maior utilidade por custo ponderado e mantém conjunto ativo mínimo | pesos e custos ainda são declarados |
| Dormant Intelligence | plano expõe conceitos ativos versus dormentes | não há store massivo persistente ainda |
| Multi-Substrate State | eventos declaram leitura/escrita em malha semântica, mundo dinâmico, episódios e programas | substratos são stores tipados mínimos |
| ProgramCell + Abstraction Compiler | induz `repeat(A,B)` apenas com compressão positiva e holdout correto | DSL inicial contém somente repetição de par |
| Population of Worlds | preserva família estrutural e ranqueia por previsão, generalização, simplicidade, novidade e falsificabilidade | não executa simulador físico completo |
| Active Experiment Planner | escolhe intervenção que separa previsões concorrentes por custo | não recebe a classe verdadeira do mundo |
| Reversible Cognition | estado e mundo guardam `before`, `after` e proveniência; `undo` restaura igualdade estrutural | journal ainda é em memória |
| Thought Profiler / Self-extending ISA | repetição verificada cria `NEW_OP_n`; guard falho deotimiza | macro não altera a ISA de CPU real |
| Location Plasticity | coativação aproxima localmente conceitos e reduz custo lógico de rota | região é lógica, não placement de hardware |
| Physical Cognitive Compiler | exato só aceita backend exato; similaridade pode aceitar HDC aproximado | CPU/HDC/reservoir são perfis, não drivers |

## Resultados desta primeira rodada

Execute:

```powershell
cargo test --lib core_v5
cargo run --bin axon_v5_lab
```

Cinco execuções produziram o mesmo relatório:

```text
SHA-256: D64F759A1FE8EECFA9AF4B00BFB704C8FC8B754FED559E55BEC14F78BF5DE260
```

Resultado determinístico atual:

| Métrica | Resultado |
|---|---:|
| ProgramCell `repeat(A,B)` em holdout | 100% |
| Compressão da sequência de treino | 2,0× |
| Famílias de mundo retidas | 3/3 |
| Intervenção discriminativa escolhida | sim |
| Rollback reversível exato | sim |
| Conceitos ativos / dormentes | 2 / 126 |
| Custo lógico de rota após coativação | 8 → 0 |
| JIT: compilação / deoptimization | sim / sim |
| Verificação exata em backend digital | sim |
| Similaridade em backend aproximado | sim |

Esses resultados somente confirmam os contratos nos mundos sintéticos
determinísticos do laboratório. Eles não medem joules, latência, precisão de
hardware analógico, nem descoberta científica aberta.

## Próximas falsificações obrigatórias

1. Instrumentar bytes, tempo e energia de runtime para produzir `Measured`.
2. Ampliar ProgramCells para DSL segura com `Sequence`, `Select`, `Emit` e
   guards; medir generalização em múltiplas sementes e holdouts.
3. Separar geração, intervenção e avaliação dos mundos causais; reportar
   precisão, recall, falsos `REFRAME`s, diversidade e custo de busca.
4. Conectar placement lógico a alocadores reais e avaliar tráfego de memória.
5. Só depois calibrar perfis para CPU, GPU, HDC, reservoir ou outros
   coprocessadores físicos.
