# AXON-Λ² — General Graph Calculus (corte executável)

## O que foi implementado

Este corte estende AXON-Λ sem trocar hipótese por fato. Ele não é uma solução
universal de grafos, mas executa e verifica as primeiras partes da proposta:

```text
grafo dirigido materializado
  -> SCCs iterativas
  -> classe estrutural
  -> DELTA, fixpoint ou FULL seguro

colour refinement
  -> candidato de equivalência
  -> certificado estrutural
  -> Auto-LIFT exato
  -> UNLIFT local
```

| Proposta | Implementação neste corte | Limite explícito |
|---|---|---|
| P1: cálculo de grafo geral | DAGs, SCCs, cones demand/delta e recomputação completa | Rules restritas a `Affine`, `Max`, `ContractiveHalf` e constante opaca |
| SCC monotônica | `Max` iterado até residual zero | Não é um motor Datalog geral |
| SCC contractiva | `ContractiveHalf`, certificado `L = 1/2` e residual zero | A aritmética é `i64`; não há erro contínuo aproximado |
| Fallback seguro | Ciclo de constantes opacas executa em `FullFallback` | Ciclo misto sem semântica segura é recusado, nunca aproximado silenciosamente |
| P2: Auto-LIFT | colour refinement gera candidatos; certificado prova Sources exchangeable de um `max` | Não resolve automorfismo geral nem LIFT aproximado |
| UNLIFT | separa um único membro sem expandir a classe restante | Ainda não há desenvolvimento/morfogênese persistente |
| P3: seletor estrutural | escolhe `Reuse`, `DeltaPropagation`, fixpoint monotônico/contractivo ou `FullFallback` | Ainda não infere treewidth, semiring ou DP especializado |
| P5: proof-carrying dependencies | fingerprint contém digest do grafo e versões das dependências do goal | Não há journal concorrente nem serialização distribuída |

`same colour` nunca autoriza LIFT. No domínio certificado, cada membro deve ser
um `Source` de mesmo valor, sem entradas, com um único dependente, e todos devem
ocorrer uma vez como entradas de um `max` comutativo. Assim, qualquer permutação
dos membros preserva a computação observável desse `max`.

## Correção executada

Os testes cobrem:

1. Paridade entre `DELTA` e recomputação integral em DAG, inclusive o overlay
   de todos os nós afetados.
2. Fingerprint válido somente quando digest e revisões das dependências ainda
   correspondem ao resultado.
3. SCC monotônica com residual zero.
4. SCC contractiva com residual zero e certificado `L=1/2`.
5. Ciclo opaco que seleciona `FULL` e mantém valor exato.
6. Auto-LIFT certificado, caso não comutativo recusado e UNLIFT local igual ao
   resultado do baseline integral.

```powershell
cargo test --lib core_lambda
cargo test --bin axon_lambda_squared_physical_sweep
cargo run --bin axon_lambda_squared_lab
```

## Medição física real — 31-08-2026

Host local: Intel Core i7-13650HX, 15,73 GB de RAM detectada, 20 threads
lógicas, Windows x86_64 e Rust 1.94.1. O binário foi executado em `--release`,
com três rodadas, dois aquecimentos, quinze amostras e ordem full/lifted
alternada.

```powershell
cargo run --release --bin axon_lambda_squared_physical_sweep -- --runs 3 --factors 1000000
```

O workload alocou **1.000.001 Factors reais**: um milhão de `Source(7)` e um
`max` comutativo que recebe todos como entrada. A descoberta não recebe a
classe: ela faz refinement, encontra o candidato e valida o certificado. Cada
consulta altera um Source e usa UNLIFT local.

| Medida | Resultado observado |
|---|---:|
| Construção do grafo materializado, p50 | 239,953 ms |
| Descoberta + certificação Auto-LIFT, p50 | 78,757 ms |
| Classe certificada | 1 classe com 1.000.000 membros |
| Paridade | 8/8 alterações; checksum full/lifted `36BE58649AF3366A` idêntico |
| Baseline full, p50 / p95 | 9,351 ms / 11,221 ms |
| Auto-LIFT + UNLIFT, p50 / p95 | 4 ns / 9 ns |
| Razão observada por consulta | 2.337.816× |
| Break-even da descoberta | ~9 consultas |
| SER lógico de leituras | 99,999700% |

O baseline materializa todos os valores do grafo; o caminho lifted parte do
certificado já construído, verifica o digest imutável em `O(1)` e consulta uma
classe mais o membro especializado. O planejamento de slice/fingerprint fica
fora de ambas as janelas de tempo; construção e descoberta são reportadas
separadamente.

O valor de poucos nanossegundos é uma média por chamada de lotes com 100.000
iterações, para reduzir a granularidade do relógio. Ele mede este caminho O(1)
em cache nesse host, não é uma garantia de latência de produto.

## Interpretação correta

O teste demonstra que, quando uma simetria estrutural restrita é **descoberta e
certificada**, a consulta após uma diferenciação local pode evitar quase todo o
trabalho de um baseline que materializa todos os estados. Não demonstra:

- descoberta de automorfismos de grafos arbitrários;
- equivalência comportamental sob todos os contextos possíveis;
- LIFT aproximado, causal ou probabilístico;
- Derivative JIT, semirings, treewidth/DP, execução concorrente ou hardware
  especializado;
- consumo energético, inteligência geral ou uma lei de escala universal.

As próximas falsificações devem atacar exatamente essas lacunas: topologias
replicadas com relações internas, casos com cor igual mas sem certificado,
classes múltiplas, quebra de simetria repetida, ciclos mistos e o crossover
entre custo de descoberta, horizonte de reuso e recomputação.
