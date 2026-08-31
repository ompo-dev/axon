# V6-X / Ψ-IR — realização física auditável

## Estado

V6-X adiciona uma camada experimental abaixo da Ω-IR: a Ψ-IR. Ela descreve
qual transformação cognitiva deve ocorrer, os requisitos de precisão e
latência, e seleciona um backend físico **por contrato**. O código está em
`src/core_v6/physical.rs` e continua isolado do runtime, TUI, persistência
`.axon` e hardware especializado. A execução da Ψ-IR e o benchmark de CPU/RAM
local são reais; p-bit, fotônica, quantum e crossbar continuam sem dispositivo
conectado.

## Primitivas introduzidas

| Primitiva | Contrato atual |
|---|---|
| `REALIZE` | `PhysicalCompiler::realize` só escolhe perfil que satisfaz operação, erro e latência. |
| Ψ-IR | `PhysicalOperation`, `PrecisionRequirement`, `PhysicalStateKind`, `PhysicalProfile` e `PhysicalPlan`. |
| Boundary Tax | `PhysicalCost` soma encode, movimento, compute, decode, verificação, resfriamento, calibração e desgaste. |
| Proveniência de custo | perfis só competem se tiverem mesma origem, unidade, fonte e calibração. |
| `UNCOMPUTE` | scratch reversível restaura o estado temporário apenas depois de preservar o resultado extraído. |
| Reversibilidade seletiva | `ERASE`, checkpoint, recompute e uncompute competem por custo; uncompute não é imposto. |

## Resultados do laboratório

```powershell
cargo test
cargo run --bin axon_v6x_lab
```

| Cenário | Resultado |
|---|---|
| Prova exata versus p-bit barato | CPU digital é selecionada. |
| P-bit com custo de conversão alto | Rejeitado pelo Boundary Tax. |
| P-bit co-localizado em sampling aproximado | Selecionado dentro da tolerância de erro. |
| Custo declarado contra medido | Decisão recusada. |
| Scratch reversível e resultado comprometido | `UNCOMPUTE` restaura o scratch. |
| Uncompute mais caro que apagar | `ERASE` vence. |

Cinco execuções independentes tiveram saída idêntica:

```text
SHA-256: 6CA5DF539F8A1E4BBFDF2CA7B882C7A399211D3A9173617AEC30DF53BD79A20F
```

## Benchmark executado na CPU local

O benchmark de referência é o sweep físico em
[`V6_X_PHYSICAL_SWEEP.md`](V6_X_PHYSICAL_SWEEP.md). Ele mede, no PC local,
decomposição da fronteira, a curva de 64 B a 64 MiB, efeito de cache,
localidade, cópias, `VERIFY`, descarte e escala do `REALIZE`.

```powershell
cargo test --bin axon_v6x_physical_sweep
cargo run --release --bin axon_v6x_physical_sweep -- --runs 3
```

O executável coleta antes do sweep e mostra no topo do relatório o host
detectado (identificador de CPU fornecido pelo SO, RAM, threads, plano de
energia e compilador), usa `Instant`, `black_box`, três aquecimentos e 15
amostras por medição. Em cada tamanho, ele exige igualdade integral entre a
pipeline materializada e o kernel fundido e imprime um checksum integral. A
documentação do sweep contém os resultados reproduzíveis e a interpretação
restrita à CPU/RAM local.

O binário anterior `axon_v6x_cpu_bench` é mantido apenas como smoke benchmark
de um cenário único; seus números não são a referência experimental atual.

## Limites deliberados

- `CpuExact` executa na CPU/RAM local. P-bit, fotônica, quantum, crossbar
  analógico e circuito reversível continuam apenas nomes de perfis: nenhum
  driver, chip ou serviço externo desses backends foi executado.
- Os tempos acima são medidos; energia, watts, joules, temperatura e eficiência
  por operação não foram medidos, pois este PC não expõe um medidor de processo
  calibrado para essas grandezas.
- `UNCOMPUTE` mede a semântica lógica de preservar resultado e restaurar scratch
  em Rust; não mede reversibilidade termodinâmica ou dissipação física.
- A seleção ainda não controla o `CognitiveRuntime`. Só será integrada depois
  de traces medidos por operação e validação contra um backend digital.

## Próximas falsificações

1. Instrumentar o `CognitiveRuntime` com os mesmos traces medidos de CPU:
   bytes, tempo, conversões, verificação e erros.
2. Calibrar energia por processo com medidor compatível e perfis `Measured`
   para um backend por vez; nunca misturá-los com
   previsões declaradas.
3. Testar uma primitive dominante de verdade, como lookup associativo ou
   sampling, antes de avaliar CIM, p-bit, FPGA ou fotônica.
4. Exigir verificação digital independente quando a realização aproximada puder
   alterar uma decisão cognitiva importante.
