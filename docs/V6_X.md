# V6-X / Ψ-IR — realização física auditável

## Estado

V6-X adiciona uma camada experimental abaixo da Ω-IR: a Ψ-IR. Ela descreve
qual transformação cognitiva deve ocorrer, os requisitos de precisão e
latência, e seleciona um backend físico **por contrato**. O código está em
`src/core_v6/physical.rs` e continua isolado do runtime, TUI, persistência
`.axon` e hardware real.

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

## Limites deliberados

- `CpuExact`, p-bit, fotônica, quantum, crossbar analógico e circuito
  reversível são nomes de perfis; nenhum driver, chip ou serviço externo é
  executado.
- As unidades de custo são abstratas e `Declared`; não são joules, watts,
  latência medida ou benchmark de eficiência.
- `UNCOMPUTE` modela a semântica lógica de preservar resultado e limpar scratch;
  não mede dissipação térmica numa CPU.
- A seleção ainda não controla o `CognitiveRuntime`. Só será integrada depois
  de traces medidos por operação e validação contra um backend digital.

## Próximas falsificações

1. Coletar traces reais de CPU: bytes, tempo, conversões, verificação e erros.
2. Calibrar perfis `Measured` para um backend por vez; nunca misturá-los com
   previsões declaradas.
3. Testar uma primitive dominante de verdade, como lookup associativo ou
   sampling, antes de avaliar CIM, p-bit, FPGA ou fotônica.
4. Exigir verificação digital independente quando a realização aproximada puder
   alterar uma decisão cognitiva importante.
