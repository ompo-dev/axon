# AXON V7-X — Contractive Morphogenic Intelligence

## Corte implementado

V7-X transforma quatro hipóteses em contratos executáveis e falsificáveis.
Não é uma alegação de AGI nem integração de hardware emergente.

```text
ContractedFactor: contrato + certificado de decisão → REALIZE
Semantic Virtual Memory: shadow build → verify → commit imutável
Versioned World: base compartilhada + delta por hipótese
Cognitive Capital: repetição verificada → programa exato → custo familiar menor
```

| Componente | Contrato verificável | Limite atual |
|---|---|---|
| `ContractedFactor` | A realização precisa refinar o contrato e manter a decisão certificada. | Custo e erro ainda são declarados. |
| `SemanticVirtualMemory` | Fatos protegidos mantêm `Summary`; o shadow é verificado antes do commit. | A troca é atômica no modelo imutável, não um swap concorrente no runtime. |
| `VersionedWorld` | Cada branch preserva seu delta e compartilha a `WorldBase`. | Usa vetores e `BTreeMap`, não a Factor Fabric completa. |
| `CognitiveCapitalRuntime` | Promoção só ocorre após equivalência interpretada verificada. | A família inicial é soma triangular; não é síntese geral. |

## Capability Envelope sem `Q(M)`

O corpus possui 1.024 clusters semânticos determinísticos. Cada cluster possui
`Summary` (64 KiB), `Compact` (256 KiB), `Full` (1 MiB) e `Exact` (4 MiB).
Os 64 fatos protegidos exigem recall mínimo em `Summary`. O catálogo exato
ainda é in-memory e pequeno, usado para verificar o experimento; os 4 GiB são
contabilidade lógica de resolução, não um arquivo frio físico nem compressão.

| Orçamento lógico | Ativo | Arquivado | Recall protegido | Recall exato completo |
|---:|---:|---:|---:|---:|
| 64 MiB | 64 MiB | 4.032 MiB | 100,0% | 0,0% |
| 128 MiB | 127 MiB | 3.968 MiB | 100,0% | 0,0% |
| 512 MiB | 511 MiB | 3.584 MiB | 100,0% | 6,2% |
| 4 GiB | 4.096 MiB | 0 MiB | 100,0% | 100,0% |

Isso valida a propriedade declarada nesse corpus: detalhe degrada sem violar
o contrato protegido. Não demonstra uma curva de memória semântica geral.

## Medição física real — 31-08-2026

Host: Intel Core i7-13650HX, 15,73 GB RAM detectada, 20 threads lógicas,
Windows x86_64, Rust 1.94.1, build `--release`. Foram três rodadas, cinco
amostras e um aquecimento para Factor, Semantic VM e Capital; COW foi uma
materialização por rodada.

```powershell
cargo test --lib core_v7x
cargo test --bin axon_v7x_physical_sweep
cargo run --bin axon_v7x_lab
cargo run --release --bin axon_v7x_physical_sweep -- --runs 3 --world-kib 32
```

### Contracted Factor

O teste varreu 8 MiB reais. `exact` e `compiled` produziram o mesmo checksum;
o caminho aproximado é muito mais curto, mas não passa no certificado.

| Exact p50 | Compiled p50 | Approximate p50 | Exact = compiled |
|---:|---:|---:|---:|
| 3,719 ms | 3,694 ms | 0,158 ms | Sim |

O kernel `compiled` ficou praticamente empatado com o `exact` (menos de 1% de
diferença nesta rodada). Custo lógico não é ganho físico; `REALIZE` só deve
usar perfis calibrados por backend e workload.

### Semantic Virtual Memory

O corpus e as consultas rodaram no CPU local, porém os budgets de 64 MiB a
4 GiB são planos lógicos: o executável não alocou 4 GiB de corpus.

| Budget lógico | Construir plano p50 | Consultar 1.024 fatos p50 | Recall protegido | Recall exato |
|---:|---:|---:|---:|---:|
| 64 MiB | 0,107 ms | 0,165 ms | 100,0% | 0,0% |
| 128 MiB | 1,964 ms | 0,166 ms | 100,0% | 0,0% |
| 512 MiB | 14,439 ms | 0,166 ms | 100,0% | 6,2% |
| 4 GiB | 34,112 ms | 0,162 ms | 100,0% | 100,0% |

### Versioned World Copy-on-Write

Cada world parte de um vetor físico de 32 KiB e muda uma palavra. Full-copy
mantém cada cópia inteira; COW mantém base compartilhada e delta. Todos os
valores mutados foram verificados em todos os branches.

| Worlds | Payload full-copy | Full-copy p50 | COW p50 | Modelo de payload full/shared |
|---:|---:|---:|---:|---:|
| 10 | 0,3 MiB | 0,113 ms | 0,095 ms | 10,0× |
| 100 | 3,1 MiB | 1,218 ms | 0,708 ms | 95,3× |
| 1.000 | 31,2 MiB | 12,506 ms | 7,209 ms | 671,9× |
| 10.000 | 312,5 MiB | 127,537 ms | 70,804 ms | 1.699,9× |

O ganho físico de construção em 10.000 worlds foi aproximadamente 1,80×. A
redução lógica de footprint foi 1.699,9×; ela não equivale a 1.699× de
aceleração, pois alocação e metadados continuam custando tempo.

### Cognitive Capital

Foram executadas 10.000 entradas variadas da família `triangular`, todas
exatas. Após três verificações interpretadas, a fórmula equivalente é promovida.
O baseline mantém uma barreira dentro do loop para impedir otimização constante.

| Baseline interpretado p50 | Após promoção p50 | Razão observada | Compiled hits | Respostas exatas |
|---:|---:|---:|---:|---:|
| 26,919 ms | 1,315 ms | 20,47× | 9.997 | 100,0% |

Isso é uma demonstração real de custo familiar menor para uma regra cuja
equivalência foi verificada, não de descoberta geral de programas ou energia.

## Próximas falsificações

1. Calibrar custo e erro por backend, payload e nível de verificação.
2. Usar payloads semânticos reais em tiers quente/frio e medir page faults,
   bytes movidos e recall cego.
3. Testar COW em DAGs de Factors com cadeias de delta, consultas e rollback.
4. Promover múltiplas famílias apenas após holdout, guards e deoptimization,
   medindo `capability(n)` e `cost(n)` juntos.
