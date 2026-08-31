# AXON-UIC architecture

AXON-UIC implementa primeira camada concreta das propostas consolidadas: computação sob orçamento contrai incerteza; uma ação ocorre apenas com certificado; uma otimização não pode reduzir correção ou aumentar autoridade.

## Fluxo executável

```text
Goal + belief bounds
        |
DecisionCertificate
   | certified?             no
   | yes                     |
 act                 choose REFINE by expected ambiguity reduction / physical cost
                              |
                         tighter bounds

Capability request
        |
observable + identifiable + reachable + conditioned + affordable + authorized?
        |
    allow / deterministic refusal

Base morphology + task regions + resource budget
        |
Pareto allocation
        |
candidate remorph pays Migration Tax and preserves SemanticContract?
        |
transaction eligible / reject
```

## Contratos

| Tipo | Regra |
|---|---|
| `Interval` | limites invertidos são erro; limites refinados devem ser subconjunto dos anteriores |
| `DecisionCertificate` | certifica somente se `lower(a*) > upper(a)` para toda ação rival |
| `ExecutionSlice` | executa somente `Backward(goal) ∩ Forward(delta)` |
| `LiftCertificate` | aceita apenas classe não vazia de valores fonte idênticos; qualquer outro caso não recebe LIFT |
| `AbstractionContract` | cada transição observável deve ficar dentro de `error_bound` |
| `CapabilityGate` | efeito pedido precisa ser subconjunto de `Authority`; gate não executa efeitos |
| `Morphology` | bytes protegidos são inviáveis de desalocar; tiers competem por utilidade marginal |
| `RemorphPolicy` | contratos iguais, dwell mínimo e benefício futuro estritamente maior que migração + histerese |
| `run_checked` | em modo verificação, resultado otimizado só passa se for igual ao exato |

## Deliberações de engenharia

- Rust stdlib apenas. Não há modelo, rede, banco ou execução de tools escondida.
- `CapabilityGate` representa autorização; não chama rede, sistema de arquivos, processos nem atuadores.
- Allocation usa fronteira Pareto finita e recusa mais de 100.000 estados. `ponytail:` busca exata pequena; trocar por solver incremental somente quando benchmark real exigir escala maior.
- `run_checked` calcula fallback exato em toda execução de verificação. Produção só poderá pular essa comparação com certificado específico, testado e versionado.
- `LiftCertificate` é deliberadamente restrito. Não afirma automorfismo geral, approximate LIFT, causal LIFT ou descoberta aberta de simetria.

## Próximos cortes científicos, não promessas

1. `RefineBench`: decidir com níveis L0/exato e medir custo por ambiguidade.
2. `LiftBench`: ampliar certificado de fontes idênticas para lanes estruturais, com paridade obrigatória.
3. `ResourceMarketBench`: workloads factual, relacional, programa e update sob orçamento variável.
4. `CausalAbstractionBench`: preservar intervenções, não só transições observadas.
5. `SecurityEffectBench`: prova automatizada de zero efeitos fora da authority.

Cada corte deve adicionar contrato, teste de regressão e benchmark local antes de alegar capacidade nova.
