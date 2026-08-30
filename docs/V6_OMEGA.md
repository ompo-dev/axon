# V6/Ω6 — núcleo experimental dirigido por Factors

## Estado

A V6/Ω6 está implementada como um núcleo Rust determinístico, single-threaded
e isolado em `src/core_v6/`. Ela não altera o runtime interativo, a TUI ou o
formato `.axon`. O objetivo é validar contratos de arquitetura antes de ligar
uma versão futura ao sistema persistente.

## O que foi implementado

| Primitiva | Contrato executável |
|---|---|
| Factor Fabric | `Factor` imutável, IDs estáveis, entradas/saídas declaradas, tipo de representação, domínio de validade, proveniência e metadados de aprendizado. |
| Mensagens seletivas | filas local, regional e global; mudanças abaixo do limiar morrem localmente; prioridade é ganho cognitivo por custo declarado. |
| Runtime esparso | `CognitiveRuntime` ativa somente o Factor destino e interrompe antes de ultrapassar o orçamento de operações/bytes. |
| Epistemic Ledger | Claims são imutáveis; revisões usam supersessão; conflitos permanecem locais; modelos específicos e válidos vencem por custo. |
| Aprendizado local | eligibility trace, vetor de ensino e consolidação rápida/lenta alteram somente o alvo elegível. |
| Program VM / Thought JIT | traço verificado recompila para programa; guards inválidos deotimizam e conservam o resultado interpretado. |
| Learnability Gate | escolhe `SOLVE`, `ADAPT`, buscar informação ou `REFRAME` a partir de residual, incerteza, persistência e ganho previsto. |
| Kernel confiável | patch não pode alterar kernel, benchmark ou verificador; exige rollback, testes e validação holdout. |

## Como reproduzir

```powershell
cargo test --lib core_v6
cargo run --bin axon_v6_lab
```

O laboratório não recebe o rótulo-oráculo dos mundos concorrentes. A escolha
ativa de experimento usa apenas desacordo entre previsões e custo declarado.

Cinco execuções independentes produziram o mesmo relatório:

```text
SHA-256: 0EFBBF9F67E0251C0747426E19B5BA103FCC792DE929F6F3B938630D3B7A99AD
```

## Resultados do corte V6.0

| Verificação | Resultado |
|---|---:|
| Fatos indexados / fato one-shot retido | 257 / sim |
| Working set para consulta local | 1 Factor |
| ABR sintético | 0,0039 |
| Supersessão / histórico | revisão nova / 2 versões |
| Modelo barato específico / modelo amplo | sim / sim |
| Mensagens suprimidas / processadas | 1 / 1 |
| Thought JIT: compilação / deoptimization / equivalência | sim / sim / sim |
| Learnability Gate / experimento ativo | completo / sim |
| Conhecimento negativo / patch seguro | retido / limitado pelo kernel |

O ABR é uma razão sintética: 1 Factor ativo de 257, com 64 bytes lógicos por
entrada. Não é medição de RAM física. Da mesma forma, todos os custos desta
rodada usam `CostVector::Declared`; não há alegação de energia, latência,
throughput ou escalabilidade industrial.

## Limites deliberados

- `BTreeMap` e estruturas em memória substituem os stores/indexes escaláveis.
- Não há persistência V6, recuperação hierárquica, wormholes, prefetch,
  micro-batching ou paralelismo regional.
- O conjunto de instruções do VM é propositalmente pequeno e seguro.
- `REFRAME` ainda é uma decisão/gatilho; geração ampla de mundos, busca de
  famílias de representação e simulador contrafactual pertencem às próximas
  rodadas experimentais.
- Nenhum backend GPU, HDC, analógico ou compilador físico está conectado.

## Próximas falsificações

1. Trocar o ledger e índice sintéticos por storage persistente com custos
   `Measured`; medir ABR, KSC, LIE e CR em escala.
2. Executar benchmarks cegos de variáveis ocultas, mudança de representação e
   aprendizado contínuo; reportar falsos `REFRAME`s, custo e generalização.
3. Expandir a Ω-IR e provar equivalência antes de promover qualquer programa
   novo para a biblioteca.
4. Só após esses dados avaliar concorrência regional, placement e backends
   físicos.
