# AXON-UIC experiments

## Demand×Delta physical sweep

Binário: `axon-uic-bench`.

Workload real em CPU/RAM local:

1. Materializa dois vetores `u64` iguais, cada um com `--mib` MiB físicos.
2. Aplica mesma sequência determinística de atualizações pontuais a ambos.
3. Baseline recalcula soma exata lendo vetor inteiro após cada update.
4. Delta atualiza soma exata com `old` e `new`, sem reler vetor inteiro.
5. Compara trace e checksum final em toda rodada. Divergência encerra benchmark.
6. Faz duas warmups e mede 1–10 rodadas, alternando ordem full/delta.
7. Delta mede 10.000 batches e normaliza tempo para uma sequência de queries. Isso evita vender resolução de relógio como performance. Primeiro batch é comparado ao trace full; depois do batch final, soma exata do vetor verifica acumulador delta completo.

Janela medida exclui criação/cópia inicial dos vetores e soma inicial do caminho delta. Inclui execução completa de cada sequência de queries. Reporta p50, p95, speedup p50 e bytes lógicos lidos.

## Como interpretar

- Mede uma lei específica: alteração local em uma redução associativa pode preservar exatidão sem reler estado inteiro.
- Vetores são físicos, tempo vem do PC local. Não é score sintético de inteligência.
- Não mede joules, GPU, energia, retrieval, qualidade cognitiva, LIFT geral, CEGAR geral ou capacidade de AGI.
- `Logical reads` descreve trabalho algorítmico da sequência; não é contador de DRAM nem lower bound de I/O.

## Comando padrão

```powershell
cargo run --release --bin axon-uic-bench -- --mib 64 --queries 20 --runs 5
```

Repetir depois de mudanças que afetem `ExecutionSlice`, acumuladores, parsing do benchmark ou contratos de fallback. Registrar hardware, comando e todos os números no commit que alterar baseline.
