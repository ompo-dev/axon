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
7. Delta mede 10.000 batches e normaliza para **um batch completo de `--queries` updates**, igual ao caminho Full. Saída também deriva ns por query dos dois batches; ambas razões de speedup precisam coincidir. Primeiro batch é comparado ao trace Full; depois do batch final, soma exata do vetor verifica acumulador Delta completo.

Janela medida exclui criação/cópia inicial dos vetores e soma inicial do caminho Delta. Inclui execução completa de cada batch de queries. Reporta p50/p95 batch, derivação por query, as duas razões e bytes lógicos lidos.

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

## Delta Algebra response curve

Binário: `axon-uic-delta-sweep`.

O analisador reconhece somente famílias declaradas no código: `SUM`, `COUNT` e `XOR` têm delta exato constante; `SORT` é global e seleciona `Full`. Ele não infere nem prova `ΔF` para programas arbitrários.

Workload local:

1. Vetor físico de `u64` com `--mib` MiB.
2. Stream determinístico de 1 a 8.000.000 eventos; cada chave recebe quatro escritas adjacentes.
3. Mede quatro caminhos: `Full`, `Full+coalesce`, `Delta` e `Delta+coalesce`.
4. `coalesce` é incluído na janela medida e conserva somente a última escrita de cada rajada adjacente.
5. Para pontos pequenos, Delta repete batches até um milhão de eventos e normaliza por batch. Valor muda por epoch; não mede repetição de no-op.
6. Primeiro batch precisa igualar checksum de `Full`; no fim, cada acumulador Delta é relido por checksum exato. Falha encerra sweep.
7. Escolha reportada compara menor p50 de `Full`/`Full+coalesce` contra menor p50 de `Delta`/`Delta+coalesce`. O contrato ainda força `Full` quando não existe delta exato.

Janela exclui cópia inicial do vetor e soma inicial do acumulador Delta. Inclui aplicação de eventos, coalescência quando usada e checksum `Full`. `Logical reads` conta somente leitura algorítmica da redução; não mede tráfego de DRAM, alocação ou energia.

Comando:

```powershell
cargo run --release --bin axon-uic-delta-sweep -- --mib 64 --runs 5 --max-updates 8000000
```

## Hybrid Oracle e Change Fabric

Binário: `axon-uic-hybrid-sweep`.

```powershell
cargo run --release --bin axon-uic-hybrid-sweep -- --mib 64 --runs 30 --hardware-id <cpu-ram-profile>
```

Workload: 64 shards de 1 MiB; oito shards densos usam `FULL_LOCAL`, um shard usa Delta coalescido, um usa Delta bruto e 54 usam `SKIP`. Toda rodada exige paridade exata entre Full, Delta, Delta coalescido, Hybrid, Oracle e Change Fabric.

Hybrid fim a fim inclui validação, índice, classificação, coalescência e execução. Oracle usa o mesmo plano pré-compilado, mas executa-o fora desse custo somente como diagnóstico; nunca é comparado como headline contra Full/Delta. Change Fabric cria esse plano durante a ingestão e reporta `ingest + query`, para não deslocar custo de compilação. Cada rodada percorre uma das seis ordens de execução determinísticas permutadas.

Métricas: `Adaptation Tax = compiler / executor`, `Oracle Gap = Hybrid fim a fim / Oracle` e `Change Fabric Adaptation Tax = ingest / query`. A verificação por checksum é medida fora de todos os timers de execução e publicada separadamente. `StrategyEvidence` recebe amostras pareadas Raw Delta/Oracle; só registra `Supported` ou `LatencyDominated` quando todos os pares concordam, senão não cria `MetaJit`.

Limite: stream é canônico e ordenado por shard; coalescência exige `FinalStateOnly`; `SUM u64` favorece incrementalização. Este sweep não usa árvore hierárquica, modelo online de custo ou DeltaForge.

## DeltaForge-SUM

Binário: `axon-uic-deltaforge-sum`.

```powershell
cargo run --release --bin axon-uic-deltaforge-sum -- --mib 64 --runs 15
```

Entrada é somente `FoldSpec::AddModU64`. Forge deriva `CommutativeGroup`, cache `ModularTotal` e regra `SubtractOldThenAddNew`; não recebe updater. Cada ponto mede três caminhos com ordem determinística permutada em seis ordens: Full, updater Raw manual e plano derivado. Os três materializam mesmo `ReplaceDelta` canônico; Full refaz fold, Raw/Forge partem de total inicial mantido. Verificação exata fica fora do timer; Forge também executa checker de certificado fora do timer.

O benchmark cobre 1.024, 1.000.000 e 4.000.000 replacements únicos que cabem no vetor. Registra todos os samples no baseline. Não mede inferência de álgebras arbitrárias, síntese de estado auxiliar para `MIN`, prova formal, transferência ou aprendizado.
