# V6-X — Physical Boundary Sweep

## Escopo e reprodução

Esta rodada mede CPU e RAM do computador local, não perfis declarados. O
executável coleta antes do sweep e registra no próprio relatório o
identificador de CPU fornecido pelo sistema, RAM, threads lógicas, plano de
energia, sistema, arquitetura e compilador. Se a política do sistema nega
acesso ao nome comercial da CPU, o identificador técnico é impresso em vez de
um modelo inventado. Os resultados abaixo foram feitos em Intel Core i7-13650HX,
16 GB RAM, Windows em modo Desempenho Máximo e build Rust `--release`.

```powershell
cargo test --bin axon_v6x_physical_sweep
cargo run --release --bin axon_v6x_physical_sweep -- --runs 3
```

Cada rodada usa três aquecimentos e 15 amostras. As tabelas registram a mediana
dos três p50 internos e a faixa entre rodadas. Não são medições de energia,
temperatura, GPU, p-bit, fotônica, crossbar analógico ou quantum.

## Resultado 1 — dissecação da fronteira, 512 KiB

| Kernel executado | p50 | Faixa p50 | Leitura |
|---|---:|---:|---|
| `fused bind: A+B→C` | 25,0 µs | 22,1–29,7 µs | Sem representação intermediária. |
| Encode de A e B | 40,2 µs | 32,5–86,2 µs | Converte sinais `i8` em bits. |
| Cópia de A' e B' | 51,8 µs | 32,4–51,9 µs | Materializa/transfere buffers. |
| XOR de A' e B' | 20,4 µs | 16,5–58,4 µs | Kernel na representação codificada. |
| Decode de C' | 7,5 µs | 7,4–8,7 µs | Volta à representação do runtime. |
| Caminho de fronteira completo | 147,8 µs | 100,7–210,7 µs | Encode → cópia → XOR → decode. |
| Alocar e tocar 8 buffers de fronteira | 2,877 ms | 2,876–2,971 ms | Mede alocação, inicialização e leitura de cada buffer. |

O caminho completo foi **5,91×** o kernel fundido nessa dissecação. Os estágios
isolados não devem ser somados como contabilidade exata: cada um foi medido em
uma execução separada, com estado de cache diferente. A conclusão suportada é
direcional: materialização, cópia e conversão dominam a diferença; não é uma
atribuição precisa de microssegundos a uma única instrução. A linha de
alocação/touch mede oito vetores, mas não entra nessa razão porque não faz parte
do caminho já pré-alocado.

## Resultado 2 — curva por tamanho

| Tamanho | Direct | Boundary | Razão boundary/direct |
|---:|---:|---:|---:|
| 64 B | 0,003 µs | 0,015 µs | 4,43× |
| 256 B | 0,011 µs | 0,025 µs | 2,36× |
| 1 KiB | 0,044 µs | 0,070 µs | 1,60× |
| 4 KiB | 0,168 µs | 0,249 µs | 1,48× |
| 16 KiB | 0,675 µs | 1,597 µs | 2,37× |
| 64 KiB | 2,688 µs | 6,062 µs | 2,26× |
| 256 KiB | 10,750 µs | 46,950 µs | 4,37× |
| 512 KiB | 31,200 µs | 132,500 µs | 4,25× |
| 1 MiB | 60,700 µs | 259,500 µs | 4,28× |
| 4 MiB | 235,300 µs | 1,488 ms | 6,33× |
| 16 MiB | 1,664 ms | 8,375 ms | 5,03× |
| 64 MiB | 7,911 ms | 33,064 ms | 4,18× |

A razão não é constante: varia de 1,48× a 6,33× nesta máquina e workload.
Portanto, o Physics Compiler deve aprender custo condicionado a tamanho e
localidade; um multiplicador global de Boundary Tax seria incorreto.

Para cada ponto, o binário também executa `assert_eq!` integral entre pipeline
e kernel fundido e imprime o checksum completo determinístico. Isso prova a
igualdade deste workload medido; não transforma a verificação amostrada L1 em
verificação exata. Como âncora de reprodução, o checksum do ponto de 512 KiB
foi `18C538C908052B0C`.

## Resultado 3 — cache, localidade e cópia

| Comparação real | Resultado | Leitura |
|---|---:|---|
| Bind quente / após sweep de 64 MiB | 26,2 / 54,3 µs | Após tentativa de evacção de cache: **2,07×**. Não é flush de cache garantido. |
| Factors agrupados / dispersos, 32 MiB | 821,2 / 1.047,3 µs | Acesso disperso foi **1,28×** mais lento. |
| `FactorHandle` / `FactorCopy`, 512 KiB | 27,1 / 41,5 µs | Copiar/materializar foi **1,53×** mais lento. |

Há evidência local para três regras de implementação: favorecer cache quente,
agrupar factors coativados e preferir handles/referências a cópias. Ainda não
prova que uma política de *memory placement learning* melhora uma tarefa
cognitiva fim a fim; mede somente o custo físico dessa escolha.

## Resultado 4 — VERIFY, descarte e REALIZE

| Trabalho | p50 | Interpretação correta |
|---|---:|---|
| L0: checar dimensão | 0,426 ns | Invariante barato, insuficiente contra corrupção de conteúdo. |
| L1: 64 posições amostradas | 25 ns | Custo baixo; sem semente secreta/aleatória, não fornece bound adversarial de falha. |
| L3: igualdade exata de 512 KiB | 17,525 µs | Verificação completa tem custo material. |
| `ZERO`, scratch 4 KiB | 14,453 ns | Operação simples de buffer nesta implementação. |
| checkpoint + restore, 4 KiB | 24,609 ns | Buffer já disponível. |
| recompute, 4 KiB | 42,187 ns | Reexecuta XOR de origem e chave. |
| `UNCOMPUTE` lógico, 4 KiB | 1,350 µs | Inclui clones e preservação de resultado de `ReversibleScratch`. |

Essas estratégias de descarte não têm a mesma semântica ou custo futuro. Em
especial, `UNCOMPUTE` não deve ser chamado de pior apenas por esta tabela: ele
preserva resultado e restaura estado. O próximo experimento deve medir custo de
vida completo por política (`DROP`, `ZERO`, `CHECKPOINT`, `RECOMPUTE`,
`UNCOMPUTE`).

`REALIZE` mediu cerca de 32,9 ns por decisão de 10 a 1.000.000 decisões numa
região; uma região de um elemento mediu 59 ns. Isso mostra overhead baixo e
linear para o perfil CPU atual, mas não autoriza realizar milhões de decisões
individuais sem agrupamento: a compilação por região continua a forma correta.

## Finding V6-X #1 — Boundary and materialization dominance

Neste i7-13650HX, para este bind denso, transformar representação e
materializar buffers elevou o custo para aproximadamente 1,5×–6,3× do kernel
fundido, conforme tamanho e cache. A evidência apoia priorizar:

```text
fusão de kernel
→ menos materialização
→ menos cópia/movimento
→ plano físico por região e contexto
```

Ela não demonstra vantagem de p-bit, fotônica ou outro acelerador; apenas
mostra que, na CPU local, o custo de fronteira é grande o bastante para ser
controlado como primeira classe.
