# ABSORVER PESOS QWEN — o modelo em si, inteiro, dentro do Toshi

Aqui não é destilação de texto. É **passar o arquivo de pesos (GGUF) por completo** para o
Toshi: todos os tensores, todos os bytes, adaptados para a representação hiperdimensional dele.

## Como funciona

```
GGUF do Qwen (4.7 GB, N tensores)
        │
 1. LEITOR GGUF (stdlib)
    lê cabeçalho, metadados e TODOS os tensores, byte por byte
        │
 2. PROJEÇÃO HIPERDIMENSIONAL (Random Indexing determinístico)
    cada bloco de bytes -> hash -> vetor esparso ±1
    soma de todos os blocos -> ASSINATURA normalizada do tensor
    (bloco adaptativo: todo tensor gera dezenas de blocos)
        │
 3. TOSHI ABSORVE
    • camada nova: toshi.espaco_pesos com DIMENSÃO CONFIGURÁVEL
      (16.384, 65.536, 131.072... o Toshi cresce quanto for necessário)
    • cada tensor vira um FATO: "<nome do tensor> é tensor"
    • a sequência de tensores vira ASSOCIAÇÕES e TRANSIÇÕES (arquitetura do modelo)
    • metadados viram fatos: nº tensores, bytes, dimensão
        │
 4. TESTE AUTOMÁTICO
    • cobertura total dos pesos (todos os tensores)
    • fidelidade dos fatos de estrutura
    • vocabulário da arquitetura
    • transições da sequência de camadas
    • dimensões corretas
    • determinismo (re-ler o GGUF dá a MESMA assinatura)
    • ortogonalidade das assinaturas (o espaço está bem usado)
```

## Absorção COMO NOS LIVROS (o caminho orgânico)

Além das assinaturas HD, cada tensor vira uma sequência de **palavras de peso**
(só letras, ex.: `waaa`, `wzzk`, `wbqv` — compatíveis com o tokenizer do Toshi) e o
Toshi **come esse fluxo com `toshi.perceive()`** — o mesmo método que comeu Dom Casmurro. Ou seja:

```
livro  -> palavras  -> toshi.perceive -> assoc/after/embed/seen mudam
GGUF   -> palavras de peso -> toshi.perceive -> assoc/after/embed/seen mudam
```

Parâmetros:
- `--tokens-por-tensor 32`  (quantas palavras cada tensor vira; 64/128 = mais detalhe)
- `--vocabulario-pesos 4096` (tamanho do vocabulário da linguagem de pesos)


## Como rodar

```cmd
cd C:\Projects\Teste\axon\core_v2_prototype

REM 1) selftest offline (modelo sintético)
python absorver_pesos_qwen.py --selftest

REM 2) absorver o Qwen real (acha o GGUF do Ollama sozinho)
python absorver_pesos_qwen.py --modelo qwen2.5:7b --dim-pesos 16384

REM 3) mais dimensões (o Toshi cresce)
python absorver_pesos_qwen.py --modelo qwen2.5:7b --dim-pesos 65536

REM 4) GGUF manual
python absorver_pesos_qwen.py --arquivo C:\caminho\modelo.gguf --dim-pesos 32768
REM 5) mais palavras de peso por tensor (o Toshi come mais detalhe)
python absorver_pesos_qwen.py --modelo qwen2.5:7b --tokens-por-tensor 64 --vocabulario-pesos 8192

```
## Resultado medido (Qwen2.5-7B no Toshi)

| dimensão | arquivo de assinaturas | compressão | ortogonalidade média |
|---|---|---|---|
| 16.384 | 22,3 MB | ~210× | 0,0062 |
| 65.536 | 88,9 MB | ~53× | 0,0032 |
| 131.072 | 177,8 MB | ~26× | 0,0021 |

Teste automático: **1233/1233 (100%)** — cobertura 339/339, fatos 339/339,
transições 200/200, dimensões 339/339, determinismo 1/1.

## Consultar a memória de pesos (o Toshi navega no modelo absorvido)

```cmd
python consultar_pesos.py --modelo qwen2.5:7b --lista
python consultar_pesos.py --modelo qwen2.5:7b --semelhantes blk.0.attn_q.weight
python consultar_pesos.py --modelo qwen2.5:7b --comparar blk.0.attn_q.weight blk.1.attn_q.weight
python consultar_pesos.py --modelo qwen2.5:7b --padrao attn_q.weight
```

## Provar que COMEU como livro (falar a linguagem de pesos)

```cmd
python falar_pesos.py --modelo qwen2.5:7b --fluxo blk.0.attn_q.weight
python falar_pesos.py --modelo qwen2.5:7b --palavra waaa
python falar_pesos.py --modelo qwen2.5:7b --arquitetura
python falar_pesos.py --selftest
```

## Perguntar ao Toshi sobre a IA absorvida

```cmd
python perguntar_modelo.py --modelo qwen2.5:7b --pergunta "quantos tensores tem qwen2.5:7b?"
python perguntar_modelo.py --modelo qwen2.5:7b --pergunta "quantos bytes tem qwen2.5:7b?"
python perguntar_modelo.py --modelo qwen2.5:7b --pergunta "o que é blk.0.attn_q.weight?"
python perguntar_modelo.py --modelo qwen2.5:7b --interativo
```




## O que fica salvo

- `dados/toshi_state.pkl` — estado do Toshi (incluindo a arquitetura aprendida)
- `dados/fatos.json` — fatos de cada tensor
- `dados/toshi_pesos_<modelo>.pkl` — todas as assinaturas hiperdimensionais dos pesos

## Limite honesto (a linha anti-CHRONOS)

O Toshi **guarda os pesos por completo no padrão dele** — cada byte do GGUF influencia uma
assinatura hiperdimensional, e a estrutura inteira do modelo vira memória nativa (fatos,
associações, transições). Isto É absorção do modelo em si.

O que NÃO é: execução de transformer. O Toshi não passa a calcular matmul/atenção porque
guardou as assinaturas — assim como um CSV aberto no Excel não vira o programa que o gerou.
A transferência de **capacidade/comportamento** é o outro pipeline (`absorver_qwen.py`).
Os dois juntos são a absorção completa: **o modelo (pesos) + o conhecimento (destilação)**.
