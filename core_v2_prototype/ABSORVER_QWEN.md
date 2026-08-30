# ABSORVER QWEN — o Toshi se alimenta de outra IA (destilação + transcrição + teste)

Não é um chat entre Qwen e Toshi. É **absorção**: o Qwen exporta conhecimento, nós
**transcrevemos** para a representação nativa do Toshi e **testamos automaticamente** o que entrou.

## Por que adaptar/transcrever (e não copiar pesos)

| | Qwen (transformer) | Toshi (VSA / Random Indexing) |
|---|---|---|
| conhecimento | matrizes de pesos (GB, opacas) | associações, transições, embeddings 1024-d e grafo de fatos |
| aprender | gradiente, offline | 1-shot, online, sem gradiente |
| memória | KV-cache volátil | persistente e leve |

Copiar os pesos do transformer para o Toshi é **impossível e errado**. O que se faz é a
**destilação do conhecimento**: o Qwen vira a fonte (o "CSV") e cada frase é convertida para
as estruturas nativas do Toshi (as "células do Excel").

## Pipeline

```
        QWEN (via Ollama)                    TOSHI
              │                                ▲
 1. EXTRATOR: pede fatos,                     │
    definições e analogias em lote            │
              │                                │
 2. TRANSCRITOR: frase ->                     │
    - tokenize + perceive (repetido) ─────────┤ associação + transição
    - contextos paralelos        ─────────────┤ significado (Random Indexing)
    - extrair(sujeito,rel,obj)   ─────────────┤ grafo factual (fatos.py)
              │                                │
 3. ABSORVEDOR: salva estado e registra       │
              │                                │
 4. BATERIA AUTOMÁTICA (antes/depois):        │
    - fidelidade dos fatos ───────────────────┤
    - evocação associativa ───────────────────┤
    - analogia (aritmética de significado) ───┤
    - cobertura de vocabulário ───────────────┤
                                               │
 5. RELATÓRIO: dados/absorcao_qwen_report.json
```

## Como rodar

```cmd
cd C:\Projects\Teste\axon\core_v2_prototype

REM 1) selftest offline (valida o pipeline sem rede)
python absorver_qwen.py --selftest

REM 2) absorção real (Qwen local via Ollama precisa estar rodando)
python absorver_qwen.py --modo tudo --temas "geografia,animais,corpo humano,tecnologia" --itens 12

REM 3) só fatos
python absorver_qwen.py --modo fatos --temas "historia,ciencia" --itens 15

REM 4) só analogias
python absorver_qwen.py --modo analogia --analogias 10

REM 5) absorção em ÁRVORE (subtemas também entram, com orçamento)
python absorver_qwen.py --expandir --temas "geografia,biologia" --profundidade 2 --max-temas 30
```

## O que o teste automático mede (a regra do RADAR: sem métrica, não entra)

1. **Fatos no grafo** — cada fato destilado foi gravado no grafo factual (fidelidade da transcrição).
2. **Fatos em linguagem** — o Toshi responde a pergunta quando ela é NÃO-ambígua
   (se o sujeito tem várias definições, só o grafo é cobrado — a pergunta é ambígua por natureza).
3. **Associações** — dado o sujeito, o objeto esperado precisa acender no top-10.
4. **Analogias** — `a está para b como c está para d` precisa aparecer no top-5 de `toshi.analogy(a,b,c)`.
5. **Vocabulário** — toda palavra nova precisa ter sido vista (cobertura).

O relatório mostra o score de cada categoria e o crescimento estrutural
(conceitos, associações, transições, fatos).

## Limite honesto

Isto destila o conhecimento que o Qwen consegue **exportar em palavras**. Não transfere a
capacidade de raciocínio do transformer, não copia pesos e não é uma "fusão de modelos".
O que entra no Toshi é o que o teste mede — nada além disso.
