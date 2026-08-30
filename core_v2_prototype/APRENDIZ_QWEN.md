# APRENDIZ QWEN — por que absorver pesos não bastava, e o ciclo que falta

## A pergunta certa

> "Se o Toshi absorveu o Qwen, por que não responde as mesmas coisas?"

Porque são duas absorções diferentes:

| O que foi absorvido | O que dá |
|---|---|
| **Pesos** (assinaturas HD + palavras de peso) | o MODELO como memória |
| **Comportamento** (perguntas/respostas) | a CAPACIDADE de responder |

Absorver pesos é guardar o dicionário inteiro em formato comprimido. Não é o mesmo
que aprender a conversar com ele. Por isso o passo seguinte é o CICLO:

```
pergunta
   │
   ▼
Toshi tenta responder da memória (fatos + QA comidas)
   │ não sabe
   ▼
Qwen (professor) responde
   │
   ▼
Toshi COME a resposta:
   • toshi.perceive(pergunta + resposta)   → associações/transições novas
   • toshi.settle(...)                     → reação em cadeia nos vizinhos
   • fatos.aprender(...)                   → memória factual crisp
   │
   ▼
próxima vez, a MESMA pergunta sai da memória dele, SEM Qwen
```

## Como rodar

```cmd
cd C:\Projects\Teste\axon\core_v2_prototype

REM selftest (prova o ciclo com professor offline)
python toshi_aprendiz.py --selftest

REM pergunta única (Qwen real via Ollama)
python toshi_aprendiz.py --pergunta "que cor tem o abacaxi?"

REM modo interativo: pergunta, ele aprende, pergunta de novo e ele responde sozinho
python toshi_aprendiz.py --interativo
```

No interativo:
```
voce> que cor tem o abacaxi?
toshi> ... (aprendi agora do Qwen)
voce> que cor tem o abacaxi?
toshi> ... 🧠 (da minha memória)
```

`/stats` mostra quantas palavras/conceitos/fatos cresceram. O estado é salvo em
`toshi_state.pkl` + `fatos.json` — o aprendizado é permanente e em tempo real.
