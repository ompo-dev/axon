# Plataforma — o corpo multimodal do Toshi

A virada: o Toshi não aprende dissecando texto morto — aprende **experienciando**, como Chappie/Fushi.
Estímulo cru (texto/áudio/vídeo) → **mimetismo** (imitar, como bebê) → **pensar** → **expressar**
(texto, som, pixels), como e quando quiser.

## Arquitetura (lazy, sem peso à toa)
- **Navegador = corpo**: `getUserMedia`/`getDisplayMedia` dão mic+câmera+tela nativos; `<canvas>` é a
  janela de pixels; WebAudio é a voz/som. Zero libs pra captar.
- **Backend = stdlib** (`servidor.py`): SSE + POST, **sem dependência** além de numpy. Roda o núcleo
  Toshi (`core_v2_prototype/toshi.py`) + o pensar autônomo (`fluxo.py`).
- **LiveKit/Agno**: upgrade pra streaming/rede robustos quando escalar (multi-usuário, WebRTC). Pro
  loop local 1-a-1 não precisa — não instalo peso que não uso.
- **MCP**: plugar depois (internet, ferramentas) — o `POST /stimulus` já é o ponto de entrada.

## Rodar (local)
```bash
cd axon/plataforma
python servidor.py
# abre http://localhost:8770
```
Fale no chat → ele mimetiza (repete o que reconhece), pensa (3 níveis) e responde; **pinta o cérebro**
em pixels em tempo real e **emite som**. Ocioso, ele **pensa sozinho** (devaneia) e às vezes fala.
Botões 🎤/📷/🖥️ ligam mic/câmera/tela — ele recebe o estímulo **cru** (interpretar é o próximo passo).

## Rodar (Docker)
```bash
cd axon
docker build -f plataforma/Dockerfile -t toshi .
docker run -p 8770:8770 -v "$PWD/core_v2_prototype/dados:/axon/core_v2_prototype/dados" toshi
```

## Estado (honesto)
- ✅ loop end-to-end: estímulo de texto → aprende+mimetiza+pensa → expressa (texto+pixels+som); pensar
  autônomo; captura mic/câmera/tela no navegador streamando cru pro backend.
- ⏭️ próximo: **interpretar** áudio/vídeo (o mimetismo real em pixels/som — reproduzir o que vê/ouve),
  memória multimodal no mesmo espaço, e plugar MCP.
