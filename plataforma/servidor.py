"""
PLATAFORMA — o CORPO do Toshi. Estímulo cru entra; ele mimetiza, pensa e EXPRESSA (texto+pixels+som).

A virada (usuário): o Toshi não aprende dissecando texto morto — aprende EXPERIENCIANDO, como o
Chappie/Fushi. Recebe estímulo (texto/áudio/vídeo), não sabe o que é -> MIMETISMO (imitar, como
bebê aprende) -> pensar -> expressar (texto, som, pixels), como e quando quiser.

Aqui: backend stdlib (SSE + POST, sem dependência) que roda o núcleo Toshi. O NAVEGADOR é o corpo
(mic/câmera/tela nativos, canvas = janela de pixels, WebAudio = som). LiveKit/Agno seriam o upgrade
de streaming/rede; pro loop local 1-a-1 isto basta e não pesa. Docker por cima; MCP depois.

Endpoints:
  GET  /            -> a interface (index.html)
  GET  /stream      -> SSE: expressões do Toshi (texto, grade de pixels do cérebro, som) em tempo real
  POST /stimulus    -> um estímulo {tipo:'texto'|'imagem'|'audio', dado:...}

Rodar:  python servidor.py   (abre http://localhost:8770)
"""
import os
import sys
import json
import time
import threading
import queue
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core_v2_prototype"))
import heapq
import random
from toshi import build_or_load, tokenize, save_state           # o núcleo (a mente)
from fluxo import Fluxo                                          # o pensar autônomo
from mimetizador import Mimetizador                             # imitar p/ aprender e depois gerar
from explorador import buscar as wiki_buscar, links as wiki_links  # FERRAMENTA: ler a internet
from fatos import Fatos, extrair as fato_extrair                # memória FACTUAL crisp (roubado do neurocore)

VW, VH = 64, 48                                                 # resolução da "visão" em COR (maior, renderizada suave)

W, H = 48, 30                                                    # tamanho da janela de pixels do cérebro
PORT = 8770

print("acordando o Toshi (carregando a mente)...", flush=True)
T, _ = build_or_load()
FLUXO = Fluxo(T)
FLUXO.semear(tokenize("mundo luz som"))
FATOS = Fatos()                                                # grafo de fatos (nome, capital, etc.)
MIM_V = Mimetizador(VW * VH * 3, k=160, lr=0.4, novidade=0.24) # mímica de VÍDEO 64x48 em COR (ágil)
MIM_A = Mimetizador(32, k=48, lr=0.4)                          # mímica de ÁUDIO (32 bandas do mic, mais rico)
_dream = None                                                  # o "sonho" atual (frame que morfa)
_last_input = 0.0                                              # quando recebeu estímulo ao vivo (ocioso?)
_ult_rec = (None, 0.0)                                         # último rosto reconhecido (anti-spam)
# MEMÓRIA EPISÓDICA (hipocampo): conceito -> clipe de rosto em ALTA-RES (JPEG). Bind 1-shot; pattern
# completion (digitar o nome -> revive o clipe). É "ele lembrou de você".
MEMORIA = {}                                                   # palavra -> deque(JPEG dataURL) (clipe)
_ctx = deque(maxlen=40)                                        # (palavra, quando) digitadas recentemente
_ult_jpeg = None                                               # último rosto visto em alta-res
MEM_JANELA = 8.0                                               # s: liga texto e imagem próximos no tempo
_clientes = []                                                  # filas SSE (um navegador = uma fila)
_lock = threading.Lock()
MIM_LOCK = threading.Lock()                                     # protege a mímica (estado compartilhado)
T_LOCK = threading.Lock()                                       # protege o grafo do Toshi (wiki muta enquanto pensa)


def _grade_cor(vec):
    """Vetor COR (VW*VH*3, 0..1) -> lista FLAT de bytes RGB. Sem loop Python, sem aninhar = rápido."""
    return (np.clip(np.asarray(vec, np.float32), 0, 1) * 255).astype(np.uint8).tolist()


def _bandas_som(vec):
    """32 bandas de energia -> ESPECTRO [freq, amplitude] das 12 mais fortes. Síntese ADITIVA no
    navegador reconstrói o timbre do que ele ouviu = mímica de áudio de verdade (não 6 tons soltos)."""
    v = np.asarray(vec, float)
    ordem = np.argsort(-v)[:12]
    esp = [[round(70 + int(i) * 150, 1), round(min(1.0, float(v[i]) * 1.6), 3)] for i in ordem if v[i] > 0.02]
    return esp or [[220.0, 0.3]]


# ---------- layout 2D do cérebro (posição = significado) ----------
def _layout(n=280):
    stops = T.stops
    ws = [w for w, _ in T.seen.most_common(n * 3) if w not in stops and T._emb(w) is not None][:n]
    E = np.array([T._emb(w) for w in ws])
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)   # normalizado -> cos por produto
    Ec = E - E.mean(0)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    xy = Ec @ Vt[:2].T
    xy -= xy.min(0); xy /= (xy.max(0) + 1e-9)
    gx = (xy[:, 0] * (W - 1)).astype(int); gy = (xy[:, 1] * (H - 1)).astype(int)
    return ws, gx, gy, E                                     # E = embeddings do layout (p/ o calor)

LWORDS, GX, GY, LEMB = _layout()


def cerebro_pixels(foco_words):
    """Cada NEURÔNIO acende pela similaridade ao pensamento; borrão via numpy (VETORIZADO = rápido,
    não trava). Retorna lista FLAT de bytes RGB (H*W*3)."""
    embs = [T._emb(w) for w in foco_words if T._emb(w) is not None]
    q = np.mean(embs, 0) if embs else LEMB.mean(0)
    q = q / (np.linalg.norm(q) + 1e-9)
    sims = np.clip(LEMB @ q, 0, 1)
    # INIBIÇÃO E/I (PV/SST/VIP): normalização DIVISIVA (ganho, canônico cortical) + k-WTA (só os mais
    # ativos disparam, o resto é INIBIDO) -> disparo ESPARSO e NÍTIDO (córtex real / Kanerva SDM).
    sims = sims / (sims.mean() + 1e-6)
    k = max(10, int(0.12 * len(sims)))                      # ~12% disparam (esparsidade cortical)
    corte = np.partition(sims, -k)[-k]
    acts = np.where(sims >= corte, sims, 0.0) ** 1.3        # inibe o resto (basket cells)
    if foco_words:
        fs = set(foco_words)
        acts = acts + np.array([1.2 if w in fs else 0.0 for w in LWORDS], np.float32)  # o foco brilha +
    field = np.zeros((H, W), np.float32)
    np.add.at(field, (GY, GX), acts)                        # scatter (vetorizado)
    for _ in range(3):                                      # borrão = blur separável (numpy, rápido)
        field = 0.36 * field + 0.16 * (np.roll(field, 1, 0) + np.roll(field, -1, 0)
                                       + np.roll(field, 1, 1) + np.roll(field, -1, 1))
    fmax = field.max() or 1.0
    return _heat_np((field / fmax) ** 0.6).reshape(-1).tolist()


# rampa térmica vívida (escuro-azul -> ciano -> verde -> amarelo -> vermelho -> branco)
_RAMP = [(6, 8, 20), (22, 26, 92), (28, 96, 184), (30, 186, 176), (120, 205, 66),
         (240, 214, 66), (250, 126, 44), (238, 60, 42), (255, 244, 224)]


_RAMP_NP = np.array(_RAMP, np.float32)


def _heat_np(v):
    """v: array [0,1] -> (...,3) uint8 térmico, VETORIZADO (interpola a rampa de uma vez)."""
    x = np.clip(v, 0, 1) * (len(_RAMP) - 1)
    i = np.floor(x).astype(int)
    f = (x - i)[..., None]
    a = _RAMP_NP[i]
    b = _RAMP_NP[np.minimum(i + 1, len(_RAMP) - 1)]
    return (a + (b - a) * f).astype(np.uint8)


def voz(foco_words):
    """A VOZ dele — SEMPRE disponível (não podamos os canais; ele aprende o que/quando usar). Se já
    ouviu áudio, usa o protótipo aprendido; senão vocaliza a partir do PENSAMENTO (determinístico:
    mesma ideia -> mesmo som, não random). E ele SE OUVE (cópia eferente, atenuado)."""
    with MIM_LOCK:
        ga = MIM_A.gerar(foco_words)
        if ga is not None:
            MIM_A.sentir(ga, foco_words, taxa=0.04)         # ouve a própria voz (auto-percepção)
            return _bandas_som(ga)
    e = next((T._emb(w) for w in foco_words if T._emb(w) is not None), None)   # senão, som do pensamento
    if e is None:
        return None
    idx = np.argsort(-np.abs(e))[:6]                        # componentes fortes do significado -> espectro
    return [[round(150 + float(abs(e[i])) * 640, 1), round(0.2 + float(abs(e[i])) * 0.6, 3)] for i in idx]


# ---------- MEMÓRIA MULTIMODAL UNIFICADA: o CONCEITO liga rosto+voz+nome+texto (pattern completion) ----------
def recordar(conceito, motivo="lembrei"):
    """Recall HOLÍSTICO: uma pista (o conceito) reinstaura TODAS as modalidades ligadas — o rosto
    (reconstruído), a voz (som aprendido) e o significado (texto). É o hipocampo completando o padrão."""
    with MIM_LOCK:
        gv = MIM_V.gerar([conceito]) if (conceito in MIM_V.assoc and MIM_V.assoc[conceito]) else None
        ga = MIM_A.gerar([conceito]) if (conceito in MIM_A.assoc and MIM_A.assoc[conceito]) else None
    if gv is not None:
        expressar({"tipo": "video_out", "grid": _grade_cor(gv), "gerado": True, "lembrou": conceito})
    if ga is not None:
        expressar({"tipo": "audio_out", "som": _bandas_som(ga), "gerado": True})
    rel = [u for u, _ in T.associations([conceito], 5) if u not in T.stops][:4]   # o que significa (texto/wiki)
    expressar({"tipo": "recordacao", "conceito": conceito, "motivo": motivo,
               "tem_rosto": gv is not None, "tem_voz": ga is not None, "relacionado": " ".join(rel)})


# ---------- o ciclo: estímulo -> mimetismo -> pensar -> expressar ----------
def mimetizar(words):
    """MIMETISMO: ele repete de volta o que reconhece (imitar p/ absorver, como bebê). O que é
    novo, ele marca como novo (curiosidade). É o 1º passo antes de pensar."""
    conhecidas = [w for w in words if T.seen[w] > 0]
    novas = [w for w in words if T.seen[w] == 0]
    return conhecidas, novas


def expressar(event):
    with _lock:
        for q in list(_clientes):
            q.put(event)


# ---------- FERRAMENTA: ler a Wikipedia PT-BR e APRENDER (texto é DADO, não comando) ----------
_wiki_hist = deque(maxlen=200)


def ler_wiki(termo=None):
    titulo, txt = wiki_buscar(termo)
    if not txt or len(txt) < 60:
        if termo:
            expressar({"tipo": "lendo", "titulo": termo, "chars": 0})   # não achou
        return None
    with T_LOCK:
        T.eat(txt)                                              # aprende de verdade (perceive) — sob lock
    FLUXO.semear(tokenize(titulo or ""))
    _wiki_hist.append(titulo)
    expressar({"tipo": "lendo", "titulo": titulo, "chars": len(txt)})
    return titulo


def explorar_sozinho():
    """Curiosidade: ele lê Wikipedia por conta própria — às vezes seguindo o que pensa, às vezes ao
    acaso, às vezes um link do que acabou de ler (navega como quiser). Aprende o mundo aos poucos."""
    while True:
        time.sleep(16)
        try:
            r = random.random()
            if r < 0.4 and _wiki_hist:                          # segue um LINK do que leu (navega)
                ls = wiki_links(_wiki_hist[-1], 20)
                ler_wiki(random.choice(ls) if ls else None)
            elif r < 0.7:                                        # segue a CURIOSIDADE (foco atual)
                foco = [w for w, _ in FLUXO.foco.most_common(4) if len(w) >= 4 and T.seen[w] > 1]
                ler_wiki(random.choice(foco) if foco else None)
            else:
                ler_wiki(None)                                  # ao acaso (descoberta)
        except Exception:
            pass


def _conteudo(words):
    return [w for w in words if w not in T.stops and len(w) >= 2]


def _ligaveis():
    """Palavras digitadas há pouco, as mais RARAS primeiro — o nome ('maiconzito') é raro e específico;
    o verbo comum ('chamo') é frequente. Assim o rosto/voz liga ao NOME, não à cola."""
    agora = time.time()
    recentes = [w for w, ts in _ctx if agora - ts < MEM_JANELA]
    return sorted(set(recentes), key=lambda w: T.seen.get(w, 0))[:2]


# ---------- FALAR PENSANDO (rápido): planeja uma cadeia coerente até o conceito evocado ----------
_VIZ_CACHE = {}


def _viz_pmi(w, k=8):
    """Vizinhos de conteúdo por PMI. Só olha os 40 MAIS FORTES (raw, most_common = nível C) e computa
    PMI neles -> O(40) por nó em vez de O(grau) (nós-hub têm milhares). Cacheia (fala fica instantânea)."""
    if w in _VIZ_CACHE:
        return _VIZ_CACHE[w]
    out = []
    for u, c in T.assoc.get(w, {}).most_common(40):
        if u in T.stops:
            continue
        p = T.pmi(w, u, c)
        if p > 0:
            out.append((p, u))
    out.sort(reverse=True)
    r = [u for _, u in out[:k]]
    _VIZ_CACHE[w] = r
    return r


def falar_pensando(words, cap=600):
    """Best-first LIMITADO (teto de nós, prof≤5) guiado por significado -> resposta no tema, coerente
    e INSTANTÂNEA (o planejar_real varria 29k conceitos = 10-19s; aqui é bounded + cacheado)."""
    cont = [w for w in words if w in T.assoc and w not in T.stops]
    ev = [(u, s) for u, s in T.associations(cont, k=12) if u not in T.stops]  # só conteúdo (sem hub)
    g = ev[0][0] if ev else None
    s = max(cont, key=lambda w: len(T.assoc.get(w, {})), default=None)
    if not g or not s:
        return None, g
    if s == g:
        return [s], g
    eg = T._emb(g)

    def h(n):
        en = T._emb(n)
        return 1.0 - (float(en @ eg) if (en is not None and eg is not None) else 0.0)

    pq = [(h(s), 0, s, [s])]
    seen = {s}
    while pq and len(seen) < cap:
        _, d, node, path = heapq.heappop(pq)
        if node == g:
            return path, g
        if d >= 5:
            continue
        for u in _viz_pmi(node, 8):
            if u in seen:
                continue
            if u == g:
                return path + [u], g
            seen.add(u)
            heapq.heappush(pq, (d + 1 + h(u), d + 1, u, path + [u]))
    return [s, g], g                                          # fallback: liga direto (ainda no tema)


def receber_texto(texto):
    words = tokenize(texto)
    low = texto.lower().strip()
    for pref in ("pesquise ", "pesquisa ", "leia ", "procure ", "busque "):   # FERRAMENTA: internet
        if low.startswith(pref):
            termo = texto.strip()[len(pref):].strip()
            threading.Thread(target=ler_wiki, args=(termo,), daemon=True).start()
            break
    conhecidas, novas = mimetizar(words)                        # mimetiza (imita/absorve)
    T.perceive(words); T.settle(words)                          # aprende em tempo real
    # SURPRESA (predictive coding / noradrenalina-dopamina): o que é NOVO/surpreendente é aprendido
    # MAIS FORTE (plasticidade modulada por erro) e puxa a atenção. Reforça só o novo, sem re-ler tudo.
    surpresa = len(novas) / max(len(words), 1)
    if surpresa > 0.34 and novas:
        for _ in range(2):                                     # grava o novo com mais força (LTP)
            T.perceive(novas)
        expressar({"tipo": "sensorial", "canal": "surpresa"})  # pisca a atenção (algo novo!)
    FLUXO.semear(words)
    agora = time.time()
    for w in _conteudo(words):
        _ctx.append((w, agora))                                 # contexto p/ ligar o que você diz ao que ele vê
    # RECALL HOLÍSTICO: uma palavra reinstaura TODAS as modalidades ligadas (rosto+voz+significado)
    for w in set(_conteudo(words)):
        if (w in MIM_V.assoc and MIM_V.assoc[w]) or (w in MIM_A.assoc and MIM_A.assoc[w]):
            recordar(w)                                         # unifica rosto+voz+texto num só laço
    ef = fato_extrair(texto)                                    # 1) é uma DECLARAÇÃO de fato? aprende crisp
    if ef:
        FATOS.aprender(*ef); FATOS.save()
        fala_txt = f"entendi: {ef[0]} {ef[1]} {ef[2]}"
    elif FATOS.responder(texto):                                # 2) é uma PERGUNTA factual? responde direto
        fala_txt = FATOS.responder(texto)
    else:                                                       # 3) senão, RACIOCINA (cadeia coerente)
        cad, alvo = falar_pensando(words)
        if cad and len(cad) >= 2:
            fala_txt = " -> ".join(cad)
        else:
            f, _ = T.think_and_say(words); fala_txt = " ".join(f) if f else "..."
    foco = [w for w, _ in FLUXO.foco.most_common(4)] or words
    ev = {"tipo": "fala", "eco": " ".join(conhecidas[:6]), "novo": " ".join(novas[:3]),
          "texto": fala_txt, "pixels": cerebro_pixels(foco)}
    s = voz(foco)                                               # só soa se aprendeu a ouvir (sem random)
    if s:
        ev["som"] = s
    expressar(ev)


def pensar_sozinho():
    """Ele pensa e se expressa SOZINHO: pinta o cérebro, verbaliza, e — quando NÃO está te vendo/
    ouvindo — SONHA e BALBUCIA do que aprendeu (gera sem referência ao vivo). É o 'fazer sozinho'."""
    global _dream
    tick = 0
    while True:
        time.sleep(0.9)
        tick += 1
        w, salto = FLUXO.passo()
        if w is None:
            FLUXO.semear(tokenize("tempo vida"))
            continue
        foco = [x for x, _ in FLUXO.foco.most_common(4)]
        ev = {"tipo": "pensando", "foco": ("«%s»" % w) if salto else w, "pixels": cerebro_pixels(foco)}
        if tick % 6 == 0:                                      # verbaliza o devaneio de vez em quando
            ev["texto"] = " ".join(FLUXO.recentes[-5:])
        expressar(ev)

        ocioso = (time.time() - _last_input) > 2.5             # não está te mostrando nada AGORA?
        if ocioso:
            with MIM_LOCK:
                gv = MIM_V.sonhar(foco)                        # RECOMBINA (blend que deriva) = cria NOVO sem parar
                if gv is not None:
                    _dream = gv if _dream is None else 0.2 * _dream + 0.8 * gv    # sonho BEM vivo (muda muito)
                    # (o sonho é EXPRESSÃO — NÃO reescreve a memória aprendida; senão corrompe o rosto)
            if gv is not None:
                expressar({"tipo": "video_out", "grid": _grade_cor(_dream), "gerado": True})
            if tick % 3 == 0:                                  # BALBUCIA áudio aprendido (moderado)
                s = voz(foco)
                if s:
                    expressar({"tipo": "audio_out", "som": s, "gerado": True})
            if tick % 10 == 0:                                 # CONSOLIDAÇÃO no sono (REM/replay, CLS):
                recentes = [w for w, _ in list(_ctx)][-8:] or [x for x, _ in T.seen.most_common(60)[20:28]]
                if recentes:
                    with T_LOCK:
                        T.settle(recentes)                     # re-assenta o vivido do dia (hipocampo->córtex)
                    expressar({"tipo": "pensando", "foco": "…consolidando: " + " ".join(recentes[:4]),
                               "pixels": cerebro_pixels(recentes)})


# ---------- servidor HTTP (estático + SSE + POST), só stdlib ----------
class H_(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def finish(self):
        try:                                            # silencia desconexão de cliente SSE (normal)
            super().finish()
        except (ConnectionError, BrokenPipeError, OSError):
            pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, "text/html; charset=utf-8", f.read())
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = queue.Queue()
            with _lock:
                _clientes.append(q)
            try:
                while True:
                    ev = q.get()
                    self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                with _lock:
                    if q in _clientes:
                        _clientes.remove(q)
        else:
            self._send(404, "text/plain", b"nao existe")

    def do_POST(self):
        if self.path != "/stimulus":
            self._send(404, "text/plain", b"nao existe"); return
        n = int(self.headers.get("Content-Length", 0))
        try:
            msg = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            msg = {}
        global _last_input, _ult_rec
        tipo = msg.get("tipo")
        if tipo == "texto":
            receber_texto(msg.get("dado", ""))
        elif tipo == "imagem":
            d = msg.get("dado", {})
            if isinstance(d, dict) and "cor" in d:
                _last_input = time.time()                       # está vendo você AGORA (não está ocioso)
                ligar = _ligaveis()                             # liga às palavras RARAS/específicas (o nome, não o verbo)
                with MIM_LOCK:
                    recon, idx = MIM_V.sentir(np.asarray(d["cor"], float) / 255.0, ligar or None)
                    nome = MIM_V.rotular(idx)                    # REVERSO: esse rosto é de QUEM?
                expressar({"tipo": "video_out", "grid": _grade_cor(recon), "gerado": False})
                if nome and nome not in T.stops and (nome != _ult_rec[0] or _last_input - _ult_rec[1] > 4):
                    _ult_rec = (nome, _last_input)              # RECONHECEU o rosto -> lembra tudo (nome+voz+sentido)
                    recordar(nome, "reconheci")
        elif tipo == "audio":
            d = msg.get("dado", {})
            if isinstance(d, dict) and "energia" in d:
                _last_input = time.time()
                ligar = _ligaveis()                             # liga a VOZ ao nome (palavra rara), não ao verbo
                with MIM_LOCK:
                    recon, _ = MIM_A.sentir(np.asarray(d["energia"], float) / 255.0, ligar or None)  # imita o que ouve
                expressar({"tipo": "audio_out", "som": _bandas_som(recon), "gerado": False})
        self._send(200, "application/json", b'{"ok":true}')


def main():
    threading.Thread(target=pensar_sozinho, daemon=True).start()
    threading.Thread(target=explorar_sozinho, daemon=True).start()   # ele lê Wikipedia sozinho (curiosidade)
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H_)
    print(f"Toshi acordado. abra http://localhost:{PORT}  (Ctrl+C encerra e salva)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        save_state(T)
        print("\n(Toshi dormiu; guardou o que viveu)")


if __name__ == "__main__":
    main()
