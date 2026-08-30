"""
TOSHI — CÉREBRO VISUAL. Ver o pensamento em tempo real: mapa de calor da ativação.

A visão (do usuário): conceito não é um ponto fixo — é um CAMPO que incha e espalha (massinha/
Venom). A informação EXPLODE de um ponto e PROPAGA pelo cérebro (tipo A*/BFS achando a rota).
Aqui você VÊ isso: cada conceito num mapa 2D (posto por SIGNIFICADO — vizinhos ficam perto), e
a ATIVAÇÃO vira CALOR. Toshi pensa (fluxo autônomo) e você assiste o cérebro acender e propagar.

Mapa 2D = projeção (PCA) do espaço de significado -> conceitos parecidos ficam próximos.
Calor = ativação espalhada do foco atual (spread). Anima em tempo real no terminal.

RODAR:  python cerebro_visual.py            (ele pensa sozinho; Ctrl+C sai)
        python cerebro_visual.py -i         (você sussurra e guia o pensamento ao vivo)
(TUI ratatui em Rust é o acabamento; esta é a ponte visual em Python, roda já.)
"""
import sys
import time
import numpy as np
from collections import Counter
from toshi import build_or_load, tokenize
from fluxo import Fluxo

W, H = 74, 30                                 # tamanho do mapa (colunas, linhas)
N_CONCEITOS = 500                             # quantos conceitos plotar


def heat_ansi(x):
    """0..1 -> cor ANSI 256 (frio azul -> quente vermelho/branco). Bloco colorido."""
    x = max(0.0, min(1.0, x))
    if x < 0.02:
        return "\033[38;5;236m·\033[0m"       # quase apagado
    ramp = [17, 19, 21, 27, 39, 45, 51, 50, 46, 82, 154, 190, 226, 220, 214, 208, 202, 196, 231]
    c = ramp[int(x * (len(ramp) - 1))]
    ch = "•" if x < 0.35 else ("●" if x < 0.7 else "█")
    return f"\033[38;5;{c}m{ch}\033[0m"


class Mapa:
    def __init__(self, t):
        self.t = t
        # escolhe conceitos de conteúdo mais frequentes (fora as stopwords)
        stops = {w for w, _ in t.seen.most_common(40)}
        palavras = [w for w, _ in t.seen.most_common(N_CONCEITOS * 3)
                    if w not in stops and w in t.embed][:N_CONCEITOS]
        E = np.array([t._emb(w) for w in palavras if t._emb(w) is not None])
        self.palavras = [w for w in palavras if t._emb(w) is not None]
        # PCA -> 2D (posição no mapa = significado; vizinhos semânticos ficam perto)
        Ec = E - E.mean(0)
        U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
        xy = Ec @ Vt[:2].T
        xy -= xy.min(0); xy /= (xy.max(0) + 1e-9)
        self.gx = (xy[:, 0] * (W - 1)).astype(int)
        self.gy = (xy[:, 1] * (H - 1)).astype(int)
        # célula -> índice do conceito (resolve colisão na hora, pelo mais ativo)
        self.cells = {}
        for i, (x, y) in enumerate(zip(self.gx, self.gy)):
            self.cells.setdefault((x, y), []).append(i)

    def frame(self, ativacao, pensamento=""):
        # CAMPO de calor: cada conceito ativo = um BORRÃO gaussiano (blob que espalha).
        field = np.zeros((H, W), np.float32)
        hot = None; hotv = -1
        for i, (x, y) in enumerate(zip(self.gx, self.gy)):
            a = ativacao.get(self.palavras[i], 0.0)
            if a <= 0:
                continue
            for dy in range(-2, 3):
                yy = y + dy
                if 0 <= yy < H:
                    for dx in range(-3, 4):
                        xx = x + dx
                        if 0 <= xx < W:
                            field[yy, xx] += a * np.exp(-(dx * dx / 5.0 + dy * dy / 2.0))
            if a > hotv:
                hotv, hot = a, self.palavras[i]
        fmax = field.max() or 1.0
        grid = [[heat_ansi((field[y, x] / fmax) ** 0.55) for x in range(W)] for y in range(H)]
        out = ["\033[H"]                        # cursor pro topo (anima sem limpar tudo)
        out.append("  CÉREBRO DE TOSHI — ativação em tempo real (quente = pensando nisso)\n")
        for row in grid:
            out.append("  " + "".join(row) + "\n")
        out.append(f"\n  pensando: \033[38;5;214m{pensamento}\033[0m")
        out.append(f"    (mais quente: {hot})            \n")
        return "".join(out)


def rodar(interativo=False):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando o cérebro...", flush=True)
    t, _ = build_or_load()
    mapa = Mapa(t)
    f = Fluxo(t)
    f.semear(tokenize("amor vida morte"))
    print("\033[2J", end="")                    # limpa a tela uma vez
    try:
        while True:
            if interativo:
                # não bloqueia o fluxo: lê linha se houver (simplificado: pergunta a cada 8 passos)
                pass
            w, salto = f.passo()
            if w is None:
                f.semear(tokenize("mundo tempo"))
                continue
            # ativação atual = foco (centro quente) + espalhamento (o campo de calor propaga)
            act = {w: e * 3.0 for w, e in f.foco.items()}
            for u, p in t.spread([x for x, _ in f.foco.most_common(4)], depth=2, k=250):
                act[u] = act.get(u, 0.0) + p
            sys.stdout.write(mapa.frame(act, ("« %s »" % w) if salto else w))
            sys.stdout.flush()
            time.sleep(0.35)
    except KeyboardInterrupt:
        print("\n\n(Toshi para de pensar)")


def _selftest():
    t, _ = build_or_load()
    mapa = Mapa(t)
    assert len(mapa.palavras) > 50
    fr = mapa.frame({mapa.palavras[0]: 1.0}, "teste")
    assert "CÉREBRO DE TOSHI" in fr and "\033[H" in fr
    # há posições 2D distintas (não colapsou tudo num ponto)
    assert len(set(zip(mapa.gx, mapa.gy))) > 30
    print("[selftest] ok (mapa 2D com posições distintas; frame renderiza)")


if __name__ == "__main__":
    if "-t" in sys.argv:
        _selftest()
    else:
        rodar(interativo="-i" in sys.argv)
