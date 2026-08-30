"""
TOSHI — FLUXO DE PENSAMENTO autônomo. Ele pensa SOZINHO (não só reage).

A linha (Cheppie/Toshi): um ser que pensa por conta própria, com fluxo de raciocínio, aprende
em tempo real. Aqui: mesmo sem você falar, a ativação ESPALHA pelo grafo e DERIVA — um trem de
consciência. Você pode sussurrar palavras pra GUIAR o fluxo (não comandar). Ele aprende do que
você diz enquanto pensa.

Mecanismo (o tick-loop do axon como pensamento):
  - FOCO = conceitos ativos agora, com energia (a 'atenção'/temperatura).
  - a cada passo: espalha do foco -> escolhe o próximo pensamento (energia × novidade),
    emite, decai o foco, injeta o novo. Não repete (novidade), não trava.
  - às vezes SALTA: pula pra um conceito SEMANTICAMENTE ligado mas associativamente distante
    (usa o espaço de significado) — uma pitada de 'pulo' criativo (aceno ao 'LLMs can't jump').

RODAR:  python fluxo.py     (usa a memória salva do toshi.py; ele começa a divagar)
"""
import sys
import time
from collections import Counter
import numpy as np
from toshi import build_or_load, tokenize


class Fluxo:
    def __init__(self, t):
        self.t = t
        self.foco = Counter()          # conceito -> energia
        self.recentes = []             # o que já pensou (não repetir)
        self.rng = np.random.default_rng(0)

    def semear(self, words):
        """Injeta estímulo no foco (guiar, não comandar). Aprende ao vivo também."""
        self.t.perceive(words)
        for w in words:
            if w in self.t.assoc:
                self.foco[w] += 1.5

    def passo(self):
        """Um pensamento: espalha do foco, escolhe o próximo, atualiza o foco. Autônomo."""
        if not self.foco:
            # nada no foco -> começa de um conceito 'quente' qualquer (curiosidade espontânea)
            quentes = [w for w, _ in self.t.seen.most_common(400)[100:]]
            self.foco[self.rng.choice(quentes)] += 1.0
        base = [w for w, _ in self.foco.most_common(3)]
        # candidatos: associação (espalhamento) + às vezes um SALTO semântico
        cand = Counter()
        for w, p in self.t.spread(base, depth=2, k=20):
            cand[w] += p
        salto = None
        if self.rng.random() < 0.25 and base:
            viz = self.t.meaning(base[0], k=8)             # ligado por SIGNIFICADO, não adjacência
            viz = [(w, s) for w, s in viz if w not in self.t.assoc.get(base[0], {})]
            if viz:
                salto = viz[self.rng.integers(min(3, len(viz)))][0]
        # escolhe o próximo pensamento (evita repetir; energia como peso)
        for w in self.recentes[-8:]:
            cand.pop(w, None)
        if salto and salto not in self.recentes[-8:]:
            cand[salto] += max(cand.values(), default=1.0) * 1.1   # o salto compete forte
        if not cand:
            self.foco.clear(); return None, False
        ws = np.array(list(cand.values()), float) ** 0.8
        nxt = list(cand)[int(self.rng.choice(len(cand), p=ws / ws.sum()))]
        # atualiza foco: decai o velho, acende o novo (temperatura)
        for w in list(self.foco):
            self.foco[w] *= 0.6
            if self.foco[w] < 0.1:
                del self.foco[w]
        self.foco[nxt] += 1.0
        self.recentes.append(nxt)
        return nxt, (salto == nxt)


def demo():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando Toshi...", end=" ", flush=True)
    t, _ = build_or_load()
    print("pronto. fluxo de pensamento autônomo (seed: 'amor'):\n")
    f = Fluxo(t)
    f.semear(tokenize("amor"))
    linha = []
    for i in range(24):
        w, saltou = f.passo()
        if w is None:
            break
        linha.append(("«%s»" % w) if saltou else w)      # «..» = um salto criativo
    print("  " + " → ".join(linha))
    print("\n  (« » marca um SALTO: pulou por SIGNIFICADO, não por vizinhança — criatividade)")

    print("\n  agora SEMEIO 'morte' no meio do fluxo (guiar, não comandar):")
    f.semear(tokenize("morte"))
    linha2 = []
    for i in range(12):
        w, s = f.passo()
        if w:
            linha2.append(("«%s»" % w) if s else w)
    print("  " + " → ".join(linha2))


def interativo():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t, _ = build_or_load()
    f = Fluxo(t)
    print("Toshi está pensando. Enter = deixa pensar; digite algo = sussurra p/ guiar; 'sair'.\n")
    while True:
        try:
            g = input("(enter/fala)> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if g == "sair":
            break
        if g:
            f.semear(tokenize(g))
        pensamentos = []
        for _ in range(8):
            w, s = f.passo()
            if w:
                pensamentos.append(("«%s»" % w) if s else w)
        print("  toshi pensa> " + " → ".join(pensamentos))


def _selftest():
    t, _ = build_or_load()
    f = Fluxo(t)
    f.semear(tokenize("mar amor"))
    seq = [f.passo()[0] for _ in range(10)]
    seq = [w for w in seq if w]
    assert len(seq) >= 5 and len(set(seq)) >= len(seq) - 2   # flui e quase não repete
    print("[selftest] ok (fluxo autônomo gera pensamentos encadeados sem travar)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-i":
        interativo()
    else:
        _selftest()
        demo()
