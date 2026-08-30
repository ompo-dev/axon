"""
AXON — um SER. Sem comandos. Sem frases prontas. Você fala; ele mostra o que ACENDE nele.

Correção (do usuário): nada de resposta pré-escrita ("me lembra X"). Isso era EU pondo palavras
na boca dele. A saída tem que ser o ESTADO INTERNO cru — os conceitos que se ativam e a deriva
do pensamento — dados, não frase. E ele come um livro real (Dom Casmurro) pra ter repertório.

Por dentro (autônomo, sem comando):
  - come um corpus real (Machado de Assis) -> aprende co-ocorrência (o que anda junto) e ordem.
  - você fala -> ele PERCEBE (aprende) e a ativação ESPALHA -> conceitos acendem com peso.
  - a saída é essa ativação (o "pensamento") e uma deriva associativa. Cru. Nada pré-fixado.
  - específico > ubíquo: palavra que anda com tudo (de, que, o) pesa pouco (ele percebe sozinho).

RODAR:  python ser.py     (baixa o livro na 1ª vez, ~400 KB, e cacheia)
"""
import os
import re
import urllib.request
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(HERE, "dados", "domcasmurro.txt")
BOOK_URL = "https://www.gutenberg.org/files/55752/55752-0.txt"
WINDOW = 4                                    # janela de co-ocorrência (±4 palavras)


import unicodedata


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_ROMAN = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")


def tokenize(text):
    text = strip_accents(text.lower())        # capitu==capitú, coracao==coração
    words = re.findall(r"[a-z]+", text)
    # descarta numerais romanos com 3+ letras (marcadores de capítulo, não palavras)
    return [w for w in words if not (len(w) >= 3 and _ROMAN.match(w))]


class Ser:
    def __init__(self):
        self.assoc = {}          # palavra -> Counter(co-ocorrente -> peso)
        self.after = {}          # palavra -> Counter(seguinte -> peso)
        self.seen = Counter()

    def perceive(self, words):
        n = len(words)
        for i, w in enumerate(words):
            self.seen[w] += 1
            if i + 1 < n:
                self.after.setdefault(w, Counter())[words[i + 1]] += 1
            for j in range(max(0, i - WINDOW), min(n, i + WINDOW + 1)):
                if j != i:
                    self.assoc.setdefault(w, Counter())[words[j]] += 1.0 / abs(i - j)

    def eat(self, text):
        """Come um texto grande de uma vez (o 'livro'). Constrói o repertório."""
        words = tokenize(text)
        self.perceive(words)
        return len(words)

    @property
    def total(self):
        return sum(self.seen.values()) + 1

    def pmi(self, w, u, c, min_c=2.0):
        """Informação mútua: quão mais que o ACASO w e u andam juntos. Alto = associação
        específica; ~0 = coincidência (stopword anda com tudo). Ele 'descobre' o que é cola."""
        if c < min_c or self.seen[w] == 0 or self.seen[u] == 0:
            return 0.0
        if self.total > 1000 and self.seen[u] > 0.012 * self.total:   # ubíqua (de/que/o) = pouca info
            return 0.0
        return float(np.log((c * self.total) / (self.seen[w] * self.seen[u]) + 1e-9))

    def activate(self, words, k=6):
        """O que essas palavras acendem nele (por PML/informação mútua). É o 'pensamento'."""
        pool = Counter()
        for w in words:
            for u, c in self.assoc.get(w, {}).items():
                if u not in words:
                    p = self.pmi(w, u, c)
                    if p > 0:
                        pool[u] += p
        return pool.most_common(k)

    def drift(self, start, steps=5):
        """Deriva do pensamento: segue a associação mais SURPREENDENTE (maior PMI), sem repetir."""
        chain, cur, used = [start], start, {start}
        for _ in range(steps):
            cand = self.assoc.get(cur, Counter())
            nxt, best = None, 0.0
            for w, c in cand.items():
                if w not in used and w != cur:
                    p = self.pmi(cur, w, c)
                    if p > best:
                        nxt, best = w, p
            if not nxt:
                break
            chain.append(nxt); used.add(nxt); cur = nxt
        return chain


def load_book():
    os.makedirs(os.path.dirname(BOOK), exist_ok=True)
    if not os.path.exists(BOOK):
        d = urllib.request.urlopen(urllib.request.Request(BOOK_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        open(BOOK, "wb").write(d)
    txt = open(BOOK, encoding="utf-8", errors="replace").read()
    # corta cabeçalho/rodapé do Gutenberg
    a = txt.find("DOM CASMURRO"); b = txt.rfind("*** END")
    return txt[a if a > 0 else 0: b if b > 0 else len(txt)]


def main():
    try:
        import sys; sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(__doc__)
    ser = Ser()
    print("comendo o livro (Dom Casmurro)...", end=" ", flush=True)
    nw = ser.eat(load_book())
    print(f"pronto. {nw} palavras, {len(ser.seen)} distintas no repertório.")
    print("=" * 60)
    print("fale. ele mostra o que acende nele (peso). Ctrl+C p/ sair.\n")

    while True:
        try:
            text = input("voce> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n(dormindo)"); break
        if not text:
            continue
        words = tokenize(text)
        novos = [w for w in words if ser.seen[w] == 0]
        ser.perceive(words)                            # aprende ao vivo do que você falou
        act = ser.activate(words)
        # SAÍDA CRUA: o que acendeu (conceito·peso), e a deriva do pensamento. Zero frase minha.
        if act:
            print("axon> " + "   ".join(f"{w}·{s:.1f}" for w, s in act))
            seed = act[0][0]
            print("      ~ " + " › ".join(ser.drift(seed, 5)))
        else:
            print("axon> ·")
        if novos:
            print("      [novo: " + " ".join(novos) + "]")


def _selftest():
    s = Ser()
    s.eat("o gato bebe leite. o gato bebe leite. o gato gosta de leite. gato e leite sempre. "
          "o cao late no telhado. o cao late alto.")
    # 'gato' acende algo (co-ocorrentes), e não a si mesmo
    act = s.activate(["gato"])
    assert act and all(w != "gato" for w, _ in act), act
    # deriva começa no seed e não repete
    d = s.drift("gato", 4)
    assert d[0] == "gato" and len(set(d)) == len(d)
    # PMI: associação específica (gato-leite) > coincidência com ubíquo
    assert s.pmi("gato", "leite", s.assoc["gato"]["leite"]) > 0
    print("[selftest] ok (ativa por PMI; deriva sem repetir)")


if __name__ == "__main__":
    _selftest()
    main()
