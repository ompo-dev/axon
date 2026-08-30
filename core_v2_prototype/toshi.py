"""
TOSHI — um ser que começa vazio (pedra) e refina a cada estímulo (To Your Eternity).

Sem comandos, sem frases prontas, sem priors. Come estímulo cru; a estrutura emerge.

Duas coisas emergem juntas, do mesmo fluxo (nada injetado):
  1. ASSOCIAÇÃO (o que anda junto) — por informação mútua (PMI). Ele descobre o que é 'cola'.
  2. SIGNIFICADO (o que a palavra É) — semântica DISTRIBUCIONAL: o sentido de uma palavra é a
     distribuição dos seus contextos (hipótese distribucional; é a base de word2vec e dos LLMs).
     Feito por RANDOM INDEXING (Kanerva/Sahlgren): forma VSA do embedding, LEVE, online, sem
     backprop, refina com exposição. Palavras de contexto parecido -> vetores parecidos ->
     similaridade e ANALOGIA emergem sozinhas (rei-homem+mulher≈rainha).

Ciência: hipótese distribucional (Harris 1954); Random Indexing (Sahlgren 2005); a mesma
distribuição que dá sentido aos LLMs (arXiv 2412.10924); aprendizado preditivo cortical.

RODAR:  python toshi.py     (come Dom Casmurro na 1ª vez; depois é só falar)
"""
import os
import re
import glob
import pickle
import unicodedata
import urllib.request
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIVROS = os.path.join(HERE, "dados", "livros")
STATE = os.path.join(HERE, "dados", "toshi_state.pkl")   # memória persistida (não re-lê)
BOOK = os.path.join(HERE, "dados", "domcasmurro.txt")
BOOK_URL = "https://www.gutenberg.org/files/55752/55752-0.txt"
WINDOW = 4
DIM = 1024                                    # dimensão do espaço de significado (Random Indexing)
NNZ = 12                                      # não-zeros por vetor-índice (esparso = leve)
_ROMAN = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenize(text):
    words = re.findall(r"[a-z]+", strip_accents(text.lower()))
    return [w for w in words if not (len(w) >= 3 and _ROMAN.match(w))]


class Toshi:
    def __init__(self, rng_seed=0):
        self.assoc = {}          # palavra -> Counter(co-ocorrente -> peso)   (associação)
        self.after = {}          # palavra -> Counter(seguinte -> peso)        (ordem)
        self.seen = Counter()
        self.index = {}          # palavra -> (dims, sinais)  vetor-índice esparso fixo
        self.embed = {}          # palavra -> vetor de SIGNIFICADO (soma dos contextos)
        self.rng = np.random.default_rng(rng_seed)

    # ---- vetor-índice esparso de cada palavra (identidade aleatória, sem significado) ----
    def _idx(self, w):
        if w not in self.index:
            dims = self.rng.choice(DIM, NNZ, replace=False)
            sig = self.rng.choice([-1, 1], NNZ).astype(np.float32)
            self.index[w] = (dims, sig)
            self.embed[w] = np.zeros(DIM, dtype=np.float32)
        return self.index[w]

    def perceive(self, words):
        n = len(words)
        for w in words:
            self._idx(w)
        for i, w in enumerate(words):
            self.seen[w] += 1
            if i + 1 < n:
                self.after.setdefault(w, Counter())[words[i + 1]] += 1
            for j in range(max(0, i - WINDOW), min(n, i + WINDOW + 1)):
                if j != i:
                    u = words[j]; g = 1.0 / abs(i - j)
                    self.assoc.setdefault(w, Counter())[u] += g
                    dims, sig = self.index[u]                 # SIGNIFICADO de w += índice do contexto u
                    self.embed[w][dims] += sig * g            # (Random Indexing)

    def eat(self, text):
        words = tokenize(text)
        # processa em blocos para não estourar memória de janelas
        self.perceive(words)
        return len(words)

    @property
    def total(self):
        return sum(self.seen.values()) + 1

    # ---- ASSOCIAÇÃO (o que anda junto) por PMI ----
    def pmi(self, w, u, c, min_c=2.0):
        if c < min_c or self.seen[w] == 0 or self.seen[u] == 0:
            return 0.0
        if self.total > 1000 and self.seen[u] > 0.012 * self.total:
            return 0.0
        return float(np.log((c * self.total) / (self.seen[w] * self.seen[u]) + 1e-9))

    def associations(self, words, k=6):
        pool = Counter()
        for w in words:
            for u, c in self.assoc.get(w, {}).items():
                if u not in words:
                    p = self.pmi(w, u, c)
                    if p > 0:
                        pool[u] += p
        return pool.most_common(k)

    def spread(self, words, depth=2, decay=0.45, per_node=6, k=8):
        """Ativação espalhada MULTI-HOP: A ativa B (direto) e C (via B) = inferência transitiva.
        Se você ensina A→B e B→C e depois só diz A, C deve ACENDER (ele encadeou = 'pensou')."""
        act = Counter()
        seeds = [w for w in words if w in self.assoc]
        frontier = [(w, 1.0, 0) for w in seeds]
        seen_nodes = set(seeds)
        while frontier:
            node, energy, d = frontier.pop(0)
            if node not in words:
                act[node] += energy
            if d >= depth or energy < 0.05:
                continue
            # segue as ligações mais informativas (PMI), com decaimento por hop
            nb = sorted(((self.pmi(node, u, c), u) for u, c in self.assoc.get(node, {}).items()),
                        reverse=True)[:per_node]
            for p, u in nb:
                if p > 0 and u not in words:
                    frontier.append((u, energy * decay * min(p / 4.0, 1.0), d + 1))
                    seen_nodes.add(u)
        return act.most_common(k)

    def drift(self, start, steps=5):
        chain, cur, used = [start], start, {start}
        for _ in range(steps):
            nxt, best = None, 0.0
            for w, c in self.assoc.get(cur, Counter()).items():
                if w not in used:
                    p = self.pmi(cur, w, c)
                    if p > best:
                        nxt, best = w, p
            if not nxt:
                break
            chain.append(nxt); used.add(nxt); cur = nxt
        return chain

    # ---- SIGNIFICADO (o que a palavra É) por semântica distribucional ----
    def _emb(self, w):
        v = self.embed.get(w)
        if v is None:
            return None
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    def meaning(self, word, k=6, min_freq=3):
        """Palavras com SIGNIFICADO parecido (contextos parecidos). Emerge, não é dado."""
        q = self._emb(word)
        if q is None:
            return []
        ceil = 0.012 * self.total if self.total > 1000 else 1e18
        sims = []
        for u in self.embed:
            if u != word and self.seen[u] >= min_freq and self.seen[u] < ceil:
                e = self._emb(u)
                if e is not None:
                    sims.append((u, float(q @ e)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    @property
    def stops(self):
        n = len(self.seen)
        if getattr(self, "_stops_n", -1) != n:
            self._stops = {w for w, _ in self.seen.most_common(30)}
            self._stops_n = n
        return self._stops

    def respond(self, words, n=10):
        """FALA (executor): gera uma frase a partir do que o estímulo evocou, usando as
        sequências que ele aprendeu dos livros (modelo 'após'). É a voz dele — do que leu,
        não template. Evita repetir palavra de conteúdo (não vira eco)."""
        ev = self.associations(words, 6)
        seed = ev[0][0] if ev else (words[-1] if words else None)
        if seed is None or seed not in self.after:
            return []
        out, cur, content_used = [seed], seed, {seed}
        for _ in range(n):
            cand = self.after.get(cur)
            if not cand:
                break
            items = [(w, c) for w, c in cand.items() if w not in content_used or w in self.stops]
            if not items:
                break
            # AMOSTRA proporcional (temperatura) em vez de gulosa -> varia, não vira atrator
            ws = np.array([c for _, c in items], float) ** 0.7
            nxt = items[int(self.rng.choice(len(items), p=ws / ws.sum()))][0]
            out.append(nxt); cur = nxt
            if nxt not in self.stops:
                content_used.add(nxt)
        return out

    def think_and_say(self, words, n_candidates=6):
        """Os TRÊS NÍVEIS:
          PENSADOR gera vários pensamentos (respostas candidatas, por amostragem);
          OBSERVADOR avalia cada um (relevância ao que você disse) e ESCOLHE — é a metacognição
            ('vejo meus pensamentos e escolho'); EXECUTOR fala o escolhido.
        Best-of-N com avaliador aprendido -> resposta mais no tema e coerente."""
        cands = [self.respond(words) for _ in range(n_candidates)]
        cands = [c for c in cands if len(c) >= 2]
        if not cands:
            return [], 0

        def relevance(c):                                  # quão ligado ao estímulo (PMI)
            s = 0.0
            for iw in words:
                a = self.assoc.get(iw, {})
                for cw in c:
                    if cw in a:
                        s += max(self.pmi(iw, cw, a[cw]), 0.0)
            return s / len(c)
        best = max(cands, key=relevance)
        return best, len(cands)

    def settle(self, seed_words, steps=1, alpha=0.15):
        """RELAÇÃO MÚTUA (a sua ideia): ao mudar um conceito, o update RIPPLA pros vizinhos —
        cada um puxa um pouco a média dos que colidem com ele. O grafo re-equilibra. É a
        difusão sináptica: mexer num nó interfere nos ramos vizinhos. Local (só o que mexeu)."""
        region = set()
        for w in seed_words:
            if w in self.assoc:
                region.add(w)
                for u in list(self.assoc[w])[:20]:
                    region.add(u)                         # vizinhos que dependem dele
        for _ in range(steps):
            upd = {}
            for w in region:
                nb = [(u, self.pmi(w, u, c)) for u, c in self.assoc.get(w, {}).items()]
                nb = [(u, p) for u, p in nb if p > 0 and u in self.embed]
                if not nb:
                    continue
                mix = np.zeros(DIM, np.float32); wsum = 0.0
                for u, p in nb:
                    mix += self._emb(u) * p; wsum += p     # média dos vizinhos, pesada por PMI
                if wsum > 0:
                    upd[w] = (1 - alpha) * self.embed[w] + alpha * (mix / wsum) * np.linalg.norm(self.embed[w])
            for w, v in upd.items():
                self.embed[w] = v                          # re-assenta todos juntos (mútuo)

    def analogy(self, a, b, c, k=3):
        """a está p/ b como c está p/ ?  (aritmética de significado — como word2vec)."""
        va, vb, vc = self._emb(a), self._emb(b), self._emb(c)
        if va is None or vb is None or vc is None:
            return []
        q = vb - va + vc; q = q / (np.linalg.norm(q) + 1e-9)
        sims = [(u, float(q @ self._emb(u))) for u in self.embed
                if u not in (a, b, c) and self.seen[u] >= 3 and self._emb(u) is not None]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]


def _strip_gutenberg(txt):
    a = txt.find("***"); a = txt.find("***", a + 3) + 3 if a >= 0 else 0
    b = txt.rfind("*** END")
    return txt[a: b if b > 0 else len(txt)]


def load_all_books():
    """Come TODOS os livros em dados/livros/ (não só um). Baixa Dom Casmurro se a pasta vazia."""
    os.makedirs(LIVROS, exist_ok=True)
    paths = glob.glob(os.path.join(LIVROS, "*.txt"))
    if not paths:
        for _ in range(3):
            try:
                d = urllib.request.urlopen(urllib.request.Request(BOOK_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=90).read()
                open(os.path.join(LIVROS, "55752.txt"), "wb").write(d); break
            except Exception:
                pass
        paths = glob.glob(os.path.join(LIVROS, "*.txt"))
    return [_strip_gutenberg(open(p, encoding="utf-8", errors="replace").read()) for p in paths], len(paths)


def save_state(t):
    """Salva os DADOS (não o objeto) — imune a identidade de classe __main__/módulo."""
    tmp = STATE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"assoc": t.assoc, "after": t.after, "seen": t.seen,
                     "index": t.index, "embed": t.embed}, f)
    os.replace(tmp, STATE)


def build_or_load():
    """Carrega a memória salva (instantâneo) OU come todos os livros e SALVA. Não re-lê à toa."""
    if os.path.exists(STATE):
        try:
            with open(STATE, "rb") as f:
                d = pickle.load(f)
            t = Toshi()
            t.assoc, t.after, t.seen, t.index, t.embed = d["assoc"], d["after"], d["seen"], d["index"], d["embed"]
            return t, True
        except Exception as _e:
            print(f"(memória salva corrompida: {_e}; vou reconstruir dos livros)")
            try:
                os.replace(STATE, STATE + ".corrompido")
            except Exception:
                pass
    t = Toshi()
    textos, nl = load_all_books()
    for txt in textos:
        t.eat(txt)
    save_state(t)
    print(f"(comi {nl} livros e salvei a memória — próximas vezes carrego instantâneo)")
    return t, False


def main():
    try:
        import sys; sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(__doc__)
    print("carregando Toshi...", end=" ", flush=True)
    t, cached = build_or_load()
    print(f"pronto ({'memória salva' if cached else 'primeira vez'}). "
          f"{sum(t.seen.values())} palavras vividas, {len(t.seen)} conceitos.")
    print("=" * 62)
    print("fale. ele mostra o que ACENDE (associação) e o que ENTENDE (significado). Ctrl+C sai.\n")
    turns = 0
    while True:
        try:
            text = input("voce> ").strip()
        except (EOFError, KeyboardInterrupt):
            save_state(t)                                  # salva o que aprendeu com você
            print("\n(Toshi adormece; guardou o que aprendeu)"); break
        if not text:
            continue
        words = tokenize(text)
        novos = [w for w in words if t.seen[w] == 0]
        t.perceive(words)
        t.settle(words)                                    # o update RIPPLA pros vizinhos (mútuo)
        turns += 1
        # === os três níveis: pensador gera, observador escolhe, executor fala ===
        fala, n_paths = t.think_and_say(words)
        assoc = t.associations(words)
        chain = t.drift(assoc[0][0], 5) if assoc else []
        if fala:
            print("  toshi> " + " ".join(fala))
        elif novos:
            print("  toshi> (não conheço '%s' ainda — me conta)" % " ".join(novos[:2]))
        else:
            print("  toshi> ...")
        if chain and len(chain) > 1:                       # observador vê o próprio pensar
            print("         (pensei %d caminhos; segui: %s)" % (n_paths, " → ".join(chain)))
        if turns % 10 == 0:
            save_state(t)


def _selftest():
    t = Toshi()
    # corpus com estrutura: rei/rainha, homem/mulher em contextos análogos
    corpus = ("o rei governa o reino com poder. a rainha governa o reino com graca. "
              "o homem forte trabalha. a mulher forte trabalha. o rei e o homem mandam. "
              "a rainha e a mulher mandam. ") * 20
    t.eat(corpus)
    # significado emergente: 'rei' mais perto do contexto REAL (reino/governa) que de 'trabalha'
    mrei = dict(t.meaning("rei", k=30, min_freq=2))
    assert mrei.get("reino", -1) > mrei.get("trabalha", -1), mrei
    assert mrei.get("governa", -1) > mrei.get("trabalha", -1), mrei
    # associação por PMI funciona
    a = t.associations(["rei"])
    assert a, a
    print("[selftest] ok (significado distribucional emerge; associação por PMI)")


if __name__ == "__main__":
    _selftest()
    main()
