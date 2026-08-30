"""
TOSHI — conceito com BRAÇOS (a ideia do usuário): não um ponto, um centro com braços que
esticam pra setores distantes do espaço. Resolve polissemia + relações de longo alcance.

Problema do embedding-ponto (LLM/word2vec padrão): 1 vetor por palavra = MÉDIA de todos os
sentidos. 'banco' (dinheiro) e 'banco' (assento) colapsam num ponto borrado. 'vermelho' (cor)
e 'vermelho' (raiva/política) idem. Os braços pra setores diferentes somem.

A ideia (= multi-prototype / multi-sense embedding, Reisinger-Mooney 2010, mas leve/online):
  - o conceito = CENTRO + vários BRAÇOS (facetas). Cada braço = um sentido/contexto, e alcança
    uma REGIÃO diferente do espaço (setor distante). Os braços emergem clusterizando os
    contextos em que a palavra aparece. Nada injetado.
  - assim 'vermelho' tem um braço no setor {maca, ferrari} e outro no setor {raiva, sangue},
    mesmo que esses setores estejam longe um do outro.

Prova aqui: uma palavra polissêmica SE PARTE em braços que apontam pra regiões distintas.
Roda: python bracos.py   (numpy; Random Indexing p/ os vetores)
"""
import re
import unicodedata
from collections import Counter
import numpy as np

DIM, NNZ, WINDOW = 2048, 20, 6
RNG = np.random.default_rng(1)


def tok(text):
    text = "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z]+", text)


class EspacoConceitos:
    def __init__(self):
        self.index = {}          # palavra -> vetor-índice esparso (identidade aleatória)
        self.embed = {}          # palavra -> vetor de significado (centro)
        self.seen = Counter()
        self.occ = {}            # palavra -> lista de vetores-de-contexto (uma por ocorrência)
        self.stops = set()

    def _idx(self, w):
        if w not in self.index:
            dims = RNG.choice(DIM, NNZ, replace=False)
            sig = RNG.choice([-1.0, 1.0], NNZ).astype(np.float32)
            self.index[w] = (dims, sig)
            self.embed[w] = np.zeros(DIM, np.float32)
        return self.index[w]

    @property
    def total(self):
        return sum(self.seen.values()) + 1

    def _idf(self, w):
        return float(np.log(self.total / (self.seen[w] + 1)))

    def _ctx_vec(self, words, i):
        """Vetor do CONTEXTO: só palavras de CONTEÚDO (pula a 'cola' = top-frequência), pesadas
        por raridade. Assim o contexto vira a assinatura do SENTIDO, não da gramática."""
        v = np.zeros(DIM, np.float32)
        for j in range(max(0, i - WINDOW), min(len(words), i + WINDOW + 1)):
            if j != i and words[j] not in self.stops:
                dims, sig = self._idx(words[j]); v[dims] += sig * self._idf(words[j])
        return v

    def eat(self, text):
        words = tok(text)
        for w in words:                                    # passada 1: frequências
            self._idx(w); self.seen[w] += 1
        # 'cola' = as N palavras mais frequentes (function words). Escala-invariante, ele descobre.
        K = max(8, min(25, len(self.seen) // 3))
        self.stops = {w for w, _ in self.seen.most_common(K)}
        for i, w in enumerate(words):                      # passada 2: contextos de conteúdo
            cv = self._ctx_vec(words, i)
            self.embed[w] += cv
            self.occ.setdefault(w, []).append(cv)

    def _norm(self, v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    def neighbors(self, vec, k=4, exclude=()):
        q = self._norm(vec)
        sims = [(u, float(q @ self._norm(self.embed[u]))) for u in self.embed
                if u not in exclude and self.seen[u] >= 2 and u not in self.stops]   # só conteúdo
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

    def bracos(self, word, n_arms=2, iters=15):
        """Parte os contextos da palavra em `n_arms` braços (facetas) por clustering (k-means
        cosseno). Cada braço aponta pra uma região distinta. Emerge dos dados."""
        ctxs = [self._norm(c) for c in self.occ.get(word, []) if np.linalg.norm(c) > 1e-9]
        if len(ctxs) < n_arms:
            return []
        X = np.array(ctxs)
        # k-means++ simples em cosseno (X já normalizado -> cosseno = produto interno)
        cen = X[RNG.choice(len(X), n_arms, replace=False)].copy()
        for _ in range(iters):
            assign = np.argmax(X @ cen.T, axis=1)
            for a in range(n_arms):
                pts = X[assign == a]
                if len(pts):
                    cen[a] = self._norm(pts.mean(0))
        arms = []
        for a in range(n_arms):
            size = int((assign == a).sum())
            if size:
                arms.append((size, self.neighbors(cen[a], k=4, exclude=(word,))))
        arms.sort(reverse=True)
        return arms


def demo():
    print("=" * 78)
    print("TOSHI — conceito com BRAÇOS: polissemia SE PARTE em setores distantes (a sua ideia)")
    print("=" * 78)
    es = EspacoConceitos()
    # corpus construído com polissemia clara (2 sentidos por palavra, contextos separados)
    corpus = (
        # 'banco' = dinheiro
        "o banco emprestou dinheiro ao cliente. paguei a conta no banco. o gerente do banco aprovou o credito. "
        "o banco cobrou juros altos. levei o dinheiro ao banco. o banco financiou a casa. "
        # 'banco' = assento
        "sentei no banco da praca para descansar. o banco de madeira estava quebrado. "
        "descansei no banco embaixo da arvore. o velho sentou no banco do parque. o banco da praca era verde. "
        # 'vermelho' = cor de objeto
        "a maca vermelha estava madura. a ferrari vermelha corria rapido. o vestido vermelho era lindo. "
        "comprei um carro vermelho novo. a rosa vermelha cheirava bem. "
        # 'vermelho' = raiva/sangue
        "ficou vermelho de raiva e gritou. o rosto vermelho de furia. o sangue vermelho escorria. "
        "vermelho de odio, ele bateu na mesa. a raiva deixou o rosto vermelho. "
    ) * 6

    es.eat(corpus)

    for palavra in ("banco", "vermelho"):
        print(f"\n[{palavra}] — centro (ponto-único, sentido BORRADO):")
        print("    " + ", ".join(f"{w}·{s:.2f}" for w, s in es.neighbors(es.embed[palavra], 5, exclude=(palavra,))))
        print(f"[{palavra}] — BRAÇOS (facetas que esticam pra setores distintos):")
        for size, neigh in es.bracos(palavra, n_arms=2):
            print(f"    braço ({size} contextos) -> " + ", ".join(f"{w}·{s:.2f}" for w, s in neigh))

    print("\n" + "=" * 78)
    print("O ponto-único mistura os sentidos. Os BRAÇOS separam: cada um alcança um setor")
    print("diferente do espaço (dinheiro vs assento; cor vs raiva). É multi-prototype embedding")
    print("(Reisinger-Mooney 2010) leve/online — captura o senso comum que o ponto perde.")


def _selftest():
    es = EspacoConceitos()
    c = ("o banco emprestou dinheiro. paguei no banco o credito. " * 8 +
         "sentei no banco da praca. o banco de madeira na praca. " * 8)
    es.eat(c)
    arms = es.bracos("banco", 2)
    assert len(arms) == 2, arms
    # os dois braços alcançam vizinhos DIFERENTES (setores distintos)
    n0 = {w for w, _ in arms[0][1]}; n1 = {w for w, _ in arms[1][1]}
    assert n0 != n1 and len(n0 & n1) < len(n0), (n0, n1)
    print("[selftest] ok (braços separam sentidos em setores distintos)")


if __name__ == "__main__":
    _selftest()
    demo()
