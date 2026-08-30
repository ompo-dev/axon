"""
AXON cérebro — degrau: CONTEXTO (memória de trabalho) -> geração coerente, não-eco.

Defeito da v1 (cerebro.py): predizia só da palavra atual (1ª ordem) -> repetia ("obebe pega
obebe pega"). Falta CONTEXTO. O cérebro usa memória de trabalho: as últimas coisas ficam
ativas ao mesmo tempo, com pesos diferentes (recência).

Aqui: uma MEMÓRIA DE TRABALHO = hipervetor rolante que acumula as últimas k unidades com
POSIÇÃO (permute) e decaimento. É um RNN/SSM de 1-bit, sem gradiente, 1-shot. A predição usa
o contexto inteiro -> segue os k-gramas aprendidos -> gera coerente, não em loop.

Ponte com o axon original: esse "estado rolante com decaimento" É a temperatura/amplitude do
axon; a posição (permute) é a fase. Working memory = superposição com recência (PLOS 2017).

Roda: python cerebro_contexto.py   (numpy; usa vsa_core.py)
"""
from copy import deepcopy

import numpy as np
from vsa_core import D, RNG, bind, bundle, permute, cos, ItemMemory

K = 3                                          # tamanho da janela de contexto (ordem-n)


class ContextBrain:
    def __init__(self, k=K):
        self.k = k
        self.vocab = ItemMemory()
        self.fast = []                         # binds recentes (contexto -> próximo)
        self.fast_M = None
        self.slow = []
        self.n_learn = 0

    def context(self, words):
        """Memória de trabalho das últimas k palavras, com posição. Recente = posição 0."""
        w = [x for x in words if x][-self.k:]
        if not w:
            return None
        parts = [permute(self.vocab.get(x), j) for j, x in enumerate(reversed(w))]
        return bundle(parts)

    def _stores(self):
        return ([self.fast_M] if self.fast_M is not None else []) + self.slow

    def predict(self, ctx_words):
        ctx = self.context(ctx_words)
        if ctx is None or not self._stores():
            return None, 0.0
        acc = np.zeros(D, dtype=np.int32)
        for M in self._stores():
            acc += bind(M, ctx).astype(np.int32)
        guess = np.where(acc > 0, 1, -1).astype(np.int8)
        top = self.vocab.cleanup(guess, topk=1)
        return (top[0][0], top[0][1]) if top else (None, 0.0)

    def perceive(self, units):
        units = list(units)
        for c in units:
            self.vocab.get(c)
        for i in range(len(units) - 1):
            ctx_words = units[max(0, i - self.k + 1):i + 1]
            nxt = units[i + 1]
            pred, conf = self.predict(ctx_words)
            if pred == nxt and conf > 0.08:
                continue                        # já sabia (predictive coding)
            self.fast.append(bind(self.context(ctx_words), self.vocab.get(nxt)))
            self.fast_M = bundle(self.fast); self.n_learn += 1
            if len(self.fast) >= 150:
                self.slow.append(self.fast_M); self.fast = []; self.fast_M = None
        return units

    def generate(self, seed_words, n=8):
        ctx = list(seed_words)
        out = list(seed_words)
        for _ in range(n):
            nxt, conf = self.predict(ctx)
            if nxt is None or conf < 0.03:
                break
            out.append(nxt); ctx = (ctx + [nxt])[-self.k:]
        return out

    def footprint(self):
        n = len(self.vocab.labels)
        return n, n * (D // 8) / 1024


def demo():
    print("=" * 82)
    print("AXON cérebro — CONTEXTO (memória de trabalho): geração coerente, não-eco")
    print("=" * 82)
    rng = np.random.default_rng(0)
    frases = ["obebe quer mamar", "obebe quer dormir", "mamae pega obebe",
              "papai fala com obebe", "obebe olha mamae", "mamae ama obebe"]
    corpus = " . ".join(frases[i] for i in rng.integers(0, len(frases), 150))
    units = corpus.split()
    print(f"\nentrada: {len(units)} palavras. Frases-base: {len(frases)}. Contexto k={K}.")

    b = ContextBrain()
    b.perceive(units)

    print(f"\n[GERAÇÃO COM CONTEXTO] (compare com o eco da v1):")
    for seed in (["obebe", "quer"], ["mamae", "pega"], ["papai", "fala"], ["obebe", "olha"]):
        g = b.generate(seed, n=5)
        print(f"    {seed} -> {' '.join(g)}")

    print(f"\n[APRENDE NOVO AO VIVO, sem esquecer]:")
    old = " ".join(b.generate(["obebe", "quer"], 3))
    b.perceive("vovo conta uma historia linda".split() * 8)
    print(f"    antigo: ['obebe','quer'] -> {' '.join(b.generate(['obebe','quer'],3))}  (mantém)")
    print(f"    novo:   ['vovo','conta'] -> {' '.join(b.generate(['vovo','conta'],4))}")

    n, kb = b.footprint()
    print(f"\n[PESO] {n} conceitos = {kb:.1f} KB. Sem GPU, sem gradiente, aprende no uso.")
    print("\n" + "=" * 82)
    print("Contexto ordem-n = geração segue frases aprendidas, não repete. É a working memory")
    print("do axon (recência+posição) em VSA 1-bit. Próximo: escala (SDM) + porte Rust.")


def _selftest():
    original_rng_state = deepcopy(RNG.bit_generator.state)
    try:
        RNG.bit_generator.state = np.random.default_rng(42).bit_generator.state
        b = ContextBrain(k=3)
        # aprende uma frase e a reproduz a partir do contexto
        b.perceive("a b c d a b c d a b c d".split())
        g = b.generate(["a", "b"], 3)
        assert g[:2] == ["a", "b"] and "c" in g, g
        # contexto importa: prever de (a,b) tende a 'c'
        p, _ = b.predict(["a", "b"])
        assert p == "c", p
        # aprende novo sem esquecer o antigo
        b.perceive("x y z x y z x y z".split())
        assert b.predict(["a", "b"])[0] == "c"          # não esqueceu
        assert b.predict(["x", "y"])[0] == "z"          # aprendeu o novo
    finally:
        RNG.bit_generator.state = original_rng_state
    print("[selftest] ok (contexto prevê k-grama; gera; aprende novo sem esquecer)")


if __name__ == "__main__":
    _selftest()
    demo()
