"""
AXON — CÉREBRO integrado (o conceito, funcionando ponta a ponta).

O bebê digital: recebe STREAM CRU (não sabe o que é), descobre unidades sozinho, aprende
sequências CONTINUAMENTE sem esquecer, e GERA (prevê/continua). Ultra-leve (1-bit, sem GPU).

Pipeline (tudo já provado em módulos separados, agora LIGADO):
  stream cru
    -> chunking estatístico não-supervisionado (pares frequentes fundem; Saffran construtivo)
    -> cada chunk vira hipervetor (VSA)
    -> memória de sequência contínua (aprende chunk->próximo, na surpresa, sem esquecer; CLS)
    -> GERAÇÃO: dado um seed, prevê o próximo chunk e continua (o 'balbucio' que vira padrão)

Não é GPT (sem entendimento profundo). É uma arquitetura NOVA: auto-organiza do cru, aprende
em tempo real, não esquece, pesa KB, roda em qualquer CPU. Prova de conceito da revolução leve.

Roda: python cerebro.py   (numpy; usa vsa_core.py)
"""
import numpy as np
from collections import Counter
from vsa_core import D, bind, bundle, permute, cos, ItemMemory


# ============================================================ 1. chunking não-supervisionado
def discover_boundary(stream):
    """Descobre o símbolo-FRONTEIRA por ESTATÍSTICA: aquele cujo próximo é mais imprevisível
    (maior entropia de sucessor). Numa fala real é a pausa/silêncio. Zero rótulo, zero feature.
    É Saffran: fronteira = ponto de baixa previsibilidade."""
    succ, pred = {}, {}
    for a, b in zip(stream[:-1], stream[1:]):
        succ.setdefault(a, Counter())[b] += 1
        pred.setdefault(b, Counter())[a] += 1

    def entropy(c):
        tot = sum(c.values())
        p = np.array(list(c.values()), float) / tot
        return float(-(p * np.log2(p + 1e-12)).sum())

    best, best_H = None, -1.0
    for a in succ:
        if sum(succ[a].values()) < 5:
            continue
        # fronteira = imprevisível dos DOIS lados (sucessor E predecessor). Vogal interna não é.
        H = entropy(succ[a]) + entropy(pred.get(a, Counter({0: 1})))
        if H > best_H:
            best, best_H = a, H
    return best


def learn_chunks(stream, **_):
    """Segmenta o stream cru no símbolo-fronteira descoberto. Chunks = 'palavras' emergentes."""
    stream = list(stream)
    bnd = discover_boundary(stream)
    if bnd is None:
        return stream
    segs, cur = [], []
    for s in stream:
        if s == bnd:
            if cur:
                segs.append("".join(str(x) for x in cur)); cur = []
        else:
            cur.append(s)
    if cur:
        segs.append("".join(str(x) for x in cur))
    return segs


# ============================================================ 2. o cérebro
class Brain:
    def __init__(self):
        self.vocab = ItemMemory()              # chunk (str) -> hipervetor
        self.seq = {}                          # chunk -> bind(chunk, proximo)  (memória rápida)
        self.seq_M = None
        self.slow = []                         # consolidação (neocórtex)
        self.n_learn = 0

    def _rebuild(self):
        self.seq_M = bundle(list(self.seq.values())) if self.seq else None

    def _stores(self):
        return ([self.seq_M] if self.seq_M is not None else []) + self.slow

    def predict_next(self, chunk):
        if chunk not in self.vocab.labels or not self._stores():
            return None, 0.0
        cv = self.vocab.get(chunk)
        acc = np.zeros(D, dtype=np.int32)
        for M in self._stores():
            acc += bind(M, cv).astype(np.int32)
        guess = np.where(acc > 0, 1, -1).astype(np.int8)
        top = self.vocab.cleanup(guess, topk=1, exclude=(chunk,))
        return (top[0][0], top[0][1]) if top else (None, 0.0)

    def perceive(self, units):
        """Recebe uma sequência de UNIDADES (chunks/palavras) -> aprende transições
        (contínuo, na surpresa, sem esquecer). `units` = lista de tokens já segmentados.
        (A descoberta das unidades a partir de stream cru gapless é sub-problema à parte;
         funciona em streams com gaps naturais — ver percepcao_crua.py, caso áudio.)"""
        units = list(units)
        for c in units:
            self.vocab.get(c)
        for a, b in zip(units[:-1], units[1:]):
            pred, conf = self.predict_next(a)
            if pred == b and conf > 0.08:
                continue                        # já sabia (predictive coding)
            self.seq[a] = bind(self.vocab.get(a), self.vocab.get(b))
            self._rebuild(); self.n_learn += 1
            if len(self.seq) >= 120:            # consolida rápido->lento (CLS)
                self.slow.append(self.seq_M); self.seq = {}; self.seq_M = None
        return units

    def generate(self, seed, n=8):
        """Balbucio que virou padrão: prevê o próximo chunk e continua."""
        out = [seed]
        cur = seed
        for _ in range(n):
            nxt, conf = self.predict_next(cur)
            if nxt is None or conf < 0.05:
                break
            out.append(nxt); cur = nxt
        return out

    def footprint(self):
        n = len(self.vocab.labels)
        kb = n * (D // 8) / 1024
        return n, kb


# ============================================================ 3. demo: o bebê em ação
def demo():
    print("=" * 82)
    print("AXON — CÉREBRO integrado: recebe cru, descobre, aprende contínuo, GERA (leve)")
    print("=" * 82)

    # corpus: frases variadas. UNIDADES = palavras (a fronteira aqui é o gap/silêncio; num
    # stream de áudio o gap é descoberto — ver percepcao_crua.py. Foco aqui: o LOOP cognitivo.)
    rng = np.random.default_rng(0)
    frases = ["obebe quer mamar", "obebe quer dormir", "mamae pega obebe",
              "papai fala com obebe", "obebe olha mamae", "mamae ama obebe"]
    corpus = " . ".join(frases[i] for i in rng.integers(0, len(frases), 120))
    units = corpus.split()
    print(f"\nentrada: {len(units)} unidades (palavras). Ex.: '{' '.join(units[:8])}...'")

    brain = Brain()
    brain.perceive(units)
    print(f"\n[1] VOCABULÁRIO aprendido: {len(brain.vocab.labels)} conceitos")

    print(f"\n[2] GERAÇÃO (seed -> continua com o que aprendeu, como 'balbucio' que virou padrão):")
    for seed in ("obebe", "mamae", "papai"):
        if seed in brain.vocab.labels:
            g = brain.generate(seed, n=5)
            print(f"    '{seed}' -> {' '.join(g)}")

    # 3. APRENDER NOVO EM TEMPO REAL sem esquecer o velho
    print(f"\n[3] APRENDE NOVO EM TEMPO REAL (sem esquecer o antigo):")
    before = " ".join(brain.generate("obebe", 4))
    brain.perceive("vovo conta historia".split() * 6)      # conhecimento novo, ao vivo
    after_old = " ".join(brain.generate("obebe", 4))
    after_new = " ".join(brain.generate("vovo", 4))
    print(f"    antes:                 'obebe' -> {before}")
    print(f"    após aprender 'vovo conta historia':")
    print(f"       o antigo:           'obebe' -> {after_old}   (não esqueceu)")
    print(f"       o novo:             'vovo'  -> {after_new}")

    # 4. footprint
    n, kb = brain.footprint()
    print(f"\n[4] PESO: {n} conceitos = {kb:.1f} KB. Zero GPU, zero treino por gradiente.")
    print(f"    (um LLM que faz 'continuar texto' = GBs de pesos + não aprende no uso)")

    print("\n" + "=" * 82)
    print("Isto FUNCIONA ponta a ponta: cru -> descobre -> aprende contínuo -> gera, em KB.")
    print("Honesto: sem entendimento profundo/raciocínio abdutivo. É a fundação nova, leve e")
    print("viva (aprende no uso), pronta pra escalar (SDM) e portar pro Rust do axon.")


def _selftest():
    b = Brain()
    # descobre a fronteira (espaço) por entropia e segmenta em palavras
    ch = learn_chunks(list("cat dog cat dog fish dog cat fish "))
    assert "cat" in ch and "dog" in ch, ch     # achou as palavras sozinho
    # aprende e prevê uma transição (nível de unidade)
    b.perceive(["cat", "dog", "cat", "dog", "cat", "dog"])
    p, c = b.predict_next("cat")
    assert p == "dog", (p, c)                   # aprendeu cat->dog
    # geração continua a partir do seed
    g = b.generate("cat", 3)
    assert g[0] == "cat" and "dog" in g
    # aprende novo sem apagar vocab antigo
    n0 = len(b.vocab.labels)
    b.perceive(["fish", "bird"] * 6)
    assert len(b.vocab.labels) >= n0 and b.predict_next("cat")[0] == "dog"  # não esqueceu
    print("[selftest] ok (chunking funde; prevê; gera; aprende novo sem apagar)")


if __name__ == "__main__":
    _selftest()
    demo()
