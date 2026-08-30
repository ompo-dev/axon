"""
TOSHI — melhoria #1: aprendizado PREDITIVO por ERRO (refina o significado).

Hoje Toshi acumula co-ocorrência (Hebbian) — SEM correção de erro -> embeddings ruidosos.
A pesquisa (predictive coding; Forward-Forward, Hinton 2022; word2vec/SGNS) diz: PREVER e
corrigir pelo ERRO afia a representação. Aqui: skip-gram com amostragem negativa (SGNS), a
versão preditiva/por-erro do que ele já faz — LOCAL, online, SEM backprop, leve.

Regra (por par que ele co-ocorreu (w,u), peso c):
  positivo: aproxima v_w e v_u   (eles PREVEEM um ao outro)
  negativo: afasta v_w de k palavras aleatórias (que NÃO deviam prever)
Só operações de vetor + sigmoide. Refina os embeddings que o Random Indexing só acumulou.

Mede: pares semanticamente relacionados ficam MAIS perto que aleatórios? (antes vs depois)
Roda: python predictivo.py   (usa a memória salva do toshi.py)
"""
import numpy as np
from toshi import build_or_load


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def treinar_sgns(t, epochs=3, lr=0.05, neg=5, min_freq=5, seed=0):
    """Refina t.embed por SGNS a partir das co-ocorrências já armazenadas em t.assoc."""
    rng = np.random.default_rng(seed)
    vocab = [w for w in t.embed if t.seen[w] >= min_freq]
    idx = {w: i for i, w in enumerate(vocab)}
    V = np.array([t.embed[w] for w in vocab], np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)     # começa dos vetores RI (bom chute)
    freq = np.array([t.seen[w] for w in vocab], np.float64) ** 0.75
    negp = freq / freq.sum()                                    # distribuição de negativos (word2vec)

    # lista de pares positivos (w, u, peso) só entre palavras do vocab
    pares = []
    for w in vocab:
        wi = idx[w]
        for u, c in t.assoc.get(w, {}).items():
            if u in idx and c >= 1.0:
                pares.append((wi, idx[u], min(c, 6.0)))
    pares = np.array(pares)
    print(f"  vocab={len(vocab)}  pares positivos={len(pares)}  epochs={epochs}")

    for ep in range(epochs):
        rng.shuffle(pares)
        for wi, ui, c in pares:
            wi, ui = int(wi), int(ui)
            vw, vu = V[wi], V[ui]
            # positivo: prever u a partir de w
            g = (1.0 - sigmoid(vw @ vu)) * lr * c
            gw = g * vu; gu = g * vw
            # negativos: w NÃO deve prever estes
            negs = rng.choice(len(vocab), neg, p=negp)
            for zi in negs:
                vz = V[zi]
                gn = sigmoid(vw @ vz) * lr
                gw -= gn * vz
                V[zi] -= gn * vw
            V[wi] = vw + gw; V[ui] = vu + gu
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        print(f"  epoch {ep+1}/{epochs} ok")
    return vocab, idx, V


def qualidade(vocab, idx, V, pares_rel, rng):
    """Gap semântico: cos(pares relacionados) - cos(pares aleatórios). Maior = melhor estrutura."""
    def cos(a, b):
        return float(V[idx[a]] @ V[idx[b]])
    rel = [cos(a, b) for a, b in pares_rel if a in idx and b in idx]
    words = list(idx)
    alea = []
    for _ in range(200):
        a, b = rng.choice(words, 2, replace=False)
        alea.append(float(V[idx[a]] @ V[idx[b]]))
    return np.mean(rel), np.mean(alea), np.mean(rel) - np.mean(alea)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t, _ = build_or_load()
    rng = np.random.default_rng(1)
    # pares que DEVEM estar relacionados (de Dom Casmurro & cia)
    pares_rel = [("mar", "ceu"), ("amor", "ciume"), ("olhos", "ressaca"), ("mulher", "esposa"),
                 ("morte", "vida"), ("rei", "reino"), ("padre", "seminario"), ("mae", "filho")]

    print("=" * 66)
    print("TOSHI — antes (Hebbian/acúmulo) vs depois (preditivo por erro / SGNS)")
    print("=" * 66)
    vocab = [w for w in t.embed if t.seen[w] >= 5]
    idx = {w: i for i, w in enumerate(vocab)}
    V0 = np.array([t._emb(w) for w in vocab], np.float32)
    r0, a0, g0 = qualidade(vocab, idx, V0, pares_rel, np.random.default_rng(1))
    print(f"\n[ANTES]  cos(relacionados)={r0:+.3f}  cos(aleatório)={a0:+.3f}  GAP={g0:+.3f}")

    print("\ntreinando preditivo (SGNS, sem backprop)...")
    vocab2, idx2, V = treinar_sgns(t, epochs=3)
    r1, a1, g1 = qualidade(vocab2, idx2, V, pares_rel, np.random.default_rng(1))
    print(f"\n[DEPOIS] cos(relacionados)={r1:+.3f}  cos(aleatório)={a1:+.3f}  GAP={g1:+.3f}")
    print(f"\n>>> GAP semântico {g0:+.3f} -> {g1:+.3f}  "
          f"({'MELHOROU' if g1 > g0 + 0.02 else 'não melhorou claramente'})")

    # mostra vizinhos de algumas palavras depois do refino
    print("\nvizinhos DEPOIS do refino preditivo:")
    for w in ("mar", "amor", "morte", "capitu"):
        if w in idx2:
            sims = sorted(((float(V[idx2[w]] @ V[idx2[u]]), u) for u in vocab2 if u != w),
                          reverse=True)[:5]
            print(f"  {w:<8} -> " + ", ".join(f"{u}·{s:.2f}" for s, u in sims))


if __name__ == "__main__":
    main()
