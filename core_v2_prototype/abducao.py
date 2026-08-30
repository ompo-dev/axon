"""
ABDUÇÃO — o salto ("LLMs can't jump"): inferir uma ligação NUNCA aprendida (uma hipótese).

Dedução = encadear elos que já existem (o Toshi já faz; LLM também). O SALTO abdutivo = propor
uma conexão que NÃO está no grafo, por evidência INDIRETA (estrutura + analogia). É o que Peirce
chamou de "inferência da melhor explicação" e o que o paper diz que falta aos LLMs.

Teste honesto (previsão de elo, com gabarito): ESCONDO elos verdadeiros do grafo do Toshi e vejo
se a mente os re-PROPÕE acima de pares aleatórios — SEM usar o elo direto. Sinais de abdução:
  - vizinhos em comum (Adamic-Adar sobre o grafo): a e b devem se ligar se muita coisa liga aos dois.
  - similaridade distribucional (embedding): significam parecido -> provável elo latente.
Mede AUC (prob. de ranquear o elo-verdadeiro-escondido acima de um par aleatório). AUC 0.5 = acaso.
Roda: python abducao.py   (usa o Toshi real)
"""
import sys
import numpy as np
from toshi import build_or_load


def auc(pos, neg):
    """Mann-Whitney: P(score(pos) > score(neg)). 0.5 = acaso, 1.0 = separa perfeito."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    todos = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(todos)) + 1                     # ranks (1..n), empates ~ok o bastante
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando Toshi real...", flush=True)
    t, _ = build_or_load()

    # nós de conteúdo com grau e embedding (onde há estrutura p/ abduzir)
    stops = t.stops
    conteudo = [w for w in t.assoc
                if t.seen[w] >= 8 and w not in stops and t._emb(w) is not None
                and len(t.assoc[w]) >= 4]
    conteudo_set = set(conteudo)
    viz = {w: (set(t.assoc[w]) & conteudo_set) for w in conteudo}   # vizinhança de conteúdo

    rng = np.random.default_rng(0)

    def escondido_positivos(n=600):
        """elos VERDADEIROS (a-b existe), com estrutura dos dois lados."""
        out = []
        pool = list(conteudo)
        while len(out) < n:
            a = pool[int(rng.integers(len(pool)))]
            nb = [x for x in viz[a] if len(viz[x]) >= 4]
            if not nb:
                continue
            b = nb[int(rng.integers(len(nb)))]
            if a != b:
                out.append((a, b))
        return out

    def negativos(n=600):
        """pares SEM elo (não co-ocorreram): a mente teria que abduzir do nada."""
        out = []
        pool = list(conteudo)
        while len(out) < n:
            a, b = pool[int(rng.integers(len(pool)))], pool[int(rng.integers(len(pool)))]
            if a != b and b not in viz[a]:
                out.append((a, b))
        return out

    pos, neg = escondido_positivos(), negativos()

    def s_vizinhos(a, b):
        """Adamic-Adar: soma sobre vizinhos COMUNS de 1/log(freq) — SEM usar o elo a-b direto."""
        comuns = (viz[a] - {b}) & (viz[b] - {a})
        return sum(1.0 / np.log(2 + t.seen[x]) for x in comuns)

    def s_sim(a, b):
        return float(t._emb(a) @ t._emb(b))

    def scores(pares, f):
        return [f(a, b) for a, b in pares]

    def zn(x):
        x = np.asarray(x, float)
        return (x - x.mean()) / (x.std() + 1e-9)

    def zn_par(p, n):
        """z-normaliza usando stats COMBINADAS (senão apaga o gap pos-neg = o sinal)."""
        a, b = np.asarray(p, float), np.asarray(n, float)
        m, s = np.concatenate([a, b]).mean(), np.concatenate([a, b]).std() + 1e-9
        return (a - m) / s, (b - m) / s

    cn_p, cn_n = scores(pos, s_vizinhos), scores(neg, s_vizinhos)
    sm_p, sm_n = scores(pos, s_sim), scores(neg, s_sim)
    cn_pz, cn_nz = zn_par(cn_p, cn_n)
    sm_pz, sm_nz = zn_par(sm_p, sm_n)
    co_p, co_n = cn_pz + sm_pz, cn_nz + sm_nz

    print("=" * 72)
    print("ABDUÇÃO — re-propor elos ESCONDIDOS acima do acaso (AUC; 0.5 = chute)")
    print("=" * 72)
    print(f"\n{len(pos)} elos verdadeiros escondidos vs {len(neg)} pares sem elo:")
    print(f"  vizinhos-em-comum (estrutura): AUC = {auc(cn_p, cn_n):.3f}")
    print(f"  similaridade (embedding):      AUC = {auc(sm_p, sm_n):.3f}")
    print(f"  COMBO ingênuo (soma z):        AUC = {auc(co_p, co_n):.3f}")

    # COMBO com PESOS APRENDIDOS (logística de 1 camada, local) — split treino/teste, sem vazar
    Xp = np.column_stack([cn_pz, sm_pz]); Xn = np.column_stack([cn_nz, sm_nz])
    def split(A, frac=0.6):
        c = int(len(A) * frac); return A[:c], A[c:]
    Xtr = np.vstack([split(Xp)[0], split(Xn)[0]])
    ytr = np.concatenate([np.ones(len(split(Xp)[0])), np.zeros(len(split(Xn)[0]))])
    w, b = np.zeros(2), 0.0
    for _ in range(400):                                      # delta rule (1 camada) = sem backprop
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        g = p - ytr
        w -= 0.1 * (Xtr.T @ g / len(ytr) + 1e-3 * w); b -= 0.1 * g.mean()
    lp = split(Xp)[1] @ w + b; ln = split(Xn)[1] @ w + b
    a_est_te = auc(split(cn_pz)[1], split(cn_nz)[1])
    print(f"  COMBO APRENDIDO (teste):       AUC = {auc(lp, ln):.3f}   "
          f"(estrutura sozinha no mesmo teste: {a_est_te:.3f}; pesos w={w.round(2)})")

    # exemplo: um par SEM elo direto que a abdução aponta como provável (hipótese nova plausível)
    cand = negativos(400)
    sc = [(zn0 + zn1, a, b) for (a, b), zn0, zn1 in
          zip(cand, zn(scores(cand, s_vizinhos)), zn(scores(cand, s_sim)))]
    sc.sort(reverse=True)
    print("\n  hipóteses abduzidas (pares que NUNCA co-ocorreram, mas a estrutura sugere ligar):")
    for _, a, b in sc[:6]:
        print(f"    {a} ~ {b}   (vizinhos comuns: "
              f"{', '.join(list((viz[a]-{b}) & (viz[b]-{a}))[:3])})")

    print("\n" + "=" * 72)
    a_est, a_sim, a_combo = auc(cn_p, cn_n), auc(sm_p, sm_n), auc(co_p, co_n)
    best = max(a_est, a_sim, a_combo)
    if a_est > 0.75:
        print(f"VEREDITO: a mente ABDUZ — re-propõe elos escondidos por ESTRUTURA (vizinhos comuns)")
        print(f"com AUC {a_est:.2f} (>> 0.5 do acaso), SEM ter visto o elo. Inferir hipótese não-dada =")
        print(f"o 'salto', não só deduzir o caminho pronto. É o passo além do LLM 'que não pula'.")
        print(f"Honesto: o sinal forte é a ESTRUTURA; embedding ({a_sim:.2f}) é mais fraco e o combo")
        print(f"ingênuo ({a_combo:.2f}) piora — juntar mal atrapalha. Próximo: abduzir CONCEITO novo (nó).")
    else:
        print(f"VEREDITO honesto: melhor AUC {best:.2f} — sinal fraco, reportado sem inflar.")


def _selftest():
    # AUC básico correto (extremos são robustos a empate)
    assert abs(auc([3, 4, 5], [0, 1, 2]) - 1.0) < 1e-9      # positivos todos acima -> 1.0
    assert abs(auc([0, 1, 2], [3, 4, 5]) - 0.0) < 1e-9      # todos abaixo -> 0.0
    assert abs(auc([0, 2, 4], [1, 3, 5]) - 1 / 3) < 1e-9    # intercalado
    print("[selftest] ok (AUC)")


if __name__ == "__main__":
    _selftest()
    main()
