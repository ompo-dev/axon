"""
ABDUÇÃO DE CONCEITO — o salto PLENO: inventar um NÓ (categoria latente) que explica observações.

Elo (abducao.py) = ligar dois nós existentes. CONCEITO = criar um nó NOVO: uma causa/categoria
oculta que explica por que um monte de coisas anda junto. É "mudar o espaço do problema" (Peirce;
a tese do 'LLMs can't jump'). Sem rótulo: a mente só vê observações e seus contextos, e tem que
DESCOBRIR os grupos latentes — e nomear cada um por um protótipo (Rosch).

Teste com gabarito: gero k categorias ocultas (membros que compartilham contexto), escondo os
rótulos, e mede se a mente re-descobre os grupos (pureza vs acaso 1/k) e classifica um membro NOVO.
Aprendizado é o mesmo cru do Toshi (distribucional), sem priors. Roda: python abducao_conceito.py
"""
import sys
import numpy as np
from toshi import Toshi

LETRAS = "abcdefghijklmnopqrstuvwxyz"


def kmeans_cos(X, k, iters=25, tentativas=6, seed=0):
    """k-means esférico (cos) simples — a mente agrupa por significado. Sem libs externas."""
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    melhor_lab, melhor_in = None, -1e18
    for _ in range(tentativas):
        C = Xn[rng.choice(len(Xn), k, replace=False)].copy()
        lab = None
        for _ in range(iters):
            sim = Xn @ C.T
            lab = sim.argmax(1)
            for j in range(k):
                m = Xn[lab == j]
                if len(m):
                    v = m.sum(0); C[j] = v / (np.linalg.norm(v) + 1e-9)
        inertia = float((Xn * C[lab]).sum())                 # soma de cos ao centro (maior=melhor)
        if inertia > melhor_in:
            melhor_in, melhor_lab = inertia, lab
    return melhor_lab, C


def pureza(lab, gab, k):
    """Cada cluster recebe a categoria-gabarito majoritária; fração de acertos."""
    ok = 0
    for j in range(k):
        idx = np.where(lab == j)[0]
        if len(idx):
            vals, cont = np.unique(gab[idx], return_counts=True)
            ok += cont.max()
    return ok / len(gab)


def gerar_e_aprender(t, k=6, membros=8, ctx=5, ruido=0.15, seed=0):
    """k categorias ocultas; membros compartilham contextos da sua categoria (+ ruído cruzado)."""
    rng = np.random.default_rng(seed)
    cats_ctx = [[f"c{LETRAS[c]}{LETRAS[j]}" for j in range(ctx)] for c in range(k)]
    membros_tok, gab = [], []
    for c in range(k):
        for i in range(membros):
            m = f"m{LETRAS[c]}{LETRAS[i]}"
            membros_tok.append(m); gab.append(c)
            for _ in range(4):                                # cada membro co-ocorre com seus contextos
                for cw in cats_ctx[c]:
                    cc = cw if rng.random() > ruido else rng.choice(cats_ctx[rng.integers(k)])
                    t.perceive([m, cc])
    return membros_tok, np.array(gab), cats_ctx


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 72)
    print("ABDUÇÃO DE CONCEITO — inventar a categoria latente que explica as observações")
    print("=" * 72)

    K = 6
    t = Toshi()
    membros, gab, cats_ctx = gerar_e_aprender(t, k=K)
    X = np.array([t.embed[m] for m in membros], np.float32)
    lab, C = kmeans_cos(X, K)
    pur = pureza(lab, gab, K)

    print(f"\n{len(membros)} observações, {K} categorias OCULTAS (sem rótulo):")
    print(f"  a mente re-descobriu os grupos: pureza = {pur:.0%}  (acaso ~ {1/K:.0%})")

    # classificar um membro NOVO (nunca visto) na categoria inventada certa (protótipo = Rosch)
    rng = np.random.default_rng(1)
    acertos = tot = 0
    for c in range(K):
        novo = f"n{LETRAS[c]}"                                # membro novo da categoria c
        for _ in range(6):                                    # ele co-ocorre com contextos da sua cat
            for cw in cats_ctx[c]:
                t.perceive([novo, cw])
        v = t.embed[novo]; v = v / (np.linalg.norm(v) + 1e-9)
        # protótipo de cada conceito inventado = centro do cluster; escolhe o mais próximo
        prot = np.array([X[lab == j].mean(0) for j in range(K)])
        prot = prot / (np.linalg.norm(prot, axis=1, keepdims=True) + 1e-9)
        pred = int((prot @ v).argmax())
        # o cluster 'pred' corresponde a qual categoria-gabarito? (a majoritária dele)
        idx = np.where(lab == pred)[0]
        cat_do_cluster = int(np.bincount(gab[idx]).argmax()) if len(idx) else -1
        acertos += (cat_do_cluster == c); tot += 1
    print(f"  classifica um membro NOVO no conceito inventado certo: {acertos}/{tot} = {acertos/tot:.0%}")

    print("\n" + "=" * 72)
    if pur > 0.7 and acertos / tot > 0.6:
        print(f"VEREDITO: a mente ABDUZIU CONCEITOS — inventou as {K} categorias ocultas ({pur:.0%} pureza,")
        print("acaso ~17%) só das observações, sem rótulo, e encaixa coisa nova no conceito certo. Isso é")
        print("criar um NÓ que muda o espaço do problema — o salto abdutivo pleno, além de ligar elos.")
        print("(distribucional puro, sem priors; é o princípio de protótipo de Rosch emergindo.)")
    else:
        print(f"VEREDITO honesto: pureza {pur:.0%} — sinal fraco, reportado sem inflar.")


def _selftest():
    t = Toshi()
    membros, gab, _ = gerar_e_aprender(t, k=4, membros=6, seed=2)
    X = np.array([t.embed[m] for m in membros], np.float32)
    lab, _ = kmeans_cos(X, 4)
    assert pureza(lab, gab, 4) > 0.6, "não recuperou os grupos latentes"
    print("[selftest] ok (recupera categorias ocultas por significado)")


if __name__ == "__main__":
    _selftest()
    main()
