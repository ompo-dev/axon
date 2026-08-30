"""
SINTAXE EMERGENTE (#0) — matar a cola na RAIZ: descobrir CLASSES de palavra sem rótulo.

Os 3 sinais que falharam (PMI, frequência, coerência) eram BAG-OF-CONTEXT: ignoram POSIÇÃO. O que
separa 'capitu' (substantivo) de 'aqui' (função) é o perfil POSICIONAL: quem vem à esquerda e à
direita. Substantivo: depois de artigo/prep, antes de verbo. Função tem outro perfil.

Represento cada palavra por [distribuição do vizinho-ESQUERDO, do vizinho-DIREITO] sobre as palavras
mais frequentes (as "casas" sintáticas) — o `after` do Toshi + seu transposto. Agrupo (k-means).
Classes de palavra EMERGEM. A(s) classe(s) das palavras funcionais = a cola, data-driven (não por
frequência). Isso é indução de POS distribucional (Brown/Schütze), leve, sem rótulo, sem backprop.

Valida: separa cola conhecida de conteúdo conhecido (onde freq/PMI/coerência falharam)?
Roda: python sintaxe.py   (usa o Toshi real)
"""
import sys
from collections import Counter, defaultdict
import numpy as np
from toshi import build_or_load


def kmeans(X, k, iters=40, tent=8, seed=0):
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    best_lab, best = None, -1e18
    for _ in range(tent):
        C = Xn[rng.choice(len(Xn), k, replace=False)].copy()
        lab = np.zeros(len(Xn), int)
        for _ in range(iters):
            lab = (Xn @ C.T).argmax(1)
            for j in range(k):
                m = Xn[lab == j]
                if len(m):
                    v = m.sum(0); C[j] = v / (np.linalg.norm(v) + 1e-9)
        inertia = float((Xn * C[lab]).sum())
        if inertia > best:
            best, best_lab = inertia, lab
    return best_lab


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando Toshi real...", flush=True)
    t, _ = build_or_load()

    # vizinho-esquerdo = transposto do `after` (o que PRECEDE cada palavra)
    before = defaultdict(Counter)
    for u, nxt in t.after.items():
        for v, c in nxt.items():
            before[v][u] += c

    F = 60
    feats = [w for w, _ in t.seen.most_common(F)]             # "casas" sintáticas = palavras frequentes
    fidx = {f: i for i, f in enumerate(feats)}

    def perfil(w):
        l = np.zeros(F); r = np.zeros(F)
        for u, c in before.get(w, {}).items():
            if u in fidx:
                l[fidx[u]] += c
        for u, c in t.after.get(w, {}).items():
            if u in fidx:
                r[fidx[u]] += c
        l /= (l.sum() + 1e-9); r /= (r.sum() + 1e-9)
        return np.concatenate([l, r])

    # palavras com estatística suficiente
    palavras = [w for w, _ in t.seen.most_common(1200) if t.seen[w] >= 20][:900]
    X = np.array([perfil(w) for w in palavras])
    K = 12
    lab = kmeans(X, K)
    widx = {w: i for i, w in enumerate(palavras)}

    print("=" * 70)
    print("SINTAXE EMERGENTE — classes de palavra por posição (esq+dir), sem rótulo")
    print("=" * 70)

    # gabarito p/ validar (função vs conteúdo)
    funcao = ["de", "a", "que", "e", "o", "as", "os", "um", "uma", "com", "por", "se",
              "nao", "mais", "como", "aqui", "quem", "entao", "onde", "quando"]
    conteudo = ["capitu", "mar", "escobar", "seminario", "olhos", "cavallo", "oceano",
                "ciume", "bento", "jose", "noite", "casa", "morte", "amor"]
    funcao = [w for w in funcao if w in widx]
    conteudo = [w for w in conteudo if w in widx]

    # rótulo do cluster: função se contém ÂNCORA frequente (as top-40 são funcionais) OU
    # se o gabarito-função domina. (a estrutura vem da sintaxe; a âncora só nomeia a classe.)
    ancora = set(w for w, _ in t.seen.most_common(40))
    rotulo = {}
    for j in range(K):
        membros = [w for w in palavras if lab[widx[w]] == j]
        f = sum(1 for w in funcao if lab[widx[w]] == j)
        c = sum(1 for w in conteudo if lab[widx[w]] == j)
        rotulo[j] = "func" if (any(w in ancora for w in membros) or f > c) else "cont"
    acc = np.mean([rotulo[lab[widx[w]]] == "func" for w in funcao] +
                  [rotulo[lab[widx[w]]] == "cont" for w in conteudo])

    print(f"\nseparação função vs conteúdo (onde freq/PMI/coerência deram 50%): {acc:.0%}")
    print(f"  'capitu' -> cluster {lab[widx['capitu']]} ({rotulo[lab[widx['capitu']]]})   "
          f"'aqui' -> cluster {lab[widx['aqui']]} ({rotulo[lab[widx['aqui']]]})" if
          'capitu' in widx and 'aqui' in widx else "")

    # mostra o conteúdo de alguns clusters (as classes que emergiram)
    print("\n  classes emergentes (amostra de cada cluster):")
    for j in range(K):
        membros = [w for w in palavras if lab[widx[w]] == j]
        if membros:
            freq_mass = np.mean([t.seen[w] for w in membros])
            print(f"    c{j:<2} [{rotulo[j]}] : {' '.join(membros[:8])}")

    # PAYOFF: usar a classe-FUNÇÃO como filtro de cola na abdução -> topo vira conteúdo puro
    func_set = {w for w in palavras if rotulo[lab[widx[w]]] == "func"}
    cont_w = [w for w in palavras if w not in func_set]
    cset = set(cont_w)
    vizC = {w: ((set(t.assoc.get(w, {})) & cset) - func_set) for w in cont_w}
    cand = {}
    for a in cont_w:
        viz2 = set()
        for x in vizC[a]:
            viz2 |= vizC.get(x, set())
        for b in viz2:
            if b != a and b not in vizC[a]:
                key = (a, b) if a < b else (b, a)
                if key not in cand:
                    comuns = (vizC[a] - {b}) & (vizC[b] - {a})
                    cand[key] = sum(1.0 / np.log(2 + t.seen[x]) for x in comuns)
    top = sorted(cand, key=lambda k: -cand[k])[:10]
    print("\n  PAYOFF — abdução SÓ entre conteúdo (cola filtrada pela classe-função):")
    for a, b in top:
        print(f"     {a} ~ {b}")

    print("\n" + "=" * 70)
    if acc >= 0.8:
        print(f"VEREDITO: classes de palavra EMERGIRAM da posição — separa função/conteúdo a {acc:.0%},")
        print("onde os 3 sinais bag-of-context deram ~50%. A cola vira uma CLASSE descoberta (não um")
        print("limiar de frequência que apagaria 'capitu'). Isso mata a cola na raiz E dá estrutura")
        print("gramatical emergente pro raciocínio — o #0 do radar, feito. Sem rótulo, sem backprop.")
    else:
        print(f"VEREDITO honesto: separação {acc:.0%} — fraco, reportado sem inflar.")


if __name__ == "__main__":
    main()
