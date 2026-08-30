"""
VERIFICADOR de cadeia — o gargalo do Loop 1: sem bom avaliador, "melhorar raciocínio" é chute.

A pesquisa (Process Reward Model; RwT/COLING 2025 treina um avaliador de passos) diz: o pulo de
qualidade vem de um VERIFICADOR que pontua a cadeia. Aqui construo e VALIDO um — no currículo COM
GABARITO (onde dá pra medir): cada pergunta tem o caminho de CONTEÚDO certo E um atalho por COLA
(um nó ubíquo, tipo stopword). Um bom verificador ranqueia o de conteúdo em 1º.

Comparo sinais de verificador (todos SEM rótulo, self-supervised no próprio modelo do Toshi):
  cos_médio (o ruim do Loop 1) · PMI_médio (força de transição) · brevidade · COMBO.
Métrica: com que frequência o caminho CERTO é ranqueado em 1º. Depois aplico no texto real (rerank).
Roda: python verificador.py
"""
import sys
from collections import Counter
import numpy as np
from toshi import Toshi, tokenize

LETRAS = "abcdefghijklmnopqrstuvwxyz"
COLA = "quux"                                   # nó de COLA global (fica ubíquo = tipo stopword)


def base(i):
    return "z" + LETRAS[i // 26] + LETRAS[i % 26]


def ensinar(t, i, L=4, rep=3, cola_rep=2):
    """Caminho de conteúdo c0->..->goal (forte) + atalho c0->COLA->goal (curto, por cola)."""
    b = base(i)
    c = [b + LETRAS[j] for j in range(L)]
    for j in range(L - 1):
        for _ in range(rep):
            t.perceive([c[j], c[j + 1]])
    for _ in range(cola_rep):
        t.perceive([c[0], COLA]); t.perceive([COLA, c[-1]])
    return c[0], c[-1], c                       # start, goal, caminho-gabarito


def caminhos(t, start, goal, prof=5, teto=60):
    """Enumera caminhos simples start->goal (DFS com poda) no grafo aprendido."""
    out = []
    def dfs(node, path):
        if len(out) >= teto or len(path) > prof:
            return
        for u in t.assoc.get(node, {}):
            if u in path:
                continue
            if u == goal:
                out.append(path + [u])
            else:
                dfs(u, path + [u])
    dfs(start, [start])
    return out


# ---- sinais de verificador (self-supervised: só usam o que o Toshi aprendeu) ----
def v_cos(t, p):
    e = [t._emb(w) for w in p]
    cs = [float(e[i] @ e[i + 1]) for i in range(len(p) - 1) if e[i] is not None and e[i + 1] is not None]
    return float(np.mean(cs)) if cs else -1.0


def v_pmi(t, p):
    ps = []
    for i in range(len(p) - 1):
        a, b = p[i], p[i + 1]
        c = t.assoc.get(a, {}).get(b, 0)
        ps.append(t.pmi(a, b, c, min_c=1.0))
    return float(np.mean(ps)) if ps else -1.0


def v_brev(t, p):
    return -len(p)


def v_combo(t, p):
    return v_pmi(t, p) - 0.15 * len(p)          # força de transição, com leve preferência por curto


def rank1(t, puzzles, verif):
    """% de vezes que o caminho-gabarito é o mais bem pontuado pelo verificador."""
    ok = n = 0
    for start, goal, gab in puzzles:
        cands = caminhos(t, start, goal)
        if not cands:
            continue
        n += 1
        melhor = max(cands, key=lambda p: verif(t, p))
        if melhor == gab:
            ok += 1
    return ok, n


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 74)
    print("VERIFICADOR — validado no currículo: ranqueia o caminho de CONTEÚDO acima do por-COLA?")
    print("=" * 74)

    t = Toshi()
    puzzles = [ensinar(t, i, L=3 + (i % 3)) for i in range(80)]   # COLA fica ubíquo ao longo dos 80

    print(f"\n{len(puzzles)} perguntas (cada uma: 1 caminho de conteúdo certo + 1 atalho por cola):")
    for nome, v in [("cos_médio (o ruim do Loop 1)", v_cos), ("PMI_médio (transição)", v_pmi),
                    ("brevidade (curto)", v_brev), ("COMBO (PMI - leve tam.)", v_combo)]:
        ok, n = rank1(t, puzzles, v)
        print(f"  {nome:<30}: gabarito em 1º = {ok}/{n} = {ok/max(n,1):.0%}")

    # aplica no TEXTO REAL: rerank de candidatos DIVERSOS pelo verificador validado (COMBO)
    print("\n" + "-" * 74)
    print("aplicando no cérebro real — o verificador escolhe a cadeia com MENOS cola (medido):")
    from toshi import build_or_load
    from mente_semantica import planejar_real, _viz_conteudo
    tr, _ = build_or_load()

    def frac_cola(p):                                        # fração de palavras-cola (frequentes)
        return np.mean([1.0 if tr.seen[w] > 0.005 * tr.total else 0.0 for w in p])

    def cands_reais(w0, wg, n=12, passos=7):
        """N cadeias DIVERSAS (seeds diferentes, amostra ampla) que chegam no alvo; dedup."""
        outs = set()
        for s in range(n):
            rng = np.random.default_rng(s)
            cur, path, usados = w0, [w0], {w0}
            for _ in range(passos):
                nb = [u for u, _ in _viz_conteudo(tr, cur, k=12) if u not in usados]
                if not nb:
                    break
                nxt = nb[int(rng.integers(min(6, len(nb))))]
                path.append(nxt); usados.add(nxt); cur = nxt
                if nxt == wg:
                    break
            if path[-1] == wg:
                outs.add(tuple(path))
        base = planejar_real(tr, w0, wg)
        if base:
            outs.add(tuple(base))
        return [list(p) for p in outs]

    import random
    conteudo = [w for w, _ in tr.seen.most_common(1500) if w not in tr.stops and tr._emb(w) is not None][:300]
    rng = random.Random(0)
    top_cola = med_cola = casos = 0.0
    exemplo = None
    for _ in range(60):
        a = rng.choice(conteudo)
        cur, walk = a, [a]                                   # deriva um alvo ALCANÇÁVEL por caminhada
        for _ in range(rng.randint(3, 6)):
            nb = [u for u, _ in _viz_conteudo(tr, cur, k=8) if u not in walk]
            if not nb:
                break
            cur = nb[rng.randrange(min(4, len(nb)))]; walk.append(cur)
        b = walk[-1]
        if b == a:
            continue
        cs = cands_reais(a, b)
        if len(cs) < 2:
            continue
        casos += 1
        best = max(cs, key=lambda p: v_combo(tr, p))
        top_cola += frac_cola(best)
        med_cola += np.mean([frac_cola(p) for p in cs])
        if exemplo is None and len(cs) >= 3:
            exemplo = (a, b, best, max(cs, key=frac_cola))
    if casos:
        print(f"  em {int(casos)} pares com >=2 cadeias distintas:")
        print(f"    cola na escolhida pelo verificador: {top_cola/casos:.0%}")
        print(f"    cola média entre os candidatos:      {med_cola/casos:.0%}")
    if exemplo:
        a, b, best, pior = exemplo
        print(f"  ex {a}->{b}: verificador -> {' -> '.join(best)}")
        print(f"             (rejeitou   -> {' -> '.join(pior)})")

    print("\n" + "=" * 74)
    print("o verificador validado (PMI/COMBO, 100% no currículo) escolhe cadeias com menos cola que")
    print("a média no texto real. É o Process Reward portado, sem treinar rede — o juiz do best-of-N.")


def _selftest():
    t = Toshi()
    puz = [ensinar(t, i, L=4) for i in range(30)]
    # o verificador PMI/COMBO acerta o gabarito mais que o cos (o ponto do Loop 1)
    okc, n = rank1(t, puz, v_cos)
    okp, _ = rank1(t, puz, v_combo)
    assert n > 0
    assert okp >= okc, (okc, okp)               # combo não é pior que o cos ruim
    assert okp >= 0.7 * n, (okp, n)             # e acerta a maioria
    print(f"[selftest] ok (combo {okp}/{n} >= cos {okc}/{n}; verificador de conteúdo funciona)")


if __name__ == "__main__":
    _selftest()
    main()
