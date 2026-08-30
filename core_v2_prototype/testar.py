"""
TESTAR — o elo perdido: a mente TESTA a hipótese abduzida contra evidência NOVA e independente.

Lição do crescer.py: abdução gera hipótese, não fato; integrar cru = ruído (precisão 4%). O teste
de verdade precisa de evidência que a mente AINDA NÃO usou. Aqui: aprendo o grafo de METADE dos
livros (A), abduzo links, e TESTO na OUTRA metade (B) — texto independente. Uma hipótese "a~b"
abduzida de A é CONFIRMADA se a e b realmente se relacionam em B (corroboração externa).

Mede: precisão das hipóteses corroboradas por B vs a taxa-base. Se >> base, o teste externo filtra
o que presta -> a mente pode crescer com segurança (hipótese->TESTE->memória). É active inference
em texto: prever e checar contra o mundo (aqui, texto novo). Roda: python testar.py
"""
import sys
import numpy as np
from collections import Counter
from toshi import Toshi, load_all_books


def constroi(textos):
    t = Toshi()
    for x in textos:
        t.eat(x)
    return t


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando livros e partindo em dois corpora independentes...", flush=True)
    textos, nl = load_all_books()
    if len(textos) < 4:
        print(f"preciso de >=4 livros (tenho {len(textos)}); abortando honesto."); return
    A = textos[0::2]                                          # metade A (ímpares)
    B = textos[1::2]                                          # metade B (independente)
    ta, tb = constroi(A), constroi(B)
    print(f"  corpus A: {len(A)} livros, {len(ta.seen)} conceitos | corpus B: {len(B)} livros, {len(tb.seen)}")

    stops = ta.stops
    teto = 0.0015 * ta.total                                  # teto de frequência: corta a COLA (ubíquos)
    conteudo = [w for w in ta.assoc
                if 6 <= ta.seen[w] < teto and w not in stops and len(ta.assoc[w]) >= 4
                and w in tb.assoc]                            # conceito de CONTEÚDO presente nos DOIS
    cset = set(conteudo)
    vizA = {w: (set(ta.assoc[w]) & cset) for w in conteudo}   # vizinhança só de conteúdo (sem cola)
    cooc_B = {w: set(tb.assoc.get(w, {})) for w in conteudo}  # com quem w co-ocorre em B (evidência nova)

    rng = np.random.default_rng(0)

    def score(a, b):                                          # abdução: vizinhos comuns em A (Adamic-Adar)
        comuns = (vizA[a] - {b}) & (vizA[b] - {a})
        return sum(1.0 / np.log(2 + ta.seen[x]) for x in comuns)

    # candidatos: pares a 2 hops em A, SEM elo direto em A (a mente nunca os viu juntos em A)
    cand = {}
    fontes = [conteudo[i] for i in rng.choice(len(conteudo), min(400, len(conteudo)), replace=False)]
    for a in fontes:
        viz2 = set()
        for x in vizA[a]:
            viz2 |= vizA.get(x, set())
        for b in viz2:
            if b != a and b not in vizA[a]:
                key = (a, b) if a < b else (b, a)
                cand[key] = score(*key)
    ranked = sorted(cand, key=lambda k: -cand[k])
    if not ranked:
        print("sem candidatos; abortando honesto."); return

    # TESTE: a hipótese (a,b) se confirma se a e b co-ocorrem em B (evidência independente)
    def corrobora(ab):
        a, b = ab
        return b in cooc_B.get(a, set()) or a in cooc_B.get(b, set())

    base = np.mean([corrobora(k) for k in ranked])
    print("=" * 72)
    print("TESTAR — hipóteses abduzidas de A, confirmadas por B (independente)? precisão vs base")
    print("=" * 72)
    print(f"\n{len(ranked)} hipóteses (pares sem elo em A). taxa-base de confirmação em B = {base:.0%}")
    for k in (20, 50, 100, 200):
        top = ranked[:k]
        prec = np.mean([corrobora(x) for x in top])
        print(f"  confirmadas por B @{k:<4}: {prec:.0%}   (lift {prec/max(base,1e-9):.1f}x)")

    # as CONFIRMADAS viram memória (hipótese->teste->memória); mostra exemplos
    confirmadas = [k for k in ranked[:100] if corrobora(k)]
    print(f"\n  {len(confirmadas)}/100 hipóteses do topo passaram no TESTE -> viram conhecimento seguro.")
    print("  exemplos (abduzido de A, confirmado em B):")
    for a, b in confirmadas[:6]:
        comuns = list((vizA[a] - {b}) & (vizA[b] - {a}))[:3]
        print(f"    {a} ~ {b}   (via {', '.join(comuns)})")

    print("\n" + "=" * 72)
    prec50 = np.mean([corrobora(x) for x in ranked[:50]])
    if prec50 > 2 * base and base > 0:
        print(f"VEREDITO: o TESTE externo FUNCIONA — hipóteses abduzidas de A confirmam em B a {prec50:.0%}")
        print(f"(vs base {base:.0%}, {prec50/base:.1f}x). A mente prevê relações que aparecem em texto que")
        print("NÃO usou. Isso fecha hipótese->TESTE->memória: pode crescer só integrando o que passa no")
        print("teste. É o elo que faltava entre imaginar e saber — active inference em linguagem.")
    else:
        print(f"VEREDITO honesto: confirmação@50 {prec50:.0%} vs base {base:.0%} — fraco, sem inflar.")


if __name__ == "__main__":
    main()
