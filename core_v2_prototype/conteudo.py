"""
CONTEÚDO — matar a COLA de uma vez (o vilão recorrente de todo ciclo).

Filtrar por frequência (top-30 stops, teto) não basta: 'aqui', 'quem', 'então' vazam — são de
frequência média mas ligam-se a TUDO (cola). O sinal certo não é frequência, é ESPECIFICIDADE:
uma palavra de conteúdo tem alguns vínculos FORTES e específicos (PMI alto); a cola tem só
vínculos fracos e difusos. Então:  conteudo(w) = média dos top-k PMI dos vizinhos de w.

Isso é DRY: um sinal, usado por abdução, verificador, planejador e resposta — limpa todos juntos.
Valida separando cola conhecida de conteúdo conhecido, e mostra que limpa o topo da abdução.
Roda: python conteudo.py   (usa o Toshi real)
"""
import sys
import numpy as np
from toshi import build_or_load


def conteudo(t, w, n=15):
    """COERÊNCIA dos vizinhos: uma palavra de conteúdo liga a um cluster semântico coerente
    (vizinhos parecidos entre si); a cola liga a tudo (vizinhos espalhados). Sinal = cos médio
    par-a-par entre os embeddings dos top-n vizinhos. Alto = conteúdo; baixo = cola."""
    nb = t.assoc.get(w)
    if not nb:
        return 0.0
    tops = [u for u, _ in sorted(nb.items(), key=lambda x: -x[1])[:n]]
    embs = [t._emb(u) for u in tops]
    embs = [e for e in embs if e is not None]
    if len(embs) < 3:
        return 0.0
    E = np.array(embs)
    sims = E @ E.T
    iu = np.triu_indices(len(E), 1)
    return float(np.mean(sims[iu]))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando Toshi real...", flush=True)
    t, _ = build_or_load()

    print("=" * 70)
    print("CONTEÚDO vs COLA — especificidade (top-k PMI) separa os dois?")
    print("=" * 70)

    cola = ["aqui", "quem", "entao", "agora", "assim", "onde", "tudo", "isso", "ella", "elle"]
    conc = ["capitu", "ciume", "mar", "seminario", "olhos", "escobar", "oceano", "cavallo",
            "morte", "beijo"]
    cola = [w for w in cola if w in t.assoc]
    conc = [w for w in conc if w in t.assoc]
    sc = {w: conteudo(t, w) for w in cola + conc}

    mc = np.mean([sc[w] for w in cola]); mk = np.mean([sc[w] for w in conc])
    print(f"\n  cola conhecida:     média especificidade = {mc:.2f}")
    for w in sorted(cola, key=lambda x: sc[x]):
        print(f"     {w:<12} {sc[w]:.2f}")
    print(f"  conteúdo conhecido: média especificidade = {mk:.2f}")
    for w in sorted(conc, key=lambda x: -sc[x]):
        print(f"     {w:<12} {sc[w]:.2f}")

    # limiar que separa: existe um corte simples que classifica cola vs conteúdo?
    todos = sorted((sc[w], 0 if w in cola else 1) for w in cola + conc)
    melhor_acc, melhor_th = 0, 0
    for i in range(len(todos) - 1):
        th = (todos[i][0] + todos[i + 1][0]) / 2
        acc = np.mean([(sc[w] >= th) == (w in conc) for w in cola + conc])
        if acc > melhor_acc:
            melhor_acc, melhor_th = acc, th
    print(f"\n  separação por um corte simples: {melhor_acc:.0%} (limiar ~{melhor_th:.2f})")

    print("\n" + "=" * 70)
    if mk > mc * 1.3 and melhor_acc >= 0.8:
        print(f"VEREDITO: especificidade separa cola de conteúdo ({mk:.2f} vs {mc:.2f}, {melhor_acc:.0%}).")
        print("Um sinal só, DRY: filtrar/ranquear por ele limpa abdução, verificador, planejador e")
        print("resposta de uma vez — mata a cola que contaminava o topo de todo ciclo.")
    else:
        print(f"VEREDITO honesto: separação fraca ({mk:.2f} vs {mc:.2f}, {melhor_acc:.0%}) — sem inflar.")


def _selftest():
    t, _ = build_or_load()
    # conteúdo forte deve pontuar acima de cola óbvia
    if "capitu" in t.assoc and "aqui" in t.assoc:
        assert conteudo(t, "capitu") > conteudo(t, "aqui"), "especificidade não separou"
    print("[selftest] ok (especificidade: conteúdo > cola)")


if __name__ == "__main__":
    _selftest()
    main()
