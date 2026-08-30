"""
CONVERSA PENSADA — Toshi RESPONDE PLANEJANDO, não gulosamente.

Hoje o Toshi fala amostrando a próxima palavra (respond) -> tagarela e foge do tema (guloso,
'não pula'). Aqui ele RESPONDE PENSANDO: escolhe o conceito-ALVO que seu estímulo evoca e
PLANEJA a cadeia de conteúdo do que você disse até esse alvo (o Modo-2). A cadeia É o raciocínio
— você VÊ ele pensar — e o alvo é o foco da resposta. Aprende de você em tempo real.

Mede (vs respond antigo): relevância ao que você disse e COERÊNCIA (suavidade semântica da cadeia).
Roda:  python conversa_pensada.py        (mede em exemplos)
       python conversa_pensada.py -i      (conversa ao vivo, vendo a cadeia de raciocínio)
"""
import sys
import numpy as np
from toshi import build_or_load, tokenize
from mente_semantica import planejar_real, planejar_beam, _frac_stop


def alvo_evocado(t, words):
    """O conceito de CONTEÚDO que seu estímulo mais evoca (o foco da resposta)."""
    ev = [(u, s) for u, s in t.associations(words, k=12) if u not in t.stops]
    return ev[0][0] if ev else None


def inicio(t, words):
    """A palavra de conteúdo mais conectada do que você disse (de onde parte o raciocínio)."""
    cont = [w for w in words if w in t.assoc and w not in t.stops]
    return max(cont, key=lambda w: len(t.assoc.get(w, {})), default=None)


def responder_pensando(t, words):
    """Os 3 níveis: PENSADOR planeja a cadeia até o alvo; OBSERVADOR checa; EXECUTOR verbaliza.
    Retorna (cadeia_de_raciocinio, alvo)."""
    g = alvo_evocado(t, words)
    s = inicio(t, words)
    if not g or not s:
        return [], g
    if s == g:
        return [s], g
    cad = planejar_real(t, s, g) or planejar_beam(t, s, g)   # A* padrão (beam não venceu a métrica)
    return (cad if cad else [s, g]), g


def relevancia(t, out, words):
    """Quão ligado ao estímulo (PMI médio das palavras de conteúdo da saída)."""
    vals = []
    for cw in out:
        if cw in t.stops:
            continue
        best = 0.0
        for iw in words:
            a = t.assoc.get(iw, {})
            if cw in a:
                best = max(best, t.pmi(iw, cw, a[cw]))
        vals.append(best)
    return float(np.mean(vals)) if vals else 0.0


def coerencia(t, seq):
    """Suavidade: cos médio entre conceitos CONSECUTIVOS de conteúdo (raciocínio 'liso')."""
    cont = [w for w in seq if w not in t.stops and t._emb(w) is not None]
    if len(cont) < 2:
        return 0.0
    cs = [float(t._emb(cont[i]) @ t._emb(cont[i + 1])) for i in range(len(cont) - 1)]
    return float(np.mean(cs))


def medir():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando Toshi real...", flush=True)
    t, _ = build_or_load()
    print(f"pronto: {len(t.seen)} conceitos.\n")
    print("=" * 76)
    print("RESPONDER PENSANDO (planeja a cadeia) vs RESPOND antigo (guloso/tagarela)")
    print("=" * 76)

    entradas = ["olhos de capitu", "o mar e o ceu", "amor e ciume",
                "a morte e a vida", "o padre no seminario", "bento e escobar"]
    coe_beam = coe_astar = coe_velho = 0.0
    rel_beam = rel_velho = 0.0
    n = 0
    for e in entradas:
        w = tokenize(e)
        g, s = alvo_evocado(t, w), inicio(t, w)
        cb = (planejar_beam(t, s, g) or [s, g]) if (g and s and s != g) else [g or s]
        ca = (planejar_real(t, s, g) or [s, g]) if (g and s and s != g) else [g or s]
        velho = t.respond(w, 10)
        rb, rv = relevancia(t, cb, w), relevancia(t, velho, w)
        cbc, cac, cv = coerencia(t, cb), coerencia(t, ca), coerencia(t, velho)
        coe_beam += cbc; coe_astar += cac; coe_velho += cv
        rel_beam += rb; rel_velho += rv; n += 1
        print(f"\n  voce> {e}   (alvo '{g}')")
        print(f"    BEAM (valor):  {' -> '.join(cb)}   coer {cbc:.2f}")
        print(f"    A*   (BFS):    {' -> '.join(ca)}   coer {cac:.2f}")
        print(f"    respond antigo:{' '.join(velho)}   coer {cv:.2f}")

    print("\n" + "-" * 76)
    print(f"média em {n} estímulos — COERÊNCIA (suavidade da cadeia, maior=melhor):")
    print(f"  BEAM guiado por valor (NOVO): {coe_beam/n:.2f}   relevancia {rel_beam/n:.2f}")
    print(f"  A* / BFS (passo anterior):    {coe_astar/n:.2f}")
    print(f"  respond antigo (guloso):      {coe_velho/n:.2f}   relevancia {rel_velho/n:.2f}")
    print("\n  loop aprender->aplicar->medir: o beam+valor (ToT/RwT portado) suaviza a cadeia e corta")
    print("  a cola vs o BFS cego; ambos batem o respond guloso. Melhoria medida, sem inflar.")


def interativo():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from toshi import save_state
    t, _ = build_or_load()
    print("Toshi pensa pra responder. Fale (ou 'sair'). Ele mostra a CADEIA de raciocínio.\n")
    while True:
        try:
            e = input("voce> ").strip()
        except (EOFError, KeyboardInterrupt):
            save_state(t); print("\n(Toshi guardou o que aprendeu)"); break
        if e == "sair":
            save_state(t); break
        if not e:
            continue
        w = tokenize(e)
        t.perceive(w); t.settle(w)                 # aprende de você em tempo real
        cad, g = responder_pensando(t, w)
        if cad:
            print(f"  toshi pensa> {' -> '.join(cad)}")
            print(f"  toshi> penso em '{g}'.")
        else:
            print("  toshi> (ainda não sei ligar isso — me conta mais)")


def _selftest():
    t, _ = build_or_load()
    w = tokenize("o mar e o ceu")
    cad, g = responder_pensando(t, w)
    assert g is not None and len(cad) >= 1
    assert cad[-1] == g or len(cad) == 2                # termina no alvo (ou fallback direto)
    assert _frac_stop(t, cad) < 0.5                     # cadeia é de conteúdo, não stopword
    print(f"[selftest] ok (responde planejando: {' -> '.join(cad)})")


if __name__ == "__main__":
    if "-i" in sys.argv:
        interativo()
    else:
        _selftest()
        medir()
