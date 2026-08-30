"""
TOSHI — canal de conversa + TESTE de raciocínio (ele está pensando ou só associando?).

Alimenta Toshi com vários livros, depois EU (o script) converso e testo. O teste-chave é
INFERÊNCIA TRANSITIVA: ensino A->B e B->C, digo só A, e vejo se C ACENDE (via B). Se sim, ele
encadeou — está 'atrelando as coisas', não só associando o vizinho direto.

Roda: python conversa.py   (usa toshi.py; livros em dados/livros/)
"""
import os
import glob
from toshi import Toshi, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))


def alimentar(t):
    livros = glob.glob(os.path.join(HERE, "dados", "livros", "*.txt"))
    total = 0
    for path in livros:
        txt = open(path, encoding="utf-8", errors="replace").read()
        a = txt.find("***"); a = txt.find("***", a + 3) + 3 if a >= 0 else 0
        b = txt.rfind("*** END")
        total += t.eat(txt[a: b if b > 0 else len(txt)])
    return len(livros), total


def teach(t, frase, vezes=8):
    toks = tokenize(frase)
    for _ in range(vezes):
        t.perceive(toks)


def main():
    import sys
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass
    t = Toshi()
    print("alimentando Toshi com livros...", end=" ", flush=True)
    nl, nw = alimentar(t)
    print(f"{nl} livros, {nw} palavras, {len(t.seen)} distintas.\n")

    # ================= TESTE 1: inferência transitiva CONTROLADA (palavras inventadas) =======
    print("=" * 70)
    print("TESTE 1 — INFERÊNCIA TRANSITIVA (palavras inventadas, isola o mecanismo)")
    print("  ensino:  'glim puxa dorn'   e   'dorn puxa fesk'   (A->B, B->C)")
    print("  distrator: 'karn puxa lont' (não ligado a glim)")
    print("=" * 70)
    teach(t, "glim puxa dorn", 10)
    teach(t, "dorn puxa fesk", 10)
    teach(t, "karn puxa lont", 10)
    print("\n  digo só 'glim'. o que acende? (esperado: dorn=direto, fesk=transitivo via dorn)")
    sp = dict(t.spread(["glim"], depth=2))
    for w in ("dorn", "fesk", "lont", "karn"):
        print(f"     {w:<6} ativação={sp.get(w, 0.0):.3f}")
    ok_trans = sp.get("fesk", 0) > sp.get("lont", 0) and sp.get("fesk", 0) > 0
    print(f"\n  >>> fesk (transitivo) acende acima do distrator lont? {ok_trans}  "
          f"{'-> ENCADEOU (pensou)' if ok_trans else '-> não encadeou'}")

    # ================= TESTE 2: cadeia semântica em cima do que ele LEU =======
    print("\n" + "=" * 70)
    print("TESTE 2 — cadeia sobre conhecimento REAL (ensino ponte, testo o elo distante)")
    print("=" * 70)
    # ensino uma ponte nova entre dois conceitos que ele já conhece dos livros
    teach(t, "rei manda no reino. reino tem terras. terras dao ouro", 12)
    print("  ensinei: rei->reino->terras->ouro. digo 'rei', vejo se 'ouro' (3 elos) acende:")
    sp2 = dict(t.spread(["rei"], depth=3))
    for w in ("reino", "terras", "ouro"):
        print(f"     {w:<8} ativação={sp2.get(w, 0.0):.3f}")

    # ================= CONVERSA: eu falo, ele mostra o que pensa =======
    print("\n" + "=" * 70)
    print("CONVERSA — eu falo, ele mostra o que ACENDE e o que aquilo SIGNIFICA pra ele")
    print("=" * 70)
    for fala in ("o mar", "a mulher chorou", "o amor e o ciume", "a morte chegou"):
        toks = tokenize(fala)
        t.perceive(toks)
        assoc = t.associations(toks, 5)
        mean = t.meaning(toks[-1], 4)
        print(f"\n  eu> {fala}")
        if assoc:
            print("  toshi acende> " + "  ".join(f"{w}·{s:.1f}" for w, s in assoc))
        if mean:
            print(f"  toshi entende '{toks[-1]}' como> " + "  ".join(f"{w}·{s:.2f}" for w, s in mean))

    print("\n" + "=" * 70)
    print("VEREDITO: se no TESTE 1 'fesk' acendeu (transitivo) > distrator, ele ENCADEIA —")
    print("está atrelando A->B->C, não só o vizinho direto. É o começo de 'pensar' em cadeia.")
    print("Honesto: é encadeamento ASSOCIATIVO (spreading), não lógica dedutiva formal.")


if __name__ == "__main__":
    main()
