"""
RESPONDER WIKI — o Toshi RESPONDE com a língua dele, não copia a Wikipédia.

Depois que o enxame comeu artigos, os shards guardam o que cada sub-Toshi aprendeu
(assoc/after/seen/embed). Aqui:
  1. a pergunta acha os artigos mais relevantes no índice;
  2. os shards são fundidos num Toshi só (memória wiki);
  3. ele GERA a resposta com think_and_say (pensador gera candidatos, observador
     escolhe o mais relevante) — saída própria, não um trecho colado.

USO:
  python responder_wiki.py --pergunta "o que é um asteroide?"
  python responder_wiki.py --pergunta "quem foi Machado de Assis?" --mostrar-fonte
  python responder_wiki.py --selftest
"""
import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, tokenize
from consultar_wiki import carregar_indice, buscar, _STOPS

WIKI_DIR = os.path.join(HERE, "dados", "wiki")
CACHE_TOSHI = os.path.join(WIKI_DIR, "wiki_toshi.pkl")


# ============================================================ FUNDIR SUB-TOSHI
def _novo_ou_carregado():
    shards = sorted(os.path.join(WIKI_DIR, p) for p in os.listdir(WIKI_DIR)
                    if p.startswith("shard_") and p.endswith(".pkl"))
    if not shards:
        return Toshi(), 0
    # cache inválido se algum shard mudou
    if os.path.exists(CACHE_TOSHI):
        cache_mtime = os.path.getmtime(CACHE_TOSHI)
        if all(os.path.getmtime(s) <= cache_mtime for s in shards):
            try:
                with open(CACHE_TOSHI, "rb") as f:
                    d = pickle.load(f)
                if isinstance(d, dict) and "toshi" in d:
                    return d["toshi"], d.get("paginas", 0)
                return d, len(shards)  # formato antigo
            except Exception:
                pass

    t = Toshi()
    paginas = 0
    for s in shards:
        with open(s, "rb") as f:
            d = pickle.load(f)
        paginas += d.get("paginas", 0)
        for w, c in d.get("assoc", {}).items():
            t.assoc.setdefault(w, Counter()).update(c)
        for w, c in d.get("after", {}).items():
            t.after.setdefault(w, Counter()).update(c)
        t.seen.update(d.get("seen", {}))
        for w, dims_sig in d.get("index", {}).items():
            t.index.setdefault(w, dims_sig)
        for w, v in d.get("embed", {}).items():
            t.embed[w] = t.embed.get(w, np.zeros_like(v)) + v

    with open(CACHE_TOSHI, "wb") as f:
        pickle.dump({"toshi": t, "paginas": paginas}, f)
    return t, paginas


# ============================================================ RESPOSTA PRÓPRIA
def palavras_da_entrada(entrada, k=8):
    toks = tokenize(entrada.get("titulo", ""))
    toks += [w for w in entrada.get("palavras", [])]
    vistos, saida = set(), []
    for w in toks:
        if len(w) > 2 and w not in _STOPS and w not in vistos:
            vistos.add(w)
            saida.append(w)
        if len(saida) >= k:
            break
    return saida


def responder(t, pergunta, entradas, n_candidatos=8):
    """Gera com a língua do Toshi a partir do que ele comeu. NÃO copia o artigo."""
    if not entradas:
        return None, []
    palavras = []
    for e in entradas[:3]:
        palavras += palavras_da_entrada(e, k=6)
    # sem palavras úteis no índice, usa o texto do título da pergunta
    if not palavras:
        palavras = [w for w in tokenize(pergunta) if len(w) > 2 and w not in _STOPS][:6]
    if not palavras:
        return None, []

    fala, _ = t.think_and_say(palavras, n_candidates=n_candidatos)
    if not fala or len(fala) < 3:
        fala = t.respond(palavras, n=14)
    if fala:
        return " ".join(fala), fala
    return None, palavras


def _selftest():
    print("SELFTEST — o Toshi responde com a língua dele\n")
    # um sub-Toshi sintético que comeu uma mini-Wikipédia
    t = Toshi()
    for _ in range(3):
        t.perceive(["asteroide", "e", "rocha", "no", "espaco"])
        t.perceive(["asteroide", "orbita", "o", "sol"])
        t.perceive(["asteroide", "e", "menor", "que", "planeta"])
    entrada = {"titulo": "Asteroide",
               "palavras": ["asteroide", "rocha", "espaco", "sol", "planeta"],
               "resumo": "Asteroide é um corpo rochoso que orbita o Sol.",
               "links": [], "imagens": []}
    resp, _ = responder(t, "o que é um asteroide?", [entrada])
    print(f"  toshi> {resp}")
    assert resp and "Asteroide é um corpo rochoso" not in resp  # não é cópia
    assert any(w in resp for w in ("asteroide", "rocha", "espaco", "sol", "planeta"))
    print("\n[selftest] ok — ele gerou a própria resposta, sem colar o texto")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Toshi responde usando a Wikipédia que comeu.")
    ap.add_argument("--pergunta", default="")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--mostrar-fonte", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    entradas = carregar_indice()
    if not entradas:
        print("o enxame ainda não comeu nada. Rode comer_wikipedia.py primeiro.")
        return

    t, paginas = _novo_ou_carregado()
    print(f"memória wiki: {len(entradas)} artigos no índice, "
          f"{paginas} páginas nos shards, {len(t.seen)} conceitos fundidos.\n")

    if not args.pergunta:
        print("use --pergunta \"...\"")
        return

    top = buscar(entradas, args.pergunta, args.top)
    if not top:
        print("toshi> (ainda não comi isso — continue a varredura)")
        return

    resp, _ = responder(t, args.pergunta, top)
    print(f"toshi> {resp}" if resp else "toshi> (ainda estou mastigando isso)")
    if args.mostrar_fonte:
        print("\nfontes que ele usou (sem colar):")
        for e in top:
            print(f"  • {e['titulo']}")


if __name__ == "__main__":
    main()
