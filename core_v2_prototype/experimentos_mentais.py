"""
EXPERIMENTOS MENTAIS — o Toshi gera CONEXÕES NOVAS do que acumulou.

Decorar não é inteligência. Inteligência é criar a ligação que ninguém tinha feito.
Einstein não tinha referência para a Relatividade Geral: ele tinha ACÚMULO de
experiências e fez o salto.

Aqui o Toshi faz o salto dele:
  1. varre os conceitos que comeu (Wikipédia + livros + tudo);
  2. acha pares que NUNCA foram ligados diretamente, mas que compartilham
     vizinhos estruturais (2 hops) e têm significado próximo;
  3. gera HIPÓTESES "A se liga a B";
  4. testa internamente (força estrutural + semelhança de significado);
  5. as melhores são INTEGRADAS: viram associações novas + fatos + arquivo de
     descobertas. É o "pulo" que LLM não dá.

USO:
  python experimentos_mentais.py --varrer --top 20
  python experimentos_mentais.py --criar 5
  python experimentos_mentais.py --selftest
"""
import argparse
import heapq
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, tokenize
from fatos import Fatos

HIP_FILE = os.path.join(HERE, "dados", "wiki", "hipoteses.jsonl")

# cola que NÃO gera inteligência: função + chavões da Wikipédia
_STOPS_GERAIS = {
    "para", "com", "dos", "das", "uma", "uns", "umas", "por", "sobre", "entre",
    "mais", "muito", "pouco", "como", "quando", "onde", "qual", "quais",
    "janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho",
    "agosto", "setembro", "outubro", "novembro", "dezembro",
    "segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo",
    "ficheiro", "arquivo", "imagem", "svg", "jpg", "png", "wikidata",
    "wikipedia", "commons", "predefinicao", "categoria", "referencias",
    "ligacoes", "externas", "liga", "https", "www", "pagina", "paginas",
    "artigo", "artigos", "notas", "ver", "tambem", "the", "and", "for",
    "was", "from", "with", "that", "this", "into", "were",
}


def _eh_stop(t, w):
    return w in _STOPS_GERAIS or (len(t.seen) > 500 and w in t.stops)


def conceitos(t, min_vez=3, max_n=4000):
    """Conceitos de CONTEÚDO que ele realmente viveu (sem cola)."""
    out = [w for w in t.assoc
           if len(w) > 3 and t.seen[w] >= min_vez and not _eh_stop(t, w)]
    out.sort(key=lambda w: -len(t.assoc.get(w, {})))
    return out[:max_n]


def gerar_hipoteses(t, sementes_max=350, top=20):
    """Gera as melhores ligações novas (A->B sem aresta direta)."""
    t0 = time.time()
    seeds = conceitos(t)
    if len(seeds) > sementes_max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(seeds), sementes_max, replace=False)
        seeds = [seeds[i] for i in idx]
    else:
        seeds = seeds[:sementes_max]

    heap = []
    vistos = set()
    for a in seeds:
        # vizinhos de CONTEÚDO (cola não conta como ponte)
        viz_a = {w for w in t.assoc.get(a, {})
                 if len(w) > 2 and not _eh_stop(t, w)}
        if not viz_a:
            continue
        # 2 hops: amigos dos amigos (só os mais fortes, para caber no cérebro)
        cands = set()
        for x in list(viz_a)[:15]:
            for w in list(t.assoc.get(x, {}))[:15]:
                if len(w) > 2 and not _eh_stop(t, w):
                    cands.add(w)
        for b in cands:
            if b == a or b in viz_a or _eh_stop(t, b):
                continue
            chave = (a, b) if a < b else (b, a)
            if chave in vistos:
                continue
            vistos.add(chave)
            viz_b = {w for w in t.assoc.get(b, {}) if not _eh_stop(t, w)}
            comuns = viz_a & viz_b
            if not comuns:
                continue
            aa = sum(1.0 / np.log(2 + t.seen[x]) for x in comuns)
            ea, eb = t._emb(a), t._emb(b)
            emb = float(ea @ eb) if (ea is not None and eb is not None) else 0.0
            score = 0.5 * min(aa / 3.0, 1.0) + 0.5 * max(emb, 0.0)
            item = (score, a, b, sorted(comuns, key=lambda x: -t.seen[x])[:5])
            if len(heap) < top:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    hip = sorted(heap, reverse=True)
    print(f"varredura: {len(seeds)} sementes, {len(vistos)} pares testados, "
          f"{time.time()-t0:.1f}s")
    return [{"a": a, "b": b, "score": round(s, 4), "comuns": c}
            for s, a, b, c in hip]


def artigos_juntos(entradas, a, b):
    """Em quantos artigos os dois conceitos JÁ aparecem juntos (referência direta)."""
    n = 0
    for e in entradas:
        palavras = set(e.get("palavras", [])) | set(tokenize(e.get("titulo", "")))
        if a in palavras and b in palavras:
            n += 1
    return n


def integrar(t, fatos, hipoteses):
    """Cria a ligação nova no cérebro (associação + fato) e registra a descoberta."""
    criadas = []
    for h in hipoteses:
        a, b = h["a"], h["b"]
        t.assoc.setdefault(a, {})[b] = t.assoc.get(a, {}).get(b, 0) + 1.0
        t.assoc.setdefault(b, {})[a] = t.assoc.get(b, {}).get(a, 0) + 1.0
        t.perceive([a, b])
        fatos.aprender(a, "hipotese_liga", b)
        h["quando"] = time.time()
        criadas.append(h)
    os.makedirs(os.path.dirname(HIP_FILE), exist_ok=True)
    with open(HIP_FILE, "a", encoding="utf-8") as f:
        for h in criadas:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    fatos.save()
    return criadas


def _selftest():
    print("SELFTEST — o salto (zero referências)\n")
    t = Toshi()
    f = Fatos()
    f.g = {}
    # ele viveu A e D em contextos separados, unidos só por B
    for _ in range(4):
        t.perceive(["aaaa", "bbbb", "cccc"])
        t.perceive(["dddd", "bbbb", "eeee"])
    assert "dddd" not in t.assoc.get("aaaa", {}), "aaaa e dddd já estavam ligados"
    hip = gerar_hipoteses(t, sementes_max=20, top=10)
    print("  hipóteses:", hip)
    alvo = [h for h in hip if {h["a"], h["b"]} == {"aaaa", "dddd"}]
    assert alvo, "não gerou a hipótese nova aaaa<->dddd"
    integrar(t, f, alvo[:1])
    assert "dddd" in t.assoc.get("aaaa", {}), "não integrou a ligação"
    assert ["hipotese_liga", "dddd"] in f.g.get("aaaa", [])
    print("\n[selftest] ok — criou a linha que não existia e guardou")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Experimentos mentais: o Toshi cria conexões novas.")
    ap.add_argument("--varrer", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--criar", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    # carrega o Toshi fundido (wiki + shards) e os artigos do índice
    try:
        from responder_wiki import _novo_ou_carregado, carregar_indice, CACHE_TOSHI
    except Exception as e:
        print(f"erro ao importar a memória wiki: {e}")
        return
    t, paginas = _novo_ou_carregado()
    entradas = carregar_indice()
    fatos = Fatos()
    print(f"cérebro: {len(t.seen)} conceitos, {paginas} páginas wiki, "
          f"{len(entradas)} artigos no índice")

    hip = gerar_hipoteses(t, top=args.top)
    print("\nhipóteses novas (sem ligação direta):")
    for i, h in enumerate(hip, 1):
        juntas = artigos_juntos(entradas, h["a"], h["b"])
        print(f"  {i:2d}. {h['a']} <-> {h['b']}  score={h['score']:.3f} "
              f"via={h['comuns']} artigos_juntos={juntas}")

    if args.criar > 0:
        escolhidas = [h for h in hip if artigos_juntos(entradas, h["a"], h["b"]) == 0][:args.criar]
        if not escolhidas:
            print("\n(nenhuma hipótese com zero referência direta; criando as top mesmo)")
            escolhidas = hip[:args.criar]
        criadas = integrar(t, fatos, escolhidas)
        print(f"\ncriadas {len(criadas)} conexões novas e salvas em {HIP_FILE}")
        # atualiza o cache do Toshi fundido para não perder as ligações
        try:
            with open(CACHE_TOSHI, "wb") as f:
                pickle.dump(t, f)
        except Exception as e:
            print(f"  [!] não consegui atualizar o cache: {e}")


if __name__ == "__main__":
    main()
