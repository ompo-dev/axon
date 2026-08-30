"""
CONSULTAR WIKI — pergunta ao Toshi o que o enxame comeu da Wikipédia.

Depois de comer_wikipedia.py, o índice (indice.jsonl) guarda cada artigo que os
sub-Toshi comeram: título, resumo, palavras, links e imagens. Aqui a pergunta
vira busca na MEMÓRIA do enxame (sem internet, sem Qwen).

USO:
  python consultar_wiki.py --pergunta "o que é um buraco negro?"
  python consultar_wiki.py --pergunta "quem foi Machado de Assis?" --top 5
  python consultar_wiki.py --selftest
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import tokenize

INDICE = os.path.join(HERE, "dados", "wiki", "indice.jsonl")


def carregar_indice(caminho=INDICE):
    entradas = []
    if not os.path.isfile(caminho):
        return entradas
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                try:
                    entradas.append(json.loads(linha))
                except Exception:
                    pass
    return entradas


_STOPS = {
    "que", "quem", "como", "onde", "quando", "qual", "quais", "uma", "umas",
    "uns", "para", "com", "dos", "das", "foi", "era", "sao", "ser", "mais",
    "muito", "pouco", "por", "sobre", "entre", "ele", "ela", "eles", "elas",
    "voce", "isso", "isto", "aquilo", "seu", "sua", "seus", "suas", "the",
}


def buscar(entradas, consulta, top=5):
    qtoks = {w for w in tokenize(consulta) if len(w) > 2 and w not in _STOPS}
    if not qtoks:
        return []
    pontuadas = []
    for e in entradas:
        titulo_toks = set(tokenize(e.get("titulo", "")))
        palavra_toks = set(e.get("palavras", []))
        s_titulo = len(qtoks & titulo_toks) / max(1, len(qtoks))
        s_palavras = len(qtoks & palavra_toks) / max(1, len(qtoks))
        score = 0.7 * s_titulo + 0.3 * s_palavras
        if score > 0:
            pontuadas.append((score, e))
    pontuadas.sort(key=lambda x: -x[0])
    return [e for _, e in pontuadas[:top]]


def _selftest():
    import tempfile
    print("SELFTEST — memória da Wikipédia\n")
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"titulo": "Buraco negro",
                            "resumo": "buraco negro é uma região do espaço com gravidade intensa.",
                            "palavras": ["buraco", "negro", "espaco", "gravidade"],
                            "links": [], "imagens": []}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"titulo": "Machado de Assis",
                            "resumo": "machado de assis foi um escritor brasileiro.",
                            "palavras": ["machado", "assis", "escritor", "brasileiro"],
                            "links": [], "imagens": []}, ensure_ascii=False) + "\n")
    entradas = carregar_indice(path)
    r1 = buscar(entradas, "o que é um buraco negro?")
    r2 = buscar(entradas, "quem foi machado de assis?")
    assert r1 and "Buraco negro" in r1[0]["titulo"]
    assert r2 and "Machado de Assis" in r2[0]["titulo"]
    print(f"  1) {r1[0]['titulo']}: {r1[0]['resumo']}")
    print(f"  2) {r2[0]['titulo']}: {r2[0]['resumo']}")
    print("\n[selftest] ok — a memória do enxame responde")
    os.remove(path)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Consulta a memória da Wikipédia comida pelo Toshi.")
    ap.add_argument("--pergunta", default="")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--lista", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    entradas = carregar_indice()
    print(f"memória da Wikipédia: {len(entradas)} artigos comidos.\n")

    if args.lista:
        for e in entradas[:50]:
            print(f"  • {e['titulo']}")
        return

    if not args.pergunta:
        print("use --pergunta \"...\"")
        return

    top = buscar(entradas, args.pergunta, args.top)
    if not top:
        print("(não achei isso na memória do enxame ainda — coma mais páginas)")
        return
    for i, e in enumerate(top, 1):
        print(f"{i}. {e['titulo']}")
        print(f"   {e['resumo'][:500]}")
        if e.get("imagens"):
            print(f"   imagens: {', '.join(e['imagens'][:5])}")
        print()


if __name__ == "__main__":
    main()
