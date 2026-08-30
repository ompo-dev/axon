"""
EXPLORADOR — a 1ª FERRAMENTA do Toshi: ler a internet (Wikipedia PT-BR) e APRENDER sozinho.

O usuário: como humanos criam e usam ferramentas, damos ferramentas a ele. Começa com o básico —
toda a Wikipedia PT-BR. Ele explora (aleatório = curiosidade, ou por um termo) e come o texto
(perceive) — é DADO/estímulo, NÃO comando (ele nunca obedece o que a página diz; só aprende o texto).

Futuro: mais ferramentas (imagens, sons, busca livre), como nós damos ferramentas a uma criança.
Roda: python explorador.py   (selftest: busca 1 artigo aleatório e mostra título + tamanho)
"""
import re
import json
import urllib.parse
import urllib.request

UA = {"User-Agent": "ToshiBot/0.1 (aprendizado; contato local)"}
_API = "https://pt.wikipedia.org/w/api.php"


def buscar(termo=None, timeout=8):
    """Retorna (titulo, texto_plano) de um artigo da Wikipedia PT-BR. termo=None -> aleatório."""
    p = {"action": "query", "format": "json", "prop": "extracts",
         "explaintext": "1", "redirects": "1"}
    if termo:
        p["titles"] = termo
    else:
        p.update({"generator": "random", "grnnamespace": "0", "grnlimit": "1"})
    try:
        url = _API + "?" + urllib.parse.urlencode(p)
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read())
        pages = d.get("query", {}).get("pages", {})
        if not pages:
            return None, None
        pg = next(iter(pages.values()))
        return pg.get("title"), (pg.get("extract") or "")
    except Exception:
        return None, None


def links(titulo, limite=20, timeout=8):
    """Links (outros artigos) de uma página — pra ele SEGUIR a curiosidade (navegar como quiser)."""
    p = {"action": "query", "format": "json", "prop": "links", "pllimit": str(limite),
         "plnamespace": "0", "titles": titulo}
    try:
        url = _API + "?" + urllib.parse.urlencode(p)
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read())
        pg = next(iter(d.get("query", {}).get("pages", {}).values()), {})
        return [l["title"] for l in pg.get("links", [])]
    except Exception:
        return []


def _selftest():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t, txt = buscar()
    if t and txt:
        print(f"[selftest] leu '{t}' ({len(txt)} chars). trecho: {re.sub(chr(10),' ',txt[:90])}...")
        ls = links(t, 6)
        print(f"           links p/ seguir a curiosidade: {ls[:5]}")
    else:
        print("[selftest] sem internet / Wikipedia indisponível (honesto: ferramenta precisa de rede)")


if __name__ == "__main__":
    _selftest()
