"""
FATOS — memória FACTUAL crisp do Toshi (roubado do neurocore, melhorado e integrado).

O Toshi tem semântica rica mas TAGARELA em vez de responder um fato preciso ("qual é meu nome?").
Esta camada dá o que faltava: extrai relações da fala (sujeito -relação-> objeto), guarda num grafo
e responde perguntas diretas. Complementa (não substitui) o raciocínio associativo.

  "meu nome é maicon"          -> aprende (meu nome --é--> maicon)
  "qual é meu nome?"           -> "meu nome é maicon"
  "paris é a capital de frança"-> aprende (paris --capital_de--> frança)
  "qual é a capital de frança?"-> "A capital de frança é paris"

Stdlib só. Persiste em JSON. Roda: python fatos.py  (selftest)
"""
import os
import re
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ARQ = os.path.join(HERE, "dados", "fatos.json")


def _norm(s):
    return re.sub(r"\s+", " ", s.lower().strip()).rstrip(".!?")


# declarações: (padrão, relação). Ordem importa (mais específico primeiro).
_DECL = [
    (r"^(.+?)\s+(?:é|eh)\s+a\s+capital\s+d(?:e|o|a)\s+(.+)$", "capital_de"),
    (r"^(.+?)\s+fic[ao]\s+(?:em|no|na)\s+(.+)$", "fica_em"),
    (r"^(.+?)\s+mor[ao](?:mos|am)?\s+em\s+(.+)$", "mora_em"),
    (r"^(.+?)\s+gost[ao](?:mos|am)?\s+de\s+(.+)$", "gosta_de"),
    (r"^(.+?)\s+trabalh[ao](?:mos|am)?\s+(?:com|como)\s+(.+)$", "trabalha_com"),
    (r"^(.+?)\s+t(?:enho|em|emos|êm)\s+(.+)$", "tem"),
    (r"^(.+?)\s+(?:é|eh|sou|somos|são|sao)\s+(.+)$", "é"),
]
# perguntas: (padrão, relação-alvo ou None). "meu nome eh X" vira sujeito "meu nome".
_PERG = [
    (r"qual\s+(?:é|eh)\s+a\s+capital\s+d(?:e|o|a)\s+(.+)", "capital_de"),
    (r"onde\s+(?:fica|mora|é|eh)\s+(.+)", "onde"),
    (r"quem\s+sou\s+eu", "quem_eu"),
    (r"quem\s+eu\s+sou", "quem_eu"),
    (r"quem\s+sou\s*$", "quem_eu"),
    (r"que\s+cor\s+tem\s+(.+)", "que_cor"),
    (r"quem\s+(?:é|eh)\s+(.+)", None),
    (r"o\s+que\s+(?:é|eh)\s+(.+)", "e"),
    (r"qual\s+(?:é|eh)\s+(?:o|a)?\s*(.+)", None),
]


_QW = r"^\s*(qual|quais|quem|onde|quando|como|que|o\s+que|por\s*que)\b"


def extrair(texto):
    """(sujeito, relação, objeto) de uma declaração, ou None. Pergunta NÃO é declaração."""
    t = texto.strip()
    if t.endswith("?") or re.match(_QW, t, re.IGNORECASE):
        return None
    m = re.match(r"^sou\s+(.+)$", t, re.IGNORECASE)   # "sou maicon" -> sujeito implícito "eu"
    if m:
        o = _norm(m.group(1))
        if o:
            return "eu", "é", o
    for pat, rel in _DECL:
        m = re.match(pat, t, re.IGNORECASE)
        if m:
            s, o = _norm(m.group(1)), _norm(m.group(2))
            if s and o and len(o) < 60:
                return s, rel, o
    return None


class Fatos:
    def __init__(self):
        self.g = {}                       # sujeito -> lista [rel, obj]
        self._load()

    def aprender(self, s, r, o):
        s, r, o = _norm(s), _norm(r), _norm(o)
        self.g.setdefault(s, [])
        if [r, o] not in self.g[s]:
            self.g[s].append([r, o])

    def _rev(self, rel, obj):             # quem tem (rel -> obj)? (busca reversa)
        return [s for s, es in self.g.items() for r, o in es if r == rel and o == obj]

    def responder(self, texto):
        """Resposta factual crisp, ou None (aí o Toshi raciocina normal)."""
        q = _norm(texto)
        e = extrair(texto)                # se for declaração, não é pergunta
        if e:
            return None
        for pat, rel in _PERG:
            m = re.search(pat, q)
            if not m:
                continue
            if rel == "quem_eu":          # pergunta sem grupo de captura
                for r2, o2 in self.g.get("eu", []):
                    if r2 == "é":
                        return f"eu {r2} {o2}."
                continue
            alvo = _norm(m.group(1))
            if rel == "capital_de":       # "capital de X" -> quem é a capital de X (reverso)
                r = self._rev("capital_de", alvo)
                if r:
                    return f"A capital de {alvo} é {r[0]}."
            if rel == "onde":             # prefere as relações de LOCALIZAÇÃO
                for r in ("fica_em", "mora_em"):
                    for r2, o2 in self.g.get(alvo, []):
                        if r2 == r:
                            return f"{alvo} {r2} {o2}."
            if rel == "que_cor":          # "que cor tem X?" -> aresta tem/cor
                for s in self.g:
                    s_n, a_n = _norm(s), alvo
                    if s_n == a_n or a_n in s_n or s_n in a_n:
                        for r2, o2 in self.g[s]:
                            if r2 == "tem" and o2.startswith("cor "):
                                return f"a cor de {s} é {o2[4:]}."
                continue
            if rel == "e":                # prefere a relação de DEFINIÇÃO
                for r2, o2 in self.g.get(alvo, []):
                    if r2 == "é":
                        return f"{alvo} {r2} {o2}."
            # direto: o alvo é sujeito de alguma relação
            if alvo in self.g and self.g[alvo]:
                r, o = self.g[alvo][0]
                return f"{alvo} {r} {o}."
            # reverso: o alvo é objeto de alguma relação
            for r in ("é", "capital_de", "fica_em", "mora_em", "gosta_de", "trabalha_com"):
                who = self._rev(r, alvo)
                if who:
                    return f"{who[0]} {r} {alvo}."
        return None

    def save(self):
        try:
            os.makedirs(os.path.dirname(ARQ), exist_ok=True)
            with open(ARQ, "w", encoding="utf-8") as f:
                json.dump(self.g, f, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        try:
            with open(ARQ, encoding="utf-8") as f:
                self.g = json.load(f)
        except Exception:
            self.g = {}


def _selftest():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    f = Fatos(); f.g = {}                  # limpo p/ o teste
    for frase in ["meu nome é maicon", "eu moro em gravatai",
                  "paris é a capital de frança", "eu gosto de programar"]:
        e = extrair(frase); assert e, frase; f.aprender(*e)
    casos = {
        "qual é meu nome?": "maicon",
        "onde eu moro?": "gravatai",
        "qual é a capital de frança?": "paris",
        "do que eu gosto?": None,          # forma não coberta -> Toshi raciocina (ok)
    }
    ok = 0
    for p, esp in casos.items():
        r = f.responder(p)
        hit = bool((esp is None) or (r and esp in r))
        ok += hit
        print(f"  {p:<28} -> {r}")
    assert f.responder("qual é meu nome?") and "maicon" in f.responder("qual é meu nome?")
    assert "paris" in (f.responder("qual é a capital de frança?") or "")
    print(f"[selftest] ok ({ok}/{len(casos)} respostas factuais crisp)")


if __name__ == "__main__":
    _selftest()
