"""
ABSORVER QWEN — destilação do conhecimento do Qwen para a representação nativa do Toshi.

IDÉIA (não é chat, é ABSORÇÃO — como abrir um CSV no Excel):
    O Qwen (transformer) e o Toshi (VSA/Random Indexing) têm arquiteturas DIFERENTES.
    Não copiamos pesos: extraímos o CONHECIMENTO do Qwen em lote, transcrevemos para a
    representação nativa do Toshi e depois TESTAMOS automaticamente o que foi absorvido.

PIPELINE:
    1. EXTRATOR   -> pergunta ao Qwen (via Ollama) fatos/definições/analogias em lote
    2. TRANSCRITOR-> converte cada frase do Qwen para o substrato do Toshi:
                      - associações e transições: toshi.perceive(tokens) repetido
                      - significado: Random Indexing emerge dos contextos
                      - memória factual crisp: fatos.aprender(sujeito, relação, objeto)
    3. ABSORVEDOR -> alimenta o Toshi e registra o antes/depois
    4. BATERIA    -> teste AUTOMÁTICO (a regra do RADAR: sem métrica, não entra):
                      - fidelidade dos fatos absorvidos
                      - evocação associativa (o que acende quando se pergunta)
                      - analogias (aritmética de significado)
                      - cobertura de vocabulário

USO:
    python absorver_qwen.py --modo tudo --temas "geografia,animais,corpo humano" --itens 12
    python absorver_qwen.py --selftest          (testa o pipeline sem rede)
    python absorver_qwen.py --modo fatos        (só fatos)
    python absorver_qwen.py --modo analogia     (só analogias)

HONESTO: isto NÃO transfere os pesos do transformer (impossível entre arquiteturas).
Transfere o conhecimento que o Qwen consegue EXPORTAR, na forma que o Toshi consegue
ABSORVER — e o teste automático mede exatamente quanto entrou.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, build_or_load, save_state, tokenize
from fatos import Fatos, extrair

REPORT = os.path.join(HERE, "dados", "absorcao_qwen_report.json")


# ============================================================ FONTE DE IA
class FonteQwen:
    """Fonte de conhecimento: Qwen rodando localmente via Ollama (HTTP)."""

    def __init__(self, host="http://127.0.0.1:11434", modelo="qwen2.5:7b",
                 timeout=300, num_ctx=4096):
        self.host = host.rstrip("/")
        self.modelo = modelo
        self.timeout = timeout
        self.num_ctx = num_ctx

    def disponivel(self) -> bool:
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def gerar(self, prompt, max_tokens=512, temperatura=0.35, **_):
        """Pede uma geração ao Qwen e devolve só o texto."""
        payload = json.dumps({
            "model": self.modelo,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperatura,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        print(f"    [qwen] {data.get('eval_count', '?')} tokens em {time.time() - t0:.1f}s")
        return data.get("response", "")


class FonteMock:
    """Fonte offline para selftest — conhecimento fixo, sem depender de rede."""

    nome = "mock"

    _FATOS = {
        "animais": [
            "canguru é marsupial",
            "canguru fica em australia",
            "camberra é a capital de australia",
            "gato tem pelo",
            "golfinho é mamifero",
            "tubarao fica no oceano",
        ],
        "geografia": [
            "paris é a capital de frança",
            "lisboa é a capital de portugal",
            "o rio amazonas fica no brasil",
            "o deserto do saara fica em africa",
            "o monte everest fica na asia",
            "o brasil tem estados",
        ],
        "corpo humano": [
            "o coracao tem ventriculos",
            "o pulmao fica no torax",
            "o figado fica no abdomen",
            "o olho tem retina",
            "o cerebro fica na cabeca",
        ],
    }

    _SEMANTICA = {
        "natureza": [
            "o sol aquece a terra",
            "a agua molha a planta",
            "o fogo queima a madeira",
            "o vento move as folhas",
            "a chuva rega o campo",
        ],
        "profissoes": [
            "o medico cuida do doente",
            "o professor ensina o aluno",
            "o cozinheiro prepara a comida",
            "o motorista dirige o carro",
        ],
    }

    _ANALOGIAS = [
        ("rei", "rainha", "homem", "mulher"),
        ("paris", "franca", "roma", "italia"),
        ("gato", "gatinho", "cao", "cachorrinho"),
    ]

    _CONTEXTOS = {
        ("rei", "rainha", "homem", "mulher"): [
            "o rei governa o reino",
            "a rainha governa o reino",
            "o homem trabalha na vila",
            "a mulher trabalha na vila",
        ],
        ("paris", "franca", "roma", "italia"): [
            "paris fica na franca",
            "lion fica na franca",
            "roma fica na italia",
            "milao fica na italia",
        ],
        ("gato", "gatinho", "cao", "cachorrinho"): [
            "o gato tem um gatinho",
            "a gata tem um gatinho",
            "o cao tem um cachorrinho",
            "a cadela tem um cachorrinho",
        ],
    }

    _SUBTEMAS = {
        "animais": ["mamiferos", "aves", "peixes", "repteis", "insetos"],
        "geografia": ["capitais", "rios", "montanhas", "desertos", "oceanos"],
        "corpo humano": ["coracao", "cerebro", "pulmao", "figado", "olho"],
    }

    def disponivel(self):
        return True

    def gerar(self, prompt, max_tokens=512, temperatura=0.35, tipo=None, tema=None, **_):
        if tipo == "fatos":
            return "\n".join(self._FATOS.get(tema or "animais", []))
        if tipo == "semantica":
            return "\n".join(self._SEMANTICA.get(tema or "natureza", []))
        if tipo == "analogias":
            return "\n".join(
                f"{a} esta para {b} como {c} esta para {d}"
                for a, b, c, d in self._ANALOGIAS
            )
        if tipo == "contextos":
            for quad, frases in self._CONTEXTOS.items():
                if tema and list(quad) == tema:
                    return "\n".join(frases)
            return ""
        if tipo == "subtemas":
            return "\n".join(self._SUBTEMAS.get(tema or "", []))
        return "mock"


# ============================================================ EXTRATOR
_RE_ANALOGIA = [
    re.compile(
        r"^\s*(.+?)\s+est[aá]\s+para\s+(.+?)\s+como\s+(.+?)\s+est[aá]\s+para\s+(.+?)\s*[.]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(.+?)\s+est[aá]\s+para\s+(.+?)\s+assim\s+como\s+(.+?)\s+est[aá]\s+para\s+(.+?)\s*[.]?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(.+?)\s+(?:é|eh)\s+para\s+(.+?)\s+(?:assim\s+)?como\s+(.+?)\s+(?:é|eh)\s+para\s+(.+?)\s*[.]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(.+?)\s*:\s*(.+?)\s*::\s*(.+?)\s*:\s*(.+?)\s*$"),
]


def linhas_limpas(texto, min_palavras=2, max_palavras=14):
    """Filtra a saída do Qwen: remove markdown/numeração e fica só com frases comestíveis."""
    out = []
    for raw in texto.splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        s = s.strip('"').rstrip(".;:")
        if s and min_palavras <= len(tokenize(s)) <= max_palavras:
            out.append(s)
    return out


class ExtratorQwen:
    """Pede ao Qwen conhecimento em formatos transcrevíveis para o Toshi."""

    def __init__(self, fonte):
        self.fonte = fonte

    def fatos(self, tema, n=10):
        prompt = (
            f"Tema: {tema}\n"
            f"Liste exatamente {n} fatos verdadeiros e simples, um por linha. "
            "Use APENAS uma destas formas de frase:\n"
            " - [sujeito] é [objeto]\n"
            " - [sujeito] é a capital de [lugar]\n"
            " - [sujeito] fica em [lugar]\n"
            " - [sujeito] tem [objeto]\n"
            "Sem numeração, sem markdown, sem título, sem explicação extra. "
            "Use a palavra 'é' com acento."
        )
        texto = self.fonte.gerar(prompt, max_tokens=512, tipo="fatos", tema=tema)
        return linhas_limpas(texto, min_palavras=3, max_palavras=12)

    def semantica(self, tema, n=10):
        prompt = (
            f"Tema: {tema}\n"
            f"Escreva {n} frases curtas e simples (máximo 14 palavras), uma por linha. "
            "Cada frase deve ligar duas ou três palavras importantes do tema "
            "(sujeito, verbo, objeto). Sem numeração, sem markdown, sem título."
        )
        texto = self.fonte.gerar(prompt, max_tokens=512, tipo="semantica", tema=tema)
        return linhas_limpas(texto, min_palavras=3, max_palavras=18)

    def subtemas(self, tema, n=5):
        prompt = (
            f"Liste {n} subtemas ou conceitos importantes relacionados ao tema '{tema}', "
            "um por linha, cada um com no máximo 3 palavras. "
            "Sem numeração, sem markdown, sem explicação."
        )
        texto = self.fonte.gerar(prompt, max_tokens=256, tipo="subtemas", tema=tema)
        return [s for s in linhas_limpas(texto, min_palavras=1, max_palavras=3)]

    def analogias(self, n=8):
        prompt = (
            f"Liste {n} analogias simples e verdadeiras, uma por linha, no formato:\n"
            "a esta para b como c esta para d\n"
            "Exemplo: rei esta para rainha como homem esta para mulher\n"
            "Use palavras comuns, sem numeração, sem markdown, sem explicação."
        )
        texto = self.fonte.gerar(prompt, max_tokens=512, tipo="analogias")
        quads = []
        for linha in linhas_limpas(texto, min_palavras=6, max_palavras=18):
            m = None
            for pat in _RE_ANALOGIA:
                m = pat.match(linha)
                if m:
                    break
            if m:
                quads.append(tuple(g.strip() for g in m.groups()))
        return quads

    def contextos_analogia(self, quad, n_frases=4):
        a, b, c, d = quad
        prompt = (
            f"Mostre a mesma relação entre ({a} e {b}) e entre ({c} e {d}).\n"
            f"Escreva {n_frases} frases curtas (uma por linha), com pares de frases paralelas "
            f"usando os mesmos contextos para os dois lados.\n"
            f"Exemplo para (rei,rainha) e (homem,mulher):\n"
            f"o rei governa o reino\n"
            f"a rainha governa o reino\n"
            f"o homem trabalha na vila\n"
            f"a mulher trabalha na vila\n"
            "Sem numeração, sem markdown."
        )
        texto = self.fonte.gerar(prompt, max_tokens=512, tipo="contextos", tema=list(quad))
        return linhas_limpas(texto, min_palavras=3, max_palavras=12)


# ============================================================ BATERIA DE TESTE
def _norm_fato(s):
    return re.sub(r"\s+", " ", s.lower().strip()).rstrip(".!?")


class Bateria:
    """Itens de teste automático + avaliação antes/depois da absorção."""

    def __init__(self):
        self.itens = {"fatos": [], "assoc": [], "analogias": [], "vocab": []}

    def add_fato(self, s, r, o, origem=""):
        chave = (s, r, o)
        if any((x["s"], x["r"], x["o"]) == chave for x in self.itens["fatos"]):
            return
        self.itens["fatos"].append({"s": s, "r": r, "o": o, "origem": origem})

    def add_assoc(self, palavra, esperadas, origem=""):
        # normaliza para o espaço do Toshi: minúsculas e SEM acento
        pal = tokenize(palavra)
        esps = []
        for e in esperadas:
            esps.extend(tokenize(e))
        self.itens["assoc"].append({
            "palavra": pal[-1] if pal else palavra,
            "esperadas": esps or list(esperadas),
            "origem": origem,
        })

    def add_analogia(self, a, b, c, esperada):
        def _tok(x):
            t = tokenize(x)
            return t[0] if t else x
        self.itens["analogias"].append({
            "a": _tok(a), "b": _tok(b), "c": _tok(c), "esperada": _tok(esperada),
        })

    def add_vocab(self, palavras):
        for w in palavras:
            if all(w != x["palavra"] for x in self.itens["vocab"]):
                self.itens["vocab"].append({"palavra": w})

    def _pergunta_fato(self, s, r, o):
        # só gera pergunta para relações que o Fatos cobre sem ambiguidade
        if r == "capital_de":
            return f"qual é a capital de {o}"
        if r == "fica_em":
            return f"onde fica {s}"
        if r == "mora_em":
            return f"onde mora {s}"
        if r == "é":
            return f"o que é {s}"
        return None

    def avaliar(self, toshi, fatos):
        """Roda a bateria inteira e devolve acertos, erros e score por categoria."""
        res = {"fatos": {"ok": 0, "n": 0, "detalhes": []},
               "fatos_linguagem": {"ok": 0, "n": 0, "detalhes": []},
               "assoc": {"ok": 0, "n": 0, "detalhes": []},
               "analogias": {"ok": 0, "n": 0, "detalhes": []},
               "vocab": {"ok": 0, "n": 0, "detalhes": []}}

        for item in self.itens["fatos"]:
            res["fatos"]["n"] += 1
            s, r, o = item["s"], item["r"], item["o"]
            # 1) FIDELIDADE: o fato foi gravado no grafo? (transcrição direta)
            edge = [r, o] in fatos.g.get(s, [])
            if edge:
                res["fatos"]["ok"] += 1

            # 2) RESPOSTA EM LINGUAGEM: só quando a pergunta é coberta e
            #    o sujeito tem UMA única aresta daquela relação (sem ambiguidade).
            pergunta = self._pergunta_fato(s, r, o)
            resposta = fatos.responder(pergunta) if pergunta else None
            n_mesma_rel = sum(1 for rr, _ in fatos.g.get(s, []) if rr == r)
            testavel = pergunta is not None and n_mesma_rel == 1
            if testavel:
                res["fatos_linguagem"]["n"] += 1
                hit_ling = bool(resposta and o in resposta)
                if hit_ling:
                    res["fatos_linguagem"]["ok"] += 1
                res["fatos_linguagem"]["detalhes"].append({
                    "frase": item["origem"], "s": s, "r": r, "o": o,
                    "pergunta": pergunta, "resposta": resposta, "hit": hit_ling,
                })

            res["fatos"]["detalhes"].append({
                "frase": item["origem"], "s": s, "r": r, "o": o,
                "grafo": edge, "pergunta": pergunta, "resposta": resposta,
                "hit": edge,
            })

        for item in self.itens["assoc"]:
            res["assoc"]["n"] += 1
            evocadas = [w for w, _ in toshi.associations([item["palavra"]], k=10)]
            hit = any(e in evocadas for e in item["esperadas"])
            if hit:
                res["assoc"]["ok"] += 1
            res["assoc"]["detalhes"].append({
                "palavra": item["palavra"], "esperadas": item["esperadas"],
                "evocadas": evocadas[:6], "hit": hit,
            })

        for item in self.itens["analogias"]:
            res["analogias"]["n"] += 1
            top = [w for w, _ in toshi.analogy(item["a"], item["b"], item["c"], k=5)]
            hit = item["esperada"] in top
            if hit:
                res["analogias"]["ok"] += 1
            res["analogias"]["detalhes"].append({
                "a": item["a"], "b": item["b"], "c": item["c"],
                "esperada": item["esperada"], "top5": top, "hit": hit,
            })

        for item in self.itens["vocab"]:
            res["vocab"]["n"] += 1
            hit = toshi.seen[item["palavra"]] > 0
            if hit:
                res["vocab"]["ok"] += 1
            res["vocab"]["detalhes"].append({
                "palavra": item["palavra"], "vista": toshi.seen[item["palavra"]], "hit": hit,
            })

        res["global"] = {
            "ok": sum(res[k]["ok"] for k in
                     ("fatos", "fatos_linguagem", "assoc", "analogias", "vocab")),
            "n": sum(res[k]["n"] for k in
                     ("fatos", "fatos_linguagem", "assoc", "analogias", "vocab")),
        }
        return res

    @staticmethod
    def resumo(res):
        def pct(d):
            return f"{d['ok']}/{d['n']} ({d['ok'] / d['n'] * 100:.0f}%)" if d["n"] else "—"
        return {
            "fatos_no_grafo": pct(res["fatos"]),
            "fatos_linguagem": pct(res["fatos_linguagem"]),
            "associacoes": pct(res["assoc"]),
            "analogias": pct(res["analogias"]),
            "vocabulario": pct(res["vocab"]),
            "global": pct(res["global"]),
        }


# ============================================================ ABSORVEDOR
class AbsorvedorQwen:
    """Coordena extração -> transcrição -> absorção -> teste automático."""

    def __init__(self, toshi=None, fatos=None, fonte=None, repetir=3,
                 auto_salvar=True, carregar_existente=True):
        self.fonte = fonte or FonteQwen()
        self.extrator = ExtratorQwen(self.fonte)

        if toshi is None and carregar_existente:
            self.toshi, _ = build_or_load()
        else:
            self.toshi = toshi or Toshi()
        self.fatos = fatos or Fatos()

        self.repetir = repetir
        self.auto_salvar = auto_salvar
        self.bateria = Bateria()
        self.log_absorcao = []          # tudo que foi transcrito
        self.snapshot_inicio = self.snapshot()

    # ---------- snapshot ----------
    def snapshot(self):
        return {
            "palavras_vividas": sum(self.toshi.seen.values()),
            "conceitos": len(self.toshi.seen),
            "fatos_grafo": sum(len(v) for v in self.fatos.g.values()),
            "associacoes": sum(len(v) for v in self.toshi.assoc.values()),
            "transicoes": sum(len(v) for v in self.toshi.after.values()),
        }

    # ---------- transcrição (a "célula do Excel") ----------
    def transcrever(self, frase, origem="qwen"):
        """Converte uma frase do Qwen para o substrato nativo do Toshi."""
        tokens = tokenize(frase)
        if len(tokens) < 2:
            return None

        # associação + transição + significado emergem de perceive repetido
        for _ in range(self.repetir):
            self.toshi.perceive(tokens)

        # fato crisp vai para o grafo factual
        tripla = extrair(frase)
        if tripla:
            s, r, o = tripla
            self.fatos.aprender(s, r, o)
            # transcrição adicional: a ligação sujeito->objeto vira associação
            # forte no substrato do Toshi (além do grafo crisp)
            par = tokenize(f"{s} {o}")
            if len(par) >= 2:
                for _ in range(self.repetir):
                    self.toshi.perceive(par)

        self.log_absorcao.append({"frase": frase, "tokens": tokens,
                                  "tripla": tripla, "origem": origem})
        return tokens, tripla

    def _registrar_fato_na_bateria(self, s, r, o, frase):
        s, r, o = _norm_fato(s), r, _norm_fato(o)
        self.bateria.add_fato(s, r, o, frase)
        # gatilho associativo = núcleo do sujeito ("o coracao" -> "coracao")
        toks_s = tokenize(s)
        gatilhos = [w for w in toks_s if len(w) > 2][-1:] or toks_s[-1:]
        for g in gatilhos:
            self.bateria.add_assoc(g, [o, o.split()[-1]], frase)
        self.bateria.add_vocab(tokenize(f"{s} {o}"))

    # ---------- modos de absorção ----------
    def absorver_fatos(self, temas, n_por_tema=10, salvar=True):
        """Fatos estruturados: transcrição exata + teste de fidelidade."""
        print(f"\n=== MODO FATOS ({', '.join(temas)}) ===")
        total_frases = 0
        for tema in temas:
            print(f"  tema: {tema}")
            try:
                frases = self.extrator.fatos(tema, n_por_tema)
            except Exception as e:
                print(f"    [falha na extração] {e}")
                continue
            for frase in frases:
                r = self.transcrever(frase, origem=f"qwen:fatos:{tema}")
                if r and r[1]:
                    self._registrar_fato_na_bateria(r[1][0], r[1][1], r[1][2], frase)
                total_frases += 1
            print(f"    {len(frases)} frases extraídas")
        print(f"  total absorvido: {total_frases} frases, "
              f"{self.bateria.itens['fatos'] and len(self.bateria.itens['fatos'])} fatos no teste")
        if salvar:
            self._salvar_se_auto()

    def absorver_semantica(self, temas, n_por_tema=10, salvar=True):
        """Definições/currículo: associação + significado (Random Indexing)."""
        print(f"\n=== MODO SEMÂNTICA ({', '.join(temas)}) ===")
        for tema in temas:
            print(f"  tema: {tema}")
            try:
                frases = self.extrator.semantica(tema, n_por_tema)
            except Exception as e:
                print(f"    [falha na extração] {e}")
                continue
            for frase in frases:
                r = self.transcrever(frase, origem=f"qwen:semantica:{tema}")
                if r:
                    toks = r[0]
                    # testa a primeira palavra de conteúdo como gatilho
                    gatilhos = [w for w in toks if len(w) > 3][:1]
                    for g in gatilhos:
                        esperadas = [w for w in toks if w != g and len(w) > 3][:3]
                        if esperadas:
                            self.bateria.add_assoc(g, esperadas, frase)
                            # transcrição direta: reforça a ligação gatilho->palavra-chave
                            # (igual ao modo fatos: o par vira associação forte)
                            for e in esperadas:
                                par = tokenize(f"{g} {e}")
                                if len(par) >= 2:
                                    for _ in range(self.repetir):
                                        self.toshi.perceive(par)
                    self.bateria.add_vocab(toks)
            print(f"    {len(frases)} frases extraídas")
        if salvar:
            self._salvar_se_auto()

    def absorver_analogias(self, n=8, n_frases=6, salvar=True):
        """Analogias: pares de contextos paralelos -> aritmética de significado."""
        print(f"\n=== MODO ANALOGIA ({n} quads) ===")
        quads = self.extrator.analogias(n)
        print(f"  {len(quads)} analogias válidas extraídas")
        for a, b, c, d in quads:
            frases = self.extrator.contextos_analogia((a, b, c, d), n_frases)
            # repete o bloco de contextos para reforçar o paralelismo
            for _ in range(5):
                for frase in frases:
                    self.transcrever(frase, origem=f"qwen:analogia:{a}:{b}:{c}:{d}")
            self.bateria.add_analogia(a, b, c, d)
            self.bateria.add_vocab(tokenize(f"{a} {b} {c} {d}"))
        if salvar:
            self._salvar_se_auto()

    def absorver_tudo(self, temas, n_por_tema=10, n_analogias=6):
        self.absorver_fatos(temas, n_por_tema)
        self.absorver_semantica(temas, n_por_tema)
        self.absorver_analogias(n_analogias)

    def absorver_arvore(self, temas_semente, profundidade=2, n_fatos=8,
                        n_subtemas=5, max_temas=30):
        """Absorção em árvore: cada tema vira subtemas que também são absorvidos.
        É o modo 'lote grande' — aproxima o 'comer o modelo' com orçamento explícito."""
        print(f"\n=== MODO ÁRVORE (profundidade={profundidade}, máx temas={max_temas}) ===")
        fila = [(t, 0) for t in temas_semente]
        visitados = set()
        temas_feitos = 0
        while fila and temas_feitos < max_temas:
            tema, prof = fila.pop(0)
            chave = _norm_fato(tema)
            if chave in visitados or not tema.strip():
                continue
            visitados.add(chave)
            temas_feitos += 1
            print(f"\n  ({temas_feitos}) tema: {tema} (prof {prof})")
            self.absorver_fatos([tema], n_por_tema=n_fatos, salvar=False)
            if prof < profundidade:
                subs = self.extrator.subtemas(tema, n_subtemas)
                print(f"      subtemas: {', '.join(subs[:n_subtemas]) or '(nenhum)'}")
                for st in subs[:n_subtemas]:
                    if _norm_fato(st) not in visitados:
                        fila.append((st, prof + 1))
        print(f"\n  temas absorvidos: {temas_feitos}")
        self._salvar_se_auto()

    def _salvar_se_auto(self):
        if self.auto_salvar:
            save_state(self.toshi)
            self.fatos.save()

    def salvar(self):
        save_state(self.toshi)
        self.fatos.save()
        print("  [salvo] estado do Toshi + fatos.json")

    # ---------- teste automático ----------
    def rodar_teste(self):
        """Roda a bateria e devolve (resumo, detalhes, crescimento do snapshot)."""
        res = self.bateria.avaliar(self.toshi, self.fatos)
        resumo = Bateria.resumo(res)
        snap_fim = self.snapshot()
        crescimento = {k: snap_fim[k] - self.snapshot_inicio[k] for k in snap_fim}
        return resumo, res, crescimento

    def relatorio(self, resumo, res, crescimento):
        """Imprime relatório honesto e salva JSON."""
        print("\n" + "=" * 66)
        print("TESTE AUTOMÁTICO DO TOSHI APÓS ABSORÇÃO DO QWEN")
        print("=" * 66)
        for k, v in resumo.items():
            print(f"  {k:<14} {v}")
        print("-" * 66)
        print("crescimento estrutural:")
        for k, v in crescimento.items():
            print(f"  {k:<14} +{v}")
        print("-" * 66)
        print("amostra de erros (se houver):")
        erros = 0
        for cat in ("fatos", "fatos_linguagem", "assoc", "analogias"):
            for d in res[cat]["detalhes"]:
                if not d["hit"] and erros < 5:
                    print(f"  [{cat}] {d}")
                    erros += 1
        if erros == 0:
            print("  (nenhum erro)")

        rel = {
            "quando": time.strftime("%Y-%m-%d %H:%M"),
            "fonte": getattr(self.fonte, "modelo", getattr(self.fonte, "nome", "desconhecida")),
            "resumo": resumo,
            "crescimento": crescimento,
            "snapshot_final": self.snapshot(),
            "frases_absorvidas": len(self.log_absorcao),
        }
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(rel, f, ensure_ascii=False, indent=2)
        print(f"\n[relatório salvo] {REPORT}")
        return rel


# ============================================================ SELFTEST (sem rede)
def _selftest():
    print("SELFTEST — pipeline de absorção com fonte mock (sem rede)\n")
    # Toshi/fatos limpos para medir do zero
    t = Toshi()
    f = Fatos()
    f.g = {}
    fonte = FonteMock()
    absv = AbsorvedorQwen(toshi=t, fatos=f, fonte=fonte, repetir=4,
                          auto_salvar=False, carregar_existente=False)

    absv.absorver_fatos(["animais", "geografia", "corpo humano"], n_por_tema=6)
    absv.absorver_semantica(["natureza", "profissoes"], n_por_tema=6)
    absv.absorver_analogias(n=3, n_frases=4)

    resumo, res, cresc = absv.rodar_teste()
    for k, v in resumo.items():
        print(f"  {k:<14} {v}")

    # asserts honestos: a transcrição entrou e o Toshi responde
    assert res["fatos"]["ok"] / max(1, res["fatos"]["n"]) >= 0.8, res["fatos"]
    assert res["fatos_linguagem"]["ok"] / max(1, res["fatos_linguagem"]["n"]) >= 0.8, res["fatos_linguagem"]
    assert "paris" in (f.responder("qual é a capital de frança") or "")
    assert res["vocab"]["ok"] == res["vocab"]["n"]
    assert cresc["conceitos"] > 0
    assert absv.extrator.subtemas("animais", 3), "extração de subtemas falhou"
    print("\n[selftest] ok — absorção transcrita, testada e aprovada")


# ============================================================ CLI
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Absorve conhecimento do Qwen no Toshi e testa automaticamente.")
    ap.add_argument("--modo", choices=["fatos", "semantica", "analogia", "tudo"], default="tudo")
    ap.add_argument("--temas", default="geografia,animais,corpo humano,tecnologia")
    ap.add_argument("--itens", type=int, default=10, help="itens por tema")
    ap.add_argument("--analogias", type=int, default=6, help="nº de analogias")
    ap.add_argument("--modelo", default="qwen2.5:7b")
    ap.add_argument("--repetir", type=int, default=3)
    ap.add_argument("--expandir", action="store_true",
                    help="absorve em árvore: subtemas dos temas iniciais também entram")
    ap.add_argument("--profundidade", type=int, default=2,
                    help="profundidade da árvore de temas (com --expandir)")
    ap.add_argument("--max-temas", type=int, default=30,
                    help="número máximo de temas na árvore (com --expandir)")
    ap.add_argument("--selftest", action="store_true", help="roda o teste offline com fonte mock")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    fonte = FonteQwen(modelo=args.modelo)
    print(f"verificando fonte de conhecimento: {fonte.modelo} ...")
    if not fonte.disponivel():
        print("  Ollama não respondeu em http://127.0.0.1:11434")
        print("  Inicie o Ollama (ollama serve) e confirme o modelo com: ollama list")
        return

    temas = [x.strip() for x in args.temas.split(",") if x.strip()]
    absv = AbsorvedorQwen(fonte=fonte, repetir=args.repetir)

    print(f"absorvendo do Qwen ({fonte.modelo}) para o Toshi...")
    print(f"snapshot inicial: {absv.snapshot()}")
    t0 = time.time()

    if args.expandir:
        absv.absorver_arvore(temas, profundidade=args.profundidade,
                             n_fatos=args.itens, max_temas=args.max_temas)
    else:
        if args.modo in ("fatos", "tudo"):
            absv.absorver_fatos(temas, args.itens)
        if args.modo in ("semantica", "tudo"):
            absv.absorver_semantica(temas, args.itens)
        if args.modo in ("analogia", "tudo"):
            absv.absorver_analogias(args.analogias)

    absv.salvar()
    resumo, res, cresc = absv.rodar_teste()
    absv.relatorio(resumo, res, cresc)
    print(f"\ntempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
