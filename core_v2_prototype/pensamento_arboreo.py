"""
PENSAMENTO ARBÓREO — o Toshi pensa em árvore e entende contexto. SEM Qwen.

O problema: reter conhecimento não basta. Decorar não é aprender. O Toshi precisa
INTERLIGAR experiências, lembranças e fatos quando recebe um texto/pergunta.

Este módulo dá ao Toshi:
  1. MEMÓRIA DE CONTEXTO — você manda um texto, ele lê e GUARDA o texto como episódio
     (tokens, embedding, fatos extraídos). Depois ele sabe que sua pergunta é sobre AQUELE texto.
  2. PENSAMENTO ARBÓREO — a partir da pergunta/contexto ele abre VÁRIAS linhas de raciocínio
     (associações + fatos + significado + contexto), expande em largura/profundidade,
     poda as fracas e fica com os melhores caminhos. É o "ver muitas possibilidades na cabeça".
  3. SÍNTESE PRÓPRIA — ele junta os melhores caminhos e fala o que ACHA, ligando o texto
     com o que ele já sabia. Não repete: conecta.

USO:
  python pensamento_arboreo.py --selftest
  python pensamento_arboreo.py --interativo
  python pensamento_arboreo.py --pergunta "o que você achou do texto?"
  python pensamento_arboreo.py --ler arquivo.txt
"""
import argparse
import json
import os
import pickle
import re
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, build_or_load, save_state, tokenize
from fatos import Fatos, _norm, extrair

CTX_FILE = os.path.join(HERE, "dados", "contextos_toshi.pkl")


def _eh_stop(toshi, palavra):
    """Stopwords só existem em corpus GRANDE. Bebê não pode podar as próprias ideias."""
    return len(toshi.seen) > 500 and palavra in toshi.stops


# ============================================================ NÓ DA ÁRVORE
class NoPensamento:
    __slots__ = ("conceito", "caminho", "profundidade", "score", "origem", "forca")

    def __init__(self, conceito, caminho, profundidade, score, origem, forca):
        self.conceito = conceito
        self.caminho = caminho
        self.profundidade = profundidade
        self.score = score
        self.origem = origem
        self.forca = forca

    def __repr__(self):
        return f"<{self.conceito} p={self.profundidade} s={self.score:.2f}>"


# ============================================================ MEMÓRIA DE CONTEXTO
class MemoriaContexto:
    def __init__(self, toshi, fatos):
        self.toshi = toshi
        self.fatos = fatos
        self.docs = []
        self._carregar()

    def _embed(self, toks):
        vecs = [self.toshi._emb(w) for w in toks if self.toshi._emb(w) is not None]
        if not vecs:
            return None
        v = sum(vecs) / len(vecs)
        n = float(np.linalg.norm(v))
        return (v / n).astype(float) if n > 1e-9 else None

    def ler(self, texto, origem="usuario"):
        toks = tokenize(texto)
        if not toks:
            return None
        # come o texto (assim como os livros)
        self.toshi.perceive(toks)
        self.toshi.settle(toks[:12])
        # fatos crisp que o texto declara
        fatos = []
        for frase in re.split(r"[.;!?\n]+", texto):
            e = extrair(frase)
            if e:
                self.fatos.aprender(*e)
                fatos.append(e)
        emb = self._embed(toks)
        doc = {
            "id": len(self.docs) + 1,
            "texto": texto,
            "tokens": toks,
            "emb": emb,
            "fatos": fatos,
            "ts": time.time(),
            "origem": origem,
        }
        self.docs.append(doc)
        self._salvar()
        return doc

    def _salvar(self):
        try:
            os.makedirs(os.path.dirname(CTX_FILE), exist_ok=True)
            with open(CTX_FILE, "wb") as f:
                pickle.dump(self.docs, f)
        except Exception:
            pass

    def _carregar(self):
        try:
            with open(CTX_FILE, "rb") as f:
                self.docs = pickle.load(f)
        except Exception:
            self.docs = []

    def relevante(self, texto, excluir_id=None):
        """Acha a experiência anterior mais parecida com o que ele acabou de viver.
        Nenhuma regra de pontuação: é só semelhança de significado + sobreposição."""
        q_toks = tokenize(texto)
        q_emb = self._embed(q_toks)
        q_set = set(q_toks)
        melhor, melhor_s = None, -1.0
        for d in self.docs:
            if excluir_id is not None and d.get("id") == excluir_id:
                continue
            s_emb = 0.0
            if q_emb is not None and d.get("emb") is not None:
                s_emb = float(np.dot(q_emb, np.asarray(d["emb"])))
            inter = len(q_set & set(d["tokens"]))
            s_txt = inter / max(1, min(len(q_set), len(d["tokens"])))
            s = 0.65 * s_emb + 0.35 * s_txt
            if s > melhor_s:
                melhor, melhor_s = d, s
        # se não achou por semelhança, a experiência mais RECENTE é o contexto natural
        if melhor is None and self.docs:
            anteriores = [d for d in self.docs if d.get("id") != excluir_id]
            if anteriores:
                melhor, melhor_s = anteriores[-1], 0.10
        return (melhor, melhor_s) if melhor_s >= 0.10 else (None, melhor_s)

    def palavras_chave(self, doc, k=5):
        cont = [w for w in doc["tokens"] if not _eh_stop(self.toshi, w) and len(w) > 2]
        return [w for w, _ in Counter(cont).most_common(k)]


# ============================================================ PENSADOR ARBÓREO
class PensadorArboreo:
    def __init__(self, toshi, fatos, contexto):
        self.toshi = toshi
        self.fatos = fatos
        self.ctx = contexto
        self.saltos_feitos = []   # hipóteses novas que ele CRIOU e integrou

    # ---------- utilidades ----------
    def _embed(self, toks):
        return self.ctx._embed(toks)

    def _score_candidato(self, cand, pai, q_toks, q_emb, doc, forca):
        s = 0.0
        if q_emb is not None:
            e = self.toshi._emb(cand)
            if e is not None:
                s += 0.35 * float(q_emb @ e)
        if cand in q_toks:
            s += 0.30
        if pai is not None and self.toshi._emb(pai) is not None and self.toshi._emb(cand) is not None:
            s += 0.15 * float(self.toshi._emb(pai) @ self.toshi._emb(cand))
        s += 0.20 * min(float(forca) / 4.0, 1.0)
        if doc is not None and cand in set(doc["tokens"]):
            s += 0.20
        return s

    def _vizinhos_assoc(self, no, q_toks, doc):
        cands = {}
        for w, c in sorted(self.toshi.assoc.get(no.conceito, {}).items(),
                           key=lambda kv: -kv[1]):
            if w in no.caminho or _eh_stop(self.toshi, w):
                continue
            cands[w] = max(cands.get(w, 0.0), float(c))
        return cands

    def _vizinhos_fatos(self, no):
        cands = {}
        alvo = no.conceito
        for s, arestas in self.fatos.g.items():
            if alvo in tokenize(s):
                for r, o in arestas:
                    for w in tokenize(o):
                        if w not in no.caminho:
                            cands[w] = max(cands.get(w, 0.0), 1.5)
            for r, o in arestas:
                if alvo in tokenize(o):
                    for w in tokenize(s):
                        if w not in no.caminho:
                            cands[w] = max(cands.get(w, 0.0), 1.5)
        return cands

    def _vizinhos_semanticos(self, no):
        try:
            return {u: max(0.5, s) for u, s in self.toshi.meaning(no.conceito, k=4, min_freq=1)
                    if u not in no.caminho and not _eh_stop(self.toshi, u)}
        except Exception:
            return {}

    def _raizes(self, pergunta, doc, k=4):
        q_toks = tokenize(pergunta)
        raizes = [w for w in q_toks
                  if not _eh_stop(self.toshi, w) and len(w) > 2 and w in self.toshi.assoc]
        if doc is not None:
            for w in self.ctx.palavras_chave(doc, k):
                if w not in raizes and w in self.toshi.assoc:
                    raizes.append(w)
        # mapa de calor: os pontos mais quentes também viram raízes (o raio)
        for w, _ in self.mapa_calor(pergunta, doc, top=6):
            if w not in raizes and w in self.toshi.assoc and not _eh_stop(self.toshi, w):
                raizes.append(w)
        # sem memória suficiente, usa qualquer palavra de conteúdo
        if not raizes:
            raizes = [w for w in q_toks if not _eh_stop(self.toshi, w) and len(w) > 2][:k]
        return raizes[:k]
    # ---------- mapa de calor (o raio dentro do cérebro) ----------
    def mapa_calor(self, pergunta, doc=None, top=12):
        """Propagação de ativação: os pontos mais quentes do cérebro para a pergunta."""
        q_toks = tokenize(pergunta)
        ativ = dict(self.toshi.spread(q_toks, depth=2, decay=0.45, per_node=6, k=top * 2))
        if doc is not None:
            for w in self.ctx.palavras_chave(doc, top):
                ativ[w] = max(ativ.get(w, 0.0), 1.0)
        return sorted(ativ.items(), key=lambda kv: -kv[1])[:top]

    # ---------- O SALTO (abdução): criar a linha que ainda não existia ----------
    def saltos(self, pergunta, doc=None, integrar=False, limiar=0.45,
               exigir_evidencia=False):
        """
        'LLMs can't jump' — aqui o Toshi tenta pular:
        para pares de conceitos que aparecem juntos na pergunta/texto mas NÃO têm
        ligação direta na memória, ele infere uma HIPÓTESE pela estrutura (vizinhos
        comuns + semelhança de significado + evidência de co-ocorrência).
        Se passar do limiar, ele CRIA a conexão nova e integra (com exigir_evidencia,
        só integra quando o texto novo mostra os dois juntos = a observação surpreendente).
        """
        q_toks = tokenize(pergunta)
        doc_toks = doc["tokens"] if doc is not None else []
        conceitos = []
        vistos = set()
        for w in q_toks + doc_toks:
            if w in vistos or _eh_stop(self.toshi, w):
                continue
            if w in self.toshi.assoc or w in self.toshi.embed:
                conceitos.append(w)
                vistos.add(w)

        saltos = []
        for i, a in enumerate(conceitos):
            for b in conceitos[i + 1:]:
                if b in self.toshi.assoc.get(a, {}) or a in self.toshi.assoc.get(b, {}):
                    continue
                comuns = set(self.toshi.assoc.get(a, {})) & set(self.toshi.assoc.get(b, {}))
                if not comuns:
                    continue
                aa = sum(1.0 / np.log(2 + self.toshi.seen[x]) for x in comuns)
                ea, eb = self.toshi._emb(a), self.toshi._emb(b)
                emb = float(ea @ eb) if (ea is not None and eb is not None) else 0.0
                evid = 1.0 if (doc is not None and a in set(doc_toks) and b in set(doc_toks)) else 0.0
                score = 0.45 * min(aa / 2.0, 1.0) + 0.15 * max(emb, 0.0) + 0.40 * evid
                if score < limiar:
                    continue
                if exigir_evidencia and evid == 0.0:
                    continue
                saltos.append({"a": a, "b": b, "score": score,
                               "comuns": sorted(comuns)[:5], "evidencia": bool(evid)})

        saltos.sort(key=lambda s: -s["score"])
        if integrar:
            for s in saltos:
                a, b = s["a"], s["b"]
                # cria a linha nova no cérebro
                self.toshi.assoc.setdefault(a, Counter())[b] += 1.0
                self.toshi.assoc.setdefault(b, Counter())[a] += 1.0
                self.toshi.perceive([a, b])
                self.fatos.aprender(a, "salto_para", b)
                self.saltos_feitos.append(s)
        return saltos

    # ---------- a árvore ----------
    def pensar(self, pergunta, doc=None, largura=4, profundidade=3, ramos=3):
        """
        Expande a árvore de pensamento a partir da pergunta/contexto.
        Devolve os melhores caminhos (lista de listas de conceitos).
        """
        q_toks = tokenize(pergunta)
        q_emb = self._embed(q_toks)
        raizes = self._raizes(pergunta, doc)

        if not raizes:
            return []

        folhas = []
        fronteira = []
        for r in raizes:
            no = NoPensamento(r, [r], 0, 1.0, "raiz", 1.0)
            fronteira.append(no)
            folhas.append(no)

        while fronteira:
            pai = max(fronteira, key=lambda n: (n.profundidade, n.score))
            fronteira.remove(pai)
            if pai.profundidade >= profundidade:
                continue

            cands = {}
            for origem, mapa in (
                ("assoc", self._vizinhos_assoc(pai, q_toks, doc)),
                ("fatos", self._vizinhos_fatos(pai)),
            ):
                for w, c in mapa.items():
                    cands[w] = max(cands.get(w, 0.0), c, key=float)

            if not cands:
                cands = self._vizinhos_semanticos(pai)

            filhos = []
            for w, c in cands.items():
                score = self._score_candidato(w, pai.conceito, q_toks, q_emb, doc, c)
                filhos.append(NoPensamento(w, pai.caminho + [w],
                                           pai.profundidade + 1, score,
                                           "assoc", c))
            filhos.sort(key=lambda n: -n.score)
            for f in filhos[:largura]:
                fronteira.append(f)
                folhas.append(f)

        # melhores caminhos: prefere quem REALMENTE caminhou (profundidade > 0)
        def valor(no):
            return no.score + 0.05 * no.profundidade
        com_profundidade = [n for n in folhas if n.profundidade > 0] or folhas
        melhores = sorted(com_profundidade, key=valor, reverse=True)[:ramos]
        return [n.caminho for n in melhores]

    # ---------- síntese ----------
    def _fato_sobre(self, palavra):
        for s, arestas in self.fatos.g.items():
            if palavra in tokenize(s):
                for r, o in arestas:
                    if r in ("é", "tem", "fica_em", "gosta_de", "mora_em", "capital_de"):
                        return f"{s} {r} {o}"
        return None

    def _ligacao(self, palavra, doc=None):
        # o que a palavra evoca na memória dele (associação mais forte fora do caminho)
        for u, c in sorted(self.toshi.assoc.get(palavra, {}).items(), key=lambda kv: -kv[1]):
            if not _eh_stop(self.toshi, u):
                return u
        if doc:
            for u in doc["tokens"]:
                if u != palavra and not _eh_stop(self.toshi, u):
                    return u
        return None

    def _opiniao_sobre_texto(self, doc, caminhos, pergunta):
        chaves = self.ctx.palavras_chave(doc, 4)
        if not chaves:
            return "li o texto, mas ainda não consegui formar uma opinião sobre ele."
        c1 = chaves[0]
        lig = self._ligacao(c1, doc)
        fato = self._fato_sobre(c1)
        caminho = caminhos[0] if caminhos else chaves
        partes = [
            f"achei que o texto fala de {', '.join(chaves[:3])}",
        ]
        if lig:
            partes.append(f"na minha memória, {c1} se liga a {lig}")
        if fato:
            partes.append(f"e eu já sabia que {fato}")
        partes.append(f"pensando em árvore, fui de {' -> '.join(caminho[:5])}")
        return ". ".join(partes) + "."

    def responder(self, entrada, doc_atual=None, aprender=True):
        """Responde SÓ com a memória dele. Nenhuma regra de '?' ou palavra mágica:
        a entrada vira experiência, ele acha a experiência anterior mais parecida
        e sintetiza a partir das duas. Aprende sozinho."""
        # 0) aprende com o que você disse — ninguém ensina nada pra ele
        if aprender:
            toks_entrada = tokenize(entrada)
            if toks_entrada:
                self.toshi.perceive(toks_entrada)
                self.toshi.settle(toks_entrada[:8])
            for frase in re.split(r"[.;!?\n]+", entrada):
                e = extrair(frase)
                if e:
                    self.fatos.aprender(*e)

        q = _norm(entrada)
        # 1) fato crisp direto
        r = self.fatos.responder(q)
        if r:
            return r, "fato"

        # 2) experiência anterior mais parecida (o input atual fica fora)
        doc, sim = self.ctx.relevante(
            entrada,
            excluir_id=(doc_atual.get("id") if isinstance(doc_atual, dict) else None),
        )

        # 3) árvore de pensamento (foco: contexto anterior, ou o que ele acabou de viver)
        foco = doc if doc is not None else doc_atual
        caminhos = self.pensar(entrada, doc=foco)

        # 3.5) o SALTO: ele tenta criar ligações novas (abdução) e integra as fortes
        saltos = self.saltos(entrada, doc=(doc or doc_atual), integrar=True,
                             limiar=0.45, exigir_evidencia=True)

        # 4) há uma experiência anterior relacionada -> sintetiza sobre ELA
        if doc is not None:
            resp = self._opiniao_sobre_texto(doc, caminhos, entrada)
            if saltos:
                s = saltos[0]
                resp += (f" E eu pulei: {s['a']} -> {s['b']} "
                         f"(ligação nova criada por {', '.join(s['comuns'])}).")
            return resp, "contexto"

        # 5) síntese a partir da árvore
        if caminhos:
            topico = caminhos[0][0]
            fato = self._fato_sobre(topico)
            base = (f"pensando em árvore: {' -> '.join(caminhos[0][:5])}. "
                    f"isso me leva a {caminhos[0][-1]}")
            if fato:
                base += f" (lembrando: {fato})"
            if saltos:
                s = saltos[0]
                base += f" (salto novo: {s['a']} -> {s['b']})"
            return base + ".", "arvore"

        # 6) fala associativa pura (o modo antigo, como último recurso)
        fala = self.toshi.respond(tokenize(entrada), n=10)
        if fala:
            return " ".join(fala), "associacao"
        return None, None


# ============================================================ SELFTEST
def _selftest():
    print("SELFTEST — pensamento arbóreo + contexto, SEM Qwen\n")
    t = Toshi()
    f = Fatos()
    f.g = {}
    ctx = MemoriaContexto(t, f)
    pens = PensadorArboreo(t, f, ctx)

    # conhecimento prévio (o que ele já viveu)
    for frase in ["gato é animal", "gato tem pelo", "gato mia",
                  "cachorro é animal", "cachorro late"]:
        e = extrair(frase)
        if e:
            f.aprender(*e)
        t.perceive(tokenize(frase))

    # agora ele LÊ um texto novo (contexto)
    doc = ctx.ler("o gato dorme no telhado e mia para a lua")
    assert doc is not None

    # 1) contexto: ele sabe que a pergunta fala do TEXTO
    d, s = ctx.relevante("o que você achou do texto?")
    assert d is not None, "não achou o contexto do texto"

    # 2) árvore: vários caminhos possíveis
    caminhos = pens.pensar("o que o gato faz?", doc=d)
    print("  árvore de pensamento:")
    for c in caminhos:
        print("    " + " -> ".join(c))
    assert len(caminhos) >= 2, "a árvore não abriu ramos"
    assert all(len(c) >= 2 for c in caminhos), "os caminhos não têm profundidade"

    # 3) síntese de contexto: liga o que viveu com o que já sabia
    resp, origem = pens.responder("o que você achou do texto?")
    print(f"\n  você> o que você achou do texto?")
    print(f"  toshi> {resp}")
    assert origem == "contexto"
    assert "gato" in resp and ("telhado" in resp or "lua" in resp or "mia" in resp)
    assert ("animal" in resp or "pelo" in resp or "se liga" in resp), \
        "não interligou com o conhecimento prévio"

    # 4) fato crisp continua funcionando
    r, o = pens.responder("o que é gato?")
    print(f"\n  você> o que é gato?\n  toshi> {r}")
    assert "animal" in r

    # 5) O SALTO ABDUTIVO (o "LLMs can't jump"): criar ligação que não existia
    t2 = Toshi()
    f2 = Fatos()
    f2.g = {}
    for _ in range(4):
        t2.perceive(["a", "b", "c"])
        t2.perceive(["d", "b", "e"])
    ctx2 = MemoriaContexto(t2, f2)
    pens2 = PensadorArboreo(t2, f2, ctx2)
    assert "d" not in t2.assoc.get("a", {}), "a e d não deveriam estar ligados ainda"
    # observação NOVA (a e d juntos) entra como contexto, SEM virar associação direta ainda
    doc2 = {"id": 1, "texto": "a d f", "tokens": ["a", "d", "f"],
            "emb": ctx2._embed(["a", "d", "f"]), "fatos": [], "ts": 0.0,
            "origem": "selftest"}
    ctx2.docs.append(doc2)
    saltos = pens2.saltos("a d f", doc=doc2, integrar=True,
                          limiar=0.45, exigir_evidencia=True)
    print(f"\n  salto abdutivo: {saltos}")
    assert saltos, "não criou a hipótese nova"
    assert "d" in t2.assoc.get("a", {}), "não integrou a ligação nova"
    assert len(pens2.saltos_feitos) >= 1


    print("\n[selftest] ok — lê, contextualiza, abre árvore, opina e CRIA ligações novas (o salto)")


# ============================================================ CLI
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Toshi: pensamento arbóreo e contexto. Zero Qwen.")
    ap.add_argument("--pergunta", default="")
    ap.add_argument("--ler", default="", help="arquivo de texto para ele ler e contextualizar")
    ap.add_argument("--interativo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    print("carregando Toshi...", end=" ", flush=True)
    toshi, _ = build_or_load()
    fatos = Fatos()
    ctx = MemoriaContexto(toshi, fatos)
    pens = PensadorArboreo(toshi, fatos, ctx)
    print(f"pronto ({len(toshi.seen)} conceitos, {len(ctx.docs)} textos na memória de contexto).")

    if args.ler:
        texto = open(args.ler, encoding="utf-8", errors="replace").read()
        doc = ctx.ler(texto, origem=args.ler)
        print(f"li o texto ({len(doc['tokens'])} tokens, {len(doc['fatos'])} fatos). "
              "agora pergunte o que eu achei.")
        return

    def faz(pergunta, eco=True, doc_atual=None):
        r, origem = pens.responder(pergunta, doc_atual=doc_atual,
                                   aprender=(doc_atual is None))
        if eco:
            print(f"você> {pergunta}")
        tags = {"fato": "📌", "contexto": "🧠", "arvore": "🌳", "associacao": "🔗"}
        print(f"toshi> {r} {tags.get(origem, '')}" if r else "toshi> (ainda não sei ligar isso)")

    if args.interativo:
        print("modo interativo — tudo o que você disser vira experiência; "
              "ele responde ligando com o que já viveu.\n")
        while True:
            try:
                p = input("voce> ").strip()
            except (EOFError, KeyboardInterrupt):
                save_state(toshi)
                fatos.save()
                print("\n(toshi adormece; guardou o que aprendeu)")
                break
            if not p:
                continue
            if p.lower() in ("sair", "exit", "quit"):
                save_state(toshi)
                fatos.save()
                print("tchau!")
                break
            if p.startswith("/arvore "):
                pergunta = p[len("/arvore "):]
                doc, _ = ctx.relevante(pergunta)
                for c in pens.pensar(pergunta, doc=doc):
                    print("  🌳 " + " -> ".join(c))
                continue
            if p.startswith("/mapa "):
                pergunta = p[len("/mapa "):]
                doc, _ = ctx.relevante(pergunta)
                print("  🔥 mapa de calor (raio no cérebro):")
                for w, e in pens.mapa_calor(pergunta, doc=doc):
                    print(f"    {w:<14} {e:5.2f}  {'█' * min(36, int(e * 18))}")
                continue
            if p.startswith("/saltos "):
                pergunta = p[len("/saltos "):]
                doc, _ = ctx.relevante(pergunta)
                for s in pens.saltos(pergunta, doc=doc, integrar=True,
                                     limiar=0.45, exigir_evidencia=True):
                    print(f"  ⚡ salto: {s['a']} -> {s['b']} "
                          f"(score {s['score']:.2f}, via {s['comuns']})")
                continue
            if p.startswith("/texto "):
                ctx.ler(p[len("/texto "):])
                print("toshi> li o texto.")
                continue
            # QUALQUER frase vira experiência. Ele responde a partir do que acabou
            # de viver + a experiência anterior mais parecida. Sem regra de '?'.
            doc_novo = ctx.ler(p)
            if doc_novo:
                faz(p, eco=False, doc_atual=doc_novo)
            else:
                print("toshi> ...")
            continue
    elif args.pergunta:
        print(f"você> {args.pergunta}")
        faz(args.pergunta)
    else:
        print("use --pergunta, --ler arquivo.txt, --interativo ou --selftest")


if __name__ == "__main__":
    main()
