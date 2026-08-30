"""
TOSHI APRENDIZ — o ciclo de aprendizado em tempo real com o Qwen.

Por que o Toshi não respondia as mesmas coisas que o Qwen?
  Absorver os PESOS é guardar o modelo como memória. Responder é COMPORTAMENTO.
  Comportamento se transfere assim: pergunta -> Toshi tenta -> não sabe -> Qwen ensina
  -> Toshi COME a resposta (perceive + settle, reação em cadeia) -> da próxima vez
  ele responde da PRÓPRIA memória, sem o Qwen.

Este módulo implementa o ciclo:
  1. tenta responder localmente (Fatos + memória de perguntas/respostas)
  2. se não souber, pergunta ao Qwen (professor)
  3. come a resposta:
       - toshi.perceive(pergunta + resposta)  -> associações/transições novas
       - toshi.settle(...)                    -> reação em cadeia nos vizinhos
       - fatos.aprender(...)                  -> memória factual crisp
  4. salva; na próxima vez a MESMA pergunta sai da memória dele, sem Qwen

USO:
  python toshi_aprendiz.py --pergunta "que cor tem o abacaxi?"
  python toshi_aprendiz.py --interativo
  python toshi_aprendiz.py --selftest
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, build_or_load, save_state, tokenize
from fatos import Fatos, _norm, extrair

REL_QA = "tem_resposta"


# ============================================================ FONTES DE CONHECIMENTO
class FonteOllama:
    """Professor: Qwen local via Ollama."""

    def __init__(self, host="http://127.0.0.1:11434", modelo="qwen2.5:7b", timeout=300):
        self.host = host.rstrip("/")
        self.modelo = modelo
        self.timeout = timeout

    def disponivel(self):
        try:
            with urllib.request.urlopen(self.host + "/api/tags", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def gerar(self, prompt, max_tokens=600):
        payload = json.dumps({
            "model": self.modelo, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.host + "/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            dados = json.loads(r.read().decode("utf-8"))
        return dados.get("response", "").strip()

    def perguntar(self, pergunta, max_tokens=220):
        prompt = (
            "Responda em português, em no máximo 3 frases curtas e diretas. "
            "Prefira frases declarativas simples (sujeito, verbo, objeto), "
            "que possam ser aprendidas como fatos.\n"
            f"Pergunta: {pergunta}"
        )
        return self.gerar(prompt, max_tokens)


class FonteMock:
    """Professor offline para selftest."""

    def __init__(self):
        self.chamadas = 0

    def disponivel(self):
        return True

    def perguntar(self, pergunta, max_tokens=220):
        self.chamadas += 1
        return "o abacaxi tem cor amarela e casca dura. o abacaxi é uma fruta tropical."


# ============================================================ APRENDIZ
class ToshiAprendiz:
    def __init__(self, fonte=None, auto_salvar=True, carregar_existente=True,
                 toshi=None, fatos=None, permitir_professor=False):
        self.fonte = fonte or FonteOllama()
        self.auto_salvar = auto_salvar
        # POR PADRÃO O QWEN FICA DESLIGADO: depois de alimentado, o Toshi responde só
        # da própria memória. O professor só é chamado se permitir_professor=True.
        self.permitir_professor = permitir_professor
        if toshi is None and carregar_existente:
            self.toshi, _ = build_or_load()
        else:
            self.toshi = toshi or Toshi()
        self.fatos = fatos or Fatos()
        self.inicio = {
            "palavras": sum(self.toshi.seen.values()),
            "conceitos": len(self.toshi.seen),
            "fatos": sum(len(v) for v in self.fatos.g.values()),
        }
        self.aprendidas = 0
        self._emb_cache = {}

    # ---------- memória local ----------
    def _qa_local(self, pergunta):
        q = _norm(pergunta)
        # exata primeiro, depois parcial (da mais recente para a mais antiga)
        for s, arestas in reversed(list(self.fatos.g.items())):
            for r, o in arestas:
                if r == REL_QA and _norm(s) == q:
                    return o
        for s, arestas in reversed(list(self.fatos.g.items())):
            for r, o in arestas:
                if r == REL_QA and (q in _norm(s) or _norm(s) in q):
                    return o
        return None

    def _emb_pergunta(self, pergunta):
        """Pergunta -> vetor no espaço de significado do Toshi (média das palavras)."""
        chave = _norm(pergunta)
        if chave in self._emb_cache:
            return self._emb_cache[chave]
        vecs = [self.toshi._emb(w) for w in tokenize(pergunta)
                if self.toshi._emb(w) is not None]
        if not vecs:
            self._emb_cache[chave] = None
            return None
        v = sum(vecs) / len(vecs)
        n = float(np.linalg.norm(v))
        self._emb_cache[chave] = (v / n).astype(float) if n > 1e-9 else None
        return self._emb_cache[chave]

    def _qa_semantica(self, pergunta):
        """Acha a pergunta comida mais PARECIDA (por significado, não por texto)."""
        qe = self._emb_pergunta(pergunta)
        if qe is None:
            return None
        melhor, melhor_sim = None, -1.0
        for s, arestas in self.fatos.g.items():
            for r, o in arestas:
                if r != REL_QA:
                    continue
                se = self._emb_pergunta(s)
                if se is None:
                    continue
                sim = float(np.dot(qe, se))
                if sim > melhor_sim:
                    melhor, melhor_sim = o, sim
        return melhor if melhor_sim >= 0.22 else None

    def responder_local(self, pergunta):
        """Responde SÓ com o que ele comeu. Nunca chama outra IA."""
        q = _norm(pergunta)
        # 1) memória exata de perguntas/respostas
        r = self._qa_local(q)
        if r:
            return r, "memoria_exata"
        # 2) fatos crisp (identidade, localização, definições, cor...)
        r = self.fatos.responder(q)
        if r:
            return r, "fatos"
        # 3) pergunta PARECIDA que ele comeu (busca por significado no espaço do Toshi)
        r = self._qa_semantica(q)
        if r:
            return r, "memoria_semantica"
        return None, None

    # ---------- comer a resposta (reação em cadeia) ----------
    def comer_resposta(self, pergunta, resposta):
        tq = tokenize(pergunta)
        ta = tokenize(resposta)
        # 1) a pergunta e a resposta viram experiência (como os livros)
        self.toshi.perceive(tq)
        self.toshi.perceive(ta)
        self.toshi.perceive(tq + ta)
        # 2) reação em cadeia: os vizinhos re-equilibram juntos
        self.toshi.settle(tq[:6] + ta[:6])
        # 3) cada frase declarativa da resposta vira fato crisp
        for frase in re.split(r"[.;\n]+", resposta):
            e = extrair(frase)
            if e:
                self.fatos.aprender(*e)
        # 4) a pergunta inteira vira memória de resposta (aprende em tempo real)
        self.fatos.aprender(_norm(pergunta), REL_QA, resposta)
        self.aprendidas += 1
        if self.auto_salvar:
            save_state(self.toshi)
            self.fatos.save()

    # ---------- ciclo completo ----------
    def responder(self, pergunta):
        local, origem = self.responder_local(pergunta)
        if local:
            return local, origem, False   # respondeu sozinho, SEM outra IA

        # POR PADRÃO NÃO CHAMA O QWEN. Se ele não comeu, ele diz que não comeu.
        if not self.permitir_professor:
            return None, "nao_comi", False

        if not self.fonte.disponivel():
            return None, "sem_professor", False

        resposta = self.fonte.perguntar(pergunta)
        if not resposta:
            return None, "sem_resposta", False

        self.comer_resposta(pergunta, resposta)
        return resposta, "qwen_agora", True

    def stats(self):
        return {
            "palavras_vividas": sum(self.toshi.seen.values()),
            "conceitos": len(self.toshi.seen),
            "fatos": sum(len(v) for v in self.fatos.g.values()),
            "respostas_aprendidas": self.aprendidas,
            "crescimento_palavras": sum(self.toshi.seen.values()) - self.inicio["palavras"],
            "crescimento_conceitos": len(self.toshi.seen) - self.inicio["conceitos"],
        }


# ============================================================ SELFTEST
def _selftest():
    print("SELFTEST — o Toshi responde SÓ do que comeu (Qwen desligado)\n")
    fonte = FonteMock()
    f = Fatos()
    f.g = {}
    t = Toshi()
    ap = ToshiAprendiz(fonte=fonte, auto_salvar=False, carregar_existente=False,
                       toshi=t, fatos=f, permitir_professor=False)

    # simula a alimentação que o cozinhar_qwen.py faz em lote
    ap.comer_resposta(
        "que cor tem o abacaxi?",
        "o abacaxi tem cor amarela e casca dura. o abacaxi é uma fruta tropical."
    )

    # 1) MESMA pergunta -> memória exata, professor NUNCA chamado
    r1, o1, usou1 = ap.responder("que cor tem o abacaxi?")
    print(f"  1) exata:    {r1[:60]}...  [origem={o1}]")
    assert o1 == "memoria_exata" and usou1 is False

    # 2) pergunta PARECIDA -> memória semântica (busca por significado no espaço do Toshi)
    r2, o2, usou2 = ap.responder("que cor tem um abacaxi?")
    print(f"  2) parecida: {r2[:60]}...  [origem={o2}]")
    assert o2 == "memoria_semantica" and usou2 is False

    # 3) fato novo criado pela reação em cadeia
    r3 = ap.fatos.responder("o que é o abacaxi?")
    print(f"  3) fato:     {r3}")
    assert r3 and "fruta" in r3

    # 4) QWEN NUNCA FOI CHAMADO
    assert fonte.chamadas == 0, "o Qwen não pode ser chamado depois de alimentado!"

    print(f"\n  stats: {ap.stats()}")
    print("\n[selftest] ok — 100% memória própria; o Qwen ficou DESLIGADO")


# ============================================================ CLI
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Toshi responde só da memória; Qwen desligado.")
    ap.add_argument("--pergunta", default="")
    ap.add_argument("--modelo", default="qwen2.5:7b")
    ap.add_argument("--interativo", action="store_true")
    ap.add_argument("--com-professor", action="store_true",
                    help="PERMITE chamar o Qwen quando ele não souber (modo professor)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    aprendiz = ToshiAprendiz(
        fonte=FonteOllama(modelo=args.modelo),
        permitir_professor=args.com_professor,
    )

    def faz(pergunta, eco=True):
        t0 = time.time()
        resposta, origem, usou = aprendiz.responder(pergunta)
        if eco:
            print(f"você> {pergunta}")
        if resposta:
            tag = {
                "memoria_exata": "🧠 (memória exata)",
                "memoria_semantica": "✨ (memória por significado)",
                "fatos": "📌 (fato aprendido)",
                "qwen_agora": f"🦉 (aprendi agora do Qwen em {time.time()-t0:.1f}s)",
            }.get(origem, "")
            print(f"toshi> {resposta} {tag}")
        else:
            print("toshi> (ainda não comi isso — o Qwen está desligado)")

    if args.interativo:
        if aprendiz.permitir_professor:
            print("modo interativo — responde da memória; se não souber, o Qwen ensina.\n")
        else:
            print("modo interativo — 100% memória própria. O Qwen está DESLIGADO.\n")
        while True:
            try:
                p = input("voce> ").strip()
            except (EOFError, KeyboardInterrupt):
                save_state(aprendiz.toshi)
                aprendiz.fatos.save()
                print("\n(toshi adormece; guardou o que aprendeu)")
                break
            if not p:
                continue
            if p.lower() in ("sair", "exit", "quit"):
                save_state(aprendiz.toshi)
                aprendiz.fatos.save()
                print("tchau!")
                break
            if p.lower() in ("/stats", "stats"):
                print(f"  {aprendiz.stats()}")
                continue
            # declaração direta também é aprendida na hora
            e = extrair(re.sub(r"^(oi|ola|olá)[,\s]+", "", p, flags=re.I))
            if e:
                aprendiz.fatos.aprender(*e)
                aprendiz.toshi.perceive(tokenize(p))
                aprendiz.toshi.settle(tokenize(p)[:6])
                print(f"toshi> (aprendi: {e[0]} {e[1]} {e[2]})")
                continue
            faz(p, eco=False)
    elif args.pergunta:
        faz(args.pergunta)
    else:
        print("use --pergunta \"...\" ou --interativo")


if __name__ == "__main__":
    main()
