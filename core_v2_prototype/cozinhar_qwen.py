"""
COZINHAR QWEN — refeição GIGANTE de conhecimento para o Toshi (uma vez só).

O Qwen é usado SOMENTE como fonte de comida, em lote, ANTES da conversa:
  1. pede perguntas e respostas curtas sobre um tema;
  2. pede subtemas e expande em árvore (para cobrir muito mais);
  3. o Toshi COME tudo: percebe pergunta+resposta, faz settle (reação em cadeia),
     extrai fatos e guarda a pergunta na memória de respostas;
  4. salva. Na conversa, ele responde da própria memória — Qwen desligado.

Quanto maior a refeição, menos "burro". Comece com 20-30 temas e aumente.

USO:
  python cozinhar_qwen.py --temas "ciencia,historia,geografia,tecnologia" --pares 20
  python cozinhar_qwen.py --temas "programacao,ia,matematica" --pares 25 --profundidade 2 --max-temas 50
  python cozinhar_qwen.py --selftest
"""
import argparse
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, build_or_load, save_state
from fatos import Fatos, _norm
from toshi_aprendiz import ToshiAprendiz, FonteOllama

_RE_PAR = re.compile(
    r"^\s*(?:pergunta|p|q)\s*:\s*(.+?)\s*\|\s*(?:resposta|r|a)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


def extrair_pares(texto):
    pares = []
    for linha in texto.splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", linha).strip()
        m = _RE_PAR.match(s)
        if m:
            p = m.group(1).strip().strip('"')
            r = m.group(2).strip().strip('"')
            if p and r:
                pares.append((p, r))
    return pares


class Cozinheiro:
    def __init__(self, fonte=None, auto_salvar=True):
        self.fonte = fonte or FonteOllama()
        self.auto_salvar = auto_salvar
        self.aprendiz = ToshiAprendiz(fonte=self.fonte, auto_salvar=False,
                                      permitir_professor=False)
        self.total = 0

    def _gerar(self, prompt, max_tokens=900):
        return self.fonte.gerar(prompt, max_tokens)

    def pares_do_tema(self, tema, n):
        prompt = (
            f"Gere {n} pares de pergunta e resposta CURTA sobre '{tema}', em português.\n"
            "Formato EXATO, um par por linha:\n"
            "PERGUNTA: <pergunta> | RESPOSTA: <resposta>\n"
            "Sem numeração, sem markdown, sem texto extra."
        )
        return extrair_pares(self._gerar(prompt))

    def subtemas(self, tema, n=6):
        prompt = (
            f"Liste {n} subtemas importantes de '{tema}', um por linha, "
            "cada um com no máximo 3 palavras. Sem numeração, sem markdown."
        )
        texto = self._gerar(prompt, 256)
        subs = []
        for linha in texto.splitlines():
            s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", linha).strip()
            if s and 1 <= len(s.split()) <= 3:
                subs.append(s)
        return subs

    def cozinhar_tema(self, tema, n):
        pares = self.pares_do_tema(tema, n)
        for p, r in pares:
            self.aprendiz.comer_resposta(p, r)
            self.total += 1
        print(f"  {tema}: {len(pares)} pares comidos")
        return len(pares)

    def cozinhar_arvore(self, temas, n_pares, profundidade=1, max_temas=40):
        fila = [(t, 0) for t in temas]
        visitados = set()
        feitos = 0
        while fila and feitos < max_temas:
            tema, prof = fila.pop(0)
            chave = _norm(tema)
            if chave in visitados or not tema.strip():
                continue
            visitados.add(chave)
            feitos += 1
            print(f"\n({feitos}/{max_temas}) tema: {tema} (prof {prof})")
            self.cozinhar_tema(tema, n_pares)
            if prof < profundidade:
                subs = self.subtemas(tema)
                print(f"    subtemas: {', '.join(subs[:6])}")
                for st in subs[:6]:
                    if _norm(st) not in visitados:
                        fila.append((st, prof + 1))
            if self.auto_salvar:
                self.salvar()
        self.salvar()
        print(f"\nrefeição completa: {feitos} temas, {self.total} pares no total")
        return self.total

    def salvar(self):
        save_state(self.aprendiz.toshi)
        self.aprendiz.fatos.save()
        print("  [salvo] toshi_state.pkl + fatos.json")

    def testar_memoria(self):
        """Depois de comer, responde sozinho às perguntas que comeu. Qwen fica de fora."""
        n = ok = 0
        for s, arestas in self.aprendiz.fatos.g.items():
            for r, o in arestas:
                if r == "tem_resposta":
                    n += 1
                    local, _ = self.aprendiz.responder_local(s)
                    if local:
                        ok += 1
        pct = (100.0 * ok / n) if n else 0.0
        print(f"teste da memória própria: {ok}/{n} perguntas respondidas ({pct:.0f}%)")
        return pct


class FonteMock:
    def __init__(self):
        self.chamadas = 0

    def disponivel(self):
        return True

    def gerar(self, prompt, max_tokens=900):
        self.chamadas += 1
        if "PERGUNTA:" in prompt:
            return ("PERGUNTA: o que é um cachorro? | RESPOSTA: o cachorro é um animal doméstico.\n"
                    "PERGUNTA: o que o gato faz? | RESPOSTA: o gato mia e caça ratos.")
        if "subtemas" in prompt:
            return "mamiferos\nanimais domesticos"
        return "mock"


def _selftest():
    print("SELFTEST — cozinhar sem rede\n")
    f = Fatos()
    f.g = {}
    t = Toshi()
    fonte = FonteMock()
    coz = Cozinheiro(fonte=fonte, auto_salvar=False)
    coz.aprendiz.toshi = t
    coz.aprendiz.fatos = f
    coz.cozinhar_tema("animais", 2)
    assert coz.total == 2
    assert coz.aprendiz.responder_local("o que é um cachorro?")[0] is not None
    assert coz.aprendiz.responder_local("o que o gato faz?")[0] is not None
    assert fonte.chamadas >= 1
    print("\n[selftest] ok — a comida entrou e a memória responde sozinha")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Cozinha conhecimento do Qwen em lote para o Toshi.")
    ap.add_argument("--temas", default="ciencia,historia,geografia,tecnologia,programacao,ia")
    ap.add_argument("--pares", type=int, default=20)
    ap.add_argument("--profundidade", type=int, default=1)
    ap.add_argument("--max-temas", type=int, default=40)
    ap.add_argument("--modelo", default="qwen2.5:7b")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    fonte = FonteOllama(modelo=args.modelo)
    if not fonte.disponivel():
        print("Ollama não está rodando. Inicie: ollama serve")
        return

    temas = [x.strip() for x in args.temas.split(",") if x.strip()]
    coz = Cozinheiro(fonte=fonte)
    t0 = time.time()
    coz.cozinhar_arvore(temas, args.pares, args.profundidade, args.max_temas)
    coz.testar_memoria()
    print(f"tempo total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
