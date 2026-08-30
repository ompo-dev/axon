"""
COMER WIKIPÉDIA PT-BR — enxame de sub-Toshi varrendo a Wikipédia inteira.

Cada sub-Toshi (worker) é um Toshi independente que:
  - baixa páginas da Wikipédia pt-br (texto + links + imagens);
  - COME o texto em blocos (toshi.perceive) -> associações, transições, embeddings;
  - COME os títulos dos links e das imagens (multimodal no substrato atual:
    títulos/legendas viram conceitos; pixels são o próximo encoder);
  - guarda tudo num shard próprio (dados/wiki/shard_*.pkl);
  - escreve um índice pesquisável (dados/wiki/indice.jsonl).

Depois da varredura, use consultar_wiki.py para perguntar ao Toshi o que ele comeu.

HONESTO SOBRE ESCALA: a pt.wikipedia tem ~1,1 milhão de artigos. Este enxame é
projetado para rodar por horas/dias com checkpoint. Faça primeiro um voo de teste
(--max-paginas 200) e depois suba a meta.

USO:
  python comer_wikipedia.py --workers 8 --max-paginas 200
  python comer_wikipedia.py --workers 16 --max-paginas 5000 --continuar
"""
import argparse
import glob
import json
import multiprocessing
import os
import queue
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, tokenize
from fatos import extrair

WIKI_DIR = os.path.join(HERE, "dados", "wiki")
INDICE = os.path.join(WIKI_DIR, "indice.jsonl")
PROGRESSO = os.path.join(WIKI_DIR, "progresso.json")
API = "https://pt.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "ToshiAprendiz/0.1 (aprendizado continuo; contato: local)"}


# ============================================================ ACESSO À WIKIPÉDIA
_ULTIMO_REQ = [0.0]
_REQ_LOCK = threading.Lock()


def _get(params, timeout=60):
    """Uma requisição por vez, com intervalo mínimo e retry em 429."""
    with _REQ_LOCK:
        espera = 0.2 - (time.time() - _ULTIMO_REQ[0])
        if espera > 0:
            time.sleep(espera)
        _ULTIMO_REQ[0] = time.time()

    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    ultimo_erro = None
    for tentativa in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            ultimo_erro = e
            if e.code == 429 and tentativa < 3:
                time.sleep(8 * (tentativa + 1))
                continue
            raise
    raise ultimo_erro


def obter_lote(gapcontinue=None, gapfrom=None, lote=10, rapido=False):
    """
    UM único pedido traz UM LOTE de artigos.
    modo normal: texto completo + links + imagens (pesado)
    modo RÁPIDO: só a introdução (exintro), sem links/imagens — muito mais leve.
    Continua por gapcontinue OU, se a API não devolver, por gapfrom (último título).
    """
    params = {
        "action": "query", "format": "json", "generator": "allpages",
        "gapnamespace": 0, "gaplimit": lote,
        "prop": "extracts", "explaintext": 1,
    }
    if rapido:
        params["exintro"] = 1
    else:
        params["prop"] = "extracts|links|images"
        params["pllimit"] = "max"
        params["imlimit"] = "max"
    if gapcontinue:
        params["gapcontinue"] = gapcontinue
    if gapfrom:
        params["gapfrom"] = gapfrom
    dados = _get(params)
    paginas = []
    for pid, p in dados.get("query", {}).get("pages", {}).items():
        paginas.append({
            "titulo": p.get("title", ""),
            "texto": p.get("extract", ""),
            "links": [l.get("title", "") for l in p.get("links", [])],
            "imagens": [i.get("title", "") for i in p.get("images", [])],
        })
    continuar = dados.get("continue", {}).get("gapcontinue")
    return paginas, continuar


def carregar_comidos(indice=INDICE):
    """Títulos que o enxame JÁ comeu (para nunca repetir artigo)."""
    comidos = set()
    if not os.path.isfile(indice):
        return comidos
    with open(indice, encoding="utf-8") as f:
        for linha in f:
            try:
                comidos.add(json.loads(linha).get("titulo", ""))
            except Exception:
                pass
    return comidos


# ============================================================ SUB-TOSHI
class SubToshi:
    """Um Toshi independente = um sub-agente do enxame."""

    def __init__(self, shard_id):
        self.id = shard_id
        self.toshi = Toshi()
        self.paginas = 0
        self.tokens = 0
        # CONTINUIDADE: se o shard já existia, carrega e continua aprendendo nele
        path = os.path.join(WIKI_DIR, f"shard_{shard_id}.pkl")
        if os.path.exists(path):
            try:
                import pickle as _pickle
                with open(path, "rb") as f:
                    d = _pickle.load(f)
                self.toshi.assoc = d.get("assoc", {})
                self.toshi.after = d.get("after", {})
                self.toshi.seen = d.get("seen", Counter())
                self.toshi.index = d.get("index", {})
                self.toshi.embed = d.get("embed", {})
                self.paginas = d.get("paginas", 0)
                self.tokens = d.get("tokens", 0)
            except Exception:
                pass

    def comer_texto(self, texto):
        palavras = tokenize(texto)
        # come em blocos (como toshi.eat, sem estourar a janela)
        for i in range(0, len(palavras), 2000):
            bloco = palavras[i:i + 2000]
            self.toshi.perceive(bloco)
        self.tokens += len(palavras)

    def comer_pagina(self, pagina):
        self.comer_texto(pagina["texto"])
        # links e imagens também viram experiência (multimodal de títulos)
        self.comer_texto(" ".join(pagina["links"][:80]))
        self.comer_texto(" ".join(pagina["imagens"][:40]))
        self.paginas += 1

    def salvar(self):
        path = os.path.join(WIKI_DIR, f"shard_{self.id}.pkl")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            import pickle
            pickle.dump({
                "paginas": self.paginas,
                "tokens": self.tokens,
                "assoc": self.toshi.assoc,
                "after": self.toshi.after,
                "seen": self.toshi.seen,
                "index": self.toshi.index,
                "embed": self.toshi.embed,
            }, f)
        os.replace(tmp, path)
        return path
def _trabalhador_processo(shard_id, fila, parar, indice_shard):
    """Sub-Toshi em PROCESSO próprio (paralelismo real, sem GIL)."""
    sub = SubToshi(shard_id)
    while not parar.is_set():
        try:
            pagina = fila.get(timeout=2)
        except Exception:
            continue
        if pagina is None:
            break
        try:
            sub.comer_pagina(pagina)
            palavras = tokenize(pagina["texto"])
            cont = Counter(w for w in palavras if len(w) > 2)
            entrada = {
                "titulo": pagina["titulo"],
                "resumo": pagina["texto"][:1200],
                "palavras": [w for w, _ in cont.most_common(60)],
                "links": pagina["links"][:60],
                "imagens": pagina["imagens"][:30],
            }
            with open(indice_shard, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            if sub.paginas % 25 == 0:
                print(f"  [p{shard_id}] {sub.paginas} páginas "
                      f"(última: {pagina['titulo']})", flush=True)
        except Exception as e:
            print(f"  [!p{shard_id}] erro: {e}", flush=True)
        finally:
            fila.task_done()
    sub.salvar()
    print(f"  [p{shard_id}] shard salvo: {sub.paginas} páginas, "
          f"{sub.tokens} tokens", flush=True)


# ============================================================ ENXAME
class EnxameWiki:
    def __init__(self, workers=8):
        os.makedirs(WIKI_DIR, exist_ok=True)
        self.workers = workers
        self.fila = queue.Queue(maxsize=workers * 4)
        self.lock = threading.Lock()
        self.progress_lock = threading.Lock()
        self.indice_f = open(INDICE, "a", encoding="utf-8")
        self.stats = {"paginas": 0, "erros": 0, "tokens": 0}
        self.parar = threading.Event()
        self.progresso = self._carregar_progresso()

    def _carregar_progresso(self):
        try:
            with open(PROGRESSO, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"gapcontinue": None, "gapfrom": None, "paginas": 0, "ultimo": None}

    def _salvar_progresso(self, gapcontinue, ultimo, gapfrom=None):
        payload = {"gapcontinue": gapcontinue, "gapfrom": gapfrom,
                   "paginas": self.stats["paginas"], "ultimo": ultimo}
        with self.progress_lock:
            for tentativa in range(3):
                tmp = PROGRESSO + f".{os.getpid()}.{tentativa}.tmp"
                try:
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(payload, f)
                    os.replace(tmp, PROGRESSO)
                    return
                except PermissionError:
                    time.sleep(1)
            # último recurso: gravação direta (sem replace atômico)
            try:
                with open(PROGRESSO, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            except Exception as e:
                print(f"  [!] checkpoint falhou (não é fatal): {e}")

    def _indexar(self, pagina):
        palavras = tokenize(pagina["texto"])
        cont = Counter(w for w in palavras if len(w) > 2)
        entrada = {
            "titulo": pagina["titulo"],
            "resumo": pagina["texto"][:1200],
            "palavras": [w for w, _ in cont.most_common(60)],
            "links": pagina["links"][:60],
            "imagens": pagina["imagens"][:30],
        }
        with self.lock:
            self.indice_f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            self.indice_f.flush()

    def _worker(self, shard_id):
        sub = SubToshi(shard_id)
        while not self.parar.is_set():
            try:
                pagina = self.fila.get(timeout=2)
            except queue.Empty:
                continue  # continua vivo esperando o coletor trazer mais comida
            try:
                sub.comer_pagina(pagina)
                self._indexar(pagina)
                with self.lock:
                    self.stats["paginas"] += 1
                    self.stats["tokens"] += len(tokenize(pagina["texto"]))
                    n = self.stats["paginas"]
                if n % 25 == 0:
                    print(f"  [{shard_id}] {n} páginas comidas "
                          f"(última: {pagina['titulo']})", flush=True)
            except Exception as e:
                with self.lock:
                    self.stats["erros"] += 1
                    if self.stats["erros"] <= 5:
                        print(f"  [!] erro em '{pagina.get('titulo', '?')}': {e}",
                              flush=True)
            finally:
                self.fila.task_done()
        try:
            sub.salvar()
            print(f"  [shard {shard_id}] salvo: {sub.paginas} páginas, "
                  f"{sub.tokens} tokens")
        except Exception as e:
            print(f"  [!] falha ao salvar shard {shard_id}: {e}")

    def reconstruir_shards(self, workers=8):
        """Reconstrói os cérebros dos sub-Toshi a partir do índice já comido.
        Sem internet, sem repetir artigo — usa o resumo que ficou no indice.jsonl."""
        entradas = []
        try:
            with open(INDICE, encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if linha:
                        try:
                            entradas.append(json.loads(linha))
                        except Exception:
                            pass
        except Exception:
            pass
        if not entradas:
            print("índice vazio; rode o enxame antes.")
            return

        print(f"reconstruindo {len(entradas)} artigos nos shards "
              f"com {workers} sub-Toshi...")
        subs = {i: SubToshi(i) for i in range(workers)}
        t0 = time.time()
        for idx, e in enumerate(entradas):
            sub = subs[idx % workers]
            sub.comer_pagina({
                "titulo": e.get("titulo", ""),
                "texto": e.get("resumo", ""),
                "links": e.get("links", []),
                "imagens": e.get("imagens", []),
            })
            if (idx + 1) % 500 == 0:
                print(f"  {idx + 1}/{len(entradas)} artigos re-comidos", flush=True)
        for sub in subs.values():
            sub.salvar()
        print(f"reconstrução concluída em {time.time() - t0:.1f}s")
    def _mesclar_indices_processos(self):
        """Junta os indice_shard_*.jsonl dos processos no índice oficial."""
        partes = glob.glob(os.path.join(WIKI_DIR, "indice_shard_*.jsonl"))
        if not partes:
            return
        with open(INDICE, "a", encoding="utf-8") as destino:
            for parte in partes:
                try:
                    with open(parte, encoding="utf-8") as origem:
                        shutil.copyfileobj(origem, destino)
                    os.remove(parte)
                except Exception as e:
                    print(f"  [!] falha ao mesclar {parte}: {e}")

    def comer(self, max_paginas, continuar=False, inicio=None, rapido=False, lote=None,
              processos=False):
        if lote is None:
            lote = 500 if rapido else 10
        # artigos que ele JÁ comeu: nunca repetir
        self.comidos = carregar_comidos()
        print(f"já comidos antes: {len(self.comidos)} artigos (não serão repetidos)")

        gapcontinue = self.progresso.get("gapcontinue") if continuar else None
        gapfrom = self.progresso.get("gapfrom") if continuar else None
        if inicio and not continuar:
            gapfrom = inicio
        print(f"continuando de: gapcontinue={gapcontinue or '—'} "
              f"gapfrom={gapfrom or '—'}")

        threads, processos_vivos = [], []
        if processos:
            # paralelismo REAL: cada sub-Toshi em um processo separado
            self.fila = multiprocessing.JoinableQueue(maxsize=self.workers * 4)
            self.parar = multiprocessing.Event()
            for i in range(self.workers):
                indice_shard = os.path.join(WIKI_DIR, f"indice_shard_{i}.jsonl")
                p = multiprocessing.Process(
                    target=_trabalhador_processo,
                    args=(i, self.fila, self.parar, indice_shard),
                    daemon=True,
                )
                p.start()
                processos_vivos.append(p)
        else:
            for i in range(self.workers):
                th = threading.Thread(target=self._worker, args=(i,), daemon=True)
                th.start()
                threads.append(th)

        t0 = time.time()
        servidos = 0
        while servidos < max_paginas:
            try:
                paginas, prox_gap = obter_lote(gapcontinue, gapfrom,
                                               lote=lote, rapido=rapido)
            except Exception as e:
                print(f"  [!] falha no lote (gap={gapcontinue}): {e}")
                time.sleep(8)
                continue
            if not paginas:
                print("  fim da Wikipédia alcançado (ou nada novo neste trecho).")
                break

            ultimo_titulo = paginas[-1].get("titulo", "")
            for pagina in paginas:
                titulo = pagina.get("titulo", "")
                if not titulo or titulo in self.comidos:
                    continue
                self.comidos.add(titulo)
                self.fila.put(pagina)
                servidos += 1
                if servidos >= max_paginas:
                    break
            print(f"  lote: +{len(paginas)} baixados; fila={self.fila.qsize()}; "
                  f"servidos={servidos}/{max_paginas}; prox_gap={prox_gap or '—'}",
                  flush=True)

            # plano A: continuação oficial; plano B: recomeça após o último título
            if prox_gap:
                gapcontinue, gapfrom = prox_gap, None
            else:
                gapcontinue, gapfrom = None, ultimo_titulo

            # checkpoint após cada lote (thread-safe; main thread)
            self._salvar_progresso(gapcontinue, ultimo_titulo, gapfrom=gapfrom)

        self.fila.join()
        self.parar.set()
        if processos:
            for p in processos_vivos:
                p.join(timeout=15)
            self._mesclar_indices_processos()
        else:
            for th in threads:
                th.join(timeout=5)
        self.indice_f.close()
        self.stats["paginas"] = servidos

        print("\n" + "=" * 66)
        print("ENXAME TERMINOU")
        print("=" * 66)
        print(f"  artigos NOVOS comidos: {self.stats['paginas']}")
        print(f"  erros:                 {self.stats['erros']}")
        print(f"  tempo:                 {time.time()-t0:.0f}s")
        print(f"  shards:                {WIKI_DIR}\\shard_*.pkl")
        print(f"  índice:                {INDICE}")
        self._salvar_progresso(gapcontinue, self.progresso.get("ultimo"), gapfrom=gapfrom)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Enxame de sub-Toshi comendo a Wikipédia pt-br.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-paginas", type=int, default=200)
    ap.add_argument("--continuar", action="store_true")
    ap.add_argument("--inicio", default="",
                    help="começa a partir deste título (ex.: Asteroide, Buraco negro, Brasil)")
    ap.add_argument("--rapido", action="store_true",
                    help="modo rápido: só a introdução de cada artigo, lotes de 200")
    ap.add_argument("--processos", action="store_true",
                    help="usa PROCESSOS (paralelismo real, sem GIL) em vez de threads")
    ap.add_argument("--lote", type=int, default=0,
                    help="tamanho do lote (0 = automático)")
    ap.add_argument("--reconstruir", action="store_true",
                    help="reconstrói os shards a partir do índice já comido (sem rede)")
    args = ap.parse_args()

    enxame = EnxameWiki(args.workers)
    if args.reconstruir:
        enxame.reconstruir_shards(args.workers)
        return

    print(f"enxame com {args.workers} sub-Toshi; meta: {args.max_paginas} páginas; "
          f"modo={'rápido' if args.rapido else 'normal'}; "
          f"paralelismo={'processos' if args.processos else 'threads'}\n")
    enxame.comer(args.max_paginas, args.continuar, args.inicio or None,
                 rapido=args.rapido, lote=args.lote or None,
                 processos=args.processos)


if __name__ == "__main__":
    main()
