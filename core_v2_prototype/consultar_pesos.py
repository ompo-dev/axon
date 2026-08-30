"""
CONSULTAR PESOS — o Toshi navega na memória hiperdimensional do modelo absorvido.

Depois de `absorver_pesos_qwen.py`, as assinaturas de todos os tensores estão no
espaço HD do Toshi. Este módulo PERGUNTA a essa memória:
  - "qual tensor é mais parecido com X?"   (vizinhos por cosseno)
  - "quão parecidos são X e Y?"            (comparação direta)
  - "como o padrão attn_q evolui entre camadas?" (análise por padrão)
  - estatísticas gerais da memória

USO:
  python consultar_pesos.py --modelo qwen2.5:7b --semelhantes blk.0.attn_q.weight
  python consultar_pesos.py --modelo qwen2.5:7b --comparar blk.0.attn_q.weight blk.1.attn_q.weight
  python consultar_pesos.py --modelo qwen2.5:7b --padrao attn_q.weight
  python consultar_pesos.py --lista
  python consultar_pesos.py --selftest
"""
import argparse
import glob
import hashlib
import os
import pickle
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(HERE, "dados")


# ============================================================ localiza memória salva
def _san(nome):
    return re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")


def localizar_memoria(modelo=""):
    if modelo:
        alvo = os.path.join(DADOS, f"toshi_pesos_{_san(modelo)}.pkl")
        return alvo if os.path.isfile(alvo) else None
    arquivos = glob.glob(os.path.join(DADOS, "toshi_pesos_*.pkl"))
    if not arquivos:
        return None
    return max(arquivos, key=os.path.getmtime)


# ============================================================ memória de pesos
class MemoriaPesos:
    """Carrega as assinaturas HD e responde perguntas sobre o modelo absorvido."""

    def __init__(self, arquivo=None, modelo=""):
        arquivo = arquivo or localizar_memoria(modelo)
        if not arquivo or not os.path.isfile(arquivo):
            raise FileNotFoundError(
                "nenhuma memória de pesos encontrada. Rode absorver_pesos_qwen.py primeiro."
            )
        with open(arquivo, "rb") as f:
            payload = pickle.load(f)
        self._init_payload(payload)
        self.arquivo = arquivo

    @classmethod
    def from_payload(cls, payload):
        obj = cls.__new__(cls)
        obj._init_payload(payload)
        obj.arquivo = None
        return obj

    def _init_payload(self, payload):
        self.meta = payload.get("meta", {})
        self.dim = payload.get("dimensao", 0)
        self.tensores = {
            nome: np.asarray(d["vetor"], dtype=np.float32)
            for nome, d in payload.get("tensores", {}).items()
        }
        # palavras de peso que o Toshi comeu, por tensor (para o falar_pesos.py)
        self.palavras_por_tensor = {
            nome: list(d.get("palavras", []))
            for nome, d in payload.get("tensores", {}).items()
        }
        self.nomes = list(self.tensores.keys())

    @staticmethod
    def cos(a, b):
        return float(np.dot(a, b))

    def comparar(self, a, b):
        if a not in self.tensores:
            raise KeyError(f"tensor '{a}' não está na memória")
        if b not in self.tensores:
            raise KeyError(f"tensor '{b}' não está na memória")
        return self.cos(self.tensores[a], self.tensores[b])

    def semelhantes(self, nome, k=10):
        if nome not in self.tensores:
            # tenta busca parcial
            parcial = [n for n in self.nomes if nome.lower() in n.lower()]
            if parcial:
                nome = parcial[0]
            else:
                raise KeyError(f"tensor '{nome}' não está na memória")
        alvo = self.tensores[nome]
        sims = []
        for n in self.nomes:
            if n == nome:
                continue
            sims.append((n, self.cos(alvo, self.tensores[n])))
        sims.sort(key=lambda x: -x[1])
        return sims[:k]

    @staticmethod
    def _bloco(nome):
        m = re.search(r"blk\.(\d+)\.", nome)
        return int(m.group(1)) if m else -1

    def _nomes_do_padrao(self, padrao):
        nomes = [n for n in self.nomes if padrao.lower() in n.lower()]
        return sorted(nomes, key=lambda n: (self._bloco(n), n))

    def analisar_padrao(self, padrao):
        nomes = self._nomes_do_padrao(padrao)
        if not nomes:
            return None
        out = {"padrao": padrao, "n": len(nomes), "nomes": nomes,
               "consecutivas": [], "pares_todos": []}

        for a, b in zip(nomes, nomes[1:]):
            out["consecutivas"].append({
                "a": a, "b": b, "cos": round(self.comparar(a, b), 6),
            })

        amostra = nomes[: min(len(nomes), 40)]
        for i in range(len(amostra)):
            for j in range(i + 1, len(amostra)):
                out["pares_todos"].append(self.comparar(amostra[i], amostra[j]))

        cos_seq = [x["cos"] for x in out["consecutivas"]]
        out["media_consecutiva"] = float(np.mean(cos_seq)) if cos_seq else 0.0
        out["media_abs_todos"] = float(np.mean(np.abs(out["pares_todos"]))) if out["pares_todos"] else 0.0
        out["min_consecutiva"] = min(cos_seq) if cos_seq else 0.0
        out["max_consecutiva"] = max(cos_seq) if cos_seq else 0.0
        return out

    def stats(self):
        normas = [float(np.linalg.norm(v)) for v in self.tensores.values()]
        blocos = sorted({self._bloco(n) for n in self.nomes if self._bloco(n) >= 0})
        return {
            "arquivo": self.arquivo,
            "modelo": self.meta.get("modelo", "?"),
            "tensores": len(self.tensores),
            "dimensao": self.dim,
            "bytes_originais": self.meta.get("bytes", 0),
            "blocos_camadas": len(blocos),
            "norma_min": round(min(normas), 6) if normas else 0,
            "norma_max": round(max(normas), 6) if normas else 0,
        }


# ============================================================ selftest
def _selftest():
    print("SELFTEST — consulta à memória de pesos sintética\n")
    rng = np.random.default_rng(0)
    nomes = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "output_norm.weight",
    ]
    vetores = {}
    for n in nomes:
        v = rng.normal(size=256).astype(np.float32)
        vetores[n] = v / np.linalg.norm(v)
    mem = MemoriaPesos.from_payload({
        "meta": {"modelo": "sintetico", "bytes": 1234},
        "dimensao": 256,
        "tensores": {n: {"vetor": vetores[n], "bytes": 10, "tokens": []} for n in nomes},
    })

    assert abs(mem.comparar("blk.0.attn_q.weight", "blk.0.attn_q.weight") - 1.0) < 1e-5
    top = mem.semelhantes("blk.0.attn_q.weight", k=3)
    assert top and len(top) == 3
    ana = mem.analisar_padrao("attn_q.weight")
    assert ana and ana["n"] == 2
    st = mem.stats()
    assert st["tensores"] == len(nomes) and st["dimensao"] == 256

    print("  stats:", {k: st[k] for k in ("tensores", "dimensao", "blocos_camadas")})
    print("  vizinhos de blk.0.attn_q.weight:")
    for n, c in top:
        print(f"    {n:<30} cos={c:+.4f}")
    print("  padrão attn_q.weight:", {k: ana[k] for k in ("n", "media_consecutiva")})
    print("\n[selftest] ok — o Toshi consulta a memória de pesos e acha vizinhos/evolução")


# ============================================================ CLI
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Consulta a memória hiperdimensional dos pesos absorvidos.")
    ap.add_argument("--arquivo", default="", help="arquivo .pkl da memória de pesos")
    ap.add_argument("--modelo", default="", help="nome do modelo (acha o arquivo automático)")
    ap.add_argument("--semelhantes", default="", help="nome do tensor para achar vizinhos")
    ap.add_argument("--comparar", nargs=2, default=None, help="compara dois tensores")
    ap.add_argument("--padrao", default="", help="analisa evolução de um padrão (ex: attn_q.weight)")
    ap.add_argument("--lista", action="store_true", help="lista tensores da memória")
    ap.add_argument("--selftest", action="store_true", help="teste offline")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    try:
        mem = MemoriaPesos(args.arquivo or None, args.modelo)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"memória de pesos: {mem.arquivo}")
    st = mem.stats()
    print("=" * 66)
    print("ESTATÍSTICAS DA MEMÓRIA HD")
    print("=" * 66)
    for k, v in st.items():
        print(f"  {k:<18} {v}")

    if args.lista:
        print(f"\n  tensores ({len(mem.nomes)}):")
        for n in mem.nomes:
            print(f"    {n}")
    if args.semelhantes:
        print(f"\n  vizinhos de '{args.semelhantes}':")
        try:
            for n, c in mem.semelhantes(args.semelhantes, k=10):
                print(f"    {n:<55} cos={c:+.4f}")
        except KeyError as e:
            print(f"    {e}")
    if args.comparar:
        try:
            c = mem.comparar(*args.comparar)
            print(f"\n  cos({args.comparar[0]}, {args.comparar[1]}) = {c:+.6f}")
        except KeyError as e:
            print(f"    {e}")
    if args.padrao:
        ana = mem.analisar_padrao(args.padrao)
        if not ana:
            print(f"\n  padrão '{args.padrao}' não encontrado")
        else:
            print(f"\n  padrão '{ana['padrao']}': {ana['n']} tensores")
            print(f"    cosseno médio consecutivo: {ana['media_consecutiva']:+.6f}")
            print(f"    |cos| médio entre todos:   {ana['media_abs_todos']:.6f}")
            print(f"    faixa consecutiva:         [{ana['min_consecutiva']:+.6f}, "
                  f"{ana['max_consecutiva']:+.6f}]")
            print("    evolução entre camadas (primeiras):")
            for x in ana["consecutivas"][:12]:
                print(f"      {x['a']} -> {x['b']}  cos={x['cos']:+.6f}")


if __name__ == "__main__":
    main()
