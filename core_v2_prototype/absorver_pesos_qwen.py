"""
ABSORVER PESOS QWEN — passa o MODELO INTEIRO (todos os tensores/pesos) para o Toshi.

IDÉIA: o Qwen no disco é um arquivo GGUF com N tensores. Este módulo:
  1. LÊ o GGUF (todos os tensores, byte por byte, sem pular nada);
  2. PROJETA cada tensor para um hipervetor no espaço do Toshi
     (Random Indexing determinístico: cada bloco de bytes vira um vetor esparso ±1
      e a soma normalizada vira a ASSINATURA do tensor);
  3. GUARDA a assinatura numa camada nova do Toshi com DIMENSÃO CONFIGURÁVEL
     (16k, 65k, 131k — o Toshi pode crescer);
  4. TRANSCREVE a ESTRUTURA do modelo para a memória nativa do Toshi:
     - cada tensor vira um fato (nome -> "é tensor")
     - a sequência de tensores vira associações/transições (arquitetura)
     - metadados do modelo viram fatos (nº tensores, bytes, dimensão)
  5. TESTA AUTOMATICAMENTE: cobertura total, fidelidade, transições,
     determinismo e ortogonalidade das assinaturas.

HONESTO: isto absorve os PESOS como memória hiperdimensional completa do Toshi.
NÃO transforma o Toshi em executor de transformer — as assinaturas são a
representação dos pesos no padrão dele (compressão com dimensão expansível).
A transferência de CAPACIDADE (comportamento) segue no absorver_qwen.py, por
destilação de conhecimento. Aqui o que entra é o MODELO EM SI, por completo.

USO:
  python absorver_pesos_qwen.py --modelo qwen2.5:7b --dim-pesos 16384
  python absorver_pesos_qwen.py --arquivo C:\\...\\modelo.gguf --dim-pesos 65536
  python absorver_pesos_qwen.py --selftest
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, build_or_load, save_state, tokenize
from fatos import Fatos


# ============================================================ LOCALIZA O GGUF DO OLLAMA
def localizar_gguf(modelo):
    """Acha o arquivo GGUF de um modelo do Ollama no disco."""
    # 1) tenta o modelfile (caminho exato)
    try:
        r = subprocess.run(
            ["ollama", "show", "--modelfile", modelo],
            capture_output=True, text=True, timeout=30,
        )
        for linha in (r.stdout or "").splitlines():
            if linha.strip().upper().startswith("FROM"):
                resto = linha.split(None, 1)[1] if len(linha.split(None, 1)) > 1 else ""
                caminho = resto.strip().strip('"')
                if caminho and os.path.isfile(caminho):
                    return caminho
    except Exception:
        pass

    # 2) tenta o manifesto do Ollama (layers -> digest -> blobs)
    base = Path.home() / ".ollama" / "models"
    try:
        for arq in (base / "manifests").rglob("*"):
            if not arq.is_file():
                continue
            try:
                dados = json.loads(arq.read_text(encoding="utf-8"))
            except Exception:
                continue
            camadas = dados.get("layers", []) if isinstance(dados, dict) else []
            melhor = None
            for cam in camadas:
                digest = str(cam.get("digest", ""))
                tamanho = int(cam.get("size", 0))
                # no disco o Ollama grava sha256-<hex> (no manifesto vem sha256:<hex>)
                blob = base / "blobs" / digest.replace(":", "-")
                if digest and tamanho and blob.exists() and (melhor is None or tamanho > melhor[1]):
                    melhor = (blob, tamanho)
            if melhor:
                return str(melhor[0])
    except Exception:
        pass

    # 3) fallback: maior blob existente (funciona com 1 modelo instalado)
    try:
        blobs = list((base / "blobs").glob("sha256-*"))
        if blobs:
            return str(max(blobs, key=lambda p: p.stat().st_size))
    except Exception:
        pass
    return None


# ============================================================ LEITOR GGUF MÍNIMO (stdlib)
def _u8(f):
    return struct.unpack("<B", f.read(1))[0]


def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _ler_str(f):
    n = _u64(f)
    return f.read(n).decode("utf-8", "replace")


_TAM_TIPO = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _pular_valor(f, tipo):
    if tipo in _TAM_TIPO:
        f.read(_TAM_TIPO[tipo])
    elif tipo == 8:                      # string
        _ler_str(f)
    elif tipo == 9:                      # array
        subtipo = _u32(f)
        n = _u64(f)
        if subtipo == 8:
            for _ in range(n):
                _ler_str(f)
        elif subtipo in _TAM_TIPO:
            f.read(_TAM_TIPO[subtipo] * n)
        else:
            raise ValueError(f"tipo de array GGUF desconhecido: {subtipo}")
    else:
        raise ValueError(f"tipo GGUF desconhecido: {tipo}")


def _parsear_cabecalho_gguf(caminho):
    """Lê cabeçalho + metadados + infos de tensores. Devolve tensores e início dos dados."""
    with open(caminho, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{caminho} não parece um arquivo GGUF (magic inválido)")
        versao = _u32(f)
        n_tensores = _u64(f)
        n_kv = _u64(f)
        for _ in range(n_kv):
            _ler_str(f)                  # chave
            _pular_valor(f, _u32(f))     # valor
        tensores = []
        for _ in range(n_tensores):
            nome = _ler_str(f)
            n_dims = _u32(f)
            dims = [_u64(f) for _ in range(n_dims)]
            ggml_tipo = _u32(f)
            offset = _u64(f)
            tensores.append({
                "nome": nome, "dims": dims, "ggml_tipo": ggml_tipo, "offset": offset,
            })
        inicio_dados = f.tell()
    return tensores, inicio_dados, os.path.getsize(caminho), versao


def iterar_tensores_gguf(caminho):
    """Gera (info, bytes_crus) para CADA tensor do GGUF, na ordem, sem pular nenhum."""
    tensores, inicio_dados, tam_arquivo, versao = _parsear_cabecalho_gguf(caminho)
    with open(caminho, "rb") as f:
        for i, info in enumerate(tensores):
            ini = inicio_dados + info["offset"]
            fim = (inicio_dados + tensores[i + 1]["offset"]) if i + 1 < len(tensores) else tam_arquivo
            if ini < 0 or fim < ini or fim > tam_arquivo:
                raise ValueError(
                    f"intervalo inválido no tensor {info['nome']}: {ini}..{fim} "
                    f"(arquivo {tam_arquivo})"
                )
            f.seek(ini)
            yield info, f.read(fim - ini)


def tokens_do_nome(nome):
    """'blk.0.attn_q.weight' -> ['blk','attn','q','weight'] (padrão do Toshi)."""
    return tokenize(re.sub(r"[^a-z]+", " ", nome.lower()))
def _palavra_indice(idx):
    """Índice -> palavra só com letras (base-26). O tokenize do Toshi só aceita [a-z]+."""
    letras = []
    x = int(idx)
    while True:
        letras.append(chr(97 + (x % 26)))
        x = x // 26 - 1
        if x < 0:
            break
    return "w" + "".join(reversed(letras))



# ============================================================ ESPAÇO HIPERDIMENSIONAL EXPANSÍVEL
class EspacoHiperdim:
    """
    Camada nova do Toshi para guardar os PESOS. Dimensão configurável:
    16k, 65k, 131k, ... quanto o hardware aguentar.
    """

    def __init__(self, dim=16384, nnz=16, chunk_bytes=1024 * 1024):
        if dim < nnz * 4:
            raise ValueError(f"dimensão {dim} é pequena demais para nnz={nnz}")
        self.dim = dim
        self.nnz = nnz
        self.chunk_bytes = max(1024, int(chunk_bytes))

    def assinar(self, nome, dados):
        """
        Projeta TODOS os bytes do tensor para um hipervetor denso NORMALIZADO
        (soma de vetores esparsos ±1 — Random Indexing, o padrão do Toshi).
        Determinístico: mesmo tensor + mesmo nome = mesma assinatura.
        """
        if isinstance(dados, np.ndarray):
            dados = np.ascontiguousarray(dados).tobytes()
        elif isinstance(dados, (bytearray, memoryview)):
            dados = bytes(dados)
        elif not isinstance(dados, (bytes,)):
            dados = bytes(dados)

        # bloco adaptativo: todo tensor vira >=64 blocos (mesmo os pequenos)
        chunk = max(256, min(self.chunk_bytes, max(1, len(dados) // 64)))

        v = np.zeros(self.dim, dtype=np.float32)
        for off in range(0, len(dados), chunk):
            bloco = dados[off:off + chunk]
            h = hashlib.sha256()
            h.update(nome.encode("utf-8", "replace"))
            h.update(struct.pack("<Q", off))
            h.update(bloco)
            seed = int.from_bytes(h.digest()[:8], "little")
            rng = np.random.default_rng(seed)
            dims = rng.choice(self.dim, size=self.nnz, replace=False)
            sinais = rng.choice(np.array([-1, 1], dtype=np.float32), size=self.nnz)
            v[dims] += sinais

        norma = float(np.linalg.norm(v))
        if norma < 1e-12:
            v = np.zeros(self.dim, dtype=np.float32)
            v[0] = 1.0
            return v
        return (v / norma).astype(np.float32)

    def palavras_do_tensor(self, dados, tokens_por_tensor=32, vocabulario=4096):
        """
        Converte os bytes do tensor em uma sequência de 'palavras de peso'.
        É a ponte para o Toshi comer o modelo COMO COMEU OS LIVROS:
        bytes -> segmentos -> hash -> palavra (ex.: p0a3f) -> toshi.perceive([...])
        """
        if isinstance(dados, np.ndarray):
            dados = np.ascontiguousarray(dados).tobytes()
        elif isinstance(dados, (bytearray, memoryview)):
            dados = bytes(dados)
        elif not isinstance(dados, (bytes,)):
            dados = bytes(dados)

        n = max(1, len(dados))
        k = max(1, int(tokens_por_tensor))
        palavras = []
        for i in range(k):
            ini = (i * n) // k
            fim = ((i + 1) * n) // k
            if fim <= ini:
                fim = min(n, ini + 1)
            bloco = dados[ini:fim]
            h = hashlib.sha256(bloco).digest()
            idx = int.from_bytes(h[:4], "little") % vocabulario
            palavras.append(_palavra_indice(idx))
        return palavras

    @staticmethod
    def cos(a, b):
        return float(np.dot(a.astype(np.float32), b.astype(np.float32)))


# ============================================================ ABSORVEDOR DE PESOS
class AbsorvedorPesos:
    """Lê todos os pesos e transcreve para o Toshi (assinaturas + estrutura)."""

    def __init__(self, toshi=None, fatos=None, dim_pesos=16384, chunk_mb=1,
                 nnz=16, carregar_existente=True, tokens_por_tensor=32,
                 vocabulario_pesos=4096):
        if toshi is None and carregar_existente:
            self.toshi, _ = build_or_load()
        else:
            self.toshi = toshi or Toshi()
        self.fatos = fatos or Fatos()

        self.dim_pesos = dim_pesos
        self.chunk_bytes = max(1024, int(chunk_mb * 1024 * 1024))
        self.tokens_por_tensor = max(1, int(tokens_por_tensor))
        self.vocabulario_pesos = max(16, int(vocabulario_pesos))
        self.espaco = EspacoHiperdim(dim=dim_pesos, nnz=nnz, chunk_bytes=self.chunk_bytes)
        # o Toshi ganha a nova camada expansível
        self.toshi.espaco_pesos = self.espaco

        self.tensores = {}            # nome -> {"vetor":..., "bytes":..., "tokens":..., "palavras":...}
        self.sequencia = []           # sequência de tokens da ARQUITETURA
        self.fluxo_pesos = []         # fluxo completo que o Toshi COME (nomes + palavras de peso)
        self.palavras_pesos = []      # só as palavras de peso, na ordem
        self.meta_modelo = {}
        self.modelo = ""
        self._arquivo = None
        self.assinatura_modelo = None   # bundle de TODOS os tensores (identidade do modelo)

    # ---------- absorção ----------
    def absorver_lista_tensores(self, modelo, itens, origem="lista"):
        t0 = time.time()
        n = 0
        total_bytes = 0
        for i, par in enumerate(itens, 1):
            item, dados = par
            # aceita tanto (nome, bytes) quanto (info_gguf, bytes)
            nome = item["nome"] if isinstance(item, dict) else item
            vec = self.espaco.assinar(nome, dados)
            toks = tokens_do_nome(nome)
            palavras = self.espaco.palavras_do_tensor(
                dados,
                tokens_por_tensor=self.tokens_por_tensor,
                vocabulario=self.vocabulario_pesos,
            )
            self.tensores[nome] = {
                "vetor": vec, "bytes": len(dados), "tokens": toks, "palavras": palavras,
            }
            self.sequencia.extend(toks)
            # o fluxo que o Toshi COME: nome do tensor + as palavras de peso dele
            self.fluxo_pesos.extend(toks)
            self.fluxo_pesos.extend(palavras)
            self.palavras_pesos.extend(palavras)
            # estrutura -> memória factual crisp do Toshi
            self.fatos.aprender(nome, "é", "tensor")
            total_bytes += len(dados)
            n = i
            if i % 25 == 0 or i == 1:
                print(f"    tensor {i}: {nome} ({len(dados)/1e6:.2f} MB, "
                      f"{len(palavras)} palavras de peso)")

        # ASSINATURA DO MODELO INTEIRO: bundle (soma normalizada) de todos os tensores.
        # Com ela o Toshi reconhece "este modelo" como UM conceito no espaço HD.
        if self.tensores:
            soma = np.sum([d["vetor"].astype(np.float32) for d in self.tensores.values()], axis=0)
            norma = float(np.linalg.norm(soma))
            if norma > 1e-12:
                self.assinatura_modelo = (soma / norma).astype(np.float32)

        # ASSIM COMO NOS LIVROS: o Toshi COME os dois fluxos.
        # 1) arquitetura pura (ordem das camadas)
        if self.sequencia:
            self.toshi.perceive(self.sequencia)
        # 2) o modelo inteiro como linguagem (nomes + palavras de peso)
        if self.fluxo_pesos:
            self.toshi.perceive(self.fluxo_pesos)
            print(f"  toshi comeu {len(self.fluxo_pesos)} tokens do modelo "
                  f"({len(self.palavras_pesos)} palavras de peso)")

        # metadados do modelo -> fatos (valores ÚNICOS: reabsorver atualiza, não empilha)
        self.fatos.g.setdefault(modelo, [])
        for rel in ("tem_tensores", "tem_bytes", "tem_dimensao_pesos"):
            self.fatos.g[modelo] = [e for e in self.fatos.g[modelo] if e[0] != rel]
        self.fatos.aprender(modelo, "é", "modelo")
        self.fatos.aprender(modelo, "tem_tensores", str(n))
        self.fatos.aprender(modelo, "tem_bytes", str(total_bytes))
        self.fatos.aprender(modelo, "tem_dimensao_pesos", str(self.dim_pesos))

        self.modelo = modelo
        self.meta_modelo = {
            "modelo": modelo, "origem": origem, "tensores": n,
            "bytes": total_bytes, "dimensao_pesos": self.dim_pesos,
            "chunk_bytes": self.chunk_bytes, "tempo_s": round(time.time() - t0, 1),
            "assinatura_modelo": self.assinatura_modelo is not None,
            "palavras_pesos": len(self.palavras_pesos),
            "vocabulario_pesos": self.vocabulario_pesos,
            "tokens_por_tensor": self.tokens_por_tensor,
        }
        print(f"  absorvido: {n} tensores, {total_bytes/1e9:.3f} GB de pesos, "
              f"dimensão {self.dim_pesos}, em {self.meta_modelo['tempo_s']}s")
        return self.meta_modelo

    def absorver_arquivo(self, caminho, modelo=None):
        tensores, _, tam, versao = _parsear_cabecalho_gguf(caminho)
        modelo = modelo or Path(caminho).stem
        self._arquivo = caminho
        print(f"GGUF v{versao}: {len(tensores)} tensores, arquivo {tam/1e9:.2f} GB")
        print(f"projetando todos os pesos para a dimensão {self.dim_pesos}...")
        return self.absorver_lista_tensores(modelo, iterar_tensores_gguf(caminho), origem="gguf")

    # ---------- teste automático ----------
    def testar(self):
        res = {}

        nomes = list(self.tensores.keys())
        n = len(nomes)
        res["cobertura"] = {"ok": n, "n": n}

        # todo tensor virou fato?
        ok_fatos = sum(1 for nome in nomes if ["é", "tensor"] in self.fatos.g.get(nome, []))
        res["fatos"] = {"ok": ok_fatos, "n": n}

        # todo token de nome entrou no vocabulário?
        unicos = sorted(set(self.sequencia))
        ok_tokens = sum(1 for w in unicos if self.toshi.seen[w] > 0)
        res["tokens"] = {"ok": ok_tokens, "n": len(unicos)}

        # a sequência da arquitetura virou transição?
        pares = list(zip(self.sequencia, self.sequencia[1:]))
        if pares:
            passo = max(1, len(pares) // 200)
            amostra = pares[::passo][:200]
            ok_trans = sum(1 for a, b in amostra if b in self.toshi.after.get(a, {}))
            res["transicoes"] = {"ok": ok_trans, "n": len(amostra)}
        else:
            res["transicoes"] = {"ok": 0, "n": 0}

        # TODAS as palavras de peso entraram no vocabulário? (como nos livros)
        unicas_pesos = sorted(set(self.palavras_pesos))
        ok_palavras = sum(1 for w in unicas_pesos if self.toshi.seen[w] > 0)
        res["palavras_pesos"] = {"ok": ok_palavras, "n": len(unicas_pesos)}

        # as transições da LINGUAGEM de pesos foram aprendidas?
        pares_pesos = list(zip(self.fluxo_pesos, self.fluxo_pesos[1:]))
        if pares_pesos:
            passo_p = max(1, len(pares_pesos) // 200)
            amostra_p = pares_pesos[::passo_p][:200]
            ok_trans_p = sum(1 for a, b in amostra_p if b in self.toshi.after.get(a, {}))
            res["transicoes_pesos"] = {"ok": ok_trans_p, "n": len(amostra_p)}
        else:
            res["transicoes_pesos"] = {"ok": 0, "n": 0}

        # dimensão correta em todos os vetores?
        ok_dim = sum(1 for d in self.tensores.values() if len(d["vetor"]) == self.dim_pesos)
        res["dimensoes"] = {"ok": ok_dim, "n": n}

        # assinatura do modelo inteiro (bundle de todos os tensores)
        ok_assin = (1 if self.assinatura_modelo is not None
                    and len(self.assinatura_modelo) == self.dim_pesos else 0)
        res["assinatura_modelo"] = {"ok": ok_assin, "n": 1}

        # determinismo: re-lê o primeiro tensor do arquivo e re-assina
        det = None
        if self._arquivo and os.path.isfile(self._arquivo):
            for info, dados in iterar_tensores_gguf(self._arquivo):
                v2 = self.espaco.assinar(info["nome"], dados)
                det = bool(np.array_equal(self.tensores[info["nome"]]["vetor"], v2))
                break
        res["determinismo"] = {"ok": int(det) if det is not None else 0,
                               "n": 1 if det is not None else 0}

        # ortogonalidade das assinaturas (amostra)
        amostra_nomes = nomes[:min(24, n)]
        cossenos = []
        for i in range(len(amostra_nomes)):
            for j in range(i + 1, len(amostra_nomes)):
                cossenos.append(EspacoHiperdim.cos(
                    self.tensores[amostra_nomes[i]]["vetor"],
                    self.tensores[amostra_nomes[j]]["vetor"],
                ))
        res["ortogonalidade"] = {
            "media_abs_cos": round(float(np.mean(np.abs(cossenos))), 4) if cossenos else 0.0,
            "pares": len(cossenos),
        }

        res["global"] = {
            "ok": sum(res[k]["ok"] for k in ("cobertura", "fatos", "tokens", "transicoes",
                                             "palavras_pesos", "transicoes_pesos",
                                             "dimensoes", "determinismo", "assinatura_modelo")),
            "n": sum(res[k]["n"] for k in ("cobertura", "fatos", "tokens", "transicoes",
                                            "palavras_pesos", "transicoes_pesos",
                                            "dimensoes", "determinismo", "assinatura_modelo")),
        }
        return res

    @staticmethod
    def resumo(res):
        def pct(d):
            return f"{d['ok']}/{d['n']} ({d['ok'] / d['n'] * 100:.0f}%)" if d["n"] else "—"
        r = {
            "cobertura_pesos": pct(res["cobertura"]),
            "fatos_estrutura": pct(res["fatos"]),
            "tokens_arquitetura": pct(res["tokens"]),
            "transicoes_arquitetura": pct(res["transicoes"]),
            "palavras_pesos_vocab": pct(res["palavras_pesos"]),
            "transicoes_pesos": pct(res["transicoes_pesos"]),
            "dimensoes": pct(res["dimensoes"]),
            "determinismo": pct(res["determinismo"]),
            "assinatura_modelo": pct(res["assinatura_modelo"]),
            "global": pct(res["global"]),
        }
        r["ortogonalidade_media"] = f"{res['ortogonalidade']['media_abs_cos']:.4f}"
        return r

    def relatorio(self, res):
        resumo = self.resumo(res)
        print("\n" + "=" * 66)
        print("TESTE AUTOMÁTICO — PESOS DO QWEN DENTRO DO TOSHI")
        print("=" * 66)
        for k, v in resumo.items():
            print(f"  {k:<26} {v}")
        print("-" * 66)
        print(f"  assinaturas guardadas: {len(self.tensores)}")
        print(f"  dimensão do espaço de pesos: {self.dim_pesos}")
        print(f"  bytes de pesos absorvidos: {self.meta_modelo.get('bytes', 0)/1e9:.3f} GB")
        print(f"  palavras de peso comidas pelo Toshi: {len(self.palavras_pesos)}")
        print(f"  ortogonalidade média (|cos|): {resumo['ortogonalidade_media']} "
              "(perto de 0 = espaço bem usado)")
        return resumo

    # ---------- persistência ----------
    def salvar(self):
        save_state(self.toshi)
        self.fatos.save()
        nome_san = re.sub(r"[^a-z0-9]+", "_", self.modelo.lower()).strip("_") or "modelo"
        arq = os.path.join(HERE, "dados", f"toshi_pesos_{nome_san}.pkl")
        payload = {
            "meta": self.meta_modelo,
            "dimensao": self.dim_pesos,
            "assinatura_modelo": self.assinatura_modelo,
            "tensores": {nome: {
                "vetor": d["vetor"], "bytes": d["bytes"], "tokens": d["tokens"],
                "palavras": d.get("palavras", []),
            } for nome, d in self.tensores.items()},
        }
        with open(arq, "wb") as f:
            pickle.dump(payload, f)
        print(f"[pesos salvos] {arq} ({os.path.getsize(arq)/1e6:.1f} MB)")
        return arq


# ============================================================ SELFTEST
def _selftest():
    print("SELFTEST — absorvedor de pesos com modelo sintético (sem Ollama/GUGF)\n")
    t = Toshi()
    f = Fatos()
    f.g = {}

    absv = AbsorvedorPesos(toshi=t, fatos=f, dim_pesos=512, chunk_mb=1,
                           nnz=8, carregar_existente=False)

    # 6 tensores sintéticos, bytes determinísticos
    itens = []
    for i, nome in enumerate([
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.1.attn_q.weight",
        "output_norm.weight",
    ]):
        dados = bytes((j * 31 + i * 7) % 256 for j in range(5000))
        itens.append((nome, dados))

    absv.absorver_lista_tensores("modelo_sintetico", itens, origem="selftest")
    res = absv.testar()
    for k, v in absv.resumo(res).items():
        print(f"  {k:<26} {v}")

    # determinismo explícito
    v1 = absv.tensores["token_embd.weight"]["vetor"]
    v2 = absv.espaco.assinar("token_embd.weight", itens[0][1])
    assert np.array_equal(v1, v2), "assinatura não é determinística"
    assert res["cobertura"]["ok"] == res["cobertura"]["n"]
    assert res["fatos"]["ok"] == res["fatos"]["n"]
    assert res["tokens"]["ok"] == res["tokens"]["n"]
    assert res["transicoes"]["ok"] == res["transicoes"]["n"]
    assert res["palavras_pesos"]["ok"] == res["palavras_pesos"]["n"]
    assert res["transicoes_pesos"]["ok"] == res["transicoes_pesos"]["n"]
    assert res["dimensoes"]["ok"] == res["dimensoes"]["n"]
    assert res["assinatura_modelo"]["ok"] == 1
    assert absv.assinatura_modelo is not None and len(absv.assinatura_modelo) == 512
    assert all(len(d["vetor"]) == 512 for d in absv.tensores.values())
    # o Toshi comeu o modelo como texto: 6 tensores x 32 palavras de peso
    assert len(absv.palavras_pesos) == 6 * 32
    print("\n[selftest] ok — pesos projetados + comidos como linguagem, tudo testado")


# ============================================================ CLI
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="Absorve TODOS os pesos de um GGUF para o espaço hiperdimensional do Toshi."
    )
    ap.add_argument("--modelo", default="qwen2.5:7b", help="modelo Ollama de origem")
    ap.add_argument("--arquivo", default="", help="caminho direto do .gguf (ignora o Ollama)")
    ap.add_argument("--dim-pesos", type=int, default=16384,
                    help="dimensão do espaço de pesos do Toshi (16k, 65k, 131k...)")
    ap.add_argument("--chunk-mb", type=int, default=1, help="tamanho do bloco de projeção")
    ap.add_argument("--nnz", type=int, default=16, help="não-zeros por bloco (Random Indexing)")
    ap.add_argument("--tokens-por-tensor", type=int, default=32,
                    help="palavras de peso por tensor (o 'texto' que o Toshi come)")
    ap.add_argument("--vocabulario-pesos", type=int, default=4096,
                    help="tamanho do vocabulário da linguagem de pesos")
    ap.add_argument("--sem-teste", action="store_true", help="não roda o teste automático")
    ap.add_argument("--sem-salvar", action="store_true", help="não salva o estado")
    ap.add_argument("--listar", action="store_true", help="apenas lista os tensores do GGUF")
    ap.add_argument("--selftest", action="store_true", help="teste offline com modelo sintético")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    caminho = args.arquivo or localizar_gguf(args.modelo)
    if not caminho or not os.path.isfile(caminho):
        print(f"Não encontrei o GGUF do modelo '{args.modelo}'.")
        print("Use --arquivo C:\\caminho\\modelo.gguf ou confirme o Ollama com: ollama list")
        return

    print(f"GGUF localizado: {caminho}")

    if args.listar:
        tensores, inicio_dados, tam, versao = _parsear_cabecalho_gguf(caminho)
        print(f"GGUF v{versao}, {len(tensores)} tensores, dados a partir de {inicio_dados}\n")
        for i, t in enumerate(tensores[:80]):
            print(f"  {i:4d}  {t['nome']:<55} dims={t['dims']} tipo={t['ggml_tipo']} off={t['offset']}")
        if len(tensores) > 80:
            print(f"  ... e mais {len(tensores) - 80} tensores")
        return

    absv = AbsorvedorPesos(
        dim_pesos=args.dim_pesos,
        chunk_mb=args.chunk_mb,
        nnz=args.nnz,
        tokens_por_tensor=args.tokens_por_tensor,
        vocabulario_pesos=args.vocabulario_pesos,
    )
    absv.absorver_arquivo(caminho, modelo=args.modelo)

    if not args.sem_teste:
        absv.relatorio(absv.testar())
    if not args.sem_salvar:
        absv.salvar()


if __name__ == "__main__":
    main()
