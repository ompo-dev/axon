"""
PERGUNTAR MODELO — o Toshi responde sobre a IA que absorveu.

Usa as DUAS memórias que ele construiu:
  - Fatos (grafo factual): quantos tensores? quantos bytes? o que é X?
  - MemoriaPesos (assinaturas HD): qual tensor é semelhante a X?

USO:
  python perguntar_modelo.py --modelo qwen2.5:7b --pergunta "quantos tensores tem qwen2.5:7b?"
  python perguntar_modelo.py --modelo qwen2.5:7b --pergunta "o que é blk.0.attn_q.weight?"
  python perguntar_modelo.py --modelo qwen2.5:7b --pergunta "qual tensor é semelhante a blk.0.attn_q.weight?"
  python perguntar_modelo.py --selftest
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from fatos import Fatos, _norm, extrair
from consultar_pesos import MemoriaPesos, localizar_memoria


def _sujeito_aproximado(fatos, alvo):
    """Acha o sujeito no grafo, aceitando nome parcial."""
    alvo = _norm(alvo)
    for s in fatos.g:
        if _norm(s) == alvo or alvo in _norm(s):
            return s
    return None


def _aresta(fatos, sujeito, relacao):
    # devolve a ÚLTIMA aresta da relação (a mais recente, no caso de reabsorção)
    for r, o in reversed(fatos.g.get(sujeito, [])):
        if r == relacao:
            return o
    return None


def responder(mem, fatos, pergunta):
    q = _norm(pergunta)

    m = re.search(r"quantos tensores tem\s+(.+)", q)
    if m:
        s = _sujeito_aproximado(fatos, m.group(1))
        o = _aresta(fatos, s, "tem_tensores") if s else None
        return f"{s} tem {o} tensores." if o is not None else None

    m = re.search(r"quantos bytes tem\s+(.+)", q)
    if m:
        s = _sujeito_aproximado(fatos, m.group(1))
        o = _aresta(fatos, s, "tem_bytes") if s else None
        if o is not None:
            return f"{s} tem {int(o)/1e9:.3f} GB de pesos ({o} bytes)."

    m = re.search(r"qual dimensao de pesos tem\s+(.+)", q)
    if m:
        s = _sujeito_aproximado(fatos, m.group(1))
        o = _aresta(fatos, s, "tem_dimensao_pesos") if s else None
        return f"{s} usa dimensão {o} no espaço de pesos." if o is not None else None

    m = re.search(r"o que é\s+(.+)", q)
    if m:
        # deixa o Fatos responder (ele cobre as definições crisp)
        return fatos.responder(q)

    m = re.search(r"qual tensor é semelhante a\s+(.+)", q)
    if m:
        nome = m.group(1).strip()
        exato = nome if nome in mem.tensores else None
        if not exato:
            parciais = [n for n in mem.nomes if nome.lower() in n.lower()]
            exato = parciais[0] if parciais else None
        if exato:
            top = mem.semelhantes(exato, k=3)
            return (f"vizinhos de {exato}: " +
                    ", ".join(f"{n} (cos {c:+.4f})" for n, c in top))

    # qualquer outra pergunta que o Fatos souber responder
    return fatos.responder(q)


def _selftest():
    print("SELFTEST — o Toshi responde sobre a IA absorvida\n")
    f = Fatos()
    f.g = {}
    f.aprender("modelo_sintetico", "é", "modelo")
    f.aprender("modelo_sintetico", "tem_tensores", "7")
    f.aprender("modelo_sintetico", "tem_bytes", "123456789")
    f.aprender("modelo_sintetico", "tem_dimensao_pesos", "512")
    f.aprender("blk.0.attn_q.weight", "é", "tensor")

    import numpy as np
    rng = np.random.default_rng(0)
    tensores = {}
    for nome in ["token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight"]:
        v = rng.normal(size=256).astype(np.float32)
        tensores[nome] = {"vetor": v / np.linalg.norm(v), "bytes": 100, "tokens": [], "palavras": []}
    mem = MemoriaPesos.from_payload({
        "meta": {"modelo": "modelo_sintetico"}, "dimensao": 256, "tensores": tensores,
    })

    casos = [
        ("quantos tensores tem modelo_sintetico?", "7"),
        ("quantos bytes tem modelo_sintetico?", "123456789"),
        ("qual dimensao de pesos tem modelo_sintetico?", "512"),
        ("o que é blk.0.attn_q.weight?", "tensor"),
        ("qual tensor é semelhante a blk.0.attn_q.weight?", "vizinhos"),
    ]
    for p, esp in casos:
        r = responder(mem, f, p)
        print(f"  Q: {p}\n  R: {r}")
        assert r and esp in r, (p, r)
    print("\n[selftest] ok — ele responde sobre o que comeu")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Pergunta ao Toshi sobre o modelo absorvido.")
    ap.add_argument("--modelo", default="qwen2.5:7b")
    ap.add_argument("--arquivo", default="")
    ap.add_argument("--pergunta", default="")
    ap.add_argument("--interativo", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    try:
        mem = MemoriaPesos(args.arquivo or None, args.modelo)
    except FileNotFoundError as e:
        print(e)
        return
    fatos = Fatos()

    def faz(pergunta, eco=True):
        r = responder(mem, fatos, pergunta)
        if eco:
            print(f"você> {pergunta}")
        print(f"toshi> {r if r else '(ainda não sei responder isso)'}")

    if args.interativo:
        print("modo interativo — pergunte sobre o modelo absorvido. Ctrl+C sai.\n")
        while True:
            try:
                p = input("voce> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\ntchau!")
                break
            if not p:
                continue
            if p.lower() in ("sair", "exit", "quit"):
                break
            # se for uma declaração ("meu nome é maicon"), APRENDE na hora
            decl = re.sub(r"^(oi|ola|olá)[,\s]+", "", p, flags=re.IGNORECASE)
            e = extrair(decl) or extrair(p)
            if e:
                fatos.aprender(*e)
                print(f"toshi> (aprendi: {e[0]} {e[1]} {e[2]})")
                continue
            faz(p, eco=False)
    elif args.pergunta:
        faz(args.pergunta)
    else:
        print("use --pergunta \"quantos tensores tem qwen2.5:7b?\" ou --interativo")


if __name__ == "__main__":
    main()
