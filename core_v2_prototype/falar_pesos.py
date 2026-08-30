"""
FALAR PESOS — prova que o Toshi COMEU o modelo como linguagem.

Depois de `absorver_pesos_qwen.py`, as palavras de peso entraram no MESMO substrato
que os livros (assoc, after, embed, seen). Aqui o Toshi USA essa memória:
  - gera uma sequência na linguagem de pesos (fala bruta pelo que comeu)
  - mostra o que uma palavra de peso evoca (associações brutas)
  - mostra o significado distribucional de uma palavra de peso
  - mostra a arquitetura que ele aprendeu (blk -> attn -> q -> weight...)

USO:
  python falar_pesos.py --modelo qwen2.5:7b --fluxo blk.0.attn_q.weight
  python falar_pesos.py --modelo qwen2.5:7b --palavra waaa
  python falar_pesos.py --modelo qwen2.5:7b --arquitetura
  python falar_pesos.py --selftest
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from toshi import Toshi, build_or_load, tokenize
from consultar_pesos import MemoriaPesos, localizar_memoria


# ---------- funções de "fala" direto no substrato aprendido ----------
def vizinhos_brutos(toshi, palavra, k=10):
    """O que acende junto com a palavra (contagem bruta, sem filtro PMI)."""
    return sorted(toshi.assoc.get(palavra, {}).items(), key=lambda kv: -kv[1])[:k]


def drift_bruto(toshi, inicio, passos=12):
    """Caminha pelas transições mais fortes que o Toshi comeu (sem filtro)."""
    if inicio not in toshi.after:
        return []
    caminho = [inicio]
    atual = inicio
    for _ in range(passos):
        cand = toshi.after.get(atual)
        if not cand:
            break
        atual = max(cand, key=lambda w: cand[w])
        caminho.append(atual)
    return caminho


def fala_bruta(toshi, sementes, n=12):
    """Gera uma sequência na linguagem de pesos a partir de sementes."""
    if not sementes:
        return []
    atual = sementes[-1]
    if atual not in toshi.after:
        return []
    saida = [atual]
    for _ in range(n):
        cand = toshi.after.get(atual)
        if not cand:
            break
        atual = max(cand, key=lambda w: cand[w])
        saida.append(atual)
    return saida


def achar_tensor(mem, nome):
    if nome in mem.tensores:
        return nome
    parcial = [n for n in mem.nomes if nome.lower() in n.lower()]
    return parcial[0] if parcial else None


# ---------- selftest ----------
def _selftest():
    print("SELFTEST — o Toshi fala a linguagem de pesos\n")
    t = Toshi()
    # mesmo formato do fluxo real: nome + palavras de peso, repetido como exposição
    fluxo = (["blk", "attn", "q", "weight"] + ["waaa", "wbbb", "wccc", "wddd"] * 4) * 3
    t.perceive(fluxo)

    viz = vizinhos_brutos(t, "waaa", 5)
    assert viz, "não acendeu nada"
    caminho = drift_bruto(t, "waaa", 6)
    assert caminho and caminho[0] == "waaa"
    fala = fala_bruta(t, ["waaa", "wbbb"], 6)
    assert len(fala) >= 2
    arq = vizinhos_brutos(t, "blk", 3)
    assert arq and arq[0][0] == "attn", arq

    print("  vizinhos brutos de waaa:", viz[:4])
    print("  drift bruto:", " -> ".join(caminho))
    print("  fala bruta:", " ".join(fala))
    print("  arquitetura blk ->", [w for w, _ in arq])
    print("\n[selftest] ok — o que ele comeu, ele usa")


# ---------- CLI ----------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="O Toshi fala/usando a memória de pesos que comeu.")
    ap.add_argument("--modelo", default="qwen2.5:7b")
    ap.add_argument("--arquivo", default="", help="arquivo .pkl da memória de pesos")
    ap.add_argument("--fluxo", default="", help="tensor para gerar na linguagem de pesos")
    ap.add_argument("--palavra", default="", help="palavra de peso para evocar/significado")
    ap.add_argument("--arquitetura", action="store_true", help="mostra a arquitetura aprendida")
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

    print(f"carregando Toshi...", end=" ", flush=True)
    toshi, cache = build_or_load()
    print(f"pronto ({sum(toshi.seen.values())} palavras vividas, {len(toshi.seen)} conceitos)")

    if args.fluxo:
        nome = achar_tensor(mem, args.fluxo)
        if not nome:
            print(f"tensor '{args.fluxo}' não encontrado")
            return
        palavras = mem.palavras_por_tensor.get(nome, [])
        if not palavras:
            print(f"este arquivo de pesos é antigo e não tem as palavras. Reabsorva o modelo.")
            return
        print("=" * 66)
        print(f"TOSHI FALA O TENSOR: {nome}")
        print("=" * 66)
        print(f"palavras de peso do tensor ({len(palavras)}): "
              f"{' '.join(palavras[:8])} ...")
        print(f"\n1) evoca (associações brutas das primeiras palavras):")
        for w in palavras[:3]:
            viz = vizinhos_brutos(toshi, w, 6)
            print(f"   {w:<10} -> {[v for v, _ in viz]}")
        print(f"\n2) caminha (transições mais fortes):")
        for w in palavras[:2]:
            print(f"   {' -> '.join(drift_bruto(toshi, w, 12))}")
        print(f"\n3) fala (geração a partir do tensor):")
        print(f"   {' '.join(fala_bruta(toshi, palavras[:2], 14))}")
        print(f"\n4) significado (Random Indexing):")
        for w in palavras[:2]:
            sig = toshi.meaning(w, k=6, min_freq=1)
            print(f"   {w:<10} ~ {[(u, round(s, 3)) for u, s in sig]}")

    if args.palavra:
        w = args.palavra.strip()
        print("=" * 66)
        print(f"TOSHI CONHECE A PALAVRA DE PESO: {w}")
        print("=" * 66)
        print(f"vista {toshi.seen.get(w, 0)} vez(es)")
        print(f"\n1) evoca: {[v for v, _ in vizinhos_brutos(toshi, w, 10)]}")
        print(f"\n2) caminha: {' -> '.join(drift_bruto(toshi, w, 14))}")
        print(f"\n3) fala: {' '.join(fala_bruta(toshi, [w], 12))}")
        print(f"\n4) significado: {[(u, round(s, 3)) for u, s in toshi.meaning(w, k=8, min_freq=1)]}")

    if args.arquitetura:
        print("=" * 66)
        print("ARQUITETURA QUE O TOSHI COMEU")
        print("=" * 66)
        for w in ("blk", "attn", "ffn", "weight", "bias", "token"):
            viz = vizinhos_brutos(toshi, w, 6)
            if viz:
                print(f"  {w:<8} -> {[v for v, _ in viz]}")
            else:
                print(f"  {w:<8} -> (não comeu)")


if __name__ == "__main__":
    main()
