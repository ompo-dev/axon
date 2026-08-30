"""
TERNARIZAR os embeddings do Toshi (float32 -> {-1,0,+1}) — a quantização extrema, medida.

Pesquisa (BitNet b1.58, Ma et al. 2024; Word2Spike 2025): pesos/embeddings ternários {-1,0,+1}
(~1.58 bit) batem o full-precision e o Word2Spike preserva ~97% da similaridade semântica. Aqui
aplico ao Toshi real: absmean quantization (o esquema do BitNet). Mede:
  - MEMÓRIA: float32 (4096 B/conceito) vs ternário bit-packed (2 bits/dim = 256 B) = 16x menor.
  - SEMÂNTICA preservada? gap(relacionados-aleatórios) e sobreposição de vizinhos, float vs ternário.
Se a estrutura se mantém, é o caminho do "imprimir um cérebro pesando quase nada".
Roda: python efficiency_ternary.py   (usa o Toshi real)
"""
import sys
import numpy as np
from toshi import build_or_load, DIM


def ternarizar(V):
    """absmean (BitNet b1.58): escala = média(|v|) por vetor; q = clip(round(v/escala),-1,1)."""
    escala = np.mean(np.abs(V), axis=1, keepdims=True) + 1e-9
    return np.clip(np.round(V / escala), -1, 1).astype(np.int8)


def cos_rows(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return An, Bn


def gap(vocab, idx, V, pares, rng):
    def c(a, b):
        return float(V[idx[a]] @ V[idx[b]])
    rel = [c(a, b) for a, b in pares if a in idx and b in idx]
    ws = list(idx)
    alea = [float(V[idx[a]] @ V[idx[b]])
            for a, b in (rng.choice(ws, 2, replace=False) for _ in range(300))]
    return np.mean(rel), np.mean(alea), np.mean(rel) - np.mean(alea)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando Toshi real...", flush=True)
    t, _ = build_or_load()
    vocab = [w for w in t.embed if t.seen[w] >= 5 and np.linalg.norm(t.embed[w]) > 1e-9]
    idx = {w: i for i, w in enumerate(vocab)}
    Vf = np.array([t.embed[w] for w in vocab], np.float32)
    Vt = ternarizar(Vf).astype(np.float32)                 # ternário como vetor p/ medir

    Vfn, _ = cos_rows(Vf, Vf)
    Vtn, _ = cos_rows(Vt, Vt)

    print("=" * 72)
    print("TERNARIZAR embeddings (BitNet absmean) — memória e semântica, medido")
    print("=" * 72)

    # memória
    bytes_f = DIM * 4
    bytes_t = DIM // 4                                      # 2 bits/dim, bit-packed
    print(f"\nmemória por conceito: float32 {bytes_f} B  ->  ternário packed {bytes_t} B  "
          f"({bytes_f/bytes_t:.0f}x menor)")
    esparso = float(np.mean(Vt == 0))
    print(f"  ({esparso:.0%} das dimensões viram 0 -> ainda mais compressível; ops viram XOR/popcount)")

    # semântica: gap
    pares = [("mar", "ceu"), ("amor", "ciume"), ("olhos", "ressaca"), ("morte", "vida"),
             ("padre", "seminario"), ("mae", "filho"), ("bento", "capitu"), ("mar", "vento")]
    rf = gap(vocab, idx, Vfn, pares, np.random.default_rng(1))
    rt = gap(vocab, idx, Vtn, pares, np.random.default_rng(1))
    print(f"\ngap semântico (relacionados - aleatórios; maior=melhor estrutura):")
    print(f"  float32 : rel {rf[0]:+.3f}  alea {rf[1]:+.3f}  GAP {rf[2]:+.3f}")
    print(f"  ternário: rel {rt[0]:+.3f}  alea {rt[1]:+.3f}  GAP {rt[2]:+.3f}   "
          f"({rt[2]/rf[2]:.0%} do float)")

    # semântica: sobreposição de vizinhos top-5 (float vs ternário)
    rng = np.random.default_rng(0)
    amostra = [vocab[i] for i in rng.choice(len(vocab), 40, replace=False)]
    jac = []
    for w in amostra:
        i = idx[w]
        vf = np.argsort(-(Vfn @ Vfn[i]))[1:6]
        vt = np.argsort(-(Vtn @ Vtn[i]))[1:6]
        jac.append(len(set(vf) & set(vt)) / 5.0)
    print(f"\nvizinhos preservados (top-5, float vs ternário): {np.mean(jac):.0%} em média")

    print("\n" + "=" * 72)
    if rt[2] > 0.6 * rf[2] and np.mean(jac) > 0.5:
        print(f"VEREDITO: ternarizar mantém a estrutura semântica ({rt[2]/rf[2]:.0%} do gap, "
              f"{np.mean(jac):.0%} dos vizinhos) por ~{bytes_f/bytes_t:.0f}x menos memória. Confirma")
        print("BitNet/Word2Spike no NOSSO sistema: o cérebro do Toshi cabe ternário, quase de graça.")
    else:
        print("VEREDITO honesto: a ternarização degradou demais aqui — reportado sem inflar.")


if __name__ == "__main__":
    main()
