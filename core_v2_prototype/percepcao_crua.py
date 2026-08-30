"""
AXON core v2 — degrau 3 (CORRETO): percepção CRUA, modality-agnostic. O princípio do bebê.

Correção de rumo (do usuário): axon NÃO sabe o que é texto/áudio/vídeo. Recebe estímulo cru.
Não damos 'features' (assento, pernas) — isso seria semântica humana injetada (cheating).
Como um bebê: recebe um STREAM de valores, não sabe o que são, e DESCOBRE a estrutura sozinho.

Ciência real: aprendizado estatístico infantil (Saffran, Aslin, Newport 1996). Bebês segmentam
fala contínua (sem pausas entre palavras) em 'palavras' usando PROBABILIDADE DE TRANSIÇÃO:
dentro de uma palavra as transições são previsíveis (alta prob); entre palavras, imprevisíveis
(baixa prob). O bebê põe a fronteira onde a previsibilidade DESPENCA. Zero rótulos.

Aqui: stream de bytes (pode ser texto, áudio, pixels — o sistema NÃO sabe qual).
  1. cada símbolo cru -> hipervetor aleatório fixo ('neurônio sensorial', sem significado).
  2. estatística online de transição (bigramas).
  3. segmenta em CHUNKS nos mínimos locais de previsibilidade (Saffran) -> 'palavras' emergentes.
  4. cada chunk vira um hipervetor p/ o núcleo cognitivo.
Modality-agnostic: bytes são bytes. O MESMO código acha estrutura em qualquer stream.

Roda: python percepcao_crua.py   (numpy; usa vsa_core.py)
"""
import numpy as np
from vsa_core import D, bind, bundle, permute, cos, ItemMemory


class RawPerception:
    """Recebe stream cru, não sabe o que é, descobre chunks por estatística (como bebê)."""
    def __init__(self):
        self.sym = ItemMemory()                # símbolo cru -> hipervetor (atribuído on-the-fly)
        self.uni = {}                          # contagem de símbolos
        self.bi = {}                           # contagem de bigramas (a,b)
        self.chunks = ItemMemory()             # 'palavras' emergentes -> hipervetor

    def _count(self, stream):
        for i, s in enumerate(stream):
            self.uni[s] = self.uni.get(s, 0) + 1
            if i + 1 < len(stream):
                key = (s, stream[i + 1])
                self.bi[key] = self.bi.get(key, 0) + 1

    def transition_prob(self, a, b):
        """P(b|a) = count(ab)/count(a). Alta dentro de palavra, baixa entre palavras."""
        if self.uni.get(a, 0) == 0:
            return 0.0
        return self.bi.get((a, b), 0) / self.uni[a]

    def segment(self, stream):
        """Segmenta em mínimos locais de previsibilidade (Saffran). SEM rótulos."""
        self._count(stream)
        tp = [self.transition_prob(stream[i], stream[i + 1]) for i in range(len(stream) - 1)]
        thr = float(np.mean(tp))               # só corta onde a previsibilidade cai ABAIXO da média
        boundaries = set()
        for i in range(1, len(tp) - 1):
            # fronteira = mínimo local E baixa previsibilidade absoluta (Saffran: dip real)
            if tp[i] < tp[i - 1] and tp[i] <= tp[i + 1] and tp[i] < thr:
                boundaries.add(i + 1)
        # monta segmentos
        segs, start = [], 0
        for b in sorted(boundaries):
            segs.append(stream[start:b]); start = b
        segs.append(stream[start:])
        return [s for s in segs if s]

    def encode_chunk(self, chunk):
        """Chunk cru -> hipervetor: bind posicional dos símbolos (ordem importa) + id do chunk."""
        parts = [permute(self.sym.get(str(s)), k) for k, s in enumerate(chunk)]
        v = bundle(parts) if parts else self.sym.get("<empty>")
        key = "".join(str(c) for c in chunk)
        self.chunks.labels.append(key) if key not in self.chunks.labels else None
        return v


def boundary_f1(pred_segs, true_words):
    """Compara fronteiras descobertas vs reais (word boundary detection)."""
    def bounds(segs):
        b, pos = set(), 0
        for s in segs[:-1]:
            pos += len(s); b.add(pos)
        return b
    pred_b = bounds(pred_segs)
    true_b = bounds([list(w) for w in true_words])
    if not pred_b and not true_b:
        return 1.0
    tp = len(pred_b & true_b)
    prec = tp / max(len(pred_b), 1); rec = tp / max(len(true_b), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def demo():
    print("=" * 84)
    print("AXON core v2 — PERCEPÇÃO CRUA: descobre 'palavras' de um stream, como um bebê")
    print("=" * 84)

    # corpus: frases repetidas (o bebê ouve as mesmas coisas muitas vezes).
    # ENTRADA = stream de caracteres SEM ESPAÇOS (fala contínua). O sistema não sabe o que é.
    frases = [
        "obebeaprende", "obebecome", "obebedorme", "amaebrinca", "opaifala",
        "obebeaprende", "amaebrinca", "obebecome", "opaifala", "obebedorme",
        "obebeaprende", "obebecome", "amaebrinca", "obebedorme", "opaifala",
    ]
    true_words = ["obebe", "aprende", "come", "dorme", "amae", "brinca", "opai", "fala"]
    stream = list("".join(frases))             # stream cru de chars, sem fronteiras
    print(f"\nstream cru ({len(stream)} símbolos, SEM espaços): '{''.join(stream[:40])}...'")
    print("o sistema NÃO recebe onde as palavras começam/terminam. Só o fluxo.")

    p = RawPerception()
    segs = p.segment(stream)
    seg_words = ["".join(s) for s in segs]
    print(f"\n[1] CHUNKS DESCOBERTOS (por probabilidade de transição, sem rótulos):")
    from collections import Counter
    top = Counter(seg_words).most_common(10)
    print("    mais frequentes:", ", ".join(f"'{w}'({n})" for w, n in top))

    f1 = boundary_f1(segs, ["".join(frases)] and _true_segmentation(frases, true_words))
    print(f"\n[2] ACERTO DE FRONTEIRAS vs palavras reais: F1 = {f1:.2f}")
    print("    (>0.5 já mostra que emergiu estrutura de palavra do fluxo cru — como Saffran 1996)")

    # 3. modality-agnostic: MESMO código num stream 'não-texto' (ex.: 'áudio' quantizado)
    print("\n[3] MODALITY-AGNOSTIC: mesmo mecanismo num stream numérico (finge ser 'áudio'):")
    rng = np.random.default_rng(0)
    motifs = [[10, 11, 12], [20, 21], [30, 31, 32, 33]]     # 'padrões sonoros' recorrentes
    audio = []
    for _ in range(40):
        audio += motifs[rng.integers(len(motifs))]
    p2 = RawPerception()
    segs2 = p2.segment(audio)
    from collections import Counter as C2
    top2 = C2(tuple(s) for s in segs2).most_common(5)
    print("    'motifs' descobertos:", ", ".join(f"{list(m)}({n})" for m, n in top2))
    print("    => o MESMO código achou os padrões recorrentes num stream que não é texto.")

    print("\n" + "=" * 84)
    print("Isto é o princípio do bebê: recebe estímulo CRU, não sabe o que é, e a estrutura")
    print("EMERGE da estatística. Nada de features dadas à mão. Modality-agnostic e leve.")
    print("Próximo: chunks -> hipervetores -> núcleo cognitivo (composição/memória já prontos).")


def _true_segmentation(frases, true_words):
    """Segmentação verdadeira (só p/ avaliar F1) — o sistema NÃO usa isto."""
    out = []
    for f in frases:
        i = 0
        while i < len(f):
            for w in sorted(true_words, key=len, reverse=True):
                if f[i:i + len(w)] == w:
                    out.append(list(w)); i += len(w); break
            else:
                out.append([f[i]]); i += 1
    return out


def _selftest():
    p = RawPerception()
    # stream com um padrão repetido -> transição interna alta, fronteira na quebra
    stream = list("abcabcabcabc")
    segs = p.segment(stream)
    assert len(segs) >= 2 and all(segs)
    # símbolo cru vira hipervetor consistente
    v1 = p.sym.get("a"); v2 = p.sym.get("a")
    assert np.array_equal(v1, v2)
    # transição interna 'a->b' (sempre juntos) > transição de quebra 'c->a'
    p._count(list("abcabcabcabc"))
    assert p.transition_prob("a", "b") > p.transition_prob("c", "a") - 1e-9
    print("[selftest] ok (segmenta stream cru; símbolo->hipervetor estável; TP interna alta)")


if __name__ == "__main__":
    _selftest()
    demo()
