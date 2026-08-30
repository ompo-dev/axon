"""
MIMETIZADOR — aprender REPRODUZINDO, depois GERAR. Uma peça, serve texto/áudio/vídeo (DRY).

Neurociência (fundamenta): NEURÔNIOS-ESPELHO (Rizzolatti) — os mesmos disparam ao VER e ao FAZER:
percepção e produção compartilham a MESMA representação. Aqui: um dicionário ONLINE de protótipos;
sentir(v) e gerar() usam os mesmos protótipos (espelho). CÓPIA EFERENTE / corollary discharge: ao
gerar (agir), a mente se percebe (ouve a própria voz) mas aprende disso ATENUADO (distingue 'eu' de
'mundo') — refina a motricidade sem virar ruído (balbucio: gerar+ouvir-se+refinar).

Ágil (aprender rápido, como pediu): protótipo NOVO na NOVIDADE (RMS por-dim > limiar) e taxa alta.
Vetorizado (numpy): distância a todos os protótipos de uma vez. Sem backprop.
Roda: python mimetizador.py  (selftest: erro de reconstrução cai; gera; novidade cria protótipo)
"""
from collections import Counter
import numpy as np


class Mimetizador:
    def __init__(self, dim, k=64, lr=0.35, novidade=0.30, seed=0):
        self.dim, self.k, self.lr, self.nov = dim, k, lr, novidade
        self.P = np.zeros((0, dim), np.float32)              # protótipos (o vocabulário), cresce
        self.assoc = {}                                      # conceito -> Counter(protótipo) p/ gerar
        self.rng = np.random.default_rng(seed)
        self._w = None                                       # pesos do 'sonho' (blend que deriva)
        self._noise = None                                   # perturbação do sonho (deriva no tempo)

    def sentir(self, v, foco=None, taxa=None):
        """Recebe vetor cru; se for NOVO cria protótipo (aprende rápido), senão aproxima o mais
        próximo (imita). taxa < lr = auto-percepção atenuada (cópia eferente). Devolve (reconstrução, i)."""
        v = np.asarray(v, np.float32)
        if len(self.P) == 0:
            self.P = v[None, :].copy(); j = 0
        else:
            d = np.sum((self.P - v) ** 2, axis=1)            # distância a TODOS (vetorizado)
            j = int(d.argmin())
            if d[j] > (self.nov ** 2) * self.dim and len(self.P) < self.k:
                self.P = np.vstack([self.P, v[None, :]]); j = len(self.P) - 1   # NOVIDADE -> novo protótipo
            else:
                self.P[j] += (self.lr if taxa is None else taxa) * (v - self.P[j])
        if foco:
            for c in foco:
                self.assoc.setdefault(c, Counter())[j] += 1
        return self.P[j].copy(), j

    def gerar(self, foco=None):
        """Evoca um protótipo — o ligado ao que pensa (espelho: pensar evoca o que viu/ouviu)."""
        if len(self.P) == 0:
            return None
        if foco:
            for c in foco:
                if c in self.assoc and self.assoc[c]:
                    return self.P[self.assoc[c].most_common(1)[0][0]].copy()
        return self.P[int(self.rng.integers(len(self.P)))].copy()

    def sonhar(self, foco=None, drift=0.25):
        """IMAGINAR contínuo: um BLEND dos protótipos cujos pesos DERIVAM (passeio aleatório) e são
        puxados pelo pensamento (foco). Cria combinações NOVAS (in-betweens) sem parar — mesmo com
        poucos protótipos. É recombinação (o córtex pré-frontal recombina o que os espelhos evocam)."""
        if len(self.P) == 0:
            return None
        if self._w is None or len(self._w) != len(self.P):
            self._w = np.ones(len(self.P), np.float32)
        self._w += self.rng.normal(0, drift, len(self.P)).astype(np.float32)
        if foco:
            for c in foco:                                   # o pensamento puxa o sonho pro que evoca
                for i, _ in self.assoc.get(c, Counter()).most_common(2):
                    if i < len(self._w):
                        self._w[i] += drift * 3
        self._w = np.clip(self._w, 0.02, None)
        w = self._w / self._w.sum()
        base = (w[:, None] * self.P).sum(0)
        # o sonho RESPIRA: perturbação que deriva no tempo (AR1) -> muda SEMPRE, mesmo com 1 protótipo
        if self._noise is None or len(self._noise) != self.dim:
            self._noise = np.zeros(self.dim, np.float32)
        self._noise = 0.8 * self._noise + 0.2 * self.rng.normal(0, 0.11, self.dim).astype(np.float32)
        return np.clip(base + self._noise, 0.0, 1.0)

    def rotular(self, idx, minimo=2):
        """REVERSO (espelho): dado um protótipo (ex: o rosto que acabei de ver), qual CONCEITO é?
        Permite RECONHECER (ver o rosto -> lembrar o nome). Liga as modalidades num espaço só."""
        best, bc = None, 0
        for c, cnt in self.assoc.items():
            v = cnt.get(idx, 0)
            if v > bc:
                bc, best = v, c
        return best if bc >= minimo else None

    def erro(self, v):
        v = np.asarray(v, np.float32)
        if len(self.P) == 0:
            return float("inf")
        return float(np.min(np.sum((self.P - v) ** 2, axis=1))) ** 0.5


def _selftest():
    rng = np.random.default_rng(1)
    dim, K = 40, 6
    centros = rng.normal(0, 1, (K, dim)).astype(np.float32)
    m = Mimetizador(dim, k=32, lr=0.3)
    amostra = lambda: (centros[rng.integers(K)] + rng.normal(0, 0.12, dim)).astype(np.float32)
    err0 = np.mean([m.erro(amostra()) for _ in range(50)])
    for _ in range(1500):
        m.sentir(amostra())
    err1 = np.mean([m.erro(amostra()) for _ in range(200)])
    assert err1 < 1.2, (err0, err1)
    assert K <= len(m.P) <= 32, len(m.P)                     # criou ~K protótipos (novidade), não explodiu
    g = m.gerar()
    assert min(np.linalg.norm(g - c) for c in centros) < 1.5
    # auto-percepção atenuada muda menos que a normal
    p = m.P[0].copy(); m.sentir(m.P[0] + 1.0, taxa=0.01); d_self = np.linalg.norm(m.P[0] - p)
    print(f"[selftest] ok (erro {err0:.2f}->{err1:.2f}; {len(m.P)} protótipos; gera; auto-percepção atenua {d_self:.3f})")


if __name__ == "__main__":
    _selftest()
