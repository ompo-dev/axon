"""
MENTE — uma mente só: modelo-de-mundo que GENERALIZA + raciocínio MULTI-PASSO com lookahead.

O usuário: "nada impede que ajam juntos numa mesma mente, pois é isso que o cérebro é." Certo.
E o salto além do pensador (1 passo) é o RACIOCÍNIO EM CADEIA com DESVIO: às vezes p/ chegar
no objetivo você tem que ir "pra trás" primeiro (ligar um gate, dar a volta). Um agente REATIVO
(guloso, 1 passo — como um LLM que só prevê o próximo) TRAVA nisso. Só o PLANEJAMENTO multi-passo
(simular no modelo, achar o caminho) resolve. É o cerne do "LLMs can't jump".

Fundamentado (2025-26): LeCun/JEPA Modo-2 (planejar simulando rollouts); MAP/Nature 09/2025
(módulos tipo-PFC planejando em grafo); EBM avalia trajetória parcial e CORRIGE o rumo.

Mundo com PORTÕES: alternar um interruptor 'travado' exige uma luz ligada AGORA. A mente:
  1. explora e APRENDE (sem backprop): luz=f(interruptores) e portão=f(luzes). Generaliza.
  2. PENSADOR: busca (BFS no modelo) uma SEQUÊNCIA de passos até o objetivo, respeitando portões.
  3. OBSERVADOR: executa e, se um passo trava (modelo errou o portão), RE-PLANEJA (anti-derailar).

Compara 3 mentes: DECORADOR (só o que visitou) · REATIVO (guloso 1 passo) · PENSADOR (multi-passo).
Roda: python mente.py   (numpy)
"""
import sys
from collections import deque
import numpy as np

RNG = np.random.default_rng(3)


# ============================================================ o MUNDO (dinâmica + portões ocultos)
class Mundo:
    def __init__(self, n_sw=12, n_luz=5, seed=0):
        rng = np.random.default_rng(seed)
        self.n_sw, self.n_luz = n_sw, n_luz
        self.regras = []                                    # luz = AND/OR de literais (negáveis)
        for _ in range(n_luz):
            k = int(rng.integers(2, 4))
            ins = rng.choice(n_sw, k, replace=False)
            neg = rng.integers(0, 2, k)
            op = "AND" if rng.random() < 0.5 else "OR"
            self.regras.append((ins, neg, op))
        # portões: cada interruptor ou é livre (-1) ou exige uma luz ON p/ ser alternado
        self.gate = [int(rng.integers(-1, n_luz)) if rng.random() < 0.55 else -1
                     for _ in range(n_sw)]

    def luzes(self, sw):
        out = np.zeros(self.n_luz, int)
        for j, (ins, neg, op) in enumerate(self.regras):
            lit = sw[ins] ^ neg
            out[j] = int(lit.all()) if op == "AND" else int(lit.any())
        return out

    def permitido(self, sw, i):
        g = self.gate[i]
        return g < 0 or self.luzes(sw)[g] == 1              # portão: precisa da luz g ligada

    def passo(self, sw, i):
        """Alterna i SE o portão deixar. Retorna (novo_estado, mudou?)."""
        if self.permitido(sw, i):
            nsw = sw.copy(); nsw[i] ^= 1
            return nsw, True
        return sw, False


# ============================================================ MODELO (perceptrons, regra delta, sem backprop)
class Perc:
    def __init__(self, n_in, n_out):
        self.W = np.zeros((n_out, n_in + 1))

    def _f(self, x):
        return np.concatenate([x * 2 - 1, [1.0]])

    def prever(self, x):
        return (self.W @ self._f(x) > 0).astype(int)

    def aprender(self, x, y, lr=0.1):
        f = self._f(x)
        self.W += lr * np.outer(y - (self.W @ f > 0).astype(int), f)


# ============================================================ a MENTE (as três + modelo aprendido)
class Mente:
    def __init__(self, mundo):
        self.mundo = mundo
        self.mluz = Perc(mundo.n_sw, mundo.n_luz)           # luz = f(interruptores)
        self.mgate = Perc(mundo.n_luz, mundo.n_sw)          # permitido_i = f(luzes atuais)
        self.visto = {}                                     # decorador: luzes(tuple)->config
        self.buf_luz, self.buf_gate = [], []

    def explorar(self, passos=1500, replay=8, reinicia=25):
        """EXECUTOR + aprendizado: passeio respeitando portões REAIS; aprende luz e portão.
        Reinicia de config aleatória a cada 'reinicia' passos p/ cobrir o espaço (senão o portão
        prende a exploração numa região e o modelo de luz fica torto)."""
        sw = RNG.integers(0, 2, self.mundo.n_sw)
        for t in range(passos):
            if t % reinicia == 0:
                sw = RNG.integers(0, 2, self.mundo.n_sw)    # teleporte: cobre o espaço p/ aprender luz
            luz = self.mundo.luzes(sw)
            self.buf_luz.append((sw.copy(), luz))
            self.visto[tuple(luz)] = sw.copy()
            i = int(RNG.integers(self.mundo.n_sw))
            perm = np.array([1 if self.mundo.permitido(sw, j) else 0 for j in range(self.mundo.n_sw)])
            self.buf_gate.append((luz.copy(), perm))        # aprende quando cada toggle é liberado
            nsw, _ = self.mundo.passo(sw, i)
            sw = nsw
        for _ in range(replay):                             # consolida (sono/CLS), local, sem backprop
            RNG.shuffle(self.buf_luz); RNG.shuffle(self.buf_gate)
            for x, y in self.buf_luz:
                self.mluz.aprender(x, y)
            for x, y in self.buf_gate:
                self.mgate.aprender(x, y)

    # ---------- PENSADOR: busca multi-passo no MODELO (Modo-2) ----------
    def planejar(self, inicio, meta, cap=6000):
        """BFS no modelo: acha sequência de toggles de 'inicio' até luzes==meta, respeitando o
        portão PREDITO. Retorna lista de interruptores a alternar, ou None. Raciocínio em cadeia."""
        ini = tuple(int(b) for b in inicio)
        if np.array_equal(self.mluz.prever(np.array(ini)), meta):
            return []
        vis = {ini}
        fila = deque([(ini, [])])
        while fila and len(vis) < cap:
            s, caminho = fila.popleft()
            sa = np.array(s)
            perm = self.mgate.prever(self.mluz.prever(sa))  # portão predito pelo modelo
            for i in range(self.mundo.n_sw):
                if not perm[i]:
                    continue
                ns = list(s); ns[i] ^= 1; ns = tuple(ns)
                if ns in vis:
                    continue
                if np.array_equal(self.mluz.prever(np.array(ns)), meta):
                    return caminho + [i]
                vis.add(ns); fila.append((ns, caminho + [i]))
        return None

    # ---------- REATIVO: guloso 1 passo (só o próximo melhor; trava em não-monótono) ----------
    def reativo(self, inicio, meta, passos=40):
        sw = inicio.copy()
        for _ in range(passos):
            if np.array_equal(self.mundo.luzes(sw), meta):
                return True
            perm = self.mgate.prever(self.mluz.prever(sw))
            atual = int(np.sum(self.mluz.prever(sw) != meta))
            best, bi = atual, -1
            for i in range(self.mundo.n_sw):
                if not perm[i]:
                    continue
                ns = sw.copy(); ns[i] ^= 1
                d = int(np.sum(self.mluz.prever(ns) != meta))
                if d < best:
                    best, bi = d, i
            if bi < 0:                                      # nenhum passo MELHORA -> travou
                return False
            sw, _ = self.mundo.passo(sw, bi)                # age no mundo real
        return np.array_equal(self.mundo.luzes(sw), meta)

    # ---------- OBSERVADOR: executa o plano no mundo real, re-planeja se travar ----------
    def executar(self, inicio, meta, corrige=True):
        sw = inicio.copy()
        for _ in range(8):                                  # até 8 re-planejamentos
            plano = self.planejar(sw, meta)
            if plano is None:
                return False
            derrapou = False
            for i in plano:
                nsw, mudou = self.mundo.passo(sw, i)        # EXECUTOR
                sw = nsw
                if not mudou and corrige:                   # OBSERVADOR: portão barrou (modelo errou)
                    self.mgate.aprender(self.mundo.luzes(sw),
                                        np.array([1 if self.mundo.permitido(sw, j) else 0
                                                  for j in range(self.mundo.n_sw)]))
                    derrapou = True
                    break
            if np.array_equal(self.mundo.luzes(sw), meta):
                return True
            if not derrapou:
                return False
        return np.array_equal(self.mundo.luzes(sw), meta)

    def decorador(self, meta):
        return self.visto.get(tuple(meta))


# ============================================================ TESTE HONESTO
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 76)
    print("MENTE — raciocínio MULTI-PASSO com desvio: reativo (1 passo) TRAVA; só planejar resolve")
    print("=" * 76)

    N = 20
    accL, accG = [], []
    rea = pen = dec_nav = tot = 0
    pen_so = 0                 # pensador resolve E reativo trava = precisou de multi-passo (o 'pulo')
    passos_pen = []

    def alcancaveis(mundo, ini):
        """Verdade-terreno: configs (e padrões de luz) REALMENTE atingíveis de 'ini' sob os portões."""
        vis = {tuple(int(b) for b in ini)}
        fila = deque([ini]); pats = set()
        while fila:
            s = fila.popleft()
            pats.add(tuple(mundo.luzes(s)))
            for i in range(mundo.n_sw):
                nsw, mudou = mundo.passo(s, i)
                if mudou and tuple(nsw) not in vis:
                    vis.add(tuple(nsw)); fila.append(nsw)
        return vis, pats

    for s in range(N):
        mundo = Mundo(seed=s)
        m = Mente(mundo)
        m.explorar(1500)
        cfgs = [RNG.integers(0, 2, 12) for _ in range(200)]
        accL.append(np.mean([np.array_equal(m.mluz.prever(c), mundo.luzes(c)) for c in cfgs]))
        accG.append(np.mean([np.array_equal(
            m.mgate.prever(mundo.luzes(c)),
            np.array([1 if mundo.permitido(c, j) else 0 for j in range(12)])) for c in cfgs]))

        rng_m = np.random.default_rng(500 + s)
        for _ in range(30):
            ini = RNG.integers(0, 2, 12)
            _, pats = alcancaveis(mundo, ini)               # metas SÓ entre as REALMENTE atingíveis (justo)
            pats = [p for p in pats if not np.array_equal(mundo.luzes(ini), p)]
            if not pats:
                continue
            meta = np.array(pats[rng_m.integers(len(pats))])
            tot += 1
            ok_rea = m.reativo(ini.copy(), meta)
            ok_pen = m.executar(ini.copy(), meta)
            # foil de memória, mas AGORA tem que NAVEGAR até a config lembrada (guloso rumo a ela):
            c = m.decorador(meta)
            ok_dec = c is not None and m.reativo(ini.copy(), mundo.luzes(c))
            rea += ok_rea; pen += ok_pen; dec_nav += ok_dec
            if ok_pen and not ok_rea:
                pen_so += 1
                p = m.planejar(ini.copy(), meta)
                if p:
                    passos_pen.append(len(p))

    print(f"\nmodelo (sem backprop): luz {np.mean(accL):.0%} acc  ·  portão {np.mean(accG):.0%} acc")
    print(f"\nmetas REALMENTE ATINGÍVEIS do início ({tot}) — todos têm que NAVEGAR sob os portões:")
    print(f"  MEMÓRIA→guloso (lembra a config, tenta chegar): {dec_nav}/{tot} = {dec_nav/tot:.0%}")
    print(f"  REATIVO (guloso, 1 passo de lookahead):         {rea}/{tot} = {rea/tot:.0%}")
    print(f"  PENSADOR (multi-passo + desvio):                {pen}/{tot} = {pen/tot:.0%}")
    print(f"\n>>> o PULO — metas que o PENSADOR resolve e o REATIVO TRAVA: {pen_so}")
    if passos_pen:
        print(f"    exigiram cadeias de {np.mean(passos_pen):.1f} passos em média (máx {max(passos_pen)}) —")
        print(f"    incl. DESVIOS (ir 'pra trás' p/ abrir um portão). O guloso não enxerga isso.")

    print("\n" + "=" * 76)
    if pen > rea and pen > dec_nav and pen_so > 0:
        print("VEREDITO: entre metas REALMENTE atingíveis, só o raciocínio MULTI-PASSO (simular a cadeia")
        print(f"no modelo, com desvios) chega lá. O guloso/1-passo TRAVA ({pen_so} casos) e a pura memória")
        print("não sabe NAVEGAR sob os portões. É o Modo-2 (LeCun/JEPA, MAP/Nature) — do zero, sem")
        print("backprop, medido. O 'pulo' que o LLM (essencialmente guloso no próximo token) não dá.")
    else:
        print("VEREDITO honesto: sinal fraco — reportado como está, sem inflar.")


def _selftest():
    mundo = Mundo(n_sw=8, n_luz=3, seed=2)
    m = Mente(mundo)
    m.explorar(1200)
    cfgs = [RNG.integers(0, 2, 8) for _ in range(100)]
    accL = np.mean([np.array_equal(m.mluz.prever(c), mundo.luzes(c)) for c in cfgs])
    assert accL > 0.8, accL
    # planeja e executa alguma meta achável a partir de um início aleatório
    ini = RNG.integers(0, 2, 8)
    meta = mundo.luzes(RNG.integers(0, 2, 8))
    p = m.planejar(ini, meta)
    assert p is not None, "não achou plano p/ meta achável"
    assert m.executar(ini.copy(), meta), "plano não atingiu a meta no mundo real"
    print(f"[selftest] ok (modelo {accL:.0%}; planejou {len(p)} passos e atingiu a meta)")


if __name__ == "__main__":
    _selftest()
    main()
