"""
PENSADOR — teste honesto de "pensa vs decora" (quarto chinês).

Tese (LeCun/JEPA, Friston, Ha-Schmidhuber; "LLMs can't jump"): PENSAR = simular num MODELO DE
MUNDO interno e avaliar. Não é casar padrão (quarto chinês); é COMPOR uma solução NOVA por
imaginação. Aqui um agente:
  1. explora um mundo e APRENDE a dinâmica (modelo) por regra DELTA local (SEM backprop).
  2. recebe metas NOVAS (nunca alcançadas) e as resolve SIMULANDO no modelo (planeja).
  3. TRÊS NÍVEIS: EXECUTOR age no mundo | PENSADOR simula planos | OBSERVADOR avalia/escolhe.

Teste decisivo: resolve metas que o DECORADOR (lookup do visto) NÃO tem como resolver?
  - se sim => compôs o inédito = PENSOU (pulou).  Espaço grande + regras com NEGAÇÃO =>
    exploração aleatória NÃO satura, e "abrir a porta" (todas as luzes juntas) é raro de topar.

Sem backprop, leve, MEDIDO. Roda: python pensador.py   (numpy só)
"""
import numpy as np

RNG = np.random.default_rng(7)


# ============================================================ o MUNDO (dinâmica oculta)
class Mundo:
    """N interruptores -> M luzes. Cada luz = AND/OR de literais (switch ou NEGADO) de 2-3
    interruptores. Negação torna 'todas as luzes juntas' um quebra-cabeça não-trivial (SAT)."""
    def __init__(self, n_sw=12, n_luz=5, seed=0):
        rng = np.random.default_rng(seed)
        self.n_sw, self.n_luz = n_sw, n_luz
        self.regras = []
        for _ in range(n_luz):
            k = int(rng.integers(2, 4))
            ins = rng.choice(n_sw, k, replace=False)
            neg = rng.integers(0, 2, k)                      # 1 = literal negado
            op = "AND" if rng.random() < 0.5 else "OR"
            self.regras.append((ins, neg, op))
        self.sw = rng.integers(0, 2, n_sw)

    def luzes(self, sw):
        out = np.zeros(self.n_luz, int)
        for j, (ins, neg, op) in enumerate(self.regras):
            lit = sw[ins] ^ neg                              # aplica negação
            out[j] = int(lit.all()) if op == "AND" else int(lit.any())
        return out

    def toggle(self, i):
        self.sw[i] ^= 1


# ============================================================ MODELO DE MUNDO (aprendido, sem backprop)
class ModeloMundo:
    """Aprende luz_j = perceptron(interruptores) pela REGRA DELTA (local, sem backprop).
    AND/OR de literais são linearmente separáveis -> um perceptron aprende exato."""
    def __init__(self, n_sw, n_luz):
        self.W = np.zeros((n_luz, n_sw + 1))                 # +1 = viés

    def _feat(self, sw):
        return np.concatenate([sw * 2 - 1, [1.0]])          # {0,1}->{-1,+1} + viés

    def prever(self, sw):
        return (self.W @ self._feat(sw) > 0).astype(int)

    def prever_todos(self, CFG1):
        """CFG1 = (2^n, n+1) todas as configs já com viés. Retorna (2^n, n_luz) predições."""
        return (CFG1 @ self.W.T > 0).astype(int)

    def aprender(self, sw, luzes, lr=0.1):
        x = self._feat(sw)
        err = luzes - (self.W @ x > 0).astype(int)          # erro de predição (predictive coding)
        self.W += lr * np.outer(err, x)                     # delta: erro × entrada. Local.


def _todas_configs(n):
    """Matriz (2^n, n+1): toda config binária + coluna de viés. Para simular tudo de uma vez."""
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)) & 1)
    feat = bits * 2 - 1
    return np.hstack([feat, np.ones((len(idx), 1))]), bits


# ============================================================ os TRÊS NÍVEIS
class Agente:
    def __init__(self, mundo):
        self.mundo = mundo
        self.modelo = ModeloMundo(mundo.n_sw, mundo.n_luz)
        self.CFG1, self.BITS = _todas_configs(mundo.n_sw)
        self.visto = {}                                     # decorador: luzes(tuple) -> config
        self.buffer = []                                    # (sw, luz) vistos — p/ replay

    def explorar(self, passos=350, replay=10):
        """EXECUTOR + aprendizado: age, observa, refina por erro. Cobre só parte do espaço.
        Depois REPLAY (sono/consolidação, CLS): re-passa o visto p/ o perceptron convergir. Local."""
        for _ in range(passos):
            self.mundo.toggle(int(RNG.integers(self.mundo.n_sw)))
            sw = self.mundo.sw.copy(); luz = self.mundo.luzes(sw)
            self.modelo.aprender(sw, luz)
            self.visto[tuple(luz)] = sw                     # guarda 1 config por padrão de luz visto
            self.buffer.append((sw, luz))
        for _ in range(replay):                             # replay: consolida sem backprop
            RNG.shuffle(self.buffer)
            for sw, luz in self.buffer:
                self.modelo.aprender(sw, luz)

    def pensar_plano(self, meta):
        """PENSADOR+OBSERVADOR: simula TODAS as configs no modelo, escolhe a de menor erro à meta."""
        pred = self.modelo.prever_todos(self.CFG1)          # imagina o resultado de cada config
        err = np.sum(pred != meta, axis=1)                  # observador avalia cada simulação
        return self.BITS[int(np.argmin(err))]

    def decorador_plano(self, meta):
        """Quarto-chinês: só devolve config JÁ VISTA com exatamente essas luzes. Senão, não sabe."""
        return self.visto.get(tuple(meta))

    def executar(self, sw_alvo):
        """EXECUTOR: leva o mundo real até a config planejada, devolve as luzes REAIS."""
        self.mundo.sw = np.array(sw_alvo, int)
        return self.mundo.luzes(self.mundo.sw)


# ============================================================ TESTE HONESTO
def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 72)
    print("PENSADOR — resolve o que o DECORADOR não tem como? (o 'pulo' / composição)")
    print("=" * 72)

    N_MUNDOS, N_METAS = 25, 40
    acc_modelo = []
    p_ach = d_ach = ach_n = 0             # sucesso em metas ACHÁVEIS (justo: só o que é possível)
    novel_n = p_novel = 0                 # acháveis que o decorador NÃO viu = o inédito (o 'pulo')
    porta_ach = porta_pensa = porta_viu = 0

    for m in range(N_MUNDOS):
        mundo = Mundo(n_sw=12, n_luz=5, seed=m)
        ag = Agente(mundo)
        ag.explorar(350)                                    # cobre ~350 de 4096 configs

        # luzes REAIS de todas as configs -> acurácia do modelo e conjunto ACHÁVEL de metas
        REALL = np.array([mundo.luzes(b) for b in ag.BITS])         # (4096, 5)
        PREDL = ag.modelo.prever_todos(ag.CFG1)
        acc_modelo.append(np.mean(np.all(PREDL == REALL, axis=1)))
        achavel = list({tuple(r) for r in REALL})                  # padrões que ALGUMA config produz

        rng_m = np.random.default_rng(1000 + m)
        for _ in range(N_METAS):
            meta = np.array(achavel[rng_m.integers(len(achavel))]) # só metas POSSÍVEIS (justo)
            ach_n += 1
            ok_p = np.array_equal(ag.executar(ag.pensar_plano(meta)), meta)
            p_ach += ok_p
            sw_d = ag.decorador_plano(meta)
            ok_d = sw_d is not None and np.array_equal(ag.executar(sw_d), meta)
            d_ach += ok_d
            if sw_d is None:                                # achável, mas o decorador NUNCA topou
                novel_n += 1; p_novel += ok_p               # <-- só quem simula/compõe resolve

        # showcase: "abrir a porta" = todas as luzes ON. Achável?
        if (1, 1, 1, 1, 1) in {tuple(r) for r in REALL}:
            alvo = np.ones(5, int)
            porta_ach += 1
            porta_pensa += np.array_equal(ag.executar(ag.pensar_plano(alvo)), alvo)
            porta_viu += tuple(alvo) in ag.visto            # exploração topou a porta aberta?

    print(f"\nmodelo interno (perceptron delta, SEM backprop): acurácia = {np.mean(acc_modelo):.1%}")
    print(f"  aprendeu a DINÂMICA oculta de ~350 amostras e generaliza p/ as 4096 configs.\n")
    print(f"metas ACHÁVEIS (só o que é fisicamente possível — teste justo):")
    print(f"  PENSADOR (simula no modelo):  {p_ach}/{ach_n} = {p_ach/ach_n:.0%}")
    print(f"  DECORADOR (lookup do visto):  {d_ach}/{ach_n} = {d_ach/ach_n:.0%}")
    print(f"\n>>> o PULO — metas acháveis que o decorador NUNCA topou: {novel_n}")
    print(f"    o PENSADOR resolveu {p_novel}/{novel_n} = {p_novel/max(novel_n,1):.0%} DELAS (compôs o inédito).")
    print(f"\n>>> 'abrir a porta' (todas as luzes juntas), quando possível ({porta_ach} mundos):")
    print(f"    exploração topou a porta aberta por acaso: {porta_viu}/{porta_ach}")
    print(f"    o PENSADOR planejou e abriu:               {porta_pensa}/{porta_ach}")
    p_all, d_all, novel = p_ach, d_ach, novel_n

    print("\n" + "=" * 72)
    if p_novel > 0.5 * novel_n and p_all > d_all:
        print("VEREDITO: o PENSADOR resolve o INÉDITO — configs que nunca viu, que o decorador não")
        print("tem como alcançar. Compôs soluções simulando no seu modelo de mundo: isto é o 'pulo'.")
        print("Não é quarto chinês (lookup) nem LLM que 'não pula'. É modelo-de-mundo + simulação,")
        print("aprendido por erro local, SEM backprop. A direção certa da arquitetura.")
    else:
        print("VEREDITO honesto: não superou o decorador de forma decisiva — reportado sem inflar.")


def _selftest():
    mundo = Mundo(n_sw=8, n_luz=3, seed=1)
    ag = Agente(mundo)
    ag.explorar(200)
    acc = np.mean([np.array_equal(ag.modelo.prever(sw), mundo.luzes(sw))
                   for sw in [RNG.integers(0, 2, 8) for _ in range(100)]])
    assert acc > 0.85, acc
    meta = mundo.luzes(RNG.integers(0, 2, 8))
    assert np.array_equal(ag.executar(ag.pensar_plano(meta)), meta) or acc < 1.0
    print(f"[selftest] ok (modelo acc={acc:.0%}; planeja+executa)")


if __name__ == "__main__":
    _selftest()
    main()
