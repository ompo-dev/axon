"""
AXON core v2 — degrau 2: aprendizado CONTÍNUO, na surpresa, sem esquecer.

Fundamento neurocientífico (pesquisado, não inventado):
  - Predictive coding (Friston): o cérebro PREVÊ; aprende do ERRO de predição.
    => só liga/aprende quando erra (surpresa). Poupa capacidade; não re-aprende o sabido.
  - Complementary Learning Systems (McClelland-McNaughton-O'Reilly 1995): hipocampo = rápido,
    1-shot, episódico; neocórtex = lento, semântico. Dois armazéns evitam esquecimento
    catastrófico — o que LLM NÃO faz (não aprende no uso).
  - Binding por sincronia gama (fase): a bind ⊗ da VSA é a versão algébrica.

Aqui: memória heteroassociativa VSA (cue->alvo) que
  1) PREVÊ o alvo de um cue;
  2) aprende (liga) só quando ERRA (predictive coding);
  3) NÃO esquece fatos antigos ao aprender novos (o teste que LLM falha);
  4) mede honestamente a CAPACIDADE (onde satura) e como o 2º armazém (CLS) estende.

Roda: python predictive_cls.py   (numpy; usa vsa_core.py)
"""
import numpy as np
from vsa_core import D, rand_vec, bind, bundle, cos, ItemMemory


class ContinualLearner:
    """Memória heteroassociativa VSA com aprendizado na surpresa + consolidação CLS.
    'Hipocampo' = episódios separados por cue (pattern separation -> permite ATUALIZAR).
    'Neocórtex' = bundles consolidados (muitos episódios juntos; distribuído, não-atualizável)."""
    def __init__(self, slot_capacity=100):
        self.mem = ItemMemory()
        self.fast = {}                         # cue_label -> bind(cue,alvo)  (1 episódio por cue)
        self.fast_M = None
        self.slow_slots = []                   # neocórtex consolidado
        self.slot_capacity = slot_capacity
        self.n_learned = 0
        self.n_skipped = 0

    def _rebuild_fast(self):
        self.fast_M = bundle(list(self.fast.values())) if self.fast else None

    def _stores(self):
        return ([self.fast_M] if self.fast_M is not None else []) + self.slow_slots

    def predict(self, cue):
        cv = self.mem.get(cue)
        if not self._stores():
            return None, 0.0
        acc = np.zeros(D, dtype=np.int32)
        for M in self._stores():
            acc += bind(M, cv).astype(np.int32)
        guess = np.where(acc > 0, 1, -1).astype(np.int8)
        top = self.mem.cleanup(guess, topk=1, exclude=(cue,))
        return (top[0][0], top[0][1]) if top else (None, 0.0)

    def observe(self, cue, target):
        """Predictive coding: prevê; aprende SÓ se a predição está errada (surpresa)."""
        self.mem.get(target)
        pred, conf = self.predict(cue)
        # "sabe" = prediz certo E com confiança (evita pular por acerto de sorte, cos~0)
        if pred == target and conf > 0.08:
            self.n_skipped += 1
            return False                        # já sabia; não aprende
        # aprende/atualiza: substitui o episódio daquele cue (pattern separation)
        self.fast[cue] = bind(self.mem.get(cue), self.mem.get(target))
        self._rebuild_fast()
        self.n_learned += 1
        if len(self.fast) >= self.slot_capacity:   # consolida: rápido -> neocórtex
            self.slow_slots.append(self.fast_M)
            self.fast = {}; self.fast_M = None
        return True

    def accuracy(self, facts):
        ok = sum(1 for c, t in facts if self.predict(c)[0] == t)
        return ok / len(facts)


# ============================================================ experimentos honestos
def make_facts(n, seed):
    rng = np.random.default_rng(seed)
    return [(f"cue{rng.integers(10**9)}", f"alvo{rng.integers(10**9)}") for _ in range(n)]


def demo():
    print("=" * 82)
    print("AXON core v2 — aprendizado CONTÍNUO na surpresa, sem esquecer (predictive coding + CLS)")
    print("=" * 82)

    # 1. SEM ESQUECER: aprende fatos em stream; recorda TODOS depois (o teste que LLM falha)
    print("\n[1] SEM ESQUECIMENTO CATASTRÓFICO: aprende N fatos em stream 1-shot; recorda todos?")
    print(f"    {'N fatos':>8}{'acc (todos)':>13}{'acertos recentes':>18}{'slots CLS':>11}")
    for N in (20, 40, 80, 160, 320):
        cl = ContinualLearner()
        facts = make_facts(N, seed=N)
        for c, t in facts:
            cl.observe(c, t)
        acc_all = cl.accuracy(facts)
        acc_recent = cl.accuracy(facts[-20:])
        print(f"    {N:>8}{acc_all:>13.1%}{acc_recent:>18.1%}{len(cl.slow_slots):>11}")
    print("    (acc alta e estável mesmo com N grande = NÃO esquece. Cai = saturou a capacidade.)")

    # 2. APRENDER NA SURPRESA: re-apresentar fatos conhecidos não gasta capacidade
    print("\n[2] PREDICTIVE CODING: re-ver fato conhecido NÃO gasta capacidade (aprende só na surpresa)")
    cl = ContinualLearner()
    facts = make_facts(30, seed=7)
    for c, t in facts:
        cl.observe(c, t)
    learned1 = cl.n_learned
    for c, t in facts:                          # re-apresenta TUDO de novo
        cl.observe(c, t)
    print(f"    1ª passada: aprendeu {learned1} fatos novos.")
    print(f"    2ª passada (mesmos fatos): aprendeu +{cl.n_learned - learned1}, pulou {cl.n_skipped} (já sabia).")
    print("    => não desperdiça capacidade re-aprendendo o conhecido. É o 'aprende do erro'.")

    # 3. ATUALIZAÇÃO: corrigir um fato muda a resposta na hora (plasticidade em tempo real)
    print("\n[3] ATUALIZAÇÃO EM TEMPO REAL: corrigir um fato muda a predição imediatamente")
    cl = ContinualLearner()
    cl.observe("capital_franca", "paris")
    print(f"    capital_franca -> {cl.predict('capital_franca')[0]}")
    cl.observe("capital_franca", "lyon")        # 'correção'/atualização
    print(f"    (após atualizar) capital_franca -> {cl.predict('capital_franca')[0]}")

    print("\n" + "=" * 82)
    print("LLM padrão NÃO faz [1]-[3] no uso: precisa re-treino, esquece, não atualiza em tempo real.")
    print("Aqui é 1-shot, na surpresa, sem esquecer, leve. Limite honesto: capacidade finita por")
    print("bundle (satura); o 2º armazém (CLS) adia, não elimina. Escalar bem é trabalho real.")


def _selftest():
    cl = ContinualLearner()
    facts = make_facts(15, seed=1)
    for c, t in facts:
        cl.observe(c, t)
    assert cl.accuracy(facts) > 0.95, cl.accuracy(facts)     # aprende e recorda
    # aprende na surpresa: re-apresentar não gasta
    before = cl.n_learned
    for c, t in facts:
        cl.observe(c, t)
    assert cl.n_learned == before, (before, cl.n_learned)    # nada novo aprendido
    assert cl.n_skipped >= len(facts)
    # atualização em tempo real
    cl.observe("k", "v1"); assert cl.predict("k")[0] == "v1"
    cl.observe("k", "v2"); assert cl.predict("k")[0] == "v2"
    print("[selftest] ok (aprende+recorda; aprende só na surpresa; atualiza em tempo real)")


if __name__ == "__main__":
    _selftest()
    demo()
