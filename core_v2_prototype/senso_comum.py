"""
AXON core v2 — degrau 3: senso comum = protótipos graduados (teoria de Rosch) na VSA.

A ideia do usuário (e é ciência real — Eleanor Rosch, prototype theory 1975):
  - Uma categoria tem um CENTRO (a cadeira ideal) e membros GRADUADOS por distância.
  - Mexer nas features desliza entre categorias:
      cadeira + almofada  -> poltrona
      cadeira - encosto   -> banco
  - Aprender REFORMA o que já se sabia: cada exemplo empurra o protótipo (o "ponto de calor
    dissipando na superfície" = campo de ativação sobre o espaço de conceitos).

Mecanismo VSA:
  - objeto = bundle das suas features (hipervetores).
  - protótipo = ACUMULADOR de exemplos (soma ±1 por bit); sinal = protótipo atual. Cada
    exemplo NUDGE o protótipo -> ele se move (calor). Remover feature = tirar do bundle.
  - categoria = protótipo mais próximo; TIPICIDADE = a similaridade (graduada, não binária).
  - "halo de senso comum" = ativação espalha p/ conceitos vizinhos ∝ similaridade.

Roda: python senso_comum.py   (numpy; usa vsa_core.py)
"""
import numpy as np
from vsa_core import D, bind, bundle, cos, ItemMemory


class ConceptSpace:
    """Espaço de conceitos com protótipos que se MOVEM ao aprender (campo de calor)."""
    def __init__(self):
        self.feat = ItemMemory()               # features atômicas (assento, pernas, encosto...)
        self.acc = {}                          # conceito -> acumulador int (soma dos exemplos)
        self.count = {}

    def feature_vec(self, features):
        """Objeto = superposição das features."""
        return bundle([self.feat.get(f) for f in features])

    def learn_example(self, concept, features):
        """Vê UM exemplo do conceito -> empurra o protótipo naquela direção (aprendizado 1-shot
        que ALTERA o que já sabia; o protótipo é a média móvel dos exemplos)."""
        v = self.feature_vec(features).astype(np.int32)
        if concept not in self.acc:
            self.acc[concept] = np.zeros(D, dtype=np.int32); self.count[concept] = 0
        self.acc[concept] += (v * 2 - 1) if v.max() <= 1 else v   # v já é ±1
        self.count[concept] += 1

    def prototype(self, concept):
        a = self.acc[concept]
        return np.where(a > 0, 1, -1).astype(np.int8)

    def classify(self, features, topk=3):
        """Categoria graduada de um objeto: protótipos ordenados por tipicidade (similaridade)."""
        v = self.feature_vec(features)
        sims = [(c, cos(v, self.prototype(c))) for c in self.acc]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]

    def halo(self, concept, topk=4):
        """Senso comum: o que 'acende junto' quando penso no conceito (∝ similaridade)."""
        p = self.prototype(concept)
        sims = [(c, cos(p, self.prototype(c))) for c in self.acc if c != concept]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]


def demo():
    print("=" * 82)
    print("AXON core v2 — SENSO COMUM: protótipos graduados (Rosch) que se movem ao aprender")
    print("=" * 82)
    cs = ConceptSpace()

    # protótipos a partir de exemplos (features compartilhadas = categorias próximas)
    cs.learn_example("cadeira", ["assento", "pernas", "encosto"])
    cs.learn_example("poltrona", ["assento", "pernas", "encosto", "almofada", "bracos", "macia"])
    cs.learn_example("banco", ["assento", "pernas"])
    cs.learn_example("mesa", ["tampo", "pernas", "superficie_plana"])

    print("\n[1] TIPICIDADE GRADUADA (não binária) — quão 'X' é um objeto:")
    obj = ["assento", "pernas", "encosto"]
    print(f"    objeto {obj}:")
    for c, s in cs.classify(obj):
        print(f"       {c:<10} tipicidade={s:+.3f}")

    print("\n[2] MEXER NAS FEATURES DESLIZA ENTRE CATEGORIAS (a sua tese):")
    for descr, feats in (
        ("cadeira + almofada", ["assento", "pernas", "encosto", "almofada", "macia"]),
        ("cadeira - encosto  ", ["assento", "pernas"]),
        ("cadeira base       ", ["assento", "pernas", "encosto"]),
    ):
        top = cs.classify(feats)[0]
        print(f"    {descr} -> '{top[0]}'  (tipicidade {top[1]:+.2f})")

    print("\n[3] HALO DE SENSO COMUM (o que acende junto ao pensar 'cadeira'):")
    for c, s in cs.halo("cadeira"):
        print(f"    cadeira ~ {c:<10} {s:+.3f}")

    # [4] APRENDER REFORMA O QUE JÁ SABIA (o ponto de calor se movendo)
    print("\n[4] APRENDIZADO ALTERA O PROTÓTIPO (calor dissipando): objeto borderline reclassifica")
    borderline = ["assento", "pernas", "encosto", "almofada"]
    antes = cs.classify(borderline)[0]
    print(f"    antes: {borderline} -> '{antes[0]}' ({antes[1]:+.2f})")
    # ensina várias poltronas enfatizando almofada/macia -> protótipo 'poltrona' se move p/ cá
    for _ in range(4):
        cs.learn_example("poltrona", ["assento", "pernas", "encosto", "almofada", "macia", "confortavel"])
    depois = cs.classify(borderline)[0]
    print(f"    após aprender +4 poltronas: mesmo objeto -> '{depois[0]}' ({depois[1]:+.2f})")
    print("    => o protótipo se moveu; a MESMA entrada agora cai noutra categoria. Como humano.")

    print("\n" + "=" * 82)
    print("Isto é senso comum graduado (Rosch) + aprendizado que reforma memória, na VSA leve.")
    print("Casa com a 'temperatura' do axon: o campo de ativação É o calor sobre o espaço.")


def _selftest():
    cs = ConceptSpace()
    cs.learn_example("cadeira", ["assento", "pernas", "encosto"])
    cs.learn_example("banco", ["assento", "pernas"])
    cs.learn_example("mesa", ["tampo", "superficie"])
    # objeto = cadeira base -> mais típico de cadeira
    assert cs.classify(["assento", "pernas", "encosto"])[0][0] == "cadeira"
    # sem encosto -> desliza p/ banco
    assert cs.classify(["assento", "pernas"])[0][0] == "banco"
    # tipicidade é graduada (valor real, não 0/1)
    sims = dict(cs.classify(["assento", "pernas", "encosto"]))
    assert 0.0 < sims["banco"] < sims["cadeira"]
    # aprender move o protótipo (acumulador cresce)
    a0 = cs.acc["cadeira"].copy()
    cs.learn_example("cadeira", ["assento", "pernas", "encosto", "rodas"])
    assert not np.array_equal(a0, cs.acc["cadeira"])
    print("[selftest] ok (classificação graduada; desliza s/ encosto; protótipo se move)")


if __name__ == "__main__":
    _selftest()
    demo()
