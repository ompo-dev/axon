"""
AXON core v2 — núcleo cognitivo VSA/HDC (Vector Symbolic Architecture / Hyperdimensional).

Por que trocar a ativação-espalhada (v3.2) por isto:
  v3.2 só ASSOCIA (X perto de Y). Não compõe, não raciocina, não liga papel-preenchedor.
  VSA COMPÕE por álgebra de vetores hiperdimensionais:
    - bind  (⊗, mult elemento a elemento): liga papel a valor  ->  cor⊗vermelho
    - bundle(+, superposição + sinal): junta fatos numa "cena"  ->  (cor⊗vermelho)+(forma⊗circulo)
    - permute (rotação): ordem/sequência  ->  a·ρ(b)·ρ²(c)
  Consultar = unbind (⊗ do inverso). Analogia = mapa de transformação. UM exemplo basta.

Propriedades (as restrições do usuário):
  - aprende em 1 exemplo, ONLINE, sem gradiente  -> tempo real
  - só add/mult de vetores  -> ULTRA-LEVE (roda em qualquer CPU)
  - representação distribuída, robusta a ruído  -> cérebro-inspirado (Kanerva/Eliasmith SPAUN)

Referências: Plate HRR 1995; Kanerva Hyperdimensional 2009; Eliasmith SPA/SPAUN 2012.
Roda: python vsa_core.py   (numpy só)
"""
import numpy as np

D = 10000                      # dimensão hiperdimensional (10k = padrão)
RNG = np.random.default_rng(42)


# ============================================================ álgebra VSA (bipolar ±1)
def rand_vec(rng=RNG):
    """Hipervetor aleatório bipolar {-1,+1}. Vetores aleatórios são quase-ortogonais em D alto."""
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=D)


def bind(a, b):
    """Liga (papel⊗valor). Elemento a elemento. Auto-inverso: bind(bind(a,b),b)=a."""
    return (a.astype(np.int16) * b.astype(np.int16)).astype(np.int8)


def bundle(vecs):
    """Superpõe (junta) por soma + sinal (maioria). O resultado LEMBRA de todos."""
    s = np.sum([v.astype(np.int32) for v in vecs], axis=0)
    out = np.where(s > 0, 1, np.where(s < 0, -1, RNG.choice([-1, 1], size=D))).astype(np.int8)
    return out


def permute(a, k=1):
    """Rotação cíclica: codifica ordem/posição (ρ). Quase-ortogonal ao original."""
    return np.roll(a, k)


def unpermute(a, k=1):
    return np.roll(a, -k)


def cos(a, b):
    return float(np.dot(a.astype(np.int32), b.astype(np.int32)) / D)


# ============================================================ memória de itens (cleanup)
class ItemMemory:
    """Mapeia rótulos <-> hipervetores e limpa vetores ruidosos de volta ao símbolo mais próximo.
    É a 'memória de conteúdo' — aprende símbolos novos em 1 exemplo."""
    def __init__(self):
        self.labels = []
        self.vecs = []

    def get(self, label, create=True):
        if label in self.labels:
            return self.vecs[self.labels.index(label)]
        if not create:
            return None
        v = rand_vec()
        self.labels.append(label); self.vecs.append(v)
        return v

    def cleanup(self, v, topk=1, exclude=()):
        """Símbolo(s) conhecido(s) mais próximo(s) do vetor ruidoso v. `exclude` = rótulos a ignorar."""
        if not self.vecs:
            return []
        sims = [(-1e9 if lab in exclude else cos(v, u)) for lab, u in zip(self.labels, self.vecs)]
        order = np.argsort(sims)[::-1][:topk]
        return [(self.labels[i], sims[i]) for i in order]


# ============================================================ o núcleo cognitivo
class AxonCoreV2:
    """Núcleo composicional: fatos estruturados, consulta, sequência, analogia — 1-shot, leve."""
    def __init__(self):
        self.mem = ItemMemory()
        self.facts = {}           # nome -> hipervetor da cena (memória associativa estruturada)

    # ---- aprender UM fato estruturado (papel=valor, ...) em 1 exemplo ----
    def learn_fact(self, name, **roles):
        """Ex.: learn_fact('maca', cor='vermelho', forma='redonda', classe='fruta').
        Codifica como bundle de bind(papel,valor). ZERO gradiente, 1 exemplo."""
        pairs = [bind(self.mem.get(role), self.mem.get(val)) for role, val in roles.items()]
        self.facts[name] = bundle(pairs)
        return self.facts[name]

    def query(self, name, role):
        """Recupera o valor de um papel: unbind e limpa. 'Que cor é a maçã?'"""
        scene = self.facts[name]
        noisy = bind(scene, self.mem.get(role))       # bind com inverso (=próprio, bipolar)
        return self.mem.cleanup(noisy, topk=1)

    # ---- sequência (ordem importa) ----
    def learn_sequence(self, name, items):
        vecs = [permute(self.mem.get(it), k) for k, it in enumerate(items)]
        self.facts[name] = bundle(vecs)

    def seq_at(self, name, k):
        return self.mem.cleanup(unpermute(self.facts[name], k), topk=1)

    # ---- analogia RELACIONAL sobre estruturas (o clássico de Kanerva) ----
    def analogy(self, item, struct_from, struct_to):
        """'O que é para struct_to o que `item` é para struct_from?'
        Ex.: analogy('paris', 'franca', 'italia) -> 'roma' (a capital análoga).
        T = struct_from ⊗ struct_to mapeia uma estrutura na outra; aplicar a `item`
        transporta o papel dele. FUNCIONA porque as estruturas compartilham papéis.
        (É o 'What is the Dollar of Mexico?' de Kanerva 2010.)"""
        T = bind(self.facts[struct_from], self.facts[struct_to])
        ans = bind(self.mem.get(item), T)
        # exclui o proprio item e os nomes das estruturas (nao sao respostas validas)
        return self.mem.cleanup(ans, topk=1, exclude=(item, struct_from, struct_to))


# ============================================================ DEMOS: o que a v3.2 NÃO faz
def demo():
    print("=" * 76)
    print("AXON core v2 (VSA/HDC) — composição, consulta, sequência, analogia (1-shot, leve)")
    print(f"D={D} | só add/mult de vetores | zero treino por gradiente")
    print("=" * 76)
    core = AxonCoreV2()

    # 1. aprende fatos ESTRUTURADOS em 1 exemplo cada
    core.learn_fact("maca", cor="vermelho", forma="redonda", classe="fruta", gosto="doce")
    core.learn_fact("banana", cor="amarelo", forma="alongada", classe="fruta", gosto="doce")
    core.learn_fact("ceu", cor="azul", forma="amplo", classe="natureza")
    print("\n[1] COMPOSIÇÃO + CONSULTA estruturada (a v3.2 só recuperava vizinhos):")
    for name, role in (("maca", "cor"), ("banana", "forma"), ("ceu", "cor"), ("maca", "classe")):
        print(f"    {role} de {name}? -> {core.query(name, role)[0]}")

    # 2. sequência (ordem) — a v3.2 não codifica ordem de verdade
    core.learn_sequence("alfabeto", ["a", "b", "c", "d", "e"])
    print("\n[2] SEQUÊNCIA (ordem importa):")
    for k in (0, 2, 4):
        print(f"    posição {k} do alfabeto -> {core.seq_at('alfabeto', k)[0]}")

    # 3. analogia RELACIONAL sobre estruturas — a v3.2 NÃO faz
    core.learn_fact("franca", capital="paris", moeda="euro", continente="europa")
    core.learn_fact("italia", capital="roma", moeda="euro", continente="europa")
    core.learn_fact("japao", capital="toquio", moeda="iene", continente="asia")
    print("\n[3] ANALOGIA relacional ('o dólar do México' de Kanerva):")
    print(f"    paris está p/ franca assim como ? p/ italia -> "
          f"{core.analogy('paris','franca','italia')[0]}  (esperado roma)")
    print(f"    euro está p/ franca assim como ? p/ japao   -> "
          f"{core.analogy('euro','franca','japao')[0]}  (esperado iene)")

    # 4. robustez a ruído (memória distribuída, cérebro-like)
    print("\n[4] ROBUSTEZ: consulta ainda funciona com 20% dos bits corrompidos:")
    scene = core.facts["maca"].copy()
    flip = RNG.random(D) < 0.20
    scene[flip] *= -1
    noisy = bind(scene, core.mem.get("cor"))
    print(f"    cor da maçã (cena 20% corrompida) -> {core.mem.cleanup(noisy)[0]}")

    print("\n" + "=" * 76)
    print("Isto o axon v3.2 (ativação-espalhada) NÃO faz: compor papel-valor, consultar")
    print("estrutura, ordem real, analogia. E aprende em 1 exemplo, sem gradiente, leve.")
    print("Limite honesto: ainda não é geração de linguagem nem 'pulo' abdutivo (ninguém tem).")
    print("É a fundação composicional certa pra construir por cima. Próximo: linguagem + multimodal.")


def _selftest():
    core = AxonCoreV2()
    core.learn_fact("x", a="p", b="q", c="r")
    assert core.query("x", "a")[0][0] == "p"
    assert core.query("x", "b")[0][0] == "q"
    # bind é auto-inverso
    u, v = rand_vec(), rand_vec()
    assert np.array_equal(bind(bind(u, v), v), u)
    # vetores aleatórios são quase-ortogonais (|cos| pequeno)
    assert abs(cos(rand_vec(), rand_vec())) < 0.05
    # sequência recupera posição
    core.learn_sequence("s", ["m", "n", "o"])
    assert core.seq_at("s", 1)[0][0] == "n"
    # analogia relacional sobre estruturas (Kanerva)
    core.learn_fact("franca", capital="paris", moeda="euro")
    core.learn_fact("italia", capital="roma", moeda="euro")
    assert core.analogy("paris", "franca", "italia")[0][0] == "roma"
    print("[selftest] ok (bind auto-inverso; quase-ortogonalidade; consulta; sequência; analogia)")


if __name__ == "__main__":
    _selftest()
    demo()
