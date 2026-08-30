"""
AXON cérebro — degrau: RACIOCINAR e IMAGINAR (as capacidades que faltavam).

Sobre o substrato VSA (composição + analogia + memória), três operações "mentais":
  1. IMAGINAR: compor um conceito NUNCA VISTO (bind/bundle de partes) e ver o que ele evoca
     (o mais próximo na memória). É simulação mental: "e se X tivesse a propriedade de Y?".
  2. INFERIR: dado parcial, completar (unbind/cleanup). "Isto voa e faz mel -> ?".
  3. PENSAR: cadeia de associações (trem de pensamento) — pular de conceito em conceito pelo
     que está mais 'quente'/próximo, gerando uma sequência de ideias.

Honesto: é raciocínio COMPOSICIONAL e ANALÓGICO real (não o "pulo" abdutivo, que ninguém tem).
Aqui as partes são dadas p/ ilustrar; no sistema pleno elas EMERGEM da percepção crua.

Roda: python raciocinio.py   (numpy; usa vsa_core.py)
"""
import numpy as np
from vsa_core import D, bind, bundle, cos, ItemMemory


class Mind:
    def __init__(self):
        self.mem = ItemMemory()
        self.concepts = {}                     # nome -> hipervetor (bundle de propriedades)

    def learn(self, name, props):
        """Aprende um conceito como superposição de propriedades (1 exemplo)."""
        self.concepts[name] = bundle([self.mem.get(p) for p in props])
        self.mem.labels.append(name) if name not in self.mem.labels else None
        self.mem.vecs.append(self.concepts[name]) if name in self.mem.labels else None
        return self.concepts[name]

    def _nearest_concept(self, v, exclude=()):
        best = sorted(((c, cos(v, h)) for c, h in self.concepts.items() if c not in exclude),
                      key=lambda x: x[1], reverse=True)
        return best

    # 1. IMAGINAR: compor algo novo e ver o que evoca
    def imagine(self, props):
        """Compõe um objeto imaginário a partir de propriedades e diz do que ele 'lembra'."""
        v = bundle([self.mem.get(p) for p in props])
        return self._nearest_concept(v)[:3]

    # 2. INFERIR: dado propriedades, qual conceito?
    def infer(self, props):
        return self.imagine(props)[0]

    # 3. PENSAR: trem de pensamento (cadeia de associações)
    def think(self, seed, steps=5):
        chain = [seed]
        cur = self.concepts[seed] if seed in self.concepts else self.mem.get(seed)
        used = {seed}
        for _ in range(steps):
            nxt = self._nearest_concept(cur, exclude=used)
            if not nxt or nxt[0][1] < 0.1:
                break
            name = nxt[0][0]
            chain.append(name); used.add(name)
            # o próximo pensamento mistura o atual com o evocado (associação encadeada)
            cur = bundle([cur, self.concepts[name]])
        return chain


def demo():
    print("=" * 80)
    print("AXON cérebro — RACIOCINAR e IMAGINAR (composicional + analógico, leve)")
    print("=" * 80)
    m = Mind()
    # conceitos como superposição de propriedades
    m.learn("passaro", ["voa", "penas", "bico", "ovos", "canta"])
    m.learn("abelha", ["voa", "faz_mel", "asas", "ferrao", "pequeno"])
    m.learn("aviao", ["voa", "metal", "motor", "asas", "grande"])
    m.learn("peixe", ["nada", "escamas", "agua", "ovos", "silencioso"])
    m.learn("morcego", ["voa", "asas", "noite", "mamifero", "silencioso"])

    print("\n[1] INFERIR (dado propriedades, qual conceito?):")
    for props in (["voa", "faz_mel"], ["nada", "escamas"], ["voa", "metal", "motor"]):
        c, s = m.infer(props)
        print(f"    {props} -> '{c}' ({s:+.2f})")

    print("\n[2] IMAGINAR (compor algo NUNCA VISTO e ver o que evoca):")
    for descr, props in (
        ("algo que voa e faz mel e é grande", ["voa", "faz_mel", "grande", "metal"]),
        ("algo que nada mas tem penas", ["nada", "penas", "ovos"]),
        ("algo que voa de noite e é silencioso", ["voa", "noite", "silencioso", "asas"]),
    ):
        top = m.imagine(props)
        print(f"    imaginar '{descr}':")
        print(f"       evoca -> " + ", ".join(f"{c}({s:+.2f})" for c, s in top))

    print("\n[3] PENSAR (trem de pensamento — cadeia de associações a partir de um conceito):")
    for seed in ("abelha", "peixe"):
        print(f"    a partir de '{seed}': {' -> '.join(m.think(seed, 4))}")

    print("\n" + "=" * 80)
    print("Isto é raciocínio composicional + analógico + imaginação (recombinar o conhecido)")
    print("num substrato de 1-bit leve. NÃO é o 'pulo' abdutivo (criar axioma novo) — ninguém")
    print("tem. É a base de 'pensar' que faltava, ligável ao resto (percepção, memória, geração).")


def _selftest():
    m = Mind()
    m.learn("passaro", ["voa", "penas", "ovos"])
    m.learn("peixe", ["nada", "escamas", "ovos"])
    m.learn("abelha", ["voa", "mel", "asas"])
    # inferir: voa+penas -> passaro; nada+escamas -> peixe
    assert m.infer(["voa", "penas"])[0] == "passaro"
    assert m.infer(["nada", "escamas"])[0] == "peixe"
    # imaginar recombinação -> evoca conceito relacionado (topo é um conceito conhecido)
    top = m.imagine(["voa", "mel"])
    assert top[0][0] in ("abelha", "passaro")
    # pensar: cadeia começa no seed e não repete
    ch = m.think("abelha", 3)
    assert ch[0] == "abelha" and len(set(ch)) == len(ch)
    print("[selftest] ok (infere; imagina recombinação; pensa em cadeia sem repetir)")


if __name__ == "__main__":
    _selftest()
    demo()
