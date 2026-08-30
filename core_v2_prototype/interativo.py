"""
AXON — modo INTERATIVO. Converse, ensine ao vivo, veja aprender.

COMO RODAR (no seu terminal):
    cd C:\\Projects\\Teste\\axon\\core_v2_prototype
    python interativo.py

COMO USAR:
  - Digite FRASES normais -> ele APRENDE na hora e mostra o que já consegue gerar.
      ex:  o bebe quer mamar
           o bebe quer dormir
           a mamae pega o bebe
  - Comandos (começam com /):
      /gerar <palavras>     continua a partir do que você deu   (ex: /gerar o bebe)
      /pensar <conceito>    trem de pensamento                  (ex: /pensar abelha)
      /imaginar <props>     compõe algo novo e diz o que evoca  (ex: /imaginar voa mel grande)
      /conceito nome: p1 p2 p3   ensina um conceito p/ raciocinar (ex: /conceito abelha: voa mel asas)
      /stats                tamanho do "cérebro"
      /ajuda                esta ajuda
      /sair
  Dica: ensine 5-10 frases repetindo palavras; depois /gerar. Ele aprende SEM esquecer e
  atualiza em tempo real (nada de re-treino). Pesa KB.
"""
import sys
from cerebro_contexto import ContextBrain
from raciocinio import Mind

AJUDA = __doc__


def main():
    brain = ContextBrain(k=3)
    mind = Mind()
    print(AJUDA)
    print("=" * 60)
    print("Cérebro vazio. Ensine frases e experimente. /ajuda p/ ver comandos.\n")

    while True:
        try:
            line = input("voce> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\ntchau!"); break
        if not line:
            continue

        if line.startswith("/"):
            parts = line[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("sair", "quit", "q"):
                print("tchau!"); break
            elif cmd in ("ajuda", "help", "h"):
                print(AJUDA)
            elif cmd == "gerar":
                seed = arg.split()
                if not seed:
                    print("  uso: /gerar <palavras>"); continue
                g = brain.generate(seed, n=8)
                print(f"  axon> {' '.join(g)}")
            elif cmd == "pensar":
                if not arg:
                    print("  uso: /pensar <conceito>"); continue
                ch = mind.think(arg.strip(), 5) if arg.strip() in mind.concepts else None
                print(f"  axon pensa> {' -> '.join(ch)}" if ch
                      else f"  (não conheço '{arg.strip()}'. ensine com /conceito {arg.strip()}: ...)")
            elif cmd == "imaginar":
                props = arg.split()
                if not props:
                    print("  uso: /imaginar <props>"); continue
                top = mind.imagine(props)
                if top:
                    print("  axon imagina> evoca " + ", ".join(f"{c}({s:+.2f})" for c, s in top))
                else:
                    print("  (ensine conceitos com /conceito antes de imaginar)")
            elif cmd == "conceito":
                if ":" not in arg:
                    print("  uso: /conceito nome: prop1 prop2 ..."); continue
                nome, props = arg.split(":", 1)
                mind.learn(nome.strip(), props.split())
                print(f"  ok, aprendi o conceito '{nome.strip()}' ({len(props.split())} propriedades)")
            elif cmd == "stats":
                n, kb = brain.footprint()
                print(f"  cérebro: {n} palavras, {len(mind.concepts)} conceitos, {kb:.1f} KB. Sem GPU.")
            else:
                print(f"  comando '?{cmd}' desconhecido. /ajuda")
            continue

        # frase normal -> aprende ao vivo e mostra o que gera
        units = line.split()
        brain.perceive(units)
        if units:
            g = brain.generate(units[:1], n=6)
            print(f"  (aprendi) axon continua '{units[0]}'> {' '.join(g)}")


if __name__ == "__main__":
    main()
