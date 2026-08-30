"""
MENTE SEMÂNTICA — a FUSÃO: o planejador multi-passo raciocina sobre os CONCEITOS do Toshi.

O usuário: "uma IA que pensa e aprende de verdade, não quarto chinês nem LLM que não pula."
Aqui as duas mentes viram UMA: o córtex associativo do Toshi (grafo de conceitos aprendido de
texto CRU, sem priors) + o PENSADOR multi-passo (busca a cadeia até o objetivo, com backtrack).

O ponto (o "pulo" em significado): às vezes a ligação LOCAL mais forte leva pro lado ERRADO
(um distrator). O GULOSO (seguir o vínculo mais forte — é o `drift` do Toshi, e é como um LLM
escolhe a próxima palavra) cai nele e TRAVA. Só PLANEJAR (buscar a cadeia inteira A->B->C->meta,
voltar quando dá em nada) chega no objetivo. E aprende EM TEMPO REAL: ensina um elo novo, ele
já raciocina por cima dele no mesmo instante.

Currículo controlado (sei as cadeias => dá pra MEDIR) + ruído de distrator. Mede planejar vs
guloso, a profundidade das cadeias, e a inferência sobre um fato recém-aprendido.
Roda: python mente_semantica.py
"""
import sys
from collections import Counter, deque
from toshi import Toshi, tokenize

LETRAS = "abcdefghijklmnopqrstuvwxyz"


class MenteSemantica:
    """Uma mente: aprende conceitos (Toshi) e RACIOCINA sobre eles (planejador multi-passo)."""
    def __init__(self):
        self.t = Toshi()

    # aprende em TEMPO REAL, do mesmo jeito cru (cada mini-fato = uma percepção isolada, p/ o elo
    # ser só entre os itens ditos juntos — sem a janela colar coisas não-ditas)
    def aprender(self, *fatos):
        for f in fatos:
            self.t.perceive(tokenize(f))

    def _viz(self, w):
        return self.t.assoc.get(w, Counter())

    def _sim(self, a, b):
        ea, eb = self.t._emb(a), self.t._emb(b)
        return float(ea @ eb) if ea is not None and eb is not None else 0.0

    # ---------- PENSADOR: busca multi-passo a cadeia start->goal (BFS, backtrack) ----------
    def planejar(self, start, goal, cap=8000):
        """Acha a CADEIA de conceitos de start até goal seguindo elos aprendidos. Retorna a lista
        (A->B->C->...->goal) ou None. É o raciocínio: compõe o caminho, não commita no 1º passo.
        (BFS = caminho mais curto; em escala vira A* guiado por significado + poda por PMI.)"""
        if start == goal:
            return [start]
        seen = {start}
        fila = deque([[start]])
        while fila and len(seen) < cap:
            caminho = fila.popleft()
            node = caminho[-1]
            # expande pelos elos mais fortes primeiro (poda o resto = tratável em grafo grande)
            for u, _ in sorted(self._viz(node).items(), key=lambda x: -x[1])[:16]:
                if u in seen:
                    continue
                if u == goal:
                    return caminho + [u]
                seen.add(u); fila.append(caminho + [u])
        return None

    # ---------- GULOSO: segue o elo mais FORTE (reativo, 1 passo — o `drift`, estilo LLM) ----------
    def guloso(self, start, goal, passos=20):
        cur, caminho, usados = start, [start], {start}
        for _ in range(passos):
            nb = [(u, c) for u, c in self._viz(cur).items() if u not in usados]
            if not nb:
                break
            nxt = max(nb, key=lambda x: x[1])[0]            # vínculo local mais forte (pode ser o distrator)
            caminho.append(nxt); usados.add(nxt); cur = nxt
            if nxt == goal:
                return caminho, True
        return caminho, goal in caminho


# ============================================================ currículo mensurável
def base(i):
    return "z" + LETRAS[i // 26] + LETRAS[i % 26]           # prefixo único por puzzle (só letras)


def ensinar_puzzle(m, i, L=5, rep=2, dist_rep=6):
    """Cadeia t0->t1->...->t{L-1} (o caminho do raciocínio) + um DISTRATOR forte e sem-saída em t0.
    Retorna (start, goal). O elo t0->distrator é mais forte que t0->t1: o guloso morde a isca."""
    b = base(i)
    chain = [b + LETRAS[j] for j in range(L)]               # zaaa, zaab, ... (únicos)
    dist = b + "z"
    for j in range(L - 1):
        for _ in range(rep):
            m.aprender(f"{chain[j]} {chain[j+1]}")          # elo de cadeia (fraco)
    for _ in range(dist_rep):
        m.aprender(f"{chain[0]} {dist}")                    # elo distrator (FORTE, sem saída)
    return chain[0], chain[-1]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 76)
    print("MENTE SEMÂNTICA — raciocinar em CONCEITOS: planejar a cadeia vs guloso morder a isca")
    print("=" * 76)

    m = MenteSemantica()
    K = 60
    puzzles = []
    for i in range(K):
        L = 3 + (i % 5)                                     # cadeias de 3..7 conceitos (profundidade)
        s, g = ensinar_puzzle(m, i, L=L)
        puzzles.append((s, g, L))

    pl_ok = gu_ok = 0
    pulo = 0
    prof = []
    por_L = {}
    for s, g, L in puzzles:
        p = m.planejar(s, g)
        _, gok = m.guloso(s, g)
        pl = p is not None and p[-1] == g
        pl_ok += pl; gu_ok += gok
        if pl and not gok:
            pulo += 1
        if pl:
            prof.append(len(p) - 1)
        d = por_L.setdefault(L, [0, 0, 0])
        d[0] += 1; d[1] += pl; d[2] += gok

    print(f"\n{K} perguntas 'ligue o conceito A ao conceito Z' (cadeias de 3..7 passos):")
    print(f"  PENSADOR (planeja a cadeia): {pl_ok}/{K} = {pl_ok/K:.0%}")
    print(f"  GULOSO   (elo mais forte):   {gu_ok}/{K} = {gu_ok/K:.0%}   <- morde o distrator e trava")
    print(f"\n>>> o PULO — perguntas que o PENSADOR resolve e o GULOSO erra: {pulo}")
    if prof:
        print(f"    cadeias raciocinadas: {sum(prof)/len(prof):.1f} passos em média (máx {max(prof)}).")
    print(f"\n  por profundidade da cadeia (planejar vs guloso):")
    for L in sorted(por_L):
        n, pl, gu = por_L[L]
        print(f"    {L} passos: planejar {pl}/{n}   guloso {gu}/{n}")

    # exemplo verbalizado: VER o raciocínio
    s, g, L = puzzles[7]
    cad = m.planejar(s, g)
    print(f"\n  exemplo — pergunta: ligar '{s}' a '{g}'")
    print(f"    guloso  diz: {' -> '.join(m.guloso(s, g)[0])}   (foi pro distrator, parou)")
    print(f"    pensador diz: {' -> '.join(cad)}   (a cadeia inteira: raciocinou)")

    # ---------- aprende EM TEMPO REAL: inferência sobre um elo recém-ensinado ----------
    print("\n" + "-" * 76)
    print("APRENDER EM TEMPO REAL — duas ideias soltas; ensino UM elo; ele raciocina por cima na hora:")
    m.aprender("chuva molha", "molha frio", "frio tremor")      # cadeia 1: chuva->...->tremor
    m.aprender("febre suor", "suor fraqueza", "fraqueza queda") # cadeia 2: febre->...->queda
    antes = m.planejar("chuva", "queda")
    print(f"  antes:  ligar 'chuva' a 'queda' -> {antes}   (não há elo entre as duas cadeias)")
    m.aprender("tremor febre")                                  # <== o elo novo (1 fato)
    depois = m.planejar("chuva", "queda")
    print(f"  ensino 'tremor febre' (1 fato) ...")
    print(f"  depois: ligar 'chuva' a 'queda' -> {' -> '.join(depois) if depois else None}")
    print(f"          raciocinou por cima do elo aprendido AGORA (não re-treinou nada).")

    print("\n" + "=" * 76)
    ok = pl_ok > gu_ok and pulo > 0 and depois is not None and antes is None
    if ok:
        print("VEREDITO: a mente raciocina CADEIAS de conceitos (planeja, com backtrack) onde o guloso")
        print("(o próximo-mais-forte, estilo LLM) morde a isca e trava. E aprende EM TEMPO REAL: um elo")
        print("novo muda a inferência na hora. Pensa E aprende de verdade — não é quarto chinês (lookup)")
        print("nem LLM que 'não pula'. É o Modo-2 rodando em SIGNIFICADO. A fusão Toshi+pensador: uma mente.")
    else:
        print("VEREDITO honesto: sinal parcial — reportado como está, sem inflar.")


# ============================================================ escala: raciocinar no TOSHI REAL
def _viz_conteudo(t, w, k=18):
    """Vizinhos de CONTEÚDO por PMI (poda stopwords e ubíquos) — segue só elos informativos.
    É o que torna a busca tratável no grafo-cabeludo real e evita o afogamento em stopword."""
    stops = t.stops
    out = []
    for u, c in t.assoc.get(w, {}).items():
        if u in stops:
            continue
        p = t.pmi(w, u, c)
        if p > 0:
            out.append((u, p))
    return sorted(out, key=lambda x: -x[1])[:k]


def planejar_real(t, start, goal, cap=40000):
    """A* guiado por SIGNIFICADO: custo = hops, heurística = distância semântica ao objetivo
    (1 - cos). Segue elos de conteúdo (PMI). Acha uma CADEIA de conteúdo start->goal, ou None."""
    import heapq
    eg = t._emb(goal)

    def h(n):
        en = t._emb(n)
        return 1.0 - float(en @ eg) if (en is not None and eg is not None) else 1.0

    if start not in t.assoc or eg is None:
        return None
    pq = [(h(start), 0, start, [start])]
    seen = {start}
    while pq and len(seen) < cap:
        _, g, node, path = heapq.heappop(pq)
        if node == goal:
            return path
        if g >= 8:
            continue
        for u, _p in _viz_conteudo(t, node):
            if u in seen:
                continue
            seen.add(u)
            heapq.heappush(pq, (g + 1 + h(u), g + 1, u, path + [u]))
    return None


def _frac_stop(t, path):
    return sum(1 for w in path if w in t.stops) / max(len(path), 1)


def planejar_beam(t, start, goal, beam=12, cap_depth=8):
    """Beam search guiado por VALOR (ToT/beam + verificador, portado clássico; ideia do RwT/COLING
    2025: buscar no grafo de conceitos com um avaliador). Valor do passo a->b:
        suavidade cos(a,b) + progresso cos(b,goal) - penalidade de COLA (palavra ubíqua).
    Escolhe a cadeia de maior valor até o objetivo -> mais lisa e mais de conteúdo que o BFS cego."""
    eg = t._emb(goal)
    if start not in t.assoc or eg is None:
        return None

    def val(a, b):
        ea, eb = t._emb(a), t._emb(b)
        suav = float(ea @ eb) if (ea is not None and eb is not None) else 0.0
        prog = float(eb @ eg) if eb is not None else 0.0
        cola = min(t.seen[b] / (0.02 * t.total + 1), 1.0)   # frequente demais = cola
        return suav + 0.5 * prog - 0.4 * cola

    beams = [(0.0, [start])]
    best = None
    for _ in range(cap_depth):
        cand = []
        for sc, path in beams:
            node = path[-1]
            if node == goal:
                if best is None or sc > best[0]:
                    best = (sc, path)
                continue
            for u, _p in _viz_conteudo(t, node, k=12):
                if u in path:
                    continue
                ns, npath = sc + val(node, u), path + [u]
                if u == goal and (best is None or ns > best[0]):
                    best = (ns, npath)
                cand.append((ns, npath))
        if not cand:
            break
        cand.sort(key=lambda x: -x[0])
        beams = cand[:beam]
    return best[1] if best else None


def rodar_real():
    from toshi import build_or_load
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("carregando o Toshi real (cérebro dos livros)...", flush=True)
    t, _ = build_or_load()
    print(f"pronto: {len(t.seen)} conceitos.\n")
    print("=" * 76)
    print("RACIOCÍNIO NO CÉREBRO REAL — planejar cadeia de CONTEÚDO vs guloso (drift) que se afoga")
    print("=" * 76)

    pares = [("capitu", "ciume"), ("mar", "ceu"), ("olhos", "amor"),
             ("padre", "seminario"), ("morte", "vida"), ("bento", "capitu")]
    pares = [(a, b) for a, b in pares if a in t.assoc and b in t.embed]
    for a, b in pares:
        cad = planejar_real(t, a, b)
        dr = t.drift(a, 8)
        print(f"\n  ligar '{a}' -> '{b}':")
        if cad:
            print(f"    PENSADOR: {' -> '.join(cad)}   ({len(cad)-1} passos, {_frac_stop(t,cad):.0%} stopword)")
        else:
            print(f"    PENSADOR: (não achou cadeia de conteúdo em 8 passos)")
        print(f"    GULOSO  : {' -> '.join(dr)}   (deriva; chega em '{b}'? {'sim' if b in dr else 'não'})")

    # estatística: sobre muitos pares de conteúdo, quanto o planejar conecta e quão 'limpo' (conteúdo)
    conteudo = [w for w, _ in t.seen.most_common(1500) if w not in t.stops and t._emb(w) is not None][:400]
    import random
    rng = random.Random(0)
    pl_conn = dr_conn = n = 0
    stop_pl = stop_dr = 0.0
    for _ in range(60):
        a, b = rng.sample(conteudo, 2)
        cad = planejar_real(t, a, b)
        dr = t.drift(a, 8)
        n += 1
        if cad:
            pl_conn += 1; stop_pl += _frac_stop(t, cad)
        if b in dr:
            dr_conn += 1
        stop_dr += _frac_stop(t, dr)
    print("\n" + "-" * 76)
    print(f"em {n} pares de conteúdo aleatórios:")
    print(f"  PENSADOR conecta: {pl_conn}/{n} = {pl_conn/n:.0%}   (stopword nas cadeias: {stop_pl/max(pl_conn,1):.0%})")
    print(f"  GULOSO  alcança:  {dr_conn}/{n} = {dr_conn/n:.0%}   (stopword na deriva: {stop_dr/n:.0%})")
    print("\n  leitura: o guloso deriva por atratores/stopwords e raramente cai no alvo; o pensador")
    print("  acha a cadeia de CONTEÚDO dirigida ao objetivo. É o raciocínio que falta ao Toshi tagarela.")


def _selftest():
    m = MenteSemantica()
    s, g = ensinar_puzzle(m, 0, L=5)
    p = m.planejar(s, g)
    assert p is not None and p[-1] == g and len(p) >= 4, p     # achou a cadeia multi-passo
    _, gok = m.guloso(s, g)
    assert not gok, "guloso deveria travar no distrator"       # o guloso morde a isca
    # tempo real: elo novo cria inferência nova  (tokens sem colidir com numeral romano)
    m.aprender("aaa bbb", "ppp qqq")
    assert m.planejar("aaa", "qqq") is None
    m.aprender("bbb ppp")
    assert m.planejar("aaa", "qqq") is not None
    print("[selftest] ok (planeja a cadeia; guloso trava; elo novo => inferência nova em tempo real)")


if __name__ == "__main__":
    if "--real" in sys.argv:
        rodar_real()
    else:
        _selftest()
        main()
