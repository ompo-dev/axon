"""
COMER DUMP — a Wikipédia INTEIRA, local, sem API e sem 429.

Baixa UMA vez o dump oficial da Wikipédia pt-br:
    https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2
Depois lê o XML comprimido em STREAMING (nunca carrega tudo na RAM) e os
sub-Toshi comem em PROCESSOS paralelos. Sem rede, sem limite de requisição:
a velocidade vira só CPU/disco.

USO:
  python comer_dump.py --baixar --workers 12 --max-paginas 5000
  python comer_dump.py --workers 12 --max-paginas 50000 --continuar
  python comer_dump.py --selftest
"""
import argparse
import bz2
import glob
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from comer_wikipedia import SubToshi, carregar_comidos, INDICE, WIKI_DIR

DUMP_URL = ("https://dumps.wikimedia.org/ptwiki/latest/"
            "ptwiki-latest-pages-articles.xml.bz2")
DUMP_PATH = os.path.join(WIKI_DIR, "ptwiki-latest-pages-articles.xml.bz2")
XML_PATH = os.path.join(WIKI_DIR, "ptwiki-latest-pages-articles.xml")
PROGRESSO_DUMP = os.path.join(WIKI_DIR, "progresso_dump.json")


# ============================================================ DOWNLOAD
def baixar_dump(destino=DUMP_PATH, url=DUMP_URL):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    if os.path.isfile(destino) and os.path.getsize(destino) > 100 * 1024 * 1024:
        print(f"dump já existe: {destino} ({os.path.getsize(destino)/1e9:.2f} GB)")
        return destino
    print(f"baixando {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ToshiAprendiz/0.1"})
    tmp = destino + ".part"
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        lidos = 0
        t0 = time.time()
        while True:
            bloco = r.read(1024 * 1024)
            if not bloco:
                break
            f.write(bloco)
            lidos += len(bloco)
            if total and lidos % (50 * 1024 * 1024) < len(bloco):
                pct = 100.0 * lidos / total
                v = lidos / max(1, time.time() - t0) / 1e6
                print(f"  {lidos/1e9:.2f}/{total/1e9:.2f} GB ({pct:.0f}%) "
                      f"{v:.1f} MB/s", flush=True)
    os.replace(tmp, destino)
    print(f"download concluído: {os.path.getsize(destino)/1e9:.2f} GB")
    return destino


def descomprimir_dump(origem=DUMP_PATH, destino=XML_PATH):
    """Descomprime o .bz2 para .xml UMA vez. Depois a leitura fica muito mais
    rápida (sem descompressão serial no meio da varredura)."""
    if os.path.isfile(destino) and os.path.getsize(destino) > 100 * 1024 * 1024:
        print(f"xml já existe: {destino} ({os.path.getsize(destino)/1e9:.2f} GB)")
        return destino
    if not os.path.isfile(origem):
        print(f"dump não encontrado: {origem}")
        return None
    print(f"descomprimindo {origem} -> {destino} ...")
    tmp = destino + ".part"
    t0 = time.time()
    with bz2.open(origem, "rb") as entrada, open(tmp, "wb") as saida:
        while True:
            bloco = entrada.read(4 * 1024 * 1024)
            if not bloco:
                break
            saida.write(bloco)
            tamanho = saida.tell()
            if tamanho % (200 * 1024 * 1024) < 4 * 1024 * 1024:
                v = tamanho / max(1, time.time() - t0) / 1e6
                print(f"  {tamanho/1e9:.2f} GB ... {v:.1f} MB/s", flush=True)
    os.replace(tmp, destino)
    print(f"descompressão concluída: {os.path.getsize(destino)/1e9:.2f} GB "
          f"em {time.time()-t0:.0f}s")
    return destino


def _iterar_paginas(source):
    """Parser streaming; usa lxml (C) se disponível, senão ElementTree."""
    try:
        from lxml import etree as LET
        for _, elem in LET.iterparse(source, events=("end",), tag="{*}page"):
            yield elem
    except ImportError:
        for _, elem in ET.iterparse(source, events=("end",)):
            if _tag(elem) == "page":
                yield elem


def _comer_xml_apagando(caminho, fila, comidos, max_paginas, workers,
                        reescrever_bytes=1024 * 1024 * 1024):
    """
    Come o XML e VAI APAGANDO o que já foi consumido:
    a cada reescrever_bytes, o arquivo é substituído pelo restante não lido.
    No fim, o .xml some do disco — fica só o cérebro (shards + índice).
    """
    try:
        from lxml import etree as LET
        parse = LET.fromstring
    except ImportError:
        parse = ET.fromstring

    servidos = lidos = 0
    while True:
        try:
            f = open(caminho, "rb")
        except FileNotFoundError:
            break
        with f:
            buffer = b""
            lido_total = 0
            lote = []
            fim_arquivo = False
            while True:
                bloco = f.read(4 * 1024 * 1024)
                if not bloco:
                    fim_arquivo = True
                    break
                buffer += bloco
                lido_total += len(bloco)
                while True:
                    ini = buffer.find(b"<page>")
                    fim = buffer.find(b"</page>")
                    if ini == -1 or fim == -1:
                        break
                    fim += len(b"</page>")
                    raw = buffer[ini:fim]
                    buffer = buffer[fim:]
                    lidos += 1
                    try:
                        elem = parse(raw)
                    except Exception:
                        continue
                    bruta = pagina_bruta_do_xml(elem)
                    if bruta is None:
                        continue
                    if bruta["titulo"] in comidos:
                        continue
                    comidos.add(bruta["titulo"])
                    lote.append(bruta)
                    servidos += 1
                    if len(lote) >= 100:
                        fila.put(lote)
                        lote = []
                    if servidos % 5000 == 0:
                        print(f"  {servidos}/{max_paginas} enviados "
                              f"(lidos {lidos}; xml restante ~"
                              f"{(os.path.getsize(caminho)-(lido_total-len(buffer)))/1e9:.2f} GB)",
                              flush=True)
                    if servidos >= max_paginas:
                        break
                if servidos >= max_paginas:
                    break

            consumido_ate = lido_total - len(buffer)
            if lote:
                fila.put(lote)

            if fim_arquivo:
                try:
                    os.remove(caminho)
                    print(f"  [limpeza] {os.path.basename(caminho)} "
                          f"consumido e APAGADO", flush=True)
                except Exception as e:
                    print(f"  [!] não consegui apagar o xml: {e}")
                break

            if consumido_ate >= reescrever_bytes:
                tmp = caminho + ".part"
                with open(caminho, "rb") as origem, open(tmp, "wb") as destino:
                    origem.seek(consumido_ate)
                    shutil.copyfileobj(origem, destino, 4 * 1024 * 1024)
                os.replace(tmp, caminho)
                print(f"  [limpeza] xml encolheu para "
                      f"{os.path.getsize(caminho)/1e9:.2f} GB", flush=True)
            else:
                break

    return servidos, lidos


def dividir_xml_em_partes(origem=XML_PATH, parte_bytes=256 * 1024 * 1024):
    """Divide o XML gigante em partes menores. Cada parte é apagada ao ser comida."""
    if not os.path.isfile(origem):
        print(f"xml não encontrado: {origem}")
        return []
    partes = []
    with open(origem, "rb") as entrada:
        idx = 0
        atual = 0
        saida = None
        while True:
            bloco = entrada.read(4 * 1024 * 1024)
            if not bloco:
                break
            if saida is None or (atual + len(bloco) > parte_bytes and atual > 0):
                if saida is not None:
                    saida.close()
                caminho = os.path.join(WIKI_DIR, f"xml_parte_{idx:04d}.xml")
                saida = open(caminho, "wb")
                partes.append(caminho)
                idx += 1
                atual = 0
            saida.write(bloco)
            atual += len(bloco)
        if saida is not None:
            saida.close()
    print(f"dividido em {len(partes)} partes "
          f"({[f'{os.path.getsize(p)/1e9:.2f}GB' for p in partes[:3]]}...)")
    return partes


def _comer_xml_partes(fila, comidos, max_paginas, workers):
    """Come as partes na ordem e APAGA cada parte assim que termina de lê-la."""
    try:
        from lxml import etree as LET
        parse = LET.fromstring
    except ImportError:
        parse = ET.fromstring

    partes = sorted(glob.glob(os.path.join(WIKI_DIR, "xml_parte_*.xml")))
    if not partes:
        return 0, 0
    print(f"{len(partes)} partes para comer (cada uma é apagada após ser lida)")

    servidos = lidos = 0
    sobra = b""
    for caminho in partes:
        parou_por_limite = False
        buffer = sobra
        sobra = b""
        lote = []
        with open(caminho, "rb") as f:
            while True:
                bloco = f.read(4 * 1024 * 1024)
                if not bloco:
                    break
                buffer += bloco
                while True:
                    ini = buffer.find(b"<page>")
                    fim = buffer.find(b"</page>")
                    if ini == -1 or fim == -1:
                        break
                    fim += len(b"</page>")
                    raw = buffer[ini:fim]
                    buffer = buffer[fim:]
                    lidos += 1
                    try:
                        elem = parse(raw)
                    except Exception:
                        continue
                    bruta = pagina_bruta_do_xml(elem)
                    if bruta is None:
                        continue
                    if bruta["titulo"] in comidos:
                        continue
                    comidos.add(bruta["titulo"])
                    lote.append(bruta)
                    servidos += 1
                    if len(lote) >= 100:
                        fila.put(lote)
                        lote = []
                    if servidos % 5000 == 0:
                        print(f"  {servidos}/{max_paginas} enviados "
                              f"(lidos {lidos})", flush=True)
                    if servidos >= max_paginas:
                        parou_por_limite = True
                        break
                if parou_por_limite:
                    break
            sobra = buffer
        if lote:
            fila.put(lote)
        if parou_por_limite:
            print(f"  [limpeza] parte {os.path.basename(caminho)} mantida "
                  f"(ainda falta comida)")
            break
        os.remove(caminho)
        print(f"  [limpeza] parte {os.path.basename(caminho)} comida e APAGADA",
              flush=True)
    return servidos, lidos


# ============================================================ PARSER WIKITEXTO
def _remover_templates(texto):
    """Remove {{...}} respeitando aninhamento."""
    out, i, prof = [], 0, 0
    while i < len(texto):
        if texto[i:i + 2] == "{{":
            prof += 1
            i += 2
            continue
        if prof and texto[i:i + 2] == "}}":
            prof -= 1
            i += 2
            continue
        if prof == 0:
            out.append(texto[i])
        i += 1
    return "".join(out)


def limpar_wikitexto(texto):
    t = _remover_templates(texto)
    t = re.sub(r"<ref[^>]*/?>.*?</ref>", " ", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[\[(?:Ficheiro|Imagem|File|Image):[^\]]+\]\]", " ", t, flags=re.I)
    t = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)
    t = re.sub(r"\{\|.*?\|\}", " ", t, flags=re.S)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"^=+\s*[^=]*\s*=+$", " ", t, flags=re.M)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def extrair_links_imagens(wikitexto):
    imagens = re.findall(
        r"\[\[(?:Ficheiro|Imagem|File|Image):([^\]|#]+)",
        wikitexto, flags=re.I)
    links = re.findall(r"\[\[([^\]|:#]+)(?:\|[^\]]+)?\]\]", wikitexto)
    return links, imagens


def _tag(elem):
    return elem.tag.rsplit("}", 1)[-1] if isinstance(elem.tag, str) else ""


def pagina_do_xml(elem, completo=False):
    if _tag(elem) != "page":
        return None
    ns = titulo = texto = None
    for filho in list(elem):
        nome = _tag(filho)
        if nome == "ns":
            ns = int((filho.text or "0").strip() or "0")
        elif nome == "title":
            titulo = (filho.text or "").strip()
        elif nome == "revision":
            for neto in list(filho):
                if _tag(neto) == "text":
                    texto = neto.text or ""
    if ns != 0 or not titulo or not texto:
        return None
    if texto.lstrip().upper().startswith("#REDIRECT"):
        return None
    links, imagens = extrair_links_imagens(texto)
    limpo = limpar_wikitexto(texto)
    if not completo:
        limpo = limpo[:1200]
    return {
        "titulo": titulo,
        "texto": limpo,
        "links": list(dict.fromkeys(links))[:80],
        "imagens": list(dict.fromkeys(imagens))[:40],
    }


def pagina_bruta_do_xml(elem):
    """Extrai SÓ o cru (título + wikitexto). A limpeza pesada acontece nos workers."""
    if _tag(elem) != "page":
        return None
    ns = titulo = texto = None
    for filho in list(elem):
        nome = _tag(filho)
        if nome == "ns":
            ns = int((filho.text or "0").strip() or "0")
        elif nome == "title":
            titulo = (filho.text or "").strip()
        elif nome == "revision":
            for neto in list(filho):
                if _tag(neto) == "text":
                    texto = neto.text or ""
    if ns != 0 or not titulo or not texto:
        return None
    if texto.lstrip().upper().startswith("#REDIRECT"):
        return None
    return {"titulo": titulo, "texto_cru": texto}


# ============================================================ WORKER PROCESSO
def _worker_dump(shard_id, fila, parar, indice_shard, completo=False):
    sub = SubToshi(shard_id)
    while not parar.is_set():
        try:
            lote = fila.get(timeout=2)
        except Exception:
            continue
        if lote is None:
            break
        try:
            for bruta in lote:
                texto_limpo = limpar_wikitexto(bruta["texto_cru"])
                if not completo:
                    texto_limpo = texto_limpo[:1200]
                links, imagens = extrair_links_imagens(bruta["texto_cru"])
                pagina = {
                    "titulo": bruta["titulo"],
                    "texto": texto_limpo,
                    "links": list(dict.fromkeys(links))[:80],
                    "imagens": list(dict.fromkeys(imagens))[:40],
                }
                sub.comer_pagina(pagina)
                with open(indice_shard, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "titulo": pagina["titulo"],
                        "resumo": pagina["texto"][:1200],
                        "palavras": list(dict.fromkeys(
                            [w for w in re.findall(r"[a-z]+", pagina["texto"].lower())
                             if len(w) > 2]))[:60],
                        "links": pagina["links"][:60],
                        "imagens": pagina["imagens"][:30],
                    }, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [!p{shard_id}] erro: {e}", flush=True)
        finally:
            fila.task_done()
    sub.salvar()
    print(f"  [p{shard_id}] shard salvo: {sub.paginas} páginas", flush=True)


# ============================================================ SELFTEST
def _selftest():
    print("SELFTEST — parser do dump\n")
    cru = ("'''X''' é um [[planeta]] do [[Sistema Solar|sistema solar]]. "
           "{{Info|a=b}} <ref>nota</ref> [[Ficheiro:X.jpg|thumb|legenda]]")
    limpo = limpar_wikitexto(cru)
    print("  limpo:", limpo)
    assert "X é um planeta" in limpo
    assert "Info" not in limpo and "nota" not in limpo and "thumb" not in limpo
    links, imagens = extrair_links_imagens(cru)
    assert "planeta" in links and "Sistema Solar" in links
    assert imagens and imagens[0].startswith("X.jpg")

    xml = ("<mediawiki><page><title>Teste</title><ns>0</ns>"
           "<revision><text>'''Teste''' é um [[conceito]].</text></revision></page>"
           "<page><title>Discussão</title><ns>1</ns><revision><text>ignorar</text>"
           "</revision></page></mediawiki>")
    paginas = []
    raiz = ET.fromstring(xml)
    for page in raiz.findall("page"):
        p = pagina_do_xml(page)
        if p:
            paginas.append(p)
    assert len(paginas) == 1 and paginas[0]["titulo"] == "Teste"
    assert "Teste é um conceito" in paginas[0]["texto"]
    print("\n[selftest] ok — wikitexto vira comida limpa")


# ============================================================ MAIN
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Come o dump local da Wikipédia pt-br.")
    ap.add_argument("--baixar", action="store_true")
    ap.add_argument("--descomprimir", action="store_true",
                    help="descomprime o .bz2 para .xml uma vez (leitura futura muito mais rápida)")
    ap.add_argument("--arquivo", default=DUMP_PATH)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-paginas", type=int, default=5000)
    ap.add_argument("--completo", action="store_true",
                    help="come o texto inteiro, não só a introdução")
    ap.add_argument("--apagar-fontes", action="store_true",
                    help="apaga o .bz2 na descompressão e o .xml ENQUANTO consome")
    ap.add_argument("--espaco-minimo-gb", type=float, default=2.0,
                    help="para a varredura se o disco livre ficar abaixo disso")
    ap.add_argument("--continuar", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.baixar:
        baixar_dump(args.arquivo)

    if args.descomprimir:
        descomprimir_dump(args.arquivo, XML_PATH)
        if args.apagar_fontes and os.path.abspath(args.arquivo) != os.path.abspath(XML_PATH):
            try:
                os.remove(args.arquivo)
                print(f"  [limpeza] {os.path.basename(args.arquivo)} apagado "
                      f"(o xml já existe)")
            except Exception as e:
                print(f"  [!] não consegui apagar o .bz2: {e}")
        return

    # prefere o .xml já descomprimido (muito mais rápido)
    if args.arquivo.endswith(".xml"):
        fonte_path = args.arquivo
    elif os.path.isfile(XML_PATH):
        print("xml descomprimido encontrado; usando ele (muito mais rápido).")
        fonte_path = XML_PATH
    else:
        fonte_path = args.arquivo

    if not os.path.isfile(fonte_path):
        print(f"dump não encontrado: {fonte_path}")
        print("baixe com: python comer_dump.py --baixar")
        print("ou descomprima com: python comer_dump.py --descomprimir")
        return

    comidos = carregar_comidos()
    print(f"já comidos antes: {len(comidos)} artigos (não serão repetidos)")

    fila = multiprocessing.JoinableQueue(maxsize=args.workers * 4)
    parar = multiprocessing.Event()
    processos = []
    for i in range(args.workers):
        indice_shard = os.path.join(WIKI_DIR, f"indice_dump_{i}.jsonl")
        p = multiprocessing.Process(
            target=_worker_dump,
            args=(i, fila, parar, indice_shard, args.completo), daemon=True)
        p.start()
        processos.append(p)

    t0 = time.time()
    servidos = lidos = 0
    print(f"abrindo dump: {fonte_path} ...")
    if args.apagar_fontes and fonte_path.endswith(".xml"):
        # MODO INTELIGENTE: come e apaga o XML enquanto consome
        servidos, lidos = _comer_xml_apagando(
            fonte_path, fila, comidos, args.max_paginas, args.workers)
    else:
        # lxml exige arquivo BINÁRIO (ele mesmo decodifica o XML)
        if fonte_path.endswith(".xml"):
            fonte = open(fonte_path, "rb")
        else:
            fonte = bz2.open(fonte_path, "rb")
        lote = []
        with fonte as f:
            for elem in _iterar_paginas(f):
                lidos += 1
                bruta = pagina_bruta_do_xml(elem)
                elem.clear()
                if bruta is None:
                    continue
                if bruta["titulo"] in comidos:
                    continue
                comidos.add(bruta["titulo"])
                lote.append(bruta)
                servidos += 1
                if len(lote) >= 100:
                    fila.put(lote)
                    lote = []
                if servidos % 5000 == 0:
                    print(f"  {servidos}/{args.max_paginas} artigos enviados "
                          f"(lidos do dump: {lidos})", flush=True)
                if servidos >= args.max_paginas:
                    break
        if lote:
            fila.put(lote)

    fila.join()
    parar.set()
    for p in processos:
        p.join(timeout=15)

    # mescla índices dos processos no índice oficial
    partes = glob.glob(os.path.join(WIKI_DIR, "indice_dump_*.jsonl"))
    with open(INDICE, "a", encoding="utf-8") as destino:
        for parte in partes:
            try:
                with open(parte, encoding="utf-8") as origem:
                    shutil.copyfileobj(origem, destino)
                os.remove(parte)
            except Exception as e:
                print(f"  [!] mesclagem {parte}: {e}")

    with open(PROGRESSO_DUMP, "w", encoding="utf-8") as f:
        json.dump({"lidos": lidos, "servidos": servidos}, f)

    print("\n" + "=" * 66)
    print("DUMP TERMINOU")
    print("=" * 66)
    print(f"  artigos NOVOS comidos: {servidos}")
    print(f"  páginas lidas do dump: {lidos}")
    print(f"  tempo:                 {time.time()-t0:.0f}s")
    print(f"  shards:                {WIKI_DIR}\\shard_*.pkl")
    print(f"  índice:                {INDICE}")


if __name__ == "__main__":
    main()
