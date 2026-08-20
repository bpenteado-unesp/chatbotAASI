"""
Ingestão dos PDFs sobre AASI para o vector store (ChromaDB).
Roda 1x (ou toda vez que os PDFs mudarem).

Este módulo foi calibrado em cima de manuais reais de fabricante (ex: GN
ReSound), que têm uma característica importante: cada página do arquivo PDF
é, na prática, um "spread" com DUAS páginas impressas lado a lado, em duas
colunas. Ignorar isso faz o texto de colunas diferentes se misturar.

Uso:
    python ingest.py --pdf_dir ./pdfs --db_dir ./chroma_db
"""
import argparse
import re
from pathlib import Path

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer

from config import CAMINHO_ASSETS, CAMINHO_CHROMA_DB, NOME_MODELO_EMBEDDING

PADRAO_TITULO = re.compile(
    r"^(?P<numero>\d{1,2}(\.\d{1,2}){0,3})\s+(?P<titulo>[A-ZÀ-Ú][^\n]{2,70})$"
)


def _parece_titulo_valido(titulo: str) -> bool:
    """
    Filtro extra além do regex: rejeita "falsos positivos" comuns, como
    tabelas de especificação técnica ou limites regulatórios que batem no
    padrão "número + texto iniciando em maiúscula" mas não são títulos de
    seção de verdade (ex: "28 V/m; 1720, 1845, 28 V/m; 1720, 1845,").

    Um título de seção real quase sempre tem letras minúsculas normais
    (é uma frase em português) e poucos dígitos. Uma linha de tabela
    técnica tende a ser dominada por números, unidades e pontuação.
    """
    letras_minusculas = sum(1 for c in titulo if c.islower())
    digitos = sum(1 for c in titulo if c.isdigit())
    if letras_minusculas < 3:
        return False
    if digitos > len(titulo) * 0.25:
        return False
    return True


def _pagina_aproximada(indice_spread: int, lado: str) -> int | None:
    """
    Estima o número de página impressa a partir do índice do spread no PDF.
    Retorna None para a capa (spread 0, sem numeração).

    Aviso: esta fórmula é uma aproximação. Em documentos longos, pode haver
    trechos onde a numeração real diverge (ex: seções com numeração própria,
    encartes). Por isso a citação principal usa o título da seção, que é
    texto real e não depende dessa conta; a página é só um complemento.
    """
    if indice_spread == 0:
        return None
    base = indice_spread * 2
    return base if lado == "esquerda" else base + 1


def _extrair_linhas_da_metade(pagina, lado: str, bboxes_tabelas: list[tuple]) -> list[str]:
    largura = pagina.width
    meio = largura / 2

    def fora_de_tabela(obj):
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        for (x0, top, x1, bottom) in bboxes_tabelas:
            if x0 <= cx <= x1 and top <= cy <= bottom:
                return False
        return True

    pagina_filtrada = pagina.filter(fora_de_tabela)

    if lado == "esquerda":
        recorte = pagina_filtrada.crop((0, 0, meio, pagina.height))
    else:
        recorte = pagina_filtrada.crop((meio, 0, largura, pagina.height))
    texto = recorte.extract_text() or ""
    return [linha.strip() for linha in texto.split("\n") if linha.strip()]


def _extrair_tabelas_da_pagina(pagina, indice_spread: int, nome_arquivo: str) -> list[dict]:
    """
    Extrai tabelas preservando linhas/colunas. Cada LINHA da tabela vira um
    chunk próprio (não a tabela inteira, nem grupos de várias linhas).

    A primeira coluna é "preenchida para baixo" (denormalizada) quando a
    célula estiver vazia — padrão comum em tabelas com células mescladas,
    como um guia de sintoma/causa/solução onde o sintoma só aparece
    escrito na primeira linha do grupo. Sem isso, uma linha como
    "A bateria ainda está boa? | Substitua-a por outra." perderia o
    contexto de qual sintoma ela resolve.

    Por que uma linha por chunk, e não a tabela inteira ou grupos por
    sintoma: cada linha de causa/solução é, na prática, uma pergunta e
    resposta atômica. Juntar várias causas diferentes num só chunk dilui o
    embedding — o vetor passa a representar uma média de vários assuntos,
    prejudicando a busca por um deles específico (ex: uma pergunta sobre
    "a bateria ainda está boa" perde força se estiver misturada, no mesmo
    chunk, com "orelha cheia de cerume" e "tubo obstruído").
    """
    resultado = []
    meio = pagina.width / 2
    for tabela in pagina.find_tables():
        linhas = tabela.extract()
        if not linhas or len(linhas) < 2:
            continue

        centro_x = (tabela.bbox[0] + tabela.bbox[2]) / 2
        lado = "esquerda" if centro_x < meio else "direita"
        pagina_num = _pagina_aproximada(indice_spread, lado)

        for linha_texto in _tabela_para_linhas_denormalizadas(linhas):
            resultado.append({
                "texto": linha_texto,
                "tipo": "tabela",
                "pagina": pagina_num,
            })
    return resultado


def _tabela_para_linhas_denormalizadas(linhas: list[list]) -> list[str]:
    def limpar(celula):
        if celula is None:
            return ""
        return str(celula).replace("\n", " ").strip()

    cabecalho = [limpar(c) for c in linhas[0]]

    resultado = []
    valor_atual_primeira_coluna = ""
    for linha in linhas[1:]:
        celulas = [limpar(c) for c in linha]

        if celulas[0]:
            valor_atual_primeira_coluna = celulas[0]
        elif valor_atual_primeira_coluna:
            celulas[0] = valor_atual_primeira_coluna

        if not any(celulas):
            continue

        partes = [
            f"{cabecalho[i]}: {celulas[i]}"
            for i in range(len(cabecalho))
            if i < len(celulas) and celulas[i]
        ]
        if partes:
            resultado.append(" | ".join(partes))

    return resultado


def extrair_conteudo(pdf_path: str) -> list[dict]:
    """
    Retorna uma lista de blocos de conteúdo na ordem de leitura correta:
    cada bloco é {"texto": str, "tipo": "linha"|"titulo"|"tabela", "pagina": int|None}
    """
    blocos = []
    with pdfplumber.open(pdf_path) as pdf:
        for indice, pagina in enumerate(pdf.pages):
            bboxes_tabelas = [t.bbox for t in pagina.find_tables() if t.extract() and len(t.extract()) >= 2]

            for lado in ("esquerda", "direita"):
                pagina_num = _pagina_aproximada(indice, lado)
                linhas = _extrair_linhas_da_metade(pagina, lado, bboxes_tabelas)
                for linha in linhas:
                    m = PADRAO_TITULO.match(linha)
                    tipo = "titulo" if (m and _parece_titulo_valido(m.group("titulo"))) else "linha"
                    blocos.append({"texto": linha, "tipo": tipo, "pagina": pagina_num})

            # tabelas da página inteira (associadas a esquerda/direita por posição)
            blocos.extend(_extrair_tabelas_da_pagina(pagina, indice, Path(pdf_path).name))

    return blocos


def dividir_em_chunks(blocos: list[dict], tamanho_max: int = 700) -> list[dict]:
    """
    Chunking por seção: cada título encontrado inicia um novo chunk, que
    acumula as linhas seguintes até o próximo título ou até o teto de
    tamanho. Tabelas sempre viram um chunk próprio (não fragmentamos tabela).

    Cada chunk retornado guarda o título da seção (quando existe) e a
    página aproximada de início/fim — a citação prioriza o título, que é
    texto real e sempre confiável, e usa a página como complemento.
    """
    chunks = []
    titulo_atual = None
    linhas_atual: list[str] = []
    paginas_atual: list[int] = []

    def fechar_chunk():
        if not linhas_atual:
            return
        chunks.append({
            "texto": "\n".join(linhas_atual),
            "titulo_secao": titulo_atual,
            "pagina_inicio": min(p for p in paginas_atual if p) if any(paginas_atual) else None,
            "pagina_fim": max(p for p in paginas_atual if p) if any(paginas_atual) else None,
            "tipo": "texto",
        })

    for bloco in blocos:
        if bloco["tipo"] == "tabela":
            fechar_chunk()
            linhas_atual, paginas_atual, titulo_atual = [], [], titulo_atual
            chunks.append({
                "texto": bloco["texto"],
                "titulo_secao": titulo_atual,
                "pagina_inicio": bloco["pagina"],
                "pagina_fim": bloco["pagina"],
                "tipo": "tabela",
            })
            continue

        if bloco["tipo"] == "titulo":
            fechar_chunk()
            linhas_atual, paginas_atual = [], []
            titulo_atual = bloco["texto"]

        candidato_tamanho = len("\n".join(linhas_atual + [bloco["texto"]]))
        if candidato_tamanho > tamanho_max and linhas_atual:
            fechar_chunk()
            linhas_atual, paginas_atual = [], []

        linhas_atual.append(bloco["texto"])
        if bloco["pagina"]:
            paginas_atual.append(bloco["pagina"])

    fechar_chunk()
    chunks = [c for c in chunks if c["texto"].strip()]
    return _fundir_titulos_orfaos(chunks)


def _fundir_titulos_orfaos(chunks: list[dict]) -> list[dict]:
    """
    Funde com o chunk seguinte qualquer chunk "órfão": uma seção cujo
    texto é só o próprio título, sem corpo real (comum quando um título
    de seção existe apenas como "container" — o conteúdo de verdade mora
    todo nas subseções seguintes, ex: "7.5 Funcionamento do aparelho
    auditivo" sem nenhum parágrafo próprio antes de "7.5.1 ...").

    Por que isso importa: um chunk assim tem um embedding que representa
    só o título, que pode "ecoar" as palavras de uma pergunta de forma
    enganosa (alta similaridade sem ter conteúdo real pra oferecer),
    disputando espaço no resultado da busca com chunks substanciais.
    """
    resultado = []
    pendente = None  # chunk órfão aguardando ser fundido com o próximo

    for chunk in chunks:
        if pendente is not None:
            if chunk["tipo"] == "texto":
                chunk = {
                    **chunk,
                    "texto": f"{pendente['texto']}\n\n{chunk['texto']}",
                    "pagina_inicio": pendente["pagina_inicio"] or chunk["pagina_inicio"],
                }
            else:
                # não funde título órfão com tabela; mantém o título sozinho
                resultado.append(pendente)
            pendente = None

        eh_orfao = (
            chunk["tipo"] == "texto"
            and chunk["titulo_secao"]
            and chunk["texto"].strip() == chunk["titulo_secao"].strip()
        )
        if eh_orfao:
            pendente = chunk
        else:
            resultado.append(chunk)

    if pendente is not None:
        resultado.append(pendente)  # órfão no fim do documento, sem próximo pra fundir

    return resultado


def main(pdf_dir: str, db_dir: str):
    print("Carregando modelo de embeddings multilíngue (paraphrase-multilingual-MiniLM-L12-v2)...")
    modelo = SentenceTransformer(NOME_MODELO_EMBEDDING, cache_folder=str(CAMINHO_ASSETS))

    client = chromadb.PersistentClient(path=db_dir)
    colecao = client.get_or_create_collection(
        "aasi_docs", metadata={"hnsw:space": "cosine"}
    )

    pdfs = list(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        print(f"Nenhum PDF encontrado em {pdf_dir}")
        return

    total_chunks = 0
    for pdf_path in pdfs:
        print(f"Processando: {pdf_path.name}")
        blocos = extrair_conteudo(str(pdf_path))
        chunks = dividir_em_chunks(blocos)

        if not chunks:
            print(f"  Aviso: nenhum conteúdo extraído de {pdf_path.name}")
            continue

        textos = [c["texto"] for c in chunks]
        embeddings = modelo.encode(textos, normalize_embeddings=True).tolist()
        ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
        metadados = [
            {
                "fonte": pdf_path.name,
                "chunk_index": i,
                "titulo_secao": c["titulo_secao"] or "",
                "pagina_inicio": c["pagina_inicio"] or 0,
                "pagina_fim": c["pagina_fim"] or 0,
                "tipo": c["tipo"],
            }
            for i, c in enumerate(chunks)
        ]

        colecao.add(
            ids=ids,
            embeddings=embeddings,
            documents=textos,
            metadatas=metadados,
        )
        total_chunks += len(chunks)

    print(f"Ingestão concluída. {total_chunks} chunks salvos em {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", default="./data/pdfs")
    parser.add_argument("--db_dir", default=CAMINHO_CHROMA_DB)
    args = parser.parse_args()
    main(args.pdf_dir, args.db_dir)