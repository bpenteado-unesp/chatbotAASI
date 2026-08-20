"""
Núcleo do RAG: recebe pergunta já filtrada pela triagem,
busca chunks relevantes e gera resposta ancorada nos PDFs,
sempre citando fonte + seção do manual (+ página aproximada
como complemento), e recusando responder quando a evidência
for insuficiente.

Geração de resposta via Gemini API (google-genai).

Nota sobre a citação: a página impressa é aproximada (calculada a partir
da posição no PDF), porque documentos reais às vezes têm numeração
irregular. Por isso a seção do manual (texto real, sempre exato) é a
referência principal — a página é só um complemento para ajudar a
localizar mais rápido.
"""
import re
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

from config import (
    CAMINHO_ASSETS,
    CAMINHO_CHROMA_DB,
    NOME_MODELO_EMBEDDING,
    carregar_chave_gemini,
    obter_nome_fonte,
)

client = genai.Client(api_key=carregar_chave_gemini())
MODELO_LLM = "gemini-3.5-flash"

modelo_embedding = SentenceTransformer(NOME_MODELO_EMBEDDING, cache_folder=str(CAMINHO_ASSETS))

chroma_client = chromadb.PersistentClient(path=CAMINHO_CHROMA_DB)
colecao = chroma_client.get_or_create_collection(
    "aasi_docs", metadata={"hnsw:space": "cosine"}
)

SYSTEM_PROMPT = """Você é um assistente de estudos sobre AASI (Aparelho de \
Amplificação Sonora Individual) para alunos do curso técnico.

Regras obrigatórias, nesta ordem de prioridade:

1. Responda SOMENTE com base nos trechos fornecidos abaixo. Nunca use \
conhecimento próprio para completar lacunas.

2. Antes de responder, avalie se os trechos fornecidos realmente sustentam \
uma resposta completa e direta à pergunta. Se os trechos forem sobre um \
assunto relacionado mas não respondem à pergunta específica, ou se a \
evidência for parcial/ambígua, responda EXATAMENTE:
"Não encontrei essa informação no material disponível. Recomendo perguntar \
ao professor ou consultar o material completo."
Não tente adivinhar nem complementar com inferência.

3. Se responder, cite a fonte de cada afirmação no formato \
[Fonte: nome_do_manual, seção "X"] logo após a informação, usando \
exatamente o nome de manual e o título de seção fornecidos no contexto. \
Isso permite ao aluno localizar o trecho original no material.

4. IMPORTANTE — os trechos de contexto podem vir de manuais de \
fabricantes/produtos DIFERENTES (o nome do manual identifica de qual). \
Se você combinar, na mesma resposta, informações vindas de manuais \
diferentes, deixe isso EXPLÍCITO no texto corrido, não só na citação \
entre colchetes — por exemplo: "Segundo o Manual GN ReSound, ... Já o \
Manual Phonak Audéo recomenda...". Nunca apresente recursos ou \
instruções específicas de um fabricante como se fossem genéricas ou \
válidas para qualquer aparelho, já que features e procedimentos variam \
entre marcas e modelos.

5. NUNCA avalie sintomas, faça diagnóstico ou dê recomendação clínica sobre \
um caso específico. Se a pergunta parecer pessoal, oriente a procurar \
atendimento clínico.

6. Seja direto e didático, como para um aluno técnico."""


STOPWORDS_PT = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "é", "em",
    "um", "uma", "uns", "umas", "para", "com", "que", "se", "no", "na",
    "nos", "nas", "por", "ou", "como", "ao", "aos", "à", "às", "sua", "seu",
    "suas", "seus", "isso", "essa", "esse", "está", "estão", "ser", "tem",
    "pode", "podem", "deve", "devem", "fazer", "faço", "então", "assim",
}

# Termos que aparecem em praticamente todo chunk deste domínio (manual de
# aparelho auditivo). Com BM25 isso passa a ser menos crítico — o IDF já
# dá peso baixo a termos que aparecem em quase todo documento, automatica-
# mente — mas mantemos como reforço extra, sem custo.
STOPWORDS_DOMINIO = {
    "aparelho", "aparelhos", "auditivo", "auditivos", "aasi", "resound",
}

STOPWORDS_PT = STOPWORDS_PT | STOPWORDS_DOMINIO


def _stem(palavra: str) -> str:
    """
    Stemming bem leve (trunca no 4º caractere). Não é linguisticamente
    rigoroso, mas resolve o caso prático que motivou isso: "apitando" e
    "apito" viram o mesmo stem ("apit"), assim como "trocar"/"troca",
    "bateria"/"baterias", "molde"/"moldes" — cobre bem os casos comuns de
    conjugação/flexão em português sem precisar de um stemmer de verdade.
    """
    return palavra[:4]


def _tokenizar(texto: str) -> list[str]:
    palavras = re.findall(r"[a-zà-úA-ZÀ-Ú]+", texto.lower())
    return [_stem(p) for p in palavras if p not in STOPWORDS_PT and len(p) > 2]


# --- Índice construído uma vez, não a cada pergunta ---
# Rebuscar e re-tokenizar toda a coleção a cada chamada seria desperdício;
# construímos os dados uma vez na carga do módulo. Se você rodar ingest.py
# de novo enquanto o app já está rodando, precisa reiniciar o processo
# para o índice ser reconstruído com o conteúdo novo.
_indice = colecao.get(include=["documents", "metadatas", "embeddings"])
_documentos_indexados = _indice["documents"]
_metadados_indexados = _indice["metadatas"]
_embeddings_indexados = np.array(_indice["embeddings"])
_corpus_tokenizado = [_tokenizar(doc) for doc in _documentos_indexados]
_bm25 = BM25Okapi(_corpus_tokenizado) if _corpus_tokenizado else None


def _pontuar_todos(pergunta: str) -> list[dict]:
    """
    Calcula o score híbrido (denso via cosseno + esparso via BM25) de TODOS
    os chunks do banco para uma pergunta, sem aplicar filtro nem corte de
    top_k — retorna a lista inteira ordenada. Usado tanto por
    buscar_contexto() quanto pelo diagnóstico (para achar a posição exata
    de um chunk específico no ranking, mesmo que ele não tenha entrado no
    resultado final).

    Por que BM25 em vez da sobreposição de palavras "na unha" que tínhamos
    antes: BM25 pondera cada termo pela sua raridade no corpus inteiro
    (IDF) e pela frequência no documento — um termo universal como
    "aparelho" automaticamente pesa pouco, sem precisar manter uma lista
    de stopwords de domínio ajustada manualmente toda vez que aparece um
    caso nôvo (o que já aconteceu 3 vezes: aparelho/auditivo, depois
    "são", depois "pode"/"apitando"×"apito").
    """
    if not _documentos_indexados:
        return []

    embedding_pergunta = np.array(
        modelo_embedding.encode([pergunta], normalize_embeddings=True)[0]
    )
    similaridades = _embeddings_indexados @ embedding_pergunta  # cosseno

    tokens_pergunta = _tokenizar(pergunta)
    if tokens_pergunta and _bm25 is not None:
        scores_bm25 = np.array(_bm25.get_scores(tokens_pergunta))
        maximo = scores_bm25.max()
        scores_bm25_norm = scores_bm25 / maximo if maximo > 0 else scores_bm25
    else:
        scores_bm25_norm = np.zeros(len(_documentos_indexados))

    candidatos = []
    for doc, meta, sim, score_bm25 in zip(
        _documentos_indexados, _metadados_indexados, similaridades, scores_bm25_norm
    ):
        sim = float(sim)
        distancia = 1 - sim
        score_bm25 = float(score_bm25)
        score_hibrido = sim * 0.6 + score_bm25 * 0.4

        candidatos.append({
            "texto": doc,
            "fonte": meta["fonte"],
            "titulo_secao": meta.get("titulo_secao") or "",
            "pagina_inicio": meta.get("pagina_inicio") or 0,
            "pagina_fim": meta.get("pagina_fim") or 0,
            "tipo": meta.get("tipo", "texto"),
            "distancia": distancia,
            "score_bm25": score_bm25,
            "score_hibrido": score_hibrido,
        })

    candidatos.sort(key=lambda c: c["score_hibrido"], reverse=True)
    return candidatos


def buscar_contexto(
    pergunta: str,
    top_k: int = 8,
    distancia_maxima: float = 0.5,
) -> list[dict]:
    """
    Busca híbrida: combina similaridade vetorial (cosseno, 60%) com BM25
    (busca por palavra-chave ponderada por raridade, 40%), calculada
    contra TODOS os chunks do banco — não um pool limitado.

    Aceita um chunk se ele passar no limiar de distância vetorial OU tiver
    um score BM25 alto (match textual forte compensa uma distância
    vetorial um pouco pior).
    """
    candidatos = _pontuar_todos(pergunta)
    selecionados = [
        c for c in candidatos
        if c["distancia"] <= distancia_maxima or c["score_bm25"] >= 0.5
    ]
    return selecionados[:top_k]


def _formatar_referencia(chunk: dict) -> str:
    partes = [f"Fonte: {obter_nome_fonte(chunk['fonte'])}"]
    if chunk["titulo_secao"]:
        partes.append(f'seção "{chunk["titulo_secao"]}"')
    if chunk["pagina_inicio"]:
        if chunk["pagina_inicio"] == chunk["pagina_fim"]:
            partes.append(f"pág. ~{chunk['pagina_inicio']}")
        else:
            partes.append(f"págs. ~{chunk['pagina_inicio']}-{chunk['pagina_fim']}")
    return ", ".join(partes)


MENSAGEM_SEM_EVIDENCIA = (
    "Não encontrei essa informação no material disponível. "
    "Recomendo perguntar ao professor ou consultar o material completo."
)


def responder(pergunta: str) -> dict:
    """
    Retorna um dict com:
    - resposta: texto da resposta (já inclui citação [Fonte, seção] quando aplicável)
    - fontes: lista de trechos usados, para exibir na interface (ex: Streamlit)
    """
    chunks = buscar_contexto(pergunta)

    if not chunks:
        return {"resposta": MENSAGEM_SEM_EVIDENCIA, "fontes": []}

    contexto = "\n\n---\n\n".join(
        f"[{_formatar_referencia(c)}]\n{c['texto']}"
        for c in chunks
    )

    mensagem = f"""Trechos do material didático:

{contexto}

Pergunta do aluno: {pergunta}"""

    resposta = client.models.generate_content(
        model=MODELO_LLM,
        contents=mensagem,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    texto_resposta = resposta.text

    return {"resposta": texto_resposta, "fontes": chunks}