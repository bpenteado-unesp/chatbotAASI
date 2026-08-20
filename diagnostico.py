"""
Script de diagnóstico: mostra o que a busca híbrida (vetorial + BM25)
está retornando, sem gastar chamada de API do Gemini. Rode depois do
ingest.py para investigar por que as respostas estão vindo como "não
encontrei essa informação" ou por que um trecho esperado não aparece.

Uso:
    python diagnostico.py "sua pergunta de teste aqui"
    python diagnostico.py "sua pergunta" --checar "frase literal esperada"
"""
import sys

from config import CAMINHO_CHROMA_DB
from rag import buscar_contexto, colecao, _pontuar_todos

argumentos = sys.argv[1:]
frase_para_checar = None
if "--checar" in argumentos:
    i = argumentos.index("--checar")
    frase_para_checar = argumentos[i + 1]
    argumentos = argumentos[:i]

pergunta = argumentos[0] if argumentos else "Quais são os tipos de AASI?"

print(f"Pasta do ChromaDB configurada: {CAMINHO_CHROMA_DB}")
total = colecao.count()
print(f"Total de chunks na coleção 'aasi_docs': {total}")
if total == 0:
    print()
    print(">>> A coleção está VAZIA. O ingest.py não populou o banco, ou")
    print(">>> populou em um caminho diferente do que o rag.py está lendo.")
    sys.exit(0)

print()
print(f"Pergunta de teste: {pergunta!r}")
print("Busca híbrida (vetorial contra TODO o banco + BM25), via rag.buscar_contexto()")
print("-" * 70)

chunks = buscar_contexto(pergunta)
if not chunks:
    print(">>> Nenhum chunk passou no filtro. É esse o motivo do 'não encontrei'.")
else:
    for c in chunks:
        print(f"[score_hibrido={c['score_hibrido']:.3f}] "
              f"distância={c['distancia']:.4f} | score_bm25={c['score_bm25']:.2f}")
        print(f"  fonte: {c['fonte']} | seção: {c['titulo_secao'] or '(sem título)'}")
        print(f"  texto: {c['texto'][:150]}...")
        print()

# --- Checagem independente do filtro: onde um trecho específico fica no ranking? ---
if frase_para_checar:
    print("=" * 70)
    print(f'Procurando, em TODOS os chunks da coleção, a frase literal "{frase_para_checar}"...')

    ranking_completo = _pontuar_todos(pergunta)
    encontrados = [
        (posicao, c) for posicao, c in enumerate(ranking_completo, start=1)
        if frase_para_checar.lower() in c["texto"].lower()
    ]

    if not encontrados:
        print(">>> NÃO encontrado em nenhum chunk. O problema é na INGESTÃO, não na busca.")
    else:
        print(f"Encontrado(s) {len(encontrados)} chunk(s) com a frase literal, "
              f"de um total de {len(ranking_completo)} chunks no banco:")
        print()
        for posicao, c in encontrados:
            print(f"  POSIÇÃO {posicao} de {len(ranking_completo)} no ranking geral para esta pergunta")
            print(f"  fonte: {c['fonte']} | seção: {c['titulo_secao'] or '(sem título)'}")
            print(f"  distância={c['distancia']:.4f} | score_bm25={c['score_bm25']:.2f} "
                  f"| score_hibrido={c['score_hibrido']:.3f}")
            print(f"  {c['texto'][:300]}")
            print()

        if chunks:
            print(f"(O resultado final retornou {len(chunks)} chunks. Se a posição acima for maior "
                  f"que {len(chunks)}, é por isso que ficou de fora.)")