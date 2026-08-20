"""
Testa o pipeline (triagem + RAG) direto no terminal, sem precisar
configurar o WhatsApp ainda. Rode depois de ter feito o ingest.py.

Uso:
    python testar_local.py
"""
from triage import eh_pergunta_conceitual, MENSAGEM_DESVIO
from rag import responder, _formatar_referencia

print("Digite sua pergunta sobre AASI (ou 'sair' para encerrar)\n")

while True:
    pergunta = input("Aluno: ").strip()
    if pergunta.lower() == "sair":
        break

    if eh_pergunta_conceitual(pergunta):
        resultado = responder(pergunta)
        print(f"\nBot: {resultado['resposta']}")
        if resultado["fontes"]:
            print("\nTrechos usados:")
            for f in resultado["fontes"]:
                print(f"  [{_formatar_referencia(f)}]")
        print()
    else:
        print(f"\nBot: {MENSAGEM_DESVIO}\n")