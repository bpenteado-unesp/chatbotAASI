"""
Webhook do WhatsApp (via Twilio) que liga tudo:
Aluno pergunta -> triagem -> RAG (se conceitual) ou desvio (se pessoal) -> responde.

Uso local:
    uvicorn app:app --reload
Depois exponha com ngrok e configure a URL no console da Twilio.
"""
from fastapi import FastAPI, Form, Response
from twilio.twiml.messaging_response import MessagingResponse

from triage import eh_pergunta_conceitual, MENSAGEM_DESVIO
from rag import responder

app = FastAPI()


@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    pergunta = Body.strip()

    if eh_pergunta_conceitual(pergunta):
        resultado = responder(pergunta)
        texto_resposta = resultado["resposta"]
        # WhatsApp já mostra a citação [Fonte, pág.] embutida no texto,
        # já que a resposta do modelo inclui essa referência.
    else:
        texto_resposta = MENSAGEM_DESVIO

    twiml = MessagingResponse()
    twiml.message(texto_resposta)
    return Response(content=str(twiml), media_type="application/xml")


@app.get("/health")
async def health():
    return {"status": "ok"}