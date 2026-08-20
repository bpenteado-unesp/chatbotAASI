"""
Filtro de triagem.

Objetivo: nunca deixar o RAG responder quando a pergunta é, na prática,
um relato de sintoma pessoal ou pedido de avaliação clínica disfarçado
de dúvida técnica. Isso roda ANTES do RAG.

Duas camadas:
1. Palavras-chave (rápido, sem custo, pega os casos óbvios)
2. Classificador via LLM (pega os casos ambíguos)
"""
from google import genai
from google.genai import types

from config import carregar_chave_gemini

client = genai.Client(api_key=carregar_chave_gemini())
MODELO_LLM = "gemini-3.5-flash"

PALAVRAS_SINTOMA = [
    "estou sentindo", "meu ouvido", "minha orelha", "meu aparelho não",
    "meu aparelho está", "comigo", "no meu caso", "eu tenho",
    "estou com dor", "está doendo", "zumbido", "tontura",
    "não estou ouvindo", "perdi a audição", "isso é normal comigo",
    "é urgente", "socorro",
]

PROMPT_CLASSIFICADOR = """Classifique a pergunta abaixo em UMA categoria:

- CONCEITUAL: pergunta sobre teoria, funcionamento, tipos, manutenção geral de AASI (Aparelho de Amplificação Sonora Individual). Não menciona sintoma ou caso pessoal.
- SINTOMA_PESSOAL: a pessoa relata um sintoma, situação clínica própria, ou pede avaliação/diagnóstico do seu caso específico.

Responda APENAS com uma palavra: CONCEITUAL ou SINTOMA_PESSOAL.

Pergunta: {pergunta}"""


def contem_palavra_sintoma(pergunta: str) -> bool:
    p = pergunta.lower()
    return any(palavra in p for palavra in PALAVRAS_SINTOMA)


def classificar_com_llm(pergunta: str) -> str:
    resposta = client.models.generate_content(
        model=MODELO_LLM,
        contents=PROMPT_CLASSIFICADOR.format(pergunta=pergunta),
        config=types.GenerateContentConfig(
            max_output_tokens=100,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    texto = (resposta.text or "").strip().upper()

    if not texto:
        # Resposta vazia do modelo (ex: falha de rede, bloqueio de conteúdo,
        # ou qualquer outro motivo). NUNCA tratar isso como "CONCEITUAL" por
        # padrão — essa é a camada que bloqueia pergunta de sintoma pessoal,
        # e uma falha aqui deve ser conservadora (bloquear), não permissiva.
        return "SINTOMA_PESSOAL"

    return "SINTOMA_PESSOAL" if "SINTOMA" in texto else "CONCEITUAL"


def eh_pergunta_conceitual(pergunta: str) -> bool:
    """
    Retorna True apenas se a pergunta for segura para passar pelo RAG.
    Checagem por palavra-chave primeiro (barata); se não pegar nada
    suspeito, confirma com o classificador LLM para os casos ambíguos.
    """
    if contem_palavra_sintoma(pergunta):
        return False
    categoria = classificar_com_llm(pergunta)
    return categoria == "CONCEITUAL"


MENSAGEM_DESVIO = (
    "Essa parece ser uma questão sobre o seu caso pessoal, e eu não posso "
    "avaliar sintomas ou fazer diagnóstico — sou só um material de apoio "
    "com dúvidas técnicas gerais sobre AASI.\n\n"
    "Procure seu fonoaudiólogo ou o atendimento clínico da instituição "
    "para isso, tá bem? 🙂"
)