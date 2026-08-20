"""
Configuração compartilhada: caminhos padrão e carregamento de credenciais.

A chave da Gemini API é lida via st.secrets, o mecanismo nativo do
Streamlit, a partir de `.streamlit/secrets.toml` (relativo ao diretório
de trabalho de onde o script é executado). Isso funciona tanto dentro de
`streamlit run` quanto em scripts avulsos (testar_local.py, app.py) —
st.secrets só depende do arquivo existir, não exige o servidor Streamlit
rodando.
"""
from pathlib import Path

import streamlit as st

CAMINHO_CHROMA_DB = "./data/chromadb"
CAMINHO_ASSETS = Path("./assets")
CAMINHO_PDFS = Path("./data/pdfs")
NOME_MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# Nome amigável de fabricante/produto para cada PDF do acervo, usado nas
# citações e para o modelo conseguir dizer explicitamente "segundo o
# manual da GN ReSound..." em vez de citar só o nome cru do arquivo.
# Adicione uma entrada aqui para cada PDF novo em ./data/pdfs — se um
# arquivo não estiver mapeado, o nome do arquivo é usado como está.
NOMES_FONTES = {
    "manual gnresound bte.pdf": "Manual GN ReSound (retroauricular)",
    "PH_UserGuide_Audeo-I-R_92x125_PT-BR_V2.01_029-1357-43.pdf": "Manual Phonak Audéo",
}


def obter_nome_fonte(nome_arquivo: str) -> str:
    return NOMES_FONTES.get(nome_arquivo, nome_arquivo)


def carregar_chave_gemini() -> str:
    """
    Lê a chave da Gemini API de .streamlit/secrets.toml, no formato:

        [gemini]
        api_key = "sua_chave_aqui"
    """
    try:
        return st.secrets["gemini"]["api_key"]
    except (FileNotFoundError, KeyError) as e:
        raise RuntimeError(
            "Não encontrei a chave da Gemini API em '.streamlit/secrets.toml' "
            "(relativo à pasta de onde você rodou o script). Crie o arquivo "
            'com o conteúdo:\n\n[gemini]\napi_key = "sua_chave_aqui"'
        ) from e