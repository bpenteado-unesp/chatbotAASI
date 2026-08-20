"""
Interface Streamlit para testar o chatbot de AASI antes de plugar no WhatsApp.
Mostra também os trechos (fonte + seção + página aproximada) usados em
cada resposta, para o usuário poder validar a informação.

Duas otimizações de performance:
1. st.cache_resource — o carregamento pesado (modelo de embeddings, conexão
   com o ChromaDB, construção do índice BM25) roda só UMA VEZ por processo
   do servidor, não a cada pergunta nem a cada reinicialização de sessão.
   Isso já acontecia "de graça" por causa do cache de import do Python,
   mas usar st.cache_resource explicitamente é mais robusto (sobrevive a
   recarregamento automático de módulos em modo dev) e mostra um spinner
   de carregamento pro usuário na primeira vez.
2. st.cache_data — a resposta completa (triagem + busca + geração) fica
   em cache por pergunta EXATA (mesmo texto), por até 1h. Se dois alunos
   perguntarem a mesma coisa, a segunda vez é instantânea, sem gastar
   chamada de API do Gemini de novo.

Uso:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Tira-dúvidas AASI", page_icon="🎧")


@st.cache_resource(show_spinner="Carregando modelo de embeddings, banco vetorial e índice de busca (só na primeira vez)...")
def carregar_backend():
    import rag
    import triage
    return rag, triage


_rag, _triage = carregar_backend()
responder = _rag.responder
_formatar_referencia = _rag._formatar_referencia
eh_pergunta_conceitual = _triage.eh_pergunta_conceitual
MENSAGEM_DESVIO = _triage.MENSAGEM_DESVIO


@st.cache_data(show_spinner=False, ttl=3600)
def responder_cacheado(pergunta: str) -> dict:
    """
    Cacheia a resposta completa por pergunta exata (mesmo texto, mesma
    resposta), por até 1h. Perguntas parecidas mas não idênticas (ex:
    "como trocar a bateria" vs "como troco a bateria") NÃO compartilham
    cache — isso é uma limitação aceitável para o volume de uso esperado.
    """
    if eh_pergunta_conceitual(pergunta):
        return responder(pergunta)
    return {"resposta": MENSAGEM_DESVIO, "fontes": []}


@st.cache_data(show_spinner="Lendo PDFs do dataset...")
def listar_pdfs() -> list[dict]:
    import pdfplumber
    from config import CAMINHO_PDFS, obter_nome_fonte

    pdfs = sorted(CAMINHO_PDFS.glob("*.pdf")) if CAMINHO_PDFS.exists() else []
    info = []
    for caminho in pdfs:
        tamanho_mb = caminho.stat().st_size / (1024 * 1024)
        try:
            with pdfplumber.open(caminho) as pdf:
                n_paginas = len(pdf.pages)
        except Exception:
            n_paginas = None
        info.append({
            "arquivo": caminho.name,
            "nome_amigavel": obter_nome_fonte(caminho.name),
            "tamanho_mb": tamanho_mb,
            "paginas_pdf": n_paginas,
        })
    return info


aba_chat, aba_pdfs = st.tabs(["💬 Chat", "📚 PDFs do curso"])

with aba_chat:
    st.title("🎧 Tira-dúvidas — AASI")
    st.caption(
        "Assistente de estudo baseado no material didático do curso. "
        "Não substitui atendimento clínico."
    )

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    def _renderizar_fontes(fontes):
        with st.expander("Ver trechos usados"):
            for f in fontes:
                st.markdown(f"**{_formatar_referencia(f)}**")
                st.text(f["texto"])
                st.divider()

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["conteudo"])
            if msg.get("fontes"):
                _renderizar_fontes(msg["fontes"])

    pergunta = st.chat_input("Digite sua dúvida sobre AASI...")

    if pergunta:
        st.session_state.mensagens.append({"role": "user", "conteudo": pergunta, "fontes": None})
        with st.chat_message("user"):
            st.markdown(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                resultado = responder_cacheado(pergunta)
                resposta = resultado["resposta"]
                fontes = resultado["fontes"]
            st.markdown(resposta)
            if fontes:
                _renderizar_fontes(fontes)

        st.session_state.mensagens.append(
            {"role": "assistant", "conteudo": resposta, "fontes": fontes}
        )

    with st.sidebar:
        st.subheader("Sobre")
        st.write(
            "Este bot responde apenas dúvidas conceituais sobre AASI "
            "com base no material didático. Perguntas que parecem "
            "relato de sintoma pessoal são desviadas para atendimento clínico. "
            "Toda resposta cita a seção e a página aproximada de origem."
        )
        if st.button("Limpar conversa"):
            st.session_state.mensagens = []
            st.rerun()

with aba_pdfs:
    st.title("📚 PDFs usados neste dataset")
    st.caption("Material didático indexado para responder às dúvidas dos alunos.")

    pdfs_info = listar_pdfs()

    if not pdfs_info:
        st.info(
            "Nenhum PDF encontrado em `./data/pdfs`. Rode `python ingest.py` "
            "depois de colocar os arquivos na pasta."
        )
    else:
        for p in pdfs_info:
            with st.container(border=True):
                st.markdown(f"**{p['nome_amigavel']}**")
                detalhes = f"Arquivo: `{p['arquivo']}` · {p['tamanho_mb']:.1f} MB"
                if p["paginas_pdf"]:
                    detalhes += f" · {p['paginas_pdf']} páginas no arquivo PDF"
                st.caption(detalhes)

        st.divider()
        st.caption(
            f"Total: {len(pdfs_info)} documento(s). "
            "Para adicionar um novo PDF: coloque o arquivo em `./data/pdfs`, "
            "adicione um nome amigável em `NOMES_FONTES` (config.py) e rode "
            "`python ingest.py` novamente."
        )