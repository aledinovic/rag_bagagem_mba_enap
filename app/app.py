import streamlit as st
import pandas as pd
from datetime import datetime
import json
from io import StringIO

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -------------------------
# CONFIGURAÇÕES
# -------------------------

st.set_page_config(page_title="RAG - Bagagem Desacompanhada", page_icon="🛄")
st.title("Assistente sobre Bagagem Desacompanhada 🛄")
st.write("Faça sua pergunta sobre bagagem desacompanhada e avalie a resposta.")

PERSIST_DIRECTORY = "faiss_db"
N_DOCUMENTOS = 3

# -------------------------
# CONTROLE DE ESTADO
# -------------------------

if "resposta" not in st.session_state:
    st.session_state.resposta = ""
if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""

# -------------------------
# CARREGAR FAISS VETORIAL
# -------------------------

@st.cache_resource(show_spinner=False)
def carregar_vector_db():
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.load_local(PERSIST_DIRECTORY, embedding_engine, allow_dangerous_deserialization=True)
    return vector_db

vector_db = carregar_vector_db()

# -------------------------
# FORMATADOR DE CONTEXTO
# -------------------------

def format_docs(documentos):
    return "\n\n".join(doc.page_content for doc in documentos)

# -------------------------
# REGISTRAR FEEDBACK NO GOOGLE SHEETS
# -------------------------

def registrar_feedback_sheets(pergunta, resposta, avaliacao):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Feedback RAG Bagagem").sheet1

    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nova_linha = [data_hora, pergunta, resposta, avaliacao]
    sheet.append_row(nova_linha)

# -------------------------
# PROMPT MANUAL
# -------------------------

prompt = PromptTemplate.from_template("""
Use o contexto abaixo para responder à pergunta. 
Se a resposta não estiver no contexto, diga "Desculpe, não sei responder com base nas informações disponíveis."

Contexto:
{context}

Pergunta:
{question}
""")

# -------------------------
# PIPELINE RAG FUNCIONAL
# -------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)

retriever = vector_db.as_retriever(search_kwargs={"k": N_DOCUMENTOS})

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------------
# INTERFACE USUÁRIO
# -------------------------

pergunta_usuario = st.text_input("Digite sua pergunta:", st.session_state.pergunta)

if st.button("Perguntar"):
    pergunta_formatada = str(pergunta_usuario).strip()

    if not pergunta_formatada:
        st.warning("Por favor, digite uma pergunta válida.")
    else:
        # Debug: exibir tipo e valor da pergunta (remova depois de testar)
        st.write(f"**DEBUG**: Tipo da pergunta: {type(pergunta_formatada)}, Valor: {repr(pergunta_formatada)}")

        with st.spinner("Buscando resposta..."):
            try:
                resposta = rag_chain.invoke({"question": pergunta_formatada})
                st.session_state.resposta = resposta
                st.session_state.pergunta = pergunta_formatada
            except Exception as e:
                st.error("Falha ao processar a pergunta. Tente reformular a frase ou remover caracteres especiais.")
                st.error(f"Detalhes do erro: {e}")
                # Se quiser interromper o app aqui:
                # st.stop()

# -------------------------
# EXIBIR RESPOSTA E FEEDBACK
# -------------------------

if st.session_state.resposta:
    st.markdown("### Resposta:")
    st.write(st.session_state.resposta)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Correto"):
            registrar_feedback_sheets(
                st.session_state.pergunta,
                st.session_state.resposta,
                "correto"
            )
            st.success("Feedback registrado: 👍 Correto")
            st.session_state.resposta = ""
            st.session_state.pergunta = ""

    with col2:
        if st.button("👎 Incorreto"):
            registrar_feedback_sheets(
                st.session_state.pergunta,
                st.session_state.resposta,
                "incorreto"
            )
            st.warning("Feedback registrado: 👎 Incorreto")
            st.session_state.resposta = ""
            st.session_state.pergunta = ""
