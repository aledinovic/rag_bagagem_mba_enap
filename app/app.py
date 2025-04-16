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

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

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

# Construir pipeline RAG
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
prompt = hub.pull("rlm/rag-prompt")

rag = (
    {
        "question": RunnablePassthrough(),
        "context": vector_db.as_retriever(k=N_DOCUMENTOS) | format_docs
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Campo para digitar pergunta
pergunta_usuario = st.text_input("Digite sua pergunta:", "")

# Botão de envio
if st.button("Perguntar"):
    if pergunta_usuario:
        with st.spinner("Buscando resposta..."):
            resposta = rag.invoke(pergunta_usuario)
        st.markdown("### Resposta:")
        st.write(resposta)

        # Avaliação simples (feedback)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Correto"):
                feedback = "Correto"
                st.success("Feedback registrado: 👍 Correto")
        with col2:
            if st.button("👎 Incorreto"):
                feedback = "Incorreto"
                st.error("Feedback registrado: 👎 Incorreto")
    else:
        st.warning("Por favor, digite uma pergunta.")
