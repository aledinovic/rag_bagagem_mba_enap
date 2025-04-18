import streamlit as st
import json
import torch
import os
from io import StringIO

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# -------------------------
# CONFIGURAÇÕES
# -------------------------
#torch.classes.__path__ = []
#torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

st.set_page_config(page_title="RAG - Bagagem Desacompanhada", page_icon="🛄")
st.title("Assistente sobre Bagagem Desacompanhada 🛄")
st.write("Faça sua pergunta sobre bagagem desacompanhada e avalie a resposta.")

PERSIST_DIRECTORY = "faiss_db"
N_DOCUMENTOS = 3

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["HUGGINGFACE_TOKEN"]

# -------------------------
# CARREGAR FAISS VETORIAL
# -------------------------

@st.cache_resource(show_spinner=False)
def carregar_vector_db():
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print(type(embedding_engine))
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

prompt_template = """
Você é um auditor fiscal especializado em regras e procedimentos sobre bagagem desacompanhada trazida de viagens ao exterior, com base na legislação vigente. Sua tarefa é fornecer respostas precisas, concisas e fáceis de entender, utilizando os pedaços do contexto fornecido. Se voce não souber a resposta, informe que a resposta não pode ser fornecida com base nos dados disponíveis.

**Contexto**:  
{context}

**Pergunta**:  
{question}

**Resposta**:  
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

rag = (
    {
        "question": RunnablePassthrough(),
        "context": vector_db.as_retriever(k = n_documentos) 
                    | format_docs
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
                st.rerun()
        with col2:
            if st.button("👎 Incorreto"):
                feedback = "Incorreto"
                st.error("Feedback registrado: 👎 Incorreto")
                st.rerun()
    else:
        st.warning("Por favor, digite uma pergunta.")
