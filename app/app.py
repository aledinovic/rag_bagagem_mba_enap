import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Configurações básicas do Streamlit
st.set_page_config(page_title="RAG - Guia do Viajante", page_icon="✈️")
st.title("Assistente sobre Bagagem Desacompanhada 🛄")
st.write("Faça sua pergunta sobre bagagem desacompanhada.")

# Carregar variáveis de ambiente (OPENAI_API_KEY)
load_dotenv()
#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Definir constantes
PERSIST_DIRECTORY = "./chroma_db"
N_DOCUMENTOS = 3

@st.cache_resource(show_spinner=False)
def carregar_vector_db():
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    #vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_engine)
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_engine, collection_name="documentos")
    return vector_db

vector_db = carregar_vector_db()

# Função para formatar documentos
def format_docs(documentos):
    return "\n\n".join(documento.page_content for documento in documentos)

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

