import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Configurações básicas do Streamlit
st.set_page_config(page_title="RAG - Guia do Viajante", page_icon="✈️")
st.title("Assistente sobre Bagagem Desacompanhada 🛄")
st.write("Faça sua pergunta sobre bagagem desacompanhada.")

if "resposta" not in st.session_state:
    st.session_state.resposta = ""
if "pergunta" not in st.session_state:
    st.session_state.pergunta = ""

# Carregar variáveis de ambiente (OPENAI_API_KEY)
load_dotenv()
#OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Definir constantes
PERSIST_DIRECTORY = "./faiss_db"
N_DOCUMENTOS = 3

@st.cache_resource(show_spinner=False)
def carregar_vector_db():
    embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_db = FAISS.load_local(PERSIST_DIRECTORY, embedding_engine, allow_dangerous_deserialization=True)
    return vector_db

vector_db = carregar_vector_db()

# -------------------------
# FUNÇÃO: registrar feedback no Google Sheets
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



# Função para formatar documentos
def format_docs(documentos):
    return "\n\n".join(documento.page_content for documento in documentos)

# Construir pipeline RAG
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini")
#prompt = hub.pull("rlm/rag-prompt")

prompt = PromptTemplate.from_template("""
Use o contexto abaixo para responder à pergunta. Se a resposta não estiver contida no contexto, diga "Desculpe, não sei responder com base nas informações disponíveis."

Contexto:
{context}

Pergunta:
{question}
""")

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
pergunta_usuario = st.text_input("Digite sua pergunta:", st.session_state.pergunta)

if st.button("Perguntar"):
    if pergunta_usuario.strip():
        with st.spinner("Buscando resposta..."):
            st.session_state.resposta = rag_chain({"question": pergunta_usuario})
            st.session_state.pergunta = pergunta_usuario
    else:
        st.warning("Por favor, digite uma pergunta.")

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
