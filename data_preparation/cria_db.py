from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


dados = PyPDFLoader("perguntas_respostas_guia_viajante.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
textos = text_splitter.split_documents(dados)
embedding_engine = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vector_db = FAISS.from_documents(textos, embedding_engine)
vector_db.save_local("faiss_db")
