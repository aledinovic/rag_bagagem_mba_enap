# Guia Viajante - App RAG com Streamlit e LangChain

Este projeto fornece respostas automáticas sobre bagagem desacompanhada utilizando Retrieval-Augmented Generation (RAG).

## 📦 Estrutura do Projeto

- `data_preparation/`: scripts e dados originais para criação do banco vetorial.
- `chroma_db/`: banco vetorial persistente gerado pela rotina acima.
- `app/`: aplicação Streamlit que consome os dados do banco vetorial.

## 🚀 Como Atualizar o Banco de Dados

- Atualize ou adicione novos arquivos PDF em `data_preparation/`.
- Execute o notebook `create_db.ipynb`.
- Após executar, faça commit e push do novo banco (`chroma_db/`) para o GitHub.

## ☁️ Como Executar o App Online

- Hospedado no Streamlit Community Cloud automaticamente a partir deste repositório.

