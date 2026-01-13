# RAG Q&A Conversation With PDF Including Chat History

import os
import streamlit as st

from langchain.chains import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# -------------------------------------------------------------------
# Environment variables (Streamlit Cloud safe)
# -------------------------------------------------------------------
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Conversational RAG with PDFs")
st.title("📄 Conversational RAG With PDF Uploads")
st.write("Upload PDFs and chat with their content.")

# Groq API key (entered by user)
api_key = st.text_input("Enter your Groq API key:", type="password")

if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="Gemma2-9b-It",
)

# -------------------------------------------------------------------
# Session & chat history store
# -------------------------------------------------------------------
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]


# -------------------------------------------------------------------
# File upload
# -------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

# -------------------------------------------------------------------
# Load PDFs
# -------------------------------------------------------------------
documents = []

for uploaded_file in uploaded_files:
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(temp_path)
    documents.extend(loader.load())

# -------------------------------------------------------------------
# Text splitting & embeddings
# -------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=500,
)

splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

# -------------------------------------------------------------------
# History-aware retriever
# -------------------------------------------------------------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt,
)

# -------------------------------------------------------------------
# QA chain
# -------------------------------------------------------------------
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
)

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    question_answer_chain,
)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# -------------------------------------------------------------------
# Chat UI
# -------------------------------------------------------------------
user_input = st.text_input("Your question:")

if user_input:
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    st.write("### Assistant")
    st.write(response["answer"])

    st.write("### Chat History")
    st.write(get_session_history(session_id).messages)
