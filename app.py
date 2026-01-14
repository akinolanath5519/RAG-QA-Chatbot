# app.py
# RAG Q&A Conversation With PDF Including Chat History (LangChain v1+)

import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_chroma import Chroma  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser  # Add this import

# -------------------------------------------------------------------
# Environment variables
# -------------------------------------------------------------------
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Conversational RAG with PDFs")
st.title("📄 Conversational RAG With PDF Uploads")
st.write("Upload PDFs and chat with their content.")

# Groq API key
api_key = st.text_input("Enter your Groq API key:", type="password")
if not api_key:
    st.warning("Please enter your Groq API key")
    st.stop()

# Session ID for chat history
session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

# -------------------------------------------------------------------
# Chat history helper
# -------------------------------------------------------------------
def get_session_history(session: str) -> BaseChatMessageHistory:
    """Return chat history for a given session."""
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

# Extract plain text from Document objects
texts_for_embedding = [doc.page_content for doc in splits]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Use from_texts to avoid sending AIMessage objects
vectorstore = Chroma.from_texts(texts=texts_for_embedding, embedding=embeddings)
retriever = vectorstore.as_retriever()

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# -------------------------------------------------------------------
# History-aware retriever (rephrase question) - FIXED VERSION
# -------------------------------------------------------------------
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question. "
    "Just return the standalone question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Add StrOutputParser to convert AIMessage to string
contextualize_runnable = contextualize_q_prompt | llm | StrOutputParser()

def format_docs(docs):
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)

# Create a runnable that takes the standalone question and retrieves docs
from langchain_core.runnables import RunnablePassthrough

def retrieve_docs(standalone_question: str):
    """Retrieve documents using the standalone question."""
    return retriever.invoke(standalone_question)

# Create the full RAG chain with proper formatting
rag_chain = (
    # Step 1: Get standalone question
    RunnablePassthrough.assign(
        standalone_question=contextualize_runnable
    )
    # Step 2: Retrieve documents
    .assign(
        docs=lambda x: retriever.invoke(x["standalone_question"])
    )
    # Step 3: Format docs for context
    .assign(
        context=lambda x: format_docs(x["docs"])
    )
)

# -------------------------------------------------------------------
# QA prompt
# -------------------------------------------------------------------
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Context:\n{context}\n\n"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the final QA chain
def create_qa_input(rag_output):
    """Create input for QA chain from RAG output."""
    return {
        "context": rag_output["context"],
        "input": rag_output["input"],
        "chat_history": rag_output.get("chat_history", [])
    }

qa_chain = (
    rag_chain 
    | create_qa_input 
    | qa_prompt 
    | llm
)

# -------------------------------------------------------------------
# Full RAG pipeline with history
# -------------------------------------------------------------------
conversational_rag_chain = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# -------------------------------------------------------------------
# Streamlit Chat UI
# -------------------------------------------------------------------
# Initialize chat history if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response from RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}},
            )
            
            # Display assistant response
            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)
            
            st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})