# app.py
# RAG Q&A Conversation With PDF Including Chat History

import os
import streamlit as st
import tempfile
from pathlib import Path

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_chroma import Chroma  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("sentence-transformers is not installed. Please add it to requirements.txt")

# -------------------------------------------------------------------
# Custom CSS for Minimal Design
# -------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
        background-color: white;
    }
    
    .stChatMessage[data-testid="user-message"] {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
    }
    
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f0f9ff;
        border-left: 4px solid #0ea5e9;
    }
    
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Service Chat Bot",
    page_icon="💬",
    layout="wide"
)

# -------------------------------------------------------------------
# Main Title
# -------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1e293b; font-size: 2.5rem; margin-bottom: 0.5rem;'>
            💬 Customer Service Chat Bot
        </h1>
        <p style='color: #64748b; font-size: 1.1rem;'>
            Upload your knowledge-base PDFs and let AI assist your customers
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    api_key = st.text_input(
        "Password",
        type="password",
        help="Enter your password to enable the chatbot"
    )
    
    if not api_key:
        st.warning("Please enter your password")

    session_id = st.text_input(
        "Session ID",
        value="default_session",
        help="Unique identifier for your chat session"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This is a Customer Service Chat Bot that:
    - Learns from your uploaded PDFs
    - Answers customer questions using AI
    - Keeps chat history per session
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & LangChain")

# Stop if embeddings package missing
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.stop()

# Session store
if "store" not in st.session_state:
    st.session_state.store = {}

# -------------------------------------------------------------------
# Chat history helper
# -------------------------------------------------------------------
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# -------------------------------------------------------------------
# Custom Embeddings Class
# -------------------------------------------------------------------
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, text: str):
        text = text.replace("\n", " ").strip() or " "
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: list):
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        return [self.model.encode(t).tolist() for t in cleaned]

# -------------------------------------------------------------------
# Main Layout
# -------------------------------------------------------------------
main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.markdown("### 📤 Upload Knowledge Base (PDFs)")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Uploaded Files:**")
        for file in uploaded_files:
            st.markdown(f"📄 **{file.name}** ({file.size:,} bytes)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your password in the sidebar first")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        documents = []
                        
                        with tempfile.TemporaryDirectory() as temp_dir:
                            for uploaded_file in uploaded_files:
                                temp_path = Path(temp_dir) / uploaded_file.name
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getvalue())
                                
                                loader = PyPDFLoader(str(temp_path))
                                documents.extend(loader.load())
                        
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=5000,
                            chunk_overlap=500,
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        embeddings = SentenceTransformerEmbeddings()
                        texts = [doc.page_content for doc in splits]
                        metadatas = [doc.metadata for doc in splits]
                        
                        vectorstore = Chroma.from_texts(
                            texts=texts,
                            embedding=embeddings,
                            metadatas=metadatas,
                        )
                        
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        st.session_state.processing_complete = True
                        
                        st.success(f"✅ Processed {len(documents)} pages into {len(splits)} chunks")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    else:
        st.info("👆 Upload PDFs to train the Customer Service Chat Bot")

with main_col2:
    if uploaded_files and api_key and st.session_state.get('processing_complete'):
        st.markdown("### 💬 Customer Service Chat")
        
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "rewrite it as a standalone question."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

        qa_system_prompt = (
            "You are a helpful customer service assistant. "
            "Use the retrieved context to answer the user's question. "
            "If you don't know, say you don't know. Keep it concise.\n\n"
            "Context:\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = qa_prompt | llm

        conversational_rag_chain = RunnableWithMessageHistory(
            qa_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        chat_history = get_session_history(session_id).messages
        for message in chat_history[-10:]:
            with st.chat_message(message.type):
                st.markdown(message.content)

        if prompt := st.chat_input("Ask the Customer Service Bot..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        standalone_q = contextualize_q_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history
                        })

                        docs = st.session_state.retriever.invoke(standalone_q)
                        context = "\n\n".join(d.page_content for d in docs)

                        response = conversational_rag_chain.invoke(
                            {"input": prompt, "context": context},
                            config={"configurable": {"session_id": session_id}},
                        )

                        answer = response.content if hasattr(response, "content") else str(response)
                        st.markdown(answer)

                        with st.expander("📖 View retrieved context"):
                            st.write(context)

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        if chat_history:
            if st.button("Clear Chat", use_container_width=True):
                get_session_history(session_id).clear()
                st.rerun()

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1:
    st.caption("💬 Customer Service Chat Bot")
with f2:
    st.caption("Powered by LangChain & Groq")
with f3:
    st.caption("Upload • Chat • Support")
