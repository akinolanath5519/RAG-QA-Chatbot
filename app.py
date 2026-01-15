# app.py
# Enhanced RAG Q&A Conversation With PDFs - Beautiful UI

import os
import streamlit as st
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

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
from langchain_core.runnables import RunnablePassthrough

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("sentence-transformers is not installed. Please add it to requirements.txt")

# -------------------------------------------------------------------
# Custom CSS for Beautiful UI
# -------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        color: white;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #4a5568;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border: 2px solid #2f855a;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border: 2px solid #e2e8f0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: #edf2f7;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border: 2px dashed #cbd5e0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Stats card styling */
    .stats-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    
    /* Chat input styling */
    .stChatInput > div > div {
        border-radius: 25px;
        border: 2px solid #667eea;
        background: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliDoc AI - RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# Initialize Session State
# -------------------------------------------------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'store' not in st.session_state:
    st.session_state.store = {}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}

# -------------------------------------------------------------------
# Sidebar Configuration
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='header'><h2>⚙️ Configuration</h2></div>", unsafe_allow_html=True)
    
    # API Key Section
    st.markdown("### 🔑 API Configuration")
    api_key = st.text_input("Groq API Key", type="password", 
                          help="Enter your Groq API key to enable the chatbot")
    
    if not api_key:
        st.warning("⚠️ Please enter your Groq API key to continue")
        st.info("Get your API key from [Groq Console](https://console.groq.com)")
    
    # Session Management
    st.markdown("### 📝 Session")
    col1, col2 = st.columns(2)
    with col1:
        session_id = st.text_input("Session ID", value="session_1", 
                                  help="Unique ID for chat session")
    with col2:
        if st.button("🔄 New Session", use_container_width=True):
            if session_id in st.session_state.store:
                st.session_state.store[session_id].clear()
            st.rerun()
    
    # Embedding Model Selection
    st.markdown("### 🧠 Embedding Model")
    embedding_model = st.selectbox(
        "Select Model",
        ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2", "all-mpnet-base-v2"],
        index=0,
        help="Choose the embedding model for processing documents"
    )
    
    # Processing Settings
    st.markdown("### ⚡ Processing Settings")
    chunk_size = st.slider("Chunk Size", 1000, 10000, 5000, 500,
                          help="Size of text chunks for processing")
    chunk_overlap = st.slider("Chunk Overlap", 0, 2000, 500, 100,
                            help="Overlap between chunks for better context")
    
    # LLM Settings
    st.markdown("### 🤖 LLM Settings")
    llm_model = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Select the LLM model for responses"
    )
    
    # Statistics Display
    if st.session_state.processing_stats:
        st.markdown("### 📊 Statistics")
        stats_html = f"""
        <div class='stats-card'>
            <p><strong>📄 Documents:</strong> {st.session_state.processing_stats.get('documents', 0)}</p>
            <p><strong>📝 Chunks:</strong> {st.session_state.processing_stats.get('chunks', 0)}</p>
            <p><strong>⏱️ Processing Time:</strong> {st.session_state.processing_stats.get('time', 0):.2f}s</p>
        </div>
        """
        st.markdown(stats_html, unsafe_allow_html=True)
    
    # Clear All Button
    st.markdown("---")
    if st.button("🗑️ Clear All Data", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# -------------------------------------------------------------------
# Main Content Area
# -------------------------------------------------------------------

# Header with Gradient
st.markdown("""
<div class='header'>
    <h1>🤖 IntelliDoc AI</h1>
    <p>Intelligent Document Analysis & Conversational RAG System</p>
</div>
""", unsafe_allow_html=True)

# Check for sentence-transformers
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.error("❌ Missing required package: sentence-transformers")
    st.info("💡 Add 'sentence-transformers' to your requirements.txt")
    st.stop()

# -------------------------------------------------------------------
# Custom Embeddings Class
# -------------------------------------------------------------------
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_query(self, text: str):
        """Embed a single query."""
        text = text.replace("\n", " ").strip()
        if not text:
            text = " "
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: list):
        """Embed multiple documents."""
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        cleaned_texts = [text if text else " " for text in cleaned_texts]
        return [self.model.encode(text).tolist() for text in cleaned_texts]

# -------------------------------------------------------------------
# File Upload Section
# -------------------------------------------------------------------
st.markdown("### 📤 Upload Documents")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_files = st.file_uploader(
        "Drag and drop your PDF files here",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or multiple PDF files for analysis"
    )
with col2:
    process_btn = st.button("🚀 Process Documents", type="primary", use_container_width=True)

if uploaded_files:
    # Display uploaded files in a beautiful way
    st.markdown("#### 📋 Uploaded Files")
    for file in uploaded_files:
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            st.markdown("📄")
        with col2:
            st.markdown(f"**{file.name}**")
        with col3:
            st.markdown(f"{file.size:,} bytes")
        st.markdown("---")
    
    if process_btn and api_key:
        with st.spinner("🔧 Processing documents..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                start_time = time.time()
                documents = []
                
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Process each file
                    for i, uploaded_file in enumerate(uploaded_files):
                        status_text.text(f"📥 Loading {uploaded_file.name}...")
                        progress_bar.progress((i / len(uploaded_files)) * 0.3)
                        
                        # Save and load PDF
                        temp_path = Path(temp_dir) / uploaded_file.name
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        loader = PyPDFLoader(str(temp_path))
                        loaded_docs = loader.load()
                        documents.extend(loaded_docs)
                
                # Split documents
                status_text.text("✂️ Splitting documents into chunks...")
                progress_bar.progress(0.4)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                splits = text_splitter.split_documents(documents)
                
                # Create embeddings
                status_text.text("🧠 Creating embeddings...")
                progress_bar.progress(0.6)
                
                embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
                texts = [doc.page_content for doc in splits]
                metadatas = [doc.metadata for doc in splits]
                
                # Create vectorstore
                status_text.text("💾 Creating vector database...")
                progress_bar.progress(0.8)
                
                vectorstore = Chroma.from_texts(
                    texts=texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    collection_name=f"docs_{int(time.time())}",
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # Store in session state
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.processing_complete = True
                st.session_state.processing_stats = {
                    "documents": len(documents),
                    "chunks": len(splits),
                    "time": time.time() - start_time
                }
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Show success message
                st.balloons()
                st.success(f"✨ Successfully processed {len(documents)} pages into {len(splits)} chunks!")
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

elif process_btn:
    st.warning("⚠️ Please upload PDF files first")

# -------------------------------------------------------------------
# Chat Interface (Only show if processing is complete)
# -------------------------------------------------------------------
if st.session_state.processing_complete and api_key:
    st.markdown("---")
    
    # Chat header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 💬 Chat with Your Documents")
    with col2:
        if st.button("🗑️ Clear Chat", type="secondary", use_container_width=True):
            if session_id in st.session_state.store:
                st.session_state.store[session_id].clear()
            st.rerun()
    
    # -------------------------------------------------------------------
    # Chat History Helper
    # -------------------------------------------------------------------
    def get_session_history(session: str) -> BaseChatMessageHistory:
        """Return chat history for a given session."""
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]
    
    # -------------------------------------------------------------------
    # Initialize LLM
    # -------------------------------------------------------------------
    @st.cache_resource
    def get_llm(api_key: str, model_name: str):
        return ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.3,
            max_tokens=1024
        )
    
    llm = get_llm(api_key, llm_model)
    
    # -------------------------------------------------------------------
    # RAG Chain Components
    # -------------------------------------------------------------------
    # Contextualize question chain
    contextualize_q_system_prompt = """Given a chat history and the latest user question, 
    formulate a standalone question that can be understood without the chat history. 
    Return ONLY the standalone question, no explanations."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # Format documents
    def format_docs(docs):
        return "\n\n---\n\n".join([
            f"**Document {i+1}**\n\n{doc.page_content}\n\n*Source: Page {doc.metadata.get('page', 'N/A')}*"
            for i, doc in enumerate(docs)
        ])
    
    # QA Prompt
    qa_system_prompt = """You are an intelligent document analysis assistant. 
    Use the following retrieved context from documents to answer the question.
    
    **Guidelines:**
    1. Be concise but comprehensive
    2. Cite sources when relevant (mention Document 1, 2, etc.)
    3. If unsure, say so and suggest what information might help
    4. Maintain a professional yet friendly tone
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # -------------------------------------------------------------------
    # Main RAG Chain Function
    # -------------------------------------------------------------------
    def process_rag_query(input_text: str) -> Dict[str, Any]:
        """Process a query through the RAG pipeline."""
        # Get chat history
        chat_history = get_session_history(session_id).messages
        
        # Get standalone question
        standalone_question = contextualize_q_chain.invoke({
            "input": input_text,
            "chat_history": chat_history[-10:]  # Last 10 messages
        })
        
        # Retrieve documents
        docs = st.session_state.retriever.invoke(standalone_question)
        
        # Format context
        context = format_docs(docs)
        
        # Create QA chain
        qa_chain = qa_prompt | llm
        
        # Get response
        response = qa_chain.invoke({
            "input": input_text,
            "context": context,
            "chat_history": chat_history[-10:]
        })
        
        # Update chat history
        get_session_history(session_id).add_user_message(input_text)
        get_session_history(session_id).add_ai_message(response.content)
        
        return {
            "answer": response.content,
            "context": context,
            "standalone_question": standalone_question,
            "sources": docs
        }
    
    # -------------------------------------------------------------------
    # Display Chat History
    # -------------------------------------------------------------------
    chat_history = get_session_history(session_id).messages
    
    # Display chat messages
    for message in chat_history[-20:]:  # Show last 20 messages
        with st.chat_message(message.type):
            st.markdown(message.content)
    
    # -------------------------------------------------------------------
    # Chat Input
    # -------------------------------------------------------------------
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                try:
                    result = process_rag_query(prompt)
                    st.markdown(result["answer"])
                    
                    # Show expandable details
                    with st.expander("📊 View Details"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🔍 Standalone Question**")
                            st.info(result["standalone_question"])
                        
                        with col2:
                            st.markdown("**📚 Retrieved Sources**")
                            for i, doc in enumerate(result["sources"][:3]):
                                st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')})")
                                st.caption(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        
                        st.markdown("**📖 Full Context**")
                        st.text_area("Context", result["context"], height=200, disabled=True)
                        
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.info("Please try again or check your API key.")
    
    # -------------------------------------------------------------------
    # Sample Questions
    # -------------------------------------------------------------------
    if not chat_history:
        st.markdown("---")
        st.markdown("### 💡 Sample Questions to Get Started:")
        
        sample_questions = [
            "What are the main topics discussed in the documents?",
            "Can you summarize the key findings?",
            "What recommendations are provided?",
            "Explain the methodology used in detail.",
            "What are the limitations mentioned?"
        ]
        
        cols = st.columns(len(sample_questions))
        for idx, question in enumerate(sample_questions):
            with cols[idx]:
                if st.button(f"`{question[:30]}...`", use_container_width=True, key=f"sample_{idx}"):
                    st.session_state.sample_question = question
                    st.rerun()
        
        if 'sample_question' in st.session_state:
            st.chat_input(st.session_state.sample_question)
            del st.session_state.sample_question

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**🤖 IntelliDoc AI**")
    st.caption("Advanced Document Analysis System")
with footer_col2:
    st.markdown("**⚡ Powered by**")
    st.caption("LangChain • Groq • ChromaDB • Sentence Transformers")
with footer_col3:
    st.markdown("**📞 Support**")
    st.caption("Report issues or request features")

# -------------------------------------------------------------------
# Initial State Message
# -------------------------------------------------------------------
elif not uploaded_files:
    # Welcome message
    st.markdown("""
    <div class='card'>
        <h3>🎯 Welcome to IntelliDoc AI!</h3>
        <p>Upload your PDF documents and chat with them using our intelligent RAG system.</p>
        
        <h4>✨ Features:</h4>
        <ul>
            <li>📄 Multiple PDF support</li>
            <li>💬 Conversational chat with history</li>
            <li>🔍 Semantic search across documents</li>
            <li>⚡ Fast processing with Groq</li>
            <li>🔒 Secure & private sessions</li>
        </ul>
        
        <h4>🚀 How to use:</h4>
        <ol>
            <li>Enter your Groq API key in the sidebar</li>
            <li>Upload PDF files using the uploader above</li>
            <li>Click "Process Documents" to analyze them</li>
            <li>Start asking questions about your documents!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature showcase
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='stats-card'>
            <h4>📚 Document Analysis</h4>
            <p>Extract insights from multiple PDFs simultaneously</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stats-card'>
            <h4>💭 Smart Conversations</h4>
            <p>Context-aware responses with chat history</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stats-card'>
            <h4>⚡ Real-time Processing</h4>
            <p>Fast responses powered by Groq</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Update requirements.txt reminder
# -------------------------------------------------------------------
