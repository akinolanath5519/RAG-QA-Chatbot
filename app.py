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
from langchain_core.runnables import RunnablePassthrough

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
    /* Main background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Chat messages */
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
    
    /* Buttons */
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
    
    /* File uploader */
    .uploadedFile {
        background-color: white;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .uploadedFile:hover {
        border-color: #3b82f6;
        background-color: #f8fafc;
    }
    
    /* Cards and containers */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    /* Success/Error messages */
    .stAlert {
        border-radius: 8px;
        border: 1px solid;
    }
    
    /* Divider */
    .stDivider {
        border-color: #e2e8f0;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #3b82f6;
    }
    
    /* Chat input */
    .stChatInput > div > div {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        background-color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        color: #475569;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f8fafc;
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide"
)

# -------------------------------------------------------------------
# Main Title with Minimal Design
# -------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1e293b; font-size: 2.5rem; margin-bottom: 0.5rem;'>📄 PDF Chat Assistant</h1>
        <p style='color: #64748b; font-size: 1.1rem;'>Upload PDFs and chat with their content using AI</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar for Configuration
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # API Key
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Enter your Groq API key to enable the chatbot"
    )
    
    if not api_key:
        st.warning("Please enter your API key")
    
    # Session ID
    session_id = st.text_input(
        "Session ID",
        value="default_session",
        help="Unique identifier for your chat session"
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This tool allows you to:
    - Upload multiple PDF files
    - Ask questions about their content
    - Get AI-powered answers with context
    """)
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit & LangChain")

# Check for sentence-transformers
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.error("Missing required package: sentence-transformers")
    st.info("Add 'sentence-transformers' to your requirements.txt")
    st.stop()

# Store initialization
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
# Main Content Area
# -------------------------------------------------------------------
main_col1, main_col2 = st.columns([1, 1])

with main_col1:
    st.markdown("### 📤 Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        # Display uploaded files nicely
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Uploaded Files:**")
        for file in uploaded_files:
            st.markdown(f"📄 **{file.name}** ({file.size:,} bytes)")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process button
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your API key in the sidebar first")
            else:
                with st.spinner("Processing documents..."):
                    try:
                        documents = []
                        
                        # Create a temporary directory for PDFs
                        with tempfile.TemporaryDirectory() as temp_dir:
                            for uploaded_file in uploaded_files:
                                # Save uploaded file to temp location
                                temp_path = Path(temp_dir) / uploaded_file.name
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getvalue())
                                
                                # Load PDF
                                loader = PyPDFLoader(str(temp_path))
                                loaded_docs = loader.load()
                                documents.extend(loaded_docs)
                        
                        if not documents:
                            st.error("No documents were loaded. Please check your PDF files.")
                            st.stop()
                        
                        # Split documents
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=5000,
                            chunk_overlap=500,
                        )
                        splits = text_splitter.split_documents(documents)
                        
                        # Create embeddings
                        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                        
                        # Extract text for embedding
                        texts = [doc.page_content for doc in splits]
                        metadatas = [doc.metadata for doc in splits]
                        
                        # Create vectorstore
                        vectorstore = Chroma.from_texts(
                            texts=texts,
                            embedding=embeddings,
                            metadatas=metadatas,
                        )
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        
                        # Store in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = retriever
                        st.session_state.processing_complete = True
                        
                        st.success(f"✅ Processed {len(documents)} pages into {len(splits)} chunks")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    else:
        st.info("👆 Upload PDF files to get started")
        
        # Quick tips
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        1. Upload one or multiple PDF files
        2. Click 'Process Documents'
        3. Start asking questions in the chat
        4. View retrieved context if needed
        """)
        st.markdown("</div>", unsafe_allow_html=True)

with main_col2:
    if uploaded_files and api_key and st.session_state.get('processing_complete'):
        st.markdown("### 💬 Chat")
        
        # Initialize LLM
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
        
        # History-aware retriever
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
        
        contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
        
        def format_docs(docs):
            """Format documents for context."""
            return "\n\n".join(doc.page_content for doc in docs)
        
        def retrieve_documents(standalone_question: str):
            """Retrieve documents based on the standalone question."""
            return st.session_state.retriever.invoke(standalone_question)
        
        # QA prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Context:\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Create the full chain
        def rag_chain(input_dict):
            """Main RAG chain."""
            # Get chat history
            chat_history = get_session_history(session_id).messages
            
            # Get standalone question
            standalone_question = contextualize_q_chain.invoke({
                "input": input_dict["input"],
                "chat_history": chat_history
            })
            
            # Retrieve documents
            docs = retrieve_documents(standalone_question)
            
            # Format context
            context = format_docs(docs)
            
            return {
                "context": context,
                "input": input_dict["input"],
                "chat_history": chat_history,
                "standalone_question": standalone_question
            }
        
        # Create QA chain
        qa_chain = qa_prompt | llm
        
        # Wrap with message history
        conversational_rag_chain = RunnableWithMessageHistory(
            qa_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        
        # Display chat history
        chat_history = get_session_history(session_id).messages
        for message in chat_history[-10:]:
            with st.chat_message(message.type):
                st.markdown(message.content)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get RAG context
                        rag_context = rag_chain({"input": prompt})
                        
                        # Get answer
                        response = conversational_rag_chain.invoke(
                            {
                                "input": prompt,
                                "context": rag_context["context"],
                                "chat_history": rag_context["chat_history"]
                            },
                            config={"configurable": {"session_id": session_id}},
                        )
                        
                        # Display answer
                        if hasattr(response, "content"):
                            answer = response.content
                        else:
                            answer = str(response)
                        
                        st.markdown(answer)
                        
                        # Show context in expander
                        with st.expander("📖 View retrieved context"):
                            st.write(rag_context["context"])
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Please try again or re-upload your PDFs.")
        
        # Clear chat button at bottom
        if chat_history:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("Clear Chat", type="secondary", use_container_width=True):
                    get_session_history(session_id).clear()
                    st.rerun()

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption("📄 PDF Chat Assistant")
with footer_col2:
    st.caption("Powered by LangChain & Groq")
with footer_col3:
    st.caption("Upload • Chat • Learn")