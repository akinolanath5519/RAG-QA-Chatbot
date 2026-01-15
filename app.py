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
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Conversational RAG with PDFs")
st.title("📄 Conversational RAG With PDF Uploads")
st.write("Upload PDFs and chat with their content.")

# Check for sentence-transformers
if not SENTENCE_TRANSFORMERS_AVAILABLE:
    st.error("Missing required package: sentence-transformers")
    st.info("Add 'sentence-transformers' to your requirements.txt")
    st.stop()

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
# Custom Embeddings Class (to avoid langchain_huggingface)
# -------------------------------------------------------------------
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_query(self, text: str):
        """Embed a single query."""
        # Clean the text
        text = text.replace("\n", " ").strip()
        if not text:
            text = " "
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: list):
        """Embed multiple documents."""
        # Clean the texts
        cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
        # Ensure no empty strings
        cleaned_texts = [text if text else " " for text in cleaned_texts]
        return [self.model.encode(text).tolist() for text in cleaned_texts]

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
# Process PDFs
# -------------------------------------------------------------------
with st.spinner("Processing PDFs..."):
    documents = []
    
    # Create a temporary directory for PDFs
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Save uploaded file to temp location
            temp_path = Path(temp_dir) / uploaded_file.name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load PDF
            try:
                loader = PyPDFLoader(str(temp_path))
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
                st.success(f"✓ Loaded {len(loaded_docs)} pages from {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
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
    st.write("Creating embeddings...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Extract text for embedding
    texts = [doc.page_content for doc in splits]
    metadatas = [doc.metadata for doc in splits]
    
    # Create vectorstore
    st.write("Creating vector database...")
    vectorstore = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    st.success(f"✅ Processed {len(splits)} chunks from {len(documents)} pages")

# -------------------------------------------------------------------
# LLM
# -------------------------------------------------------------------
llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# -------------------------------------------------------------------
# History-aware retriever (rephrase question)
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

# Create standalone question chain
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

def format_docs(docs):
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------------------------
# Main RAG Chain
# -------------------------------------------------------------------
def retrieve_documents(standalone_question: str):
    """Retrieve documents based on the standalone question."""
    return retriever.invoke(standalone_question)

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

# -------------------------------------------------------------------
# Streamlit Chat UI
# -------------------------------------------------------------------
st.divider()
st.header("💬 Chat with Your Documents")

# Display existing chat history
chat_history = get_session_history(session_id).messages
for message in chat_history[-10:]:  # Show last 10 messages
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("What is your question?"):
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
                with st.expander("View retrieved context"):
                    st.write(rag_context["context"])
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please try again or re-upload your PDFs.")

# Clear chat button
if st.button("Clear Chat History"):
    get_session_history(session_id).clear()
    st.rerun()