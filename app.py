import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Streamlit UI setup
st.set_page_config(page_title="Open-Source RAG", layout="wide")
st.title("📚 Open-Source RAG System")
st.markdown("Powered by Mistral-7B, FAISS, and Sentence Transformers")

# Sidebar for document upload
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"],
        accept_multiple_files=True
    )
    process_btn = st.button("Process Documents")
    
    st.markdown("---")
    st.markdown("**Note:** Requires Ollama running locally with Mistral model")
    st.markdown("Run this first in another terminal:")
    st.code("ollama serve")

# Document processing function
@st.cache_data
def process_documents(files):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for file in files:
        with open(f"temp_{file.name}", "wb") as f:
            f.write(file.getbuffer())
        
        loader = PyPDFLoader(f"temp_{file.name}")
        pages = loader.load()
        split_docs = text_splitter.split_documents(pages)
        all_docs.extend(split_docs)
        os.remove(f"temp_{file.name}")
    
    return all_docs

# Main chat interface
if process_btn and uploaded_files:
    with st.status("Processing documents...", expanded=True) as status:
        st.write("Loading and splitting documents...")
        documents = process_documents(uploaded_files)
        
        st.write("Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        st.write("Building vector store...")
        st.session_state.vector_store = FAISS.from_documents(
            documents, embeddings
        )
        
        status.update(label="Processing complete!", state="complete")

# Chat input and response generation
if st.session_state.vector_store:
    prompt = st.chat_input("Ask about your documents...")
    
    if prompt:
        # Build RAG chain
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        
        prompt_template = """Answer the question using only this context:
        {context}
        
        Question: {question}
        
        If the answer isn't in the context, say "I don't know". 
        Provide a detailed answer in markdown format."""
        
        llm = Ollama(model="mistral", temperature=0.3)
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | PromptTemplate.from_template(prompt_template)
            | llm
        )
        
        # Display messages
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
else:
    st.info("Please upload and process documents first")