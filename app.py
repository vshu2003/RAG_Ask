import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Streamlit configuration (must be first)
st.set_page_config(
    page_title="PCS neo CN Analyzer",
    layout="wide",
    page_icon="📘"
)

# Custom CSS for enhanced UI
# st.markdown("""
# <style>
#     .stApp {
#         background-color: #f5f5f5;
#     }
#     .user-message {
#         background-color: #e3f2fd;
#         border-radius: 15px;
#         padding: 12px;
#         margin: 8px 0;
#         max-width: 80%;
#         float: right;
#     }
#     .bot-message {
#         background-color: #ffffff;
#         border-radius: 15px;
#         padding: 12px;
#         margin: 8px 0;
#         max-width: 80%;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .source-card {
#         border-left: 3px solid #4CAF50;
#         padding: 10px;
#         margin: 10px 0;
#         background-color: #f8f9fa;
#     }
#     .stSpinner > div {
#         color: #4CAF50;
#     }
#     .sidebar .sidebar-content {
#         background-color: #ffffff;
#         box-shadow: 2px 0 8px rgba(0,0,0,0.1);
#     }
# </style>
# """, unsafe_allow_html=True)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App header
st.title("📘 SIMATIC CN4100 Knowledge Analyzer")
st.caption("AI-powered document understanding with source citations")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", 
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload multiple PDFs for analysis"
    )
    process_btn = st.button("Process Documents", type="primary")
    
    st.markdown("---")
    st.markdown("**Requirements:**")
    st.markdown("""
    1. [Hugging Face Account](https://huggingface.co/)
    2. [API Token](https://huggingface.co/settings/tokens)
    """)
    
    hf_token = st.text_input("Hugging Face API Token", type="password")
    st.markdown("---")
    st.markdown("📖 **About**  \nAI document analysis system using Mistral-7B and RAG technology")

# Document processing function
@st.cache_data(show_spinner=False)
def process_documents(files):
    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    for file in files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        try:
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            split_docs = text_splitter.split_documents(pages)
            all_docs.extend(split_docs)
        finally:
            os.remove(temp_path)
    
    return all_docs

# Processing pipeline
if process_btn and uploaded_files:
    with st.status("📂 Processing Documents...", expanded=True) as status:
        st.write("🔍 Loading and splitting documents...")
        documents = process_documents(uploaded_files)
        
        st.write("🧠 Creating embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        st.write("🏗️ Building knowledge base...")
        st.session_state.vector_store = FAISS.from_documents(
            documents, embeddings
        )
        status.update(label="✅ Processing complete!", state="complete")

# Format source documents
def format_source(doc):
    page_num = doc.metadata.get('page', 'N/A')
    if isinstance(page_num, int):
        page_num += 1
    return f"📄 **Page {page_num}**: {doc.page_content[:150]}..."

# Chat interface
if st.session_state.vector_store:
    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        # Use simple single emoji avatars
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
            if role == "assistant" and "sources" in message:
                with st.expander("View Source Documents", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            {format_source(source)}
                        </div>
                        """, unsafe_allow_html=True)

    # Handle new prompt
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            if not hf_token:
                st.error("⚠️ Missing Hugging Face API Token!")
            else:
                try:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    context_docs = retriever.get_relevant_documents(prompt)
                    
                    # Build enhanced prompt
                    prompt_template = """[INST] <<SYS>>
                    You are an expert document analyst. Answer the question using only the provided context.
                    Format your answer with markdown, including bullet points when appropriate.
                    If the answer isn't in the context, clearly state "I don't know".
                    <</SYS>>

                    Context:
                    {context}

                    Question: {question} [/INST]""".format(
                        context="\n\n\n".join([d.page_content for d in context_docs]),
                        question=prompt
                    )

                    # Initialize LLM
                    llm = HuggingFaceHub(
                        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                        huggingfacehub_api_token=hf_token,
                        model_kwargs={
                            "temperature": 0.3,
                            "max_length": 1024,
                            "top_p": 0.95
                        }
                    )

                    # Generate response
                    with st.spinner("Analyzing documents..."):
                        response = llm.invoke(prompt_template)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to chat history with sources
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "sources": context_docs
                    })

                except Exception as e:
                    st.error(f"⚠️ Error generating response: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"⚠️ Error: {str(e)}"
                    })

else:
    st.info("📁 Please upload and process documents to begin analysis")
