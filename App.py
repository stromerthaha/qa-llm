import logging
import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_mistralai import ChatMistralAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import shutil


# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Streamlit UI Config
st.set_page_config(page_title="Document Question Answering System", layout="wide")
st.title("📄 Document Question Answering System")
st.caption("Upload your PDF documents, ingest them into a vector store, and ask questions based on the content.")

# Directories
UPLOAD_FOLDER = "uploaded_docs"
FAISS_DB_DIR = "faiss_indexes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_DB_DIR, exist_ok=True)

# Sidebar - LLM Model Selection
st.sidebar.header("⚡ Model & Database Settings")
model_option = st.sidebar.selectbox("Choose LLM Model", [
    "Groq (Llama3-8B)", "Google GenAI (Gemini)", "Cohere", "Mistral AI"
])

# Function to initialize model based on selection
@st.cache_resource
def initialize_model(model_option):
    if model_option == "Groq (Llama3-8B)":
        return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")
    elif model_option == "Google GenAI (Gemini)":
        return ChatGoogleGenerativeAI(model="gemini-pro")
    elif model_option == "Cohere":
        return ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"), model="command-r-plus")
    elif model_option == "Mistral AI":
        return ChatMistralAI(model="mistral-tiny", mistral_api_key=os.getenv("MISTRAL_API_KEY"))
    else:
        raise ValueError(f"Model {model_option} not supported.")

llm = initialize_model(model_option)

# Sidebar - FAISS Database Management (with delete option)
st.sidebar.subheader("📂 Database Management")
existing_dbs = [d for d in os.listdir(FAISS_DB_DIR) if os.path.isdir(os.path.join(FAISS_DB_DIR, d))]
selected_db = st.sidebar.selectbox("Select a FAISS Database", ["Create New Database"] + existing_dbs)

# Add the "Delete Database" functionality
delete_db = st.sidebar.selectbox("Or, select a FAISS Database to delete", ["None"] + existing_dbs)
if delete_db != "None":
    if st.sidebar.button("Delete Selected Database"):
        db_path_to_delete = os.path.join(FAISS_DB_DIR, delete_db)
        try:
            shutil.rmtree(db_path_to_delete)  # Remove directory and its contents
            st.sidebar.success(f"✅ Successfully deleted the database: {delete_db}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error deleting the database: {e}")

# Improved Prompt Template
prompt_template = """
You are a document assistant. Answer the user's question based on the provided context.

Context: {context}
Question: {input}

Answer the question clearly and concisely.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)
    return documents

# Function to create a new vector store
def vector_embedding(uploaded_files):
    timestamp = str(int(time.time()))
    db_name = f"faiss_db_{timestamp}"
    db_path = os.path.join(FAISS_DB_DIR, db_name)
    os.makedirs(db_path, exist_ok=True)
    
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    docs = extract_text_from_pdfs(uploaded_files)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    st.session_state.vectors.save_local(db_path)
    
    st.success(f"✅ Data has been ingested into the new vector store: {db_name}")

# Load selected FAISS database
if selected_db != "Create New Database":
    db_path = os.path.join(FAISS_DB_DIR, selected_db)
    try:
        if "embeddings" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.load_local(db_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
        st.success(f"✅ Loaded FAISS database: {selected_db}")
    except Exception as e:
        st.error(f"⚠️ Error loading FAISS database: {e}")

# Upload PDFs
uploaded_files = st.file_uploader("📂 Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
if st.button("Ingest Data into Vector Store") and uploaded_files:
    vector_embedding(uploaded_files)

# Query Section with enhanced UI and logging
st.subheader("🔍 Ask a question")
query = st.text_input("Enter your question here:")
if query:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner('Retrieving and generating answer...'):
            start = time.time()
            response = retrieval_chain.invoke({'input': query})
            end = time.time()

            response_text = response['answer'].strip()

            # Log the query and response for transparency
            logging.info(f"Query: {query} | Answer: {response_text} | Time: {round(end - start, 2)}s")

            # Handle bad responses and display to user
            if response_text.startswith("Human:") or "I couldn't find relevant information" in response_text:
                response_text = "I couldn't find relevant information in the documents."

            st.write(f"⏳ Response time: {round(end - start, 2)} seconds")
            st.write("📝 Answer:", response_text)
    else:
        st.warning("⚠️ Please ingest data first by uploading documents.")

# Enhancements to improve UX:
# 1. Instructions for users
st.sidebar.subheader("📝 Instructions")
st.sidebar.write(
    "1. Select a model for question answering (Groq, Google GenAI, Cohere, Mistral AI)."
    "\n2. Upload your PDF documents using the uploader."
    "\n3. Press the 'Ingest Data into Vector Store' button."
    "\n4. Ask a question related to the uploaded documents, and the system will provide an answer."
)

# 2. Add loading spinner to show when the model is processing
with st.spinner('Processing your question, please wait...'):
    pass  # The processing occurs in the query section above

# 3. Provide a loading indicator for file uploads
if uploaded_files:
    st.info(f"📑 {len(uploaded_files)} file(s) uploaded. Ready for ingestion.")
else:
    st.info("📂 Please upload PDF documents to start.")

# Adding Custom CSS for better frontend styling
st.markdown(""" 
    <style>
        /* Custom Styles */
        .css-1y4d2l7 { background-color: #f0f4f8; }
        .css-ffhzg2 { font-size: 1.2rem; color: #333; }
        .css-1v3oer { background-color: #FAF7F0; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
        .sidebar .sidebar-content { padding: 20px; }
        .css-vl7r5d { background-color: #f8f9fa; border-radius: 5px; }
        .stButton button { background-color: #5c6bc0; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }
        .stButton button:hover { background-color: #3f51b5; }
        .stTextInput input { border-radius: 8px; padding: 12px; }
        .stInfo { background-color: #e3f2fd; color: #1e88e5; }
        .stWarning { background-color: #fff3e0; color: #f57c00; }
    </style>
""", unsafe_allow_html=True)
