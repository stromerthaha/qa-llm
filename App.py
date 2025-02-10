import logging
import os
import time
import shutil
from dotenv import load_dotenv
import streamlit as st
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

# Sidebar - FAISS Database Management
st.sidebar.subheader("📂 Database Management")
existing_dbs = [d for d in os.listdir(FAISS_DB_DIR) if os.path.isdir(os.path.join(FAISS_DB_DIR, d))]
selected_db = st.sidebar.selectbox("Select a FAISS Database", ["Create New Database"] + existing_dbs)

# Delete Database Option
delete_db = st.sidebar.selectbox("Or, select a FAISS Database to delete", ["None"] + existing_dbs)
if delete_db != "None":
    if st.sidebar.button("Delete Selected Database"):
        db_path_to_delete = os.path.join(FAISS_DB_DIR, delete_db)
        try:
            shutil.rmtree(db_path_to_delete)
            st.sidebar.success(f"✅ Successfully deleted the database: {delete_db}")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error deleting the database: {e}")

# Business Problem Templates
prompt_template_case_review = """
You are a legal assistant reviewing all deposition documents submitted by the Defendant.
Your task is to carefully examine each document and determine which ones contain the most critical information relevant to the case.
Identify key facts, inconsistencies, or evidence that may influence the outcome.

Context: {context}
Question: {input}

Provide a clear summary highlighting the most important documents or excerpts.
"""
prompt_case_review = ChatPromptTemplate.from_template(prompt_template_case_review)

prompt_template_generate_questions = """
You are a legal assistant reviewing deposition documents to better understand the case details.
Based on the provided context, generate 3-5 specific, insightful questions that probe into key details and potential inconsistencies in the case.
These questions should help in clarifying uncertainties and identifying further lines of inquiry.

Context: {context}
Question: {input}

List the generated questions in a clear and organized manner.
"""
prompt_generate_questions = ChatPromptTemplate.from_template(prompt_template_generate_questions)

prompt_template_email_info = """
You are a legal assistant reviewing deposition documents and related emails discussing material changes.
Your task is to extract the names of each person mentioned in any email that discusses a material change.
Additionally, identify the material changes mentioned and specify the corresponding file number or page number where they appear.

Context: {context}
Question: {input}

Provide a detailed list of names along with the detected material changes and their references.
"""
prompt_email_info = ChatPromptTemplate.from_template(prompt_template_email_info)

prompt_template_general_qa = """
You are a legal expert answering questions based on legal documents.
Use the provided context to generate a well-reasoned, concise, and accurate response.

Context: {context}
Question: {input}

Provide an accurate and well-structured answer.
"""
prompt_general_qa = ChatPromptTemplate.from_template(prompt_template_general_qa)

# Business Problem Selection (Optional)
st.sidebar.subheader("📌 Select Business Problem (Optional)")
business_problem = st.sidebar.radio("Choose Business Problem (or leave empty for General QA)", [
    "Review Documents for Case Relevance", 
    "Generate Specific Case Questions", 
    "Extract Names and Material Changes from Emails"
], index=None)

# Assign Prompt Based on Selection (Default to General QA if None is chosen)
prompt_mapping = {
    "Review Documents for Case Relevance": prompt_case_review,
    "Generate Specific Case Questions": prompt_generate_questions,
    "Extract Names and Material Changes from Emails": prompt_email_info
}
prompt = prompt_mapping.get(business_problem, prompt_general_qa)

# Function to Extract Text from PDFs
def extract_text_from_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        try:
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.warning(f"⚠️ Error reading {uploaded_file.name}: {e}")
    return documents

# Function to Load an Existing FAISS Database
def load_faiss_db(db_path):
    try:
        if "embeddings" not in st.session_state:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the FAISS index
        st.session_state.vectors = FAISS.load_local(db_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
        st.success(f"✅ Successfully loaded the FAISS database from: {db_path}")
    except Exception as e:
        st.error(f"⚠️ Error loading FAISS database: {e}")

# Function to Add PDFs to an Existing FAISS Database
def append_to_existing_faiss_db(uploaded_files, db_path):
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Extract documents from PDFs
    docs = extract_text_from_pdfs(uploaded_files)
    if not docs:
        st.warning("⚠️ No documents were processed. Please check the uploaded files.")
        return

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # Load the existing FAISS index and append new documents
    try:
        st.session_state.vectors.add_documents(final_documents)
        st.session_state.vectors.save_local(db_path)
        st.success(f"✅ Successfully added new documents to the existing FAISS database: {db_path}")
    except Exception as e:
        st.error(f"⚠️ Error loading FAISS database: {e}")

# Function to Create a New FAISS Database
def create_new_faiss_db(uploaded_files):
    timestamp = str(int(time.time()))
    db_name = f"faiss_db_{timestamp}"
    db_path = os.path.join(FAISS_DB_DIR, db_name)
    os.makedirs(db_path, exist_ok=True)

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Extract and process documents
    docs = extract_text_from_pdfs(uploaded_files)
    if not docs:
        st.warning("⚠️ No documents were processed. Please check the uploaded files.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    # Create a new vector store and save it
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
    st.session_state.vectors.save_local(db_path)
    st.success(f"✅ Created a new FAISS database: {db_name}")

# Upload PDFs
uploaded_files = st.file_uploader("📂 Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
if st.button("Ingest Data into FAISS Database") and uploaded_files:
    if selected_db != "Create New Database":
        # Append to existing FAISS database
        db_path = os.path.join(FAISS_DB_DIR, selected_db)
        append_to_existing_faiss_db(uploaded_files, db_path)
    else:
        # Create a new FAISS database
        create_new_faiss_db(uploaded_files)

# Load the selected FAISS database (if any)
if selected_db != "Create New Database":
    db_path = os.path.join(FAISS_DB_DIR, selected_db)
    load_faiss_db(db_path)

# Query Section
st.subheader("🔍 Ask a question")
query = st.text_input("Enter your question here:")
if query:
    if "vectors" in st.session_state:
        retriever = st.session_state.vectors.as_retriever()
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': query})
        st.write("📝 Answer:", response['answer'].strip())
    else:
        st.warning("⚠️ Please ingest data first by uploading documents.")
