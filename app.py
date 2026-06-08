import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

load_dotenv()

APP_TITLE = "RAG PDF Q&A - Gemini"
INDEX_DIR = "faiss_index"
UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="wide")


def get_api_key() -> str:
    try:
        secret_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        secret_key = ""
    return os.getenv("GOOGLE_API_KEY") or secret_key


def save_uploaded_files(uploaded_files):
    saved_paths = []
    for uploaded_file in uploaded_files:
        target_path = UPLOAD_DIR / uploaded_file.name
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(target_path))
    return saved_paths


def load_documents(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        documents.extend(docs)
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
    return vector_store


def load_vector_store(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    return FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_qa_chain(vector_store, api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt_template = """You are a helpful assistant for answering questions from documents.
Use the context below to answer the question as accurately as possible.
If the context does not contain enough information, say: "I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def reset_index():
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)


st.title("📄 RAG PDF Q&A with Gemini + FAISS")
st.caption("Upload PDFs, build a FAISS index, and ask questions grounded in your documents using Gemini.")

api_key = get_api_key()
if not api_key:
    st.error("GOOGLE_API_KEY not found. Add it to your .env file or Streamlit secrets before using the app.")
    st.stop()

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    build_clicked = col1.button("Build index", use_container_width=True)
    reset_clicked = col2.button("Reset index", use_container_width=True)

    if reset_clicked:
        reset_index()
        st.success("FAISS index removed.")

    st.markdown("---")
    st.write("Chunk size: 500")
    st.write("Chunk overlap: 50")
    st.write("Retriever top-k: 4")
    st.write("Embedding model: models/gemini-embedding-001")
    st.write("Chat model: gemini-2.5-flash-lite")

if build_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        with st.spinner("Saving files and building vector store..."):
            pdf_paths = save_uploaded_files(uploaded_files)
            documents = load_documents(pdf_paths)
            chunks = split_documents(documents)
            vector_store = build_vector_store(chunks, api_key)
            st.session_state["vector_store"] = vector_store
            st.session_state["qa_chain"] = build_qa_chain(vector_store, api_key)
            st.session_state["chunk_count"] = len(chunks)
        st.success(f"Index built successfully from {len(pdf_paths)} PDF(s) with {len(chunks)} chunks.")

if "qa_chain" not in st.session_state and os.path.exists(INDEX_DIR):
    with st.spinner("Loading existing FAISS index..."):
        vector_store = load_vector_store(api_key)
        st.session_state["vector_store"] = vector_store
        st.session_state["qa_chain"] = build_qa_chain(vector_store, api_key)

question = st.text_input("Ask a question about your PDFs")

if question:
    if "qa_chain" not in st.session_state:
        st.warning("Build or load an index before asking questions.")
    else:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            response = st.session_state["qa_chain"]({"query": question})

        st.subheader("Answer")
        st.write(response["result"])

        st.subheader("Sources")
        for i, doc in enumerate(response["source_documents"], start=1):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page_number") or doc.metadata.get("page", "N/A")
            with st.expander(f"Source {i}: {Path(source).name} | Page: {page}"):
                st.write(doc.page_content)

with st.expander("How it works"):
    st.markdown(
        """
        1. PDFs are loaded with `UnstructuredPDFLoader`.
        2. Text is chunked using `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`.
        3. Chunks are embedded with Gemini embeddings (`models/embedding-001`).
        4. Embeddings are stored in a FAISS vector index.
        5. A LangChain `RetrievalQA` chain retrieves relevant chunks and sends them to Gemini (`gemini-1.5-flash`).
        """
    )
