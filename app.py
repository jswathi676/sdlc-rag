import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="SDLC Knowledge Assistant", layout="wide")

KB_PATH = "sdlc_kt.txt"

@st.cache_resource
def get_vector_store():
    if not os.path.exists(KB_PATH):
        raise FileNotFoundError(f"{KB_PATH} not found. Please create it first.")

    with open(KB_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def answer_question(question: str):
    vectorstore = get_vector_store()
    docs = vectorstore.similarity_search(question, k=4)

    if not docs:
        return "No relevant SDLC information was found.", []

    return docs[0].page_content, docs

st.title("SDLC Knowledge Assistant")
st.write("Ask questions related to SDLC, requirement analysis, testing, deployment, and DevOps.")

question = st.text_input("Ask a question about SDLC:")

if question:
    with st.spinner("Searching the SDLC knowledge base..."):
        answer, docs = answer_question(question)

    st.markdown("### Best match")
    st.write(answer)

    st.markdown("### Relevant passages")
    for i, doc in enumerate(docs[:3], start=1):
        st.info(f"Match {i}: {doc.page_content[:800]}")