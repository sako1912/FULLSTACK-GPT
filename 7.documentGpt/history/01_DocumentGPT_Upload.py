import time
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# upload filed을 cahches/files에 저장하고, 해당 파일을 embeddings하여 다시 caches 저장 및 vectorstore로 저장
def embed_file(file):
    file_content = file.read()
    #1. cache에 저장될 file_path 설정
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    #2. embedding cache을 위해 각 file을 파일명 기준으로 cache_dir 생성
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    #3. Split 설정
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    #4. Load the documents in cache
    loader = UnstructuredFileLoader(file_path)
    #5. Split the documents into chunks
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    #6. cache_dir에 embeddings 저장
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    #7. Store the embeddings in a vector store
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    #8. vectorstore을 retriever로 변경 후 Return the retriever
    retriever = vectorstore.as_retriever()
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)

#file을 upaload하고 winston에 대해 검색
if file:
    retriever = embed_file(file)
    s = retriever.invoke("winston")
    s