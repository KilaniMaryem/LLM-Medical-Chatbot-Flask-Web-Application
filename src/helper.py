from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_data(data):
    return DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader).load()
def get_data_chunks(data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    return text_splitter.split_documents(data)
def download_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
