from langchain_community.vectorstores import DeepLake
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import getpass

api_key = os.environ.get("OPENAI_API_KEY")
embeddings_model = OpenAIEmbeddings()
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")


def create_vector(file_path):
    loader=PyPDFLoader(file_path=file_path)
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=20)
    docs = text_splitter.split_documents(docs)
    db=DeepLake.from_documents(docs,dataset_path="hub://rian/medicaldoc",embedding=embeddings_model,overwrite=False)
    return db

file_path="/Users/rianrachmanto/miniforge3/project/docurag/documents/asthma-management-(nov-2020).pdf"
db = create_vector(file_path)
