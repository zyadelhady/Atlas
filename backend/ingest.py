from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = "data"
TABLE_NAME = "my_documents"

CONNECTION_STRING = os.environ.get("DATABASE_URL")
print(f"CONNECTION_STRING in ingest.py: {CONNECTION_STRING}")

def run_ingestion():
    print("Running ingestion...")
    loader = DirectoryLoader(DATA_DIR, glob="*.txt")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    vector_store = PGVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        table_name=TABLE_NAME,
        engine=pg_engine,
    )
    print("INGESTION DONE")


