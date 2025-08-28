from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = "data"
TABLE_NAME = "docs"

# IMPORTANT: If you change the embedding model, you must delete the old table in the database.
# You can do this by connecting to the database and running `DROP TABLE my_documents;`

CONNECTION_STRING = os.environ.get("DATABASE_URL")

def run_ingestion():
    print("Running ingestion...")
    loader = DirectoryLoader(DATA_DIR, glob="*.txt")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    vector_store = PGVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        table_name=TABLE_NAME,
        engine=pg_engine,
    )
    print("INGESTION DONE")

if __name__ == "__main__":
    run_ingestion()


