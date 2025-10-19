from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = "data"
TABLE_NAME = "docs"

# IMPORTANT: If you change the embedding model, you must delete the old table in the database.
# You can do this by connecting to the database and running `DROP TABLE docs;`

CONNECTION_STRING = os.environ.get("DATABASE_URL")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

def run_ingestion():
    print("Running ingestion with improved chunking...")

    # Load documents
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", show_progress=True)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    # IMPROVED: Smart recursive chunking with better parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Larger chunks for better context (was 500)
        chunk_overlap=200,  # More overlap for continuity (was 50)
        length_function=len,
        is_separator_regex=False,
        # Smart separators prioritize document structure
        separators=[
            "\n\n\n",  # Multiple blank lines (major sections)
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ".",       # Sentences
            ",",       # Clauses
            " ",       # Words
            "",        # Characters (fallback)
        ]
    )

    texts = text_splitter.split_documents(documents)

    # Add metadata enrichment
    for i, chunk in enumerate(texts):
        # Add chunk index for ordering
        chunk.metadata['chunk_index'] = i
        # Add source file name (extract from path)
        if 'source' in chunk.metadata:
            chunk.metadata['filename'] = os.path.basename(chunk.metadata['source'])
        # Add chunk size for debugging
        chunk.metadata['chunk_size'] = len(chunk.page_content)

    print(f"Split into {len(texts)} chunks")
    print(f"Average chunk size: {sum(len(t.page_content) for t in texts) / len(texts):.0f} characters")

    # Create embeddings
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_EMBEDDING_MODEL,
    )

    pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

    # Initialize the vectorstore table (768 is the dimension for nomic-embed-text)
    pg_engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_size=768,
    )

    # Create vector store and add documents
    vector_store = PGVectorStore.create_sync(
        table_name=TABLE_NAME,
        embedding_service=embeddings,
        engine=pg_engine,
    )

    # Add documents to the vector store
    vector_store.add_documents(texts)

    print("INGESTION DONE")
    print(f"✓ Stored {len(texts)} chunks in database")

if __name__ == "__main__":
    run_ingestion()


