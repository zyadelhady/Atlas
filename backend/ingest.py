from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

DATA_DIR = "data"
DB_DIR = "db"

def main():
    # Create the loader
    loader = DirectoryLoader(DATA_DIR, glob="*.txt")

    # Load the documents
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Create the embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the vector store
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=DB_DIR
    )
    db.persist()

if __name__ == "__main__":
    main()


# from langchain_community.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_community.vectorstores import Chroma
# from langchain.embeddings.base import Embeddings
# from dotenv import load_dotenv
# import os
# import shutil
# import google.generativeai as genai
# from typing import List
#
# load_dotenv()
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
#
# CHROMA_PATH = "chroma"
# DATA_PATH = "data"
#
# class GeminiEmbeddings(Embeddings):
#     def __init__(self, model="models/embedding-001"):
#         self.model = model
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [genai.embed_content(model=self.model, content=text, task_type="retrieval_document")["embedding"] for text in texts]
#
#     def embed_query(self, text: str) -> List[float]:
#         return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")["embedding"]
#
# def main():
#     generate_data_store()
#
# def generate_data_store():
#     documents = load_documents()
#     chunks = split_text(documents)
#     save_to_chroma(chunks)
#
# def load_documents():
#     loader = DirectoryLoader(DATA_PATH, glob="*.txt")
#     documents = loader.load()
#     return documents
#
# def split_text(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=300,
#         chunk_overlap=100,
#         length_function=len,
#         add_start_index=True,
#     )
#     chunks = text_splitter.split_documents(documents)
#     print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
#
#     document = chunks[10]
#     print(document.page_content)
#     print(document.metadata)
#
#     return chunks
#
# def save_to_chroma(chunks: list[Document]):
#     # Clear out the database first.
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)
#
#     # Create a new DB from the documents.
#     db = Chroma.from_documents(
#         chunks, GeminiEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
#
# if __name__ == "__main__":
#     main()
