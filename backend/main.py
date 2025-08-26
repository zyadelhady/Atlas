from fastapi import FastAPI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

load_dotenv()

CHROMA_DIR = "db"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

PROMPT_TEMPLATE = """
You are an assistent which answers questions based on knowledge which is provided to you.
While answering, you don't use your internal knowledge, 
but solely the information in the "The knowledge" section.
You don't mention anything to the user about the povided knowledge.

The question: {message}

Conversation history: {history}

The knowledge: {knowledge}

"""

class Query(BaseModel):
    query: str

app = FastAPI()

origins = [
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/user")
def read_root():
    return {"message": "Hello World"}

@app.post("/ai")
def ai(query: Query):
    results = chroma_db.similarity_search_with_relevance_scores(query.query, k=10)

    if len(results) == 0 or results[0][1] < 0.1:
        return {"response": "Unable to find matching results."}

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(knowledge=context_text, message=query.query,history="")

    chain = llm | StrOutputParser()
    response = chain.invoke(prompt)

    return {"response": response}
