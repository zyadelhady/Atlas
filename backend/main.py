from fastapi import FastAPI, Depends
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import SessionLocal, ChatHistory, create_db_and_tables
from ingest import run_ingestion
from typing import Optional
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
If your answer contains code, please enclose it in triple backticks (```).

The question: {message}

Conversation history: {history}

The knowledge: {knowledge}

"""

class Query(BaseModel):
    query: str
    sessionId: str

app = FastAPI()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    run_ingestion()

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

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/ai")
def ai(query: Query, db: Session = Depends(get_db)):
    results = chroma_db.similarity_search_with_relevance_scores(query.query, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    # Retrieve history from database
    formatted_history = ""
    chat_history_entries = db.query(ChatHistory).filter(ChatHistory.session_id == query.sessionId).order_by(ChatHistory.timestamp).all()
    for entry in chat_history_entries:
        formatted_history += "User: " + entry.prompt + "\n"
        formatted_history += "AI: " + entry.response + "\n"

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(knowledge=context_text, message=query.query, history=formatted_history)

    chain = llm

    async def generate():
        full_response = ""
        for chunk in chain.stream(prompt):
            full_response += chunk.content
            yield chunk.content
        
        # Store the conversation in the database after the full response is generated
        chat_entry = ChatHistory(session_id=query.sessionId, prompt=query.query, response=full_response)
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)

    return StreamingResponse(generate())

@app.get("/history/{session_id}")
def get_history(session_id: str, db: Session = Depends(get_db)):
    history = db.query(ChatHistory).filter(ChatHistory.session_id == session_id).all()
    return history
