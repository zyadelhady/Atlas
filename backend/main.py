from fastapi import FastAPI, Depends
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVectorStore, PGEngine
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import ChatHistory, create_db_and_tables, get_session
from ingest import run_ingestion

import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

CONNECTION_STRING = os.environ.get("DATABASE_URL")
print(f"CONNECTION_STRING in main.py: {CONNECTION_STRING}")
TABLE_NAME = "my_documents"

pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

# pg_engine.init_vectorstore_table(
#     table_name=TABLE_NAME,
#     vector_size=384,
# )

vector_store =  PGVectorStore.create_sync(
    table_name=TABLE_NAME,
    embedding_service=embeddings,
    engine=pg_engine,
)

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

# @app.on_event("startup")
# async def on_startup():
    # await create_db_and_tables()
    # run_ingestion()

origins = [
    "http://localhost:3001",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/ai")
async def ai(query: Query, db: AsyncSession = Depends(get_session)):
    results = await vector_store.asimilarity_search_with_relevance_scores(query.query, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    # Retrieve history from database
    formatted_history = ""
    chat_history_entries = (await db.execute(select(ChatHistory).filter(ChatHistory.session_id == query.sessionId).order_by(ChatHistory.timestamp))).scalars().all()
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
        
        chat_entry = ChatHistory(session_id=query.sessionId, prompt=query.query, response=full_response)
        db.add(chat_entry)
        await db.commit()
        await db.refresh(chat_entry)

    return StreamingResponse(generate())

@app.get("/history/{session_id}")
async def get_history(session_id: str, db: AsyncSession = Depends(get_session)):
    history = (await db.execute(select(ChatHistory).filter(ChatHistory.session_id == session_id))).scalars().all()
    return history
