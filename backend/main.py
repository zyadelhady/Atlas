from fastapi import FastAPI, Depends
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
import asyncio
from langchain_postgres import PGVectorStore, PGEngine
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database import ChatHistory, create_db_and_tables, get_session


import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

CONNECTION_STRING = os.environ.get("DATABASE_URL")
TABLE_NAME = "docs"

pg_engine = PGEngine.from_connection_string(url=CONNECTION_STRING)

# pg_engine.init_vectorstore_table(
#     table_name=TABLE_NAME,
#     vector_size=768,
# )

vector_store =  PGVectorStore.create_sync(
    table_name=TABLE_NAME,
    embedding_service=embeddings,
    engine=pg_engine,
)

PROMPT_TEMPLATE = """
You are an assistant that answers questions using the "Knowledge" section as the source of truth. 
You have three response modes depending on the type of user input:

1. Knowledge Mode:
   - For factual questions, explanations, or definitions: ONLY use what is in the Knowledge section.
   - If the Knowledge does not contain the answer, respond with: "I don’t know based on the provided information."

2. Code Mode:
   - You are allowed to generate new code examples even if they are not in the Knowledge.
   - Generated code must be correct, relevant, and consistent with the Knowledge context.
   - Always enclose code in triple backticks and specify the language (e.g., ```python).

3. Social Mode:
   - For greetings, thanks, compliments, or casual small talk, reply politely and naturally.
   - Keep responses short, friendly, and human-like.

General Rules:
- Always use clear, structured, and professional language when in Knowledge or Code Mode.
- For lists, use bullet points or tables for clarity.
- Use the Conversation History only for context, prioritizing the most recent messages.

The Question: {message}

Conversation History: {history}

Knowledge: {knowledge}
"""


class Query(BaseModel):
    query: str
    sessionId: str

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await create_db_and_tables()

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



async def get_formatted_history(session_id: str, db: AsyncSession) -> str:
    result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .order_by(ChatHistory.timestamp)
        .limit(10)
    )
    chat_history_entries = result.scalars().all()

    formatted_history = "\n".join(
        f"User: {entry.prompt}\nAI: {entry.response}"
        for entry in chat_history_entries
    )

    return formatted_history

@app.post("/ai")
async def ai(query: Query, db: AsyncSession = Depends(get_session)):
    search_task = vector_store.asimilarity_search_with_relevance_scores(query.query, k=10)
    history_task = get_formatted_history(query.sessionId, db)
    results, formatted_history = await asyncio.gather(search_task, history_task)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(knowledge=context_text, message=query.query, history=formatted_history)
    chain = llm
    async def generate():
        full_response = ""
        try:
            for chunk in chain.stream(prompt):
                full_response += chunk.content
                yield chunk.content

            chat_entry = ChatHistory(session_id=query.sessionId, prompt=query.query, response=full_response)
            db.add(chat_entry)
            await db.commit()
            await db.refresh(chat_entry)
        except SQLAlchemyError as e:
            await db.rollback()
            yield f"\n[Error saving chat history: {str(e)}]"

    return StreamingResponse(generate())

@app.get("/history/{session_id}")
async def get_history(session_id: str, db: AsyncSession = Depends(get_session)):
    history = (await db.execute(select(ChatHistory).filter(ChatHistory.session_id == session_id))).scalars().all()
    return history
