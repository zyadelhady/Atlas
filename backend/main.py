from fastapi import FastAPI, Depends
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
import asyncio
from langchain_postgres import PGVectorStore, PGEngine
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
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

# Configuration for retrieval
RELEVANCE_SCORE_THRESHOLD = 0.7  # Only use docs with similarity > 0.7
SEMANTIC_SEARCH_K = 10  # Number of docs to retrieve via semantic search
KEYWORD_SEARCH_K = 5    # Number of docs to retrieve via keyword search

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



async def preprocess_query(query: str) -> str:
    QUERY_IMPROVEMENT_PROMPT = f"""You are a query optimization assistant. Your job is to improve search queries for better retrieval from a documentation database.

Given the user's original query, create an improved version that:
1. Expands abbreviations (e.g., "df" → "DataFrame")
2. Adds key synonyms or related terms (e.g., "combine" → "merge join concatenate")
3. Clarifies ambiguous terms
4. Keeps it concise (2-3 sentences max)
5. Maintains the original intent

IMPORTANT: Only return the improved query text, nothing else. No explanations, no quotes, just the improved query.

Original query: {query}

Improved query:"""

    try:
        # Use a lightweight model call for fast query improvement
        response = llm.invoke(QUERY_IMPROVEMENT_PROMPT)
        improved_query = response.content.strip()

        # Fallback to original if improvement is too long or empty
        if not improved_query or len(improved_query) > len(query) * 3:
            return query

        print(f"[Query Preprocessing] Original: '{query}'")
        print(f"[Query Preprocessing] Improved: '{improved_query}'")

        return improved_query

    except Exception as e:
        print(f"Query preprocessing failed: {e}")
        return query  # Fallback to original query


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


async def hybrid_search(query: str, db: AsyncSession):
    """
    Performs hybrid search combining semantic similarity and keyword search.
    Returns deduplicated results with relevance score filtering.
    """
    # 0. Preprocess query using AI to improve search quality
    improved_query = await preprocess_query(query)

    # 1. Semantic search using vector similarity (with improved query)
    semantic_results = await vector_store.asimilarity_search_with_relevance_scores(
        improved_query, k=SEMANTIC_SEARCH_K
    )

    # Filter by relevance score threshold
    filtered_semantic = [
        (doc, score) for doc, score in semantic_results
        if score >= RELEVANCE_SCORE_THRESHOLD
    ]

    # 2. Keyword search using PostgreSQL full-text search (with improved query)
    # Note: LangChain creates table with name from TABLE_NAME variable
    # Default columns: content, embedding, langchain_id, langchain_metadata
    search_query = text(f"""
        SELECT content, langchain_metadata, embedding
        FROM {TABLE_NAME}
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', :query)
        LIMIT :limit
    """)

    try:
        keyword_results_raw = await db.execute(
            search_query,
            {"query": improved_query, "limit": KEYWORD_SEARCH_K}
        )
        keyword_results = keyword_results_raw.fetchall()
    except Exception as e:
        print(f"Keyword search failed: {e}")
        keyword_results = []

    # 3. Combine and deduplicate results
    # Create a dict to track unique documents by content
    unique_docs = {}

    # Add semantic results (higher priority due to relevance scores)
    for doc, score in filtered_semantic:
        content = doc.page_content
        if content not in unique_docs:
            unique_docs[content] = (doc, score)

    # Add keyword results (if not already included)
    for row in keyword_results:
        content = row[0]  # content column
        if content not in unique_docs:
            # Create a Document-like object for keyword results
            metadata = row[1] if row[1] else {}  # langchain_metadata column
            doc = Document(page_content=content, metadata=metadata)
            # Assign a default score for keyword matches (lower than semantic)
            unique_docs[content] = (doc, 0.5)

    # Return combined results sorted by score
    combined_results = sorted(unique_docs.values(), key=lambda x: x[1], reverse=True)

    return combined_results

@app.post("/ai")
async def ai(query: Query, db: AsyncSession = Depends(get_session)):
    # Use hybrid search instead of semantic-only search
    search_task = hybrid_search(query.query, db)
    history_task = get_formatted_history(query.sessionId, db)
    results, formatted_history = await asyncio.gather(search_task, history_task)

    # Extract context from filtered and deduplicated results
    if results:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    else:
        # If no results meet the threshold, inform the LLM
        context_text = "No relevant documentation found for this query."

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
