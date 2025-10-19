# Atlas - Documentation AI Chatbot

Atlas is an AI-powered chatbot designed to answer questions based on your documentation. It leverages Ollama for local LLM inference, LangChain for RAG (Retrieval Augmented Generation), PostgreSQL with pgvector for vector storage, FastAPI for the backend API, and Next.js for a responsive chat interface.

## Features

- **FastAPI Backend:** A Python backend built with FastAPI to serve AI responses.
- **Ollama Integration:** Uses Ollama for local LLM inference with support for various open-source models (llama3.2, mistral, etc.).
- **LangChain RAG:** Implements Retrieval Augmented Generation using LangChain to fetch relevant information from your documents stored in a PostgreSQL database.
- **Hybrid Search:** Combines semantic (vector) and keyword (full-text) search for improved retrieval accuracy.
- **AI-Powered Query Preprocessing:** Automatically improves user queries before retrieval.
- **PostgreSQL with pgvector for vector storage:** Efficiently stores and retrieves document embeddings, which improves memory usage.
- **PostgreSQL Chat History:** Stores all prompts and AI responses in a PostgreSQL database, organized by chat session.
- **Streaming Responses:** Provides a smooth user experience by streaming AI responses chunk by chunk to the frontend.
- **Next.js Frontend:** A modern chat interface built with React, TypeScript, and Tailwind CSS.
- **Interactive Chat UI:** Features include:
  - Real-time display of AI responses.
  - Syntax highlighting for code snippets in AI responses.
  - Pre-defined question suggestions for quick interaction.
  - Session-based history management, allowing users to continue conversations.
- **Docker Compose Orchestration:** Easily set up and run the entire application stack (backend, frontend, Ollama, database) using Docker.

## Technologies Used

**Backend:**

- Python
- FastAPI
- LangChain
- langchain-community (for Ollama integration)
- SQLAlchemy
- python-dotenv

**Frontend:**

- Next.js (React)
- TypeScript
- Tailwind CSS
- react-syntax-highlighter
- uuid

**Database:**

- PostgreSQL with pgvector

**LLM:**

- Ollama (local LLM inference)

**Orchestration:**

- Docker
- Docker Compose

## Setup and Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/zyadelhady/atlas
cd atlas
```

### 2. Configure Environment Variables (Optional)

The project uses default environment variables that work with Docker Compose. If you want to customize settings, create a `.env` file in the root directory based on `.env.example`:

```bash
cp .env.example .env
```

Default configuration:
- **DATABASE_URL:** `postgresql+psycopg://user:password@db:5432/mydatabase`
- **OLLAMA_BASE_URL:** `http://ollama:11434`
- **OLLAMA_MODEL:** `llama3.2` (for chat responses)
- **OLLAMA_EMBEDDING_MODEL:** `nomic-embed-text` (for embeddings)

### 3. Build and Run with Docker Compose

Navigate to the root of the project directory (where `docker-compose.yml` is located) and run:

```bash
docker-compose up --build -d
```

This command will:

- Build the Docker images for the backend and frontend.
- Start the Ollama service for local LLM inference.
- Start the PostgreSQL database service with pgvector extension.
- Start the FastAPI backend service.
- Start the Next.js frontend service.

### 4. Pull Ollama Models

After the services start, you need to pull the required Ollama models:

```bash
# Pull the chat model (llama3.2 or any model you prefer)
docker exec -it atlas-ollama-1 ollama pull llama3.2

# Pull the embedding model
docker exec -it atlas-ollama-1 ollama pull nomic-embed-text
```

**Note:** The first pull will take some time depending on your internet connection. Model sizes:
- `llama3.2`: ~2GB
- `nomic-embed-text`: ~274MB

**Alternative models you can use:**
- Chat models: `mistral`, `llama2`, `codellama`, `phi3`, etc.
- Embedding models: `all-minilm`, `mxbai-embed-large`, etc.

If you change models, update the `.env` file accordingly.

### 5. Ingest Your Documents

Place your documentation files (in `.txt` format) in the `backend/data` directory.

Then, run the ingestion script inside the backend container:

```bash
docker exec -it atlas-web-1 python ingest.py
```

This will load your documents, create embeddings using Ollama, and store them in the PostgreSQL database.

**Important:** If you change the embedding model after initial ingestion, you must drop the existing table:

```bash
# Connect to the database
docker exec -it atlas-db-1 psql -U user -d mydatabase

# Drop the table
DROP TABLE docs;

# Exit
\q

# Re-run ingestion
docker exec -it atlas-web-1 python ingest.py
```


## Usage

Once all services are up and data is ingested:

- **Frontend Chat UI:** Open your browser and navigate to `http://localhost:3001`.

  - Type your questions in the input field.
  - Click on suggested questions.
  - Observe AI responses streaming in real-time with code highlighting.

- **Backend API Endpoints:**
  - **AI Chat Endpoint (POST):** `http://localhost:8001/ai`
    - **Method:** `POST`
    - **Headers:** `Content-Type: application/json`
    - **Body:** `{"query": "Your question", "sessionId": "your-unique-session-id"}`
    - **Response:** Streams the AI's answer.
  - **Chat History Endpoint (GET):** `http://localhost:8001/history/{session_id}`
    - **Method:** `GET`
    - **Path Parameter:** Replace `{session_id}` with the actual session ID.
    - **Response:** Returns a JSON array of chat history entries for that session.
  - **API Documentation (Swagger UI):** `http://localhost:8001/docs`

## Development Notes

### Why I Choose This Project

I Liked this idea because i think it's something i could use daily with my work as i would create agent for something i am learning and it will help me understand it

### Time Spent and Future Improvements

This project was developed over approximately one day of focused work, with breaks for lunch. As this is my first experience creating an AI agent, writing Python and LLM code, or utilizing RAG, the learning curve was significant.

Given more time, I would prioritize the following improvements:

- **Advanced Chunking Strategies:** Explore and implement more sophisticated document chunking techniques (e.g., hierarchical chunking) to improve context preservation and retrieval accuracy.
- **Support for Diverse Document Formats:** Extend ingestion capabilities to handle a wider range of document types (e.g., PDFs, Markdown, HTML, code files) beyond plain text.
- **Enhanced UI/UX:** Further refine the chat interface for a more polished and feature-rich user experience, potentially including markdown rendering for AI responses.
- **Evaluation Metrics:** Implement evaluation metrics to quantitatively assess the performance of the RAG system and LLM responses.
- **User Authentication and Multi-tenancy:** Add user management to support multiple users and separate their chat histories and documentation.
- **Deployment Automation:** Streamline the deployment process for easier hosting and scaling.

## Development Notes

- **Python Version:** The backend Dockerfile uses Python 3.9. Ensure your type hints are compatible (e.g., use `Optional[str]` instead of `str | None`).
- **SQLAlchemy Migrations:** For production environments with evolving database schemas, consider integrating a dedicated migration tool like [Alembic](https://alembic.sqlalchemy.org/en/latest/) instead of relying solely on `Base.metadata.create_all()`.
