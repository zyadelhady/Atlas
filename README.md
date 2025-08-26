# DocsAI - Documentation AI Chatbot

DocsAI is an AI-powered chatbot designed to answer questions based on your documentation. It leverages Google's Gemini LLM, LangChain for RAG (Retrieval Augmented Generation), ChromaDB for vector storage, FastAPI for the backend API, and Next.js for a responsive chat interface.

## Features

- **FastAPI Backend:** A Python backend built with FastAPI to serve AI responses.
- **Gemini LLM Integration:** Utilizes Google's Gemini model for generating intelligent responses.
- **LangChain RAG:** Implements Retrieval Augmented Generation using LangChain to fetch relevant information from your documents stored in ChromaDB.
- **ChromaDB Vector Store:** Efficiently stores and retrieves document embeddings.
- **PostgreSQL Chat History:** Stores all prompts and AI responses in a PostgreSQL database, organized by chat session.
- **Streaming Responses:** Provides a smooth user experience by streaming AI responses chunk by chunk to the frontend.
- **Next.js Frontend:** A modern chat interface built with React, TypeScript, and Tailwind CSS.
- **Interactive Chat UI:** Features include:
  - Real-time display of AI responses.
  - Syntax highlighting for code snippets in AI responses.
  - Pre-defined question suggestions for quick interaction.
  - Session-based history management, allowing users to continue conversations.
- **Docker Compose Orchestration:** Easily set up and run the entire application stack (backend, frontend, database) using Docker.

## Technologies Used

**Backend:**

- Python
- FastAPI
- LangChain
- langchain-google-genai
- ChromaDB
- SQLAlchemy
- psycopg2-binary
- python-dotenv

**Frontend:**

- Next.js (React)
- TypeScript
- Tailwind CSS
- react-syntax-highlighter
- uuid

**Database:**

- PostgreSQL

**Orchestration:**

- Docker
- Docker Compose

## Setup and Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- A Google API Key for Gemini (get one from [Google AI Studio](https://aistudio.google.com/)).

### 1. Clone the Repository

```bash
git clone https://github.com/zyadelhady/atlas
cd atlas
```

### 2. Configure Environment Variables

Create a `.env` file in the `backend/` directory with your Google API Key:

```
# backend/.env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

**Note on PostgreSQL Credentials:** The `docker-compose.yml` file already defines default PostgreSQL credentials (`user`, `password`, `mydatabase`). If you wish to change these, update the `db` service environment variables in `docker-compose.yml` and the `DATABASE_URL` in the `web` service accordingly.

### 3. Build and Run with Docker Compose

Navigate to the root of the project directory (where `docker-compose.yml` is located) and run:

```bash
docker-compose up --build -d
```

This command will:

- Build the Docker images for the backend and frontend.
- Start the PostgreSQL database service.
- Start the FastAPI backend service.
- Start the Next.js frontend service.

### 4. Initial Data Ingestion

Before asking questions, you need to ingest your documentation into ChromaDB. The project includes a script for this.

First, ensure the backend container is running:

```bash
docker-compose ps
```

Find the name of your backend service (e.g., `docsai-web-1`). Then, run the ingestion script inside the backend container:

```bash
docker-compose exec web python ingest.py
```

This will process the `.txt` files in `backend/data/` and store their embeddings in `backend/db/chroma.sqlite3`.

## Usage

Once all services are up and data is ingested:

- **Frontend Chat UI:** Open your browser and navigate to `http://localhost:3001`.

  - Type your questions in the input field.
  - Click on suggested questions.
  - Observe AI responses streaming in real-time with code highlighting.

- **Backend API Endpoints:**
  - **AI Chat Endpoint (POST):** `http://localhost:8000/ai`
    - **Method:** `POST`
    - **Headers:** `Content-Type: application/json`
    - **Body:** `{"query": "Your question", "sessionId": "your-unique-session-id"}`
    - **Response:** Streams the AI's answer.
  - **Chat History Endpoint (GET):** `http://localhost:8000/history/{session_id}`
    - **Method:** `GET`
    - **Path Parameter:** Replace `{session_id}` with the actual session ID.
    - **Response:** Returns a JSON array of chat history entries for that session.

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
