# Chat RAG — Multi-Agent RAG Chatbot

A multi-agent RAG chatbot that lets you upload documents, search the web, and execute code — all through a conversational interface.

## Stack

- **Agent Framework**: [Pydantic AI](https://ai.pydantic.dev/)
- **LLM Providers**: Azure OpenAI, OpenAI, Ollama, LM Studio
- **Vector DB**: PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)
- **UI**: [Streamlit](https://streamlit.io/)
- **Document Parsing**: [Docling](https://github.com/DS4SD/docling)
- **Code Execution**: Sandboxed Python & Node.js executors

## Agents

| Agent | Role |
|---|---|
| **Orchestrator** | Routes requests to specialized agents, plans tasks, enables parallel execution |
| **Retrieval** | Semantic search over the knowledge base (cosine similarity) |
| **Web** | Fetches webpages (Playwright/httpx), DuckDuckGo search, saves pages to KB |
| **Code** | Executes Python/JS in sandboxed containers, captures output & plots |
| **Reformulation** | Clarifies user questions, extracts URLs, fixes typos |

## Supported Document Formats

txt, pdf, md, json, csv, docx, pptx, xlsx, html, tsv

## How to Run

Requires [Docker Compose](https://docs.docker.com/compose/install/).

1. Clone and enter the repo:

```shell
git clone https://github.com/lealre/rag-github-repo.git
cd rag-github-repo
```

2. Create a `.env` file from the example and configure your LLM provider:

```shell
mv .env-example .env
```

Key environment variables:

| Variable | Default |
|---|---|
| `LLM_PROVIDER` | `openai` (azure, ollama, lmstudio) |
| `LLM_API_KEY` | — |
| `LLM_CHAT_MODEL` | `gpt-4o` |
| `LLM_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `DATABASE_URL` | `postgres://admin:admin@localhost:5432/vector_db` |

3. Start everything:

```shell
docker compose up
```

The app will be available at `http://localhost:8501/`.

Services started: Streamlit UI, PostgreSQL (pgvector), Python executor, Node.js executor.
