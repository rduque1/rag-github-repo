"""Retrieval agent - specialized for document search and summarization."""
import pydantic_core
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from src.agents.shared import create_llm_client, Deps
from src.core.settings import settings


# Focused system prompt for retrieval
retrieval_system_prompt = """
You are a document retrieval specialist. Your job is to:
1. Search the knowledge base for relevant information
2. Summarize documents when requested
3. List available documents

Use the available tools to find and present document information.
Always cite the source document name when providing information.
Be concise and direct.
"""

# Create model
_llm_client, _chat_model = create_llm_client()
_model = OpenAIChatModel(
    model_name=_chat_model,
    provider=OpenAIProvider(openai_client=_llm_client),
)

retrieval_agent = Agent(
    _model,
    output_type=str,
    system_prompt=retrieval_system_prompt,
    deps_type=Deps,
)


@retrieval_agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve relevant document sections based on a search query."""
    response = await context.deps.openai.embeddings.create(
        input=search_query,
        model=context.deps.embedding_model,
    )

    embedding = response.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()

    if context.deps.selected_docs:
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            WHERE folder = ANY($2)
            ORDER BY embedding <=> $1 LIMIT 5
            """,
            embedding_json,
            context.deps.selected_docs,
        )
    else:
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            ORDER BY embedding <=> $1 LIMIT 5
            """,
            embedding_json,
        )

    if not rows:
        return "No documents found in the knowledge base."

    results = []
    for row in rows:
        source = row["folder"]
        content = row["content"]
        results.append(f"[Source: {source}]\n{content}")

    return '\n\n---\n\n'.join(results)


@retrieval_agent.tool
async def list_documents(context: RunContext[Deps]) -> str:
    """List all available documents in the knowledge base."""
    if context.deps.selected_docs:
        docs = context.deps.selected_docs
    else:
        rows = await context.deps.pool.fetch(
            'SELECT DISTINCT folder FROM repo ORDER BY folder'
        )
        docs = [row['folder'] for row in rows]

    if not docs:
        return "No documents found in the knowledge base."

    doc_list = '\n'.join(f'- {doc}' for doc in docs)
    return f"Available documents ({len(docs)}):\n{doc_list}"


@retrieval_agent.tool
async def summarize_documents(
    context: RunContext[Deps],
    document_name: str | None = None,
) -> str:
    """Get content from documents for summarization."""
    if document_name:
        rows = await context.deps.pool.fetch(
            "SELECT folder, content FROM repo WHERE folder = $1 ORDER BY id",
            document_name,
        )
    elif context.deps.selected_docs:
        rows = await context.deps.pool.fetch(
            "SELECT folder, content FROM repo WHERE folder = ANY($1) ORDER BY folder, id",
            context.deps.selected_docs,
        )
    else:
        rows = await context.deps.pool.fetch(
            "SELECT folder, content FROM repo ORDER BY folder, id"
        )

    if not rows:
        return "No documents found to summarize."

    # Group and return content
    results = []
    for row in rows:
        results.append(f"[Source: {row['folder']}]\n{row['content']}")

    return '\n\n---\n\n'.join(results)


async def run_retrieval(query: str, deps: Deps) -> str:
    """Run the retrieval agent with a query."""
    result = await retrieval_agent.run(query, deps=deps)
    return result.output
