"""
Assistant agent using Azure OpenAI with pgvector as database.
Supports document search, summarization, listing, and web search.
"""
import asyncio
import sys
import asyncpg
import pydantic_core
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from openai import AsyncAzureOpenAI
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from src.core.database import database_connect
from src.core.settings import settings


@dataclass
class Deps:
    openai: AsyncAzureOpenAI
    pool: asyncpg.Pool
    selected_docs: list[str] | None = None  # Filter to specific documents


system_prompt = """
# ROLE
You are a helpful AI Assistant that works with the user's uploaded documents and can also search the web.

# INTENT UNDERSTANDING
First, understand what the user wants to do:
- **Search/Question**: User wants to find specific information or ask a question about the documents → use `retrieve` tool
- **Summarize**: User wants a summary of one or more documents → use `summarize_documents` tool
- **List**: User wants to know what documents are available → use `list_documents` tool
- **Compare**: User wants to compare information across documents → use `retrieve` multiple times with different queries
- **Web Search**: User wants information not in the documents or asks about external topics → use `duckduckgo_search` tool

# RULES
1. First try to answer from documents. If not found, offer to search the web.
2. When using document info, cite the source document name, e.g., [document.pdf].
3. When using web search, cite the source URL.
4. Be concise and direct in your responses.
5. If the query is unclear, ask for clarification.

# EXAMPLES
- "What does the document say about X?" → Search intent → use `retrieve`
- "Summarize the main points" → Summarize intent → use `summarize_documents`
- "What files do I have?" → List intent → use `list_documents`
- "Compare X in document A vs B" → Compare intent → use `retrieve` for each topic
- "What is the latest news about Y?" → Web search intent → use `duckduckgo_search`
- "Search the web for Z" → Web search intent → use `duckduckgo_search`
"""


# Configure Azure OpenAI provider for pydantic-ai
azure_client = AsyncAzureOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

model = OpenAIChatModel(
    model_name=settings.AZURE_CHAT_DEPLOYMENT,
    provider=OpenAIProvider(openai_client=azure_client),
)

# Create the agent with the Azure model
agent = Agent(
    model,
    output_type=str,
    system_prompt=system_prompt,
    deps_type=Deps,
    tools=[duckduckgo_search_tool()],
)


@agent.tool
async def retrieve(context: RunContext[Deps], search_query: str) -> str:
    """Retrieve relevant document sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query to find relevant documents.
    """

    response = await context.deps.openai.embeddings.create(
        input=search_query,
        model=settings.AZURE_EMBEDDING_DEPLOYMENT,
    )

    embedding = response.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()

    # Filter by selected documents if specified
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


@agent.tool
async def list_documents(context: RunContext[Deps]) -> str:
    """List all available documents in the knowledge base.

    Use this when the user wants to know what documents are available.

    Args:
        context: The call context.
    """
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


# Max tokens for summarization context
MAX_SUMMARY_TOKENS = 6000


async def _summarize_chunk(openai: AsyncAzureOpenAI, chunk: str, source: str) -> str:
    """Generate a summary for a single chunk of content."""
    response = await openai.chat.completions.create(
        model=settings.AZURE_CHAT_DEPLOYMENT,
        messages=[
            {
                'role': 'system',
                'content': (
                    'You are a summarization assistant. Provide a concise '
                    'summary of the following content. Keep key facts and '
                    'important details. Be brief but comprehensive.'
                ),
            },
            {
                'role': 'user',
                'content': f'Summarize this content from [{source}]:\n\n{chunk}',
            },
        ],
        max_tokens=500,
    )
    return f"[{source}] {response.choices[0].message.content}"


async def _hierarchical_summarize(
    openai: AsyncAzureOpenAI,
    chunks: list[tuple[str, str]],  # List of (source, content)
) -> str:
    """
    Perform hierarchical summarization (map-reduce style).

    If content is too large, summarize chunks individually, then combine
    and summarize the summaries recursively.
    """
    # Estimate tokens (rough: 4 chars = 1 token)
    total_chars = sum(len(content) for _, content in chunks)
    estimated_tokens = total_chars // 4

    if estimated_tokens <= MAX_SUMMARY_TOKENS:
        # Content fits, return as-is for the agent to summarize
        results = []
        for source, content in chunks:
            results.append(f"[Source: {source}]\n{content}")
        return '\n\n---\n\n'.join(results)

    # Too large - summarize each chunk individually
    summaries = []
    for source, content in chunks:
        # Split very large chunks further
        if len(content) > MAX_SUMMARY_TOKENS * 4:
            # Split into sub-chunks
            sub_chunk_size = MAX_SUMMARY_TOKENS * 3
            for i in range(0, len(content), sub_chunk_size):
                sub_chunk = content[i : i + sub_chunk_size]
                summary = await _summarize_chunk(openai, sub_chunk, source)
                summaries.append((source, summary))
        else:
            summary = await _summarize_chunk(openai, content, source)
            summaries.append((source, summary))

    # Check if combined summaries fit
    combined_chars = sum(len(s) for _, s in summaries)
    if combined_chars // 4 <= MAX_SUMMARY_TOKENS:
        # Summaries fit, return them
        results = []
        for source, summary in summaries:
            results.append(summary)
        return (
            "Intermediate summaries (synthesize these for final answer):\n\n"
            + '\n\n---\n\n'.join(results)
        )

    # Still too large - recursively summarize the summaries
    return await _hierarchical_summarize(openai, summaries)


@agent.tool
async def summarize_documents(
    context: RunContext[Deps],
    document_name: str | None = None,
) -> str:
    """Get content from documents for summarization.

    Use this when the user wants a summary of the documents.
    Handles large documents by creating hierarchical summaries.

    Args:
        context: The call context.
        document_name: Optional specific document to summarize. If None,
            summarizes from all available/selected documents.
    """
    if document_name:
        # Get ALL content from specific document
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            WHERE folder = $1
            ORDER BY id
            """,
            document_name,
        )
    elif context.deps.selected_docs:
        # Get ALL content from selected documents
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            WHERE folder = ANY($1)
            ORDER BY folder, id
            """,
            context.deps.selected_docs,
        )
    else:
        # Get ALL content from all documents
        rows = await context.deps.pool.fetch(
            """
            SELECT folder, content FROM repo
            ORDER BY folder, id
            """
        )

    if not rows:
        return "No documents found to summarize."

    # Group content by document
    chunks: list[tuple[str, str]] = [
        (row['folder'], row['content']) for row in rows
    ]

    # Perform hierarchical summarization if needed
    return await _hierarchical_summarize(context.deps.openai, chunks)


async def stream_messages(
    question: str,
    selected_docs: list[str] | None = None,
    message_history: list[dict[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream messages for Streamlit interface.

    Args:
        question: The user's question.
        selected_docs: Optional list of document names to limit search to.
        message_history: Optional list of previous messages for context.
            Each message is a dict with 'role' and 'content' keys.
    """
    async with database_connect() as pool:
        deps = Deps(openai=azure_client, pool=pool, selected_docs=selected_docs)

        # Build message history for the agent
        messages: list[tuple[str, str]] = []
        if message_history:
            for msg in message_history:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    messages.append(('user', content))
                elif role == 'assistant':
                    messages.append(('assistant', content))

        async with agent.run_stream(
            question, deps=deps, message_history=messages
        ) as result:
            async for message in result.stream_text(delta=True):
                yield message


async def run_agent(question: str) -> None:
    """
    Entry point to run the agent and perform RAG based question answering.
    """
    async with database_connect() as pool:
        deps = Deps(openai=azure_client, pool=pool)
        answer = await agent.run(question, deps=deps)

    print(answer.data)


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else None

    if action == 'search':
        if len(sys.argv) == 3:
            q = sys.argv[2]
        else:
            q = 'Qual problema é resolvido no workshop de kafka?'
        asyncio.run(run_agent(q))
    else:
        print('Exiting')
