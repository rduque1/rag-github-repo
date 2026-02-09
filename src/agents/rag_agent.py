"""
RAG agent using Azure OpenAI with pgvector as database.
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
from src.core.database import database_connect
from src.core.settings import settings


@dataclass
class Deps:
    openai: AsyncAzureOpenAI
    pool: asyncpg.Pool


system_prompt = """
# ROLE
You are a helpful AI Assistant that answers questions based on the user's uploaded documents.

# RULES
1. ONLY use the retrieved document context to answer. Do not make up information.
2. If the answer is not found in the documents, say: "I couldn't find information about that in the uploaded documents."
3. Cite the source document name when providing information, e.g., [document.pdf].
4. Be concise and direct in your responses.
5. If the question is unclear, ask for clarification.

# PROCESS
1. Use the retrieve tool to search the document database for relevant content.
2. Analyze the retrieved context to find the answer.
3. Provide a clear answer with source citations.
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


async def stream_messages(question: str) -> AsyncGenerator[str, None]:
    """
    Stream messages for Streamlit interface.
    """
    async with database_connect() as pool:
        deps = Deps(openai=azure_client, pool=pool)
        async with agent.run_stream(question, deps=deps) as result:
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
