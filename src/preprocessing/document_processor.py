"""Document processor that parses files with docling and creates embeddings."""

import asyncio
from pathlib import Path

import asyncpg
import pydantic_core
from openai import AsyncAzureOpenAI, AsyncOpenAI

from src.core.database import database_connect
from src.core.settings import settings
from src.preprocessing.chunk_splitter import (
    aggregate_files_by_token,
    num_tokens_from_string,
)
from src.preprocessing.document_parser import parse_document


def _create_embedding_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    """Create the appropriate embedding client based on settings."""
    provider = settings.LLM_PROVIDER

    if provider == 'azure':
        client = AsyncAzureOpenAI(
            azure_endpoint=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            api_version=settings.LLM_API_VERSION,
        )
        return client, settings.LLM_EMBEDDING_MODEL

    # OpenAI, Ollama, and LM Studio all use OpenAI-compatible API
    base_url = settings.LLM_BASE_URL
    if provider == 'ollama' and not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=settings.LLM_API_KEY or 'not-needed',
    )
    return client, settings.LLM_EMBEDDING_MODEL


async def process_and_embed_document(
    file_path: str | Path,
    max_tokens: int = 6000,
) -> None:
    """
    Parse a document with docling and create embeddings in the database.

    :param file_path: Path to the document to process.
    :param max_tokens: Maximum tokens per chunk for embedding.
    """
    path = Path(file_path)
    content = parse_document(file_path)

    # Split content into chunks if needed
    chunks = _split_content(content, max_tokens)

    # Create embeddings and store in database
    await _embed_and_store(path.name, chunks)


def _split_content(content: str, max_tokens: int) -> list[str]:
    """
    Split content into chunks that fit within the token limit.

    :param content: Text content to split.
    :param max_tokens: Maximum tokens per chunk.
    :return: List of content chunks.
    """
    total_tokens = num_tokens_from_string(content)

    if total_tokens <= max_tokens:
        return [content]

    # Split by paragraphs and aggregate
    paragraphs = content.split('\n\n')
    chunks: list[str] = []
    current_chunk = ''

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        combined = f'{current_chunk}\n\n{paragraph}'.strip()

        if num_tokens_from_string(combined) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
        else:
            current_chunk = combined

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


async def _embed_and_store(source_name: str, chunks: list[str]) -> None:
    """
    Create embeddings for chunks and store them in the database.

    :param source_name: Name of the source document.
    :param chunks: List of text chunks to embed.
    """
    await embed_chunks(source_name, chunks)


async def embed_chunks(source_name: str, chunks: list[str]) -> None:
    """Create embeddings for chunks and store them in the database.

    This is the single source of truth for embedding logic.

    :param source_name: Name of the source document.
    :param chunks: List of text chunks to embed.
    """
    openai, embedding_model = _create_embedding_client()

    async with database_connect() as pool:
        sem = asyncio.Semaphore(10)

        try:
            async with asyncio.TaskGroup() as tg:
                for chunk in chunks:
                    tg.create_task(
                        _insert_embedding(sem, openai, pool, source_name, chunk, embedding_model)
                    )
        except ExceptionGroup as eg:
            # Log all sub-exceptions for debugging
            for exc in eg.exceptions:
                print(f"Embedding error: {exc}")
            raise eg.exceptions[0] from eg  # Re-raise first exception with context


async def _insert_embedding(
    sem: asyncio.Semaphore,
    openai: AsyncOpenAI | AsyncAzureOpenAI,
    pool: asyncpg.Pool,
    source_name: str,
    content: str,
    embedding_model: str,
) -> None:
    """Create and insert a single embedding into the database.

    :param sem: Semaphore for rate limiting.
    :param openai: OpenAI-compatible client.
    :param pool: Database connection pool.
    :param source_name: Name of the source document.
    :param content: Text content to embed.
    :param embedding_model: Name of the embedding model to use.
    """
    async with sem:
        response = await openai.embeddings.create(
            input=content,
            model=embedding_model,
        )

        embedding = response.data[0].embedding
        embedding_json = pydantic_core.to_json(embedding).decode()

        await pool.execute(
            """
            INSERT INTO repo (folder, content, embedding)
            VALUES ($1, $2, $3)
            """,
            source_name,
            content,
            embedding_json,
        )
