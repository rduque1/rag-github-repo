"""
Document processor that parses files with docling and creates embeddings.
"""

import asyncio
from pathlib import Path

import asyncpg
import pydantic_core
from openai import AsyncAzureOpenAI

from src.core.database import database_connect
from src.core.settings import settings
from src.preprocessing.chunk_splitter import (
    aggregate_files_by_token,
    num_tokens_from_string,
)
from src.preprocessing.document_parser import parse_document


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
    """
    Create embeddings for chunks and store them in the database.

    This is the single source of truth for embedding logic.

    :param source_name: Name of the source document.
    :param chunks: List of text chunks to embed.
    """
    openai = AsyncAzureOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )

    async with database_connect() as pool:
        sem = asyncio.Semaphore(10)

        async with asyncio.TaskGroup() as tg:
            for chunk in chunks:
                tg.create_task(
                    _insert_embedding(sem, openai, pool, source_name, chunk)
                )


async def _insert_embedding(
    sem: asyncio.Semaphore,
    openai: AsyncAzureOpenAI,
    pool: asyncpg.Pool,
    source_name: str,
    content: str,
) -> None:
    """
    Create and insert a single embedding into the database.

    :param sem: Semaphore for rate limiting.
    :param openai: Azure OpenAI client.
    :param pool: Database connection pool.
    :param source_name: Name of the source document.
    :param content: Text content to embed.
    """
    async with sem:
        response = await openai.embeddings.create(
            input=content,
            model=settings.AZURE_EMBEDDING_DEPLOYMENT,
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
