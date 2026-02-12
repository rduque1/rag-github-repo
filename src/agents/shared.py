"""Shared utilities and types for agents."""
from dataclasses import dataclass, field

import asyncpg
from openai import AsyncAzureOpenAI, AsyncOpenAI

from src.core.settings import settings


def create_llm_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
    """Create the appropriate LLM client based on settings."""
    provider = settings.LLM_PROVIDER

    if provider == 'azure':
        client = AsyncAzureOpenAI(
            azure_endpoint=settings.LLM_BASE_URL,
            api_key=settings.LLM_API_KEY,
            api_version=settings.LLM_API_VERSION,
        )
        return client, settings.LLM_CHAT_MODEL

    # OpenAI, Ollama, and LM Studio all use OpenAI-compatible API
    base_url = settings.LLM_BASE_URL
    if provider == 'ollama' and not base_url.endswith('/v1'):
        base_url = f"{base_url}/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=settings.LLM_API_KEY or 'not-needed',
    )
    return client, settings.LLM_CHAT_MODEL


def create_embedding_client() -> tuple[AsyncOpenAI | AsyncAzureOpenAI, str]:
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


@dataclass
class Deps:
    """Shared dependencies for all agents."""
    openai: AsyncOpenAI | AsyncAzureOpenAI
    pool: asyncpg.Pool
    embedding_model: str
    selected_docs: list[str] | None = None
    last_fetched_url: str | None = None
    memory: str | None = None
    generated_images: list[str] = field(default_factory=list)
