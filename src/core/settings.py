from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='allow'
    )

    DATABASE_URL: str = 'postgresql://admin:admin@localhost:5432/vector_db'
    MAX_TOKENS_PER_MINUITE: int = 25000

    # LLM Provider: 'azure', 'openai', 'ollama', or 'lmstudio'
    LLM_PROVIDER: Literal['azure', 'openai', 'ollama', 'lmstudio'] = 'azure'

    # Unified LLM settings (used for all providers)
    LLM_BASE_URL: str = ''  # e.g. https://xxx.openai.azure.com, http://localhost:11434, http://localhost:1234/v1
    LLM_API_KEY: str = ''  # API key (use 'ollama' or 'lmstudio' for local providers)
    LLM_CHAT_MODEL: str = 'gpt-4o'  # Chat model name/deployment
    LLM_EMBEDDING_MODEL: str = 'text-embedding-3-small'  # Embedding model name/deployment
    LLM_EMBEDDING_DIMENSIONS: int = 1536  # Embedding vector dimensions (1536 for OpenAI, 768 for nomic-embed-text)

    # Azure-specific (only needed for Azure)
    LLM_API_VERSION: str = '2024-02-01'

    # Code executor settings
    PYTHON_EXECUTOR_URL: str = 'http://localhost:5000'
    NODEJS_EXECUTOR_URL: str = 'http://localhost:3000'


settings = Settings()

# print('=== Settings Debug ===')
# for key, value in settings.model_dump().items():
#     # Mask sensitive values
#     if 'KEY' in key or 'PASSWORD' in key:
#         display = value[:4] + '***' if value else '(not set)'
#     else:
#         display = value if value else '(not set)'
#     print(f'{key}: {display}')
