from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env', env_file_encoding='utf-8', extra='allow'
    )

    DATABASE_URL: str = 'postgresql://admin:admin@localhost:5432/vector_db'
    MAX_TOKENS_PER_MINUITE: int = 25000

    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: str = ''
    AZURE_OPENAI_API_KEY: str = ''
    AZURE_OPENAI_API_VERSION: str = '2024-02-01'
    AZURE_EMBEDDING_DEPLOYMENT: str = 'text-embedding-3-small'
    AZURE_CHAT_DEPLOYMENT: str = 'gpt-4o'


settings = Settings()

# print('=== Settings Debug ===')
# for key, value in settings.model_dump().items():
#     # Mask sensitive values
#     if 'KEY' in key or 'PASSWORD' in key:
#         display = value[:4] + '***' if value else '(not set)'
#     else:
#         display = value if value else '(not set)'
#     print(f'{key}: {display}')
