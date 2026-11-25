import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()
openai_api_key= os.getenv('OPENAI_API_KEY')
class Settings(BaseSettings):
    database_url: str
    redis_url: str
    openai_api_key: str = openai_api_key
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()