from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "StoryWeaver"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # Database (direct Postgres connection for SQLAlchemy / Alembic)
    DATABASE_URL: str

    # AI Services
    GEMINI_API_KEY: str                # Parent tip generation
    WHISPER_MODEL_SIZE: str = "base"   # faster-whisper model size (self-hosted, no API key)
    WHISPER_MODEL_PATH: str | None = "ml_models/storyweaver_whisper_v1"  # our LoRA fine-tuned model (CTranslate2 format); overrides WHISPER_MODEL_SIZE when set

    # Auth
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days


settings = Settings()
