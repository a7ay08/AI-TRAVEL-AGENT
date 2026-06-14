"""Configuration settings for AI Travel Agent backend.

Uses pydantic Settings to load environment variables and provide defaults for data directory, API origins, search API, defaults, and model settings.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    DATA_DIR: str = Field(default=str(Path(__file__).parent.parent / "data"))
    API_ORIGINS: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    SEARCH_API_URL: str = "https://www.searchapi.io/api/v1/search"
    DEFAULT_ORIGIN: str = "AUH"
    FLIGHT_DAYS_AHEAD: int = 30
    FLIGHT_TIMEOUT: float = 15.0
    LLM_MODEL: str = "meta-llama-3.1-8b-instruct"
    LLM_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LLM_API_KEY: str = "lm-studio"
    SEARCHAPI_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"
