"""Central configuration — reads from environment / .env file."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All runtime settings resolved from environment variables."""

    # --- Embeddings ---
    embedding_provider: str = Field("local", env="EMBEDDING_PROVIDER")
    local_embedding_model: str = Field(
        "all-MiniLM-L6-v2", env="LOCAL_EMBEDDING_MODEL"
    )

    # --- Gemini (generation + optional embeddings) ---
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_api_version: str = Field("v1", env="GEMINI_API_VERSION")
    gemini_model: str = Field("gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        "models/gemini-embedding-001", env="GEMINI_EMBEDDING_MODEL"
    )

    # --- Vector store ---
    chroma_persist_dir: Path = Field(Path("./data/chroma_db"), env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("nextsearch", env="CHROMA_COLLECTION_NAME")

    # --- Chunking ---
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # --- Retrieval ---
    top_k: int = Field(5, env="TOP_K")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton — import this everywhere
settings = Settings()
