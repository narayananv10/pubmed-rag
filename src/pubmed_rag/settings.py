from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "abstracts.jsonl"
CHROMA_PATH = DATA_DIR / "chroma"


class PubMedConfig(BaseModel):
    query: str
    max_results: int
    batch_size: int


class ModelsConfig(BaseModel):
    embedding: str
    rerank: str
    generator: str
    judge: str


class RetrievalConfig(BaseModel):
    retrieve_pool: int
    retrieve_k: int
    rerank_k: int


class Config(BaseModel):
    pubmed: PubMedConfig
    models: ModelsConfig
    retrieval: RetrievalConfig


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ncbi_api_key: str = ""
    ncbi_email: str = ""
    voyage_api_key: str = ""
    anthropic_api_key: str = ""


def load_config(path: Path = CONFIG_PATH) -> Config:
    return Config(**yaml.safe_load(path.read_text()))


def load_secrets() -> Secrets:
    return Secrets()
