from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "qfis"
    model_path: str = "./models/v1"
    base_model_name: str = "microsoft/phi-2"
    faiss_index_path: str = "./data/processed/faiss_index"
    log_level: str = "INFO"
    backend_port: int = 8000
    frontend_port: int = 3000

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
