import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Configuration settings for RAG Agent."""
    
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CHROMA_DB_DIR: Path = DATA_DIR / "chroma_db"
    
    # Google Drive
    GDRIVE_CREDENTIALS_PATH: Path = BASE_DIR / "config" / "credentials.json"
    GDRIVE_TOKEN_PATH: Path = BASE_DIR / "config" / "token.json"
    GDRIVE_SCOPES: List[str] = Field(default=["https://www.googleapis.com/auth/drive.readonly"])
    
    # Embedding settings
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model: all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, text-embedding-3-small"
    )
    EMBEDDING_DIMENSION: int = 384  # For all-MiniLM-L6-v2
    BATCH_SIZE: int = 32
    
    # Chunking settings
    CHUNK_SIZE: int = 512  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    CHUNKING_STRATEGY: str = Field(
        default="recursive",
        description="Options: fixed, recursive, semantic"
    )
    
    # ChromaDB settings
    CHROMA_COLLECTION_NAME: str = "documents"
    CHROMA_DISTANCE_METRIC: str = "cosine"  # Options: cosine, l2, ip
    
    # Retrieval settings
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.3  # Lowered from 0.7 to be less restrictive
    USE_RERANKING: bool = True
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # API Keys (optional - only if using paid models)
    OPENAI_API_KEY: Optional[str] = None
    COHERE_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Evaluation
    EVAL_DATASET_SIZE: int = 100
    EVAL_METRICS: List[str] = Field(
        default=["precision", "recall", "ndcg", "mrr", "hit_rate"]
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        extra = 'ignore'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.CHROMA_DB_DIR,
            self.BASE_DIR / "config",
            self.BASE_DIR / "logs",
            self.BASE_DIR / "tests",
            self.BASE_DIR / "notebooks",
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        init_files = [
            self.BASE_DIR / "config" / "__init__.py",
            self.BASE_DIR / "src" / "__init__.py",
            self.BASE_DIR / "src" / "document_processor" / "__init__.py",
            self.BASE_DIR / "src" / "embeddings" / "__init__.py",
            self.BASE_DIR / "src" / "vector_store" / "__init__.py",
            self.BASE_DIR / "src" / "retrieval" / "__init__.py",
            self.BASE_DIR / "src" / "evaluation" / "__init__.py",
            self.BASE_DIR / "tests" / "__init__.py",
        ]
        
        for init_file in init_files:
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch(exist_ok=True)

# Global settings instance
settings = Settings()