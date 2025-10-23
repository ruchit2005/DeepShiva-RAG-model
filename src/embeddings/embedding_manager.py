from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from config.settings import settings
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding generation with multiple provider support."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.batch_size = settings.BATCH_SIZE
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the appropriate embedding model."""
        
        # OpenAI embeddings
        if self.model_name.startswith("text-embedding"):
            if not settings.OPENAI_API_KEY:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            logger.info(f"Using OpenAI embeddings: {self.model_name}")
            return OpenAIEmbeddings(
                model=self.model_name,
                openai_api_key=settings.OPENAI_API_KEY
            )
        
        # Cohere embeddings
        elif self.model_name.startswith("embed-"):
            if not settings.COHERE_API_KEY:
                raise ValueError("Cohere API key required for Cohere embeddings")
            logger.info(f"Using Cohere embeddings: {self.model_name}")
            return CohereEmbeddings(
                model=self.model_name,
                cohere_api_key=settings.COHERE_API_KEY
            )
        
        # HuggingFace embeddings (default, free)
        else:
            logger.info(f"Using HuggingFace embeddings: {self.model_name}")
            return HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
                encode_kwargs={
                    'normalize_embeddings': True,  # Important for cosine similarity
                    'batch_size': self.batch_size
                }
            )
    
    def embed_documents(self, texts: List[str], 
                       show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches
        batches = [
            texts[i:i + self.batch_size] 
            for i in range(0, len(texts), self.batch_size)
        ]
        
        iterator = tqdm(batches, desc="Embedding documents") if show_progress else batches
        
        for batch in iterator:
            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                # Add zero vectors for failed batches
                embeddings.extend([[0.0] * settings.EMBEDDING_DIMENSION] * len(batch))
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text
        
        Returns:
            Embedding vector
        """
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return [0.0] * settings.EMBEDDING_DIMENSION
    
    def compute_similarity(self, embedding1: List[float], 
                          embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model."""
        test_embedding = self.embed_query("test")
        return len(test_embedding)