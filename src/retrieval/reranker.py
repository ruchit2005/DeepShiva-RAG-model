from typing import List, Dict
from sentence_transformers import CrossEncoder
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class Reranker:
    """Reranks retrieved documents using cross-encoder models."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name or settings.RERANK_MODEL
        
        try:
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Reranker initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None
    
    def rerank(self, 
               query: str,
               documents: List[Dict],
               top_k: int = None) -> List[Dict]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document dictionaries
            top_k: Number of top results to return
        
        Returns:
            Reranked list of documents
        """
        if not self.model or not documents:
            return documents
        
        top_k = top_k or len(documents)
        
        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in documents]
        
        # Get reranking scores
        try:
            scores = self.model.predict(pairs)
            
            # Add reranking scores to documents
            for doc, score in zip(documents, scores):
                doc['rerank_score'] = float(score)
                doc['original_similarity'] = doc.get('similarity', 0)
            
            # Sort by reranking score
            reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            
            logger.info(f"Reranked {len(documents)} documents")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return documents[:top_k]
    
    def batch_rerank(self,
                    query: str,
                    document_batches: List[List[Dict]],
                    top_k: int = None) -> List[Dict]:
        """
        Rerank multiple batches of documents.
        
        Args:
            query: Search query
            document_batches: List of document batches
            top_k: Total number of results to return
        
        Returns:
            Combined and reranked results
        """
        all_reranked = []
        
        for batch in document_batches:
            reranked_batch = self.rerank(query, batch)
            all_reranked.extend(reranked_batch)
        
        # Final reranking of all results
        if all_reranked:
            final_reranked = self.rerank(query, all_reranked, top_k)
            return final_reranked
        
        return []