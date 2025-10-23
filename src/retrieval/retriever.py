from typing import List, Dict, Optional
from src.vector_store.chroma_manager import ChromaDBManager
from src.retrieval.reranker import Reranker
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Handles document retrieval with optional reranking."""
    
    def __init__(self, 
                 chroma_manager: ChromaDBManager,
                 use_reranking: bool = None):
        """
        Initialize retriever.
        
        Args:
            chroma_manager: ChromaDBManager instance
            use_reranking: Whether to use reranking (default from settings)
        """
        self.chroma_manager = chroma_manager
        self.use_reranking = use_reranking if use_reranking is not None else settings.USE_RERANKING
        
        if self.use_reranking:
            self.reranker = Reranker()
            logger.info("Retriever initialized with reranking")
        else:
            self.reranker = None
            logger.info("Retriever initialized without reranking")
    
    def retrieve(self, 
                query: str,
                top_k: Optional[int] = None,
                filters: Optional[Dict] = None,
                similarity_threshold: Optional[float] = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of retrieved documents with scores
        """
        top_k = top_k or settings.TOP_K
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        # Initial retrieval - get more results if reranking
        initial_k = top_k * 3 if self.use_reranking else top_k
        
        # Retrieve from ChromaDB
        results = self.chroma_manager.search(
            query=query,
            top_k=initial_k,
            filter_dict=filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity'] >= similarity_threshold
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents above threshold {similarity_threshold}")
        
        # Apply reranking if enabled
        if self.use_reranking and filtered_results:
            reranked_results = self.reranker.rerank(
                query=query,
                documents=filtered_results,
                top_k=top_k
            )
            return reranked_results[:top_k]
        
        return filtered_results[:top_k]
    
    def retrieve_with_context(self,
                             query: str,
                             top_k: Optional[int] = None,
                             context_window: int = 1) -> List[Dict]:
        """
        Retrieve documents with surrounding context chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            context_window: Number of adjacent chunks to include
        
        Returns:
            List of documents with context
        """
        # Standard retrieval
        results = self.retrieve(query=query, top_k=top_k)
        
        # For each result, try to find adjacent chunks
        # This assumes chunks have sequential chunk_id in metadata
        results_with_context = []
        
        for result in results:
            chunk_id = result['metadata'].get('chunk_id')
            source = result['metadata'].get('source')
            
            if chunk_id is not None and source:
                # Try to get adjacent chunks
                context_results = {
                    'main': result,
                    'before': [],
                    'after': []
                }
                
                # Search for adjacent chunks (simplified - in production use more sophisticated logic)
                for offset in range(1, context_window + 1):
                    # This is a simplified approach - you may need to store chunk relationships
                    pass
                
                results_with_context.append(context_results)
            else:
                results_with_context.append({'main': result, 'before': [], 'after': []})
        
        return results_with_context
    
    def retrieve_mmr(self,
                    query: str,
                    top_k: Optional[int] = None,
                    diversity_factor: float = 0.5) -> List[Dict]:
        """
        Maximal Marginal Relevance retrieval for diverse results.
        
        Args:
            query: Search query
            top_k: Number of results
            diversity_factor: Balance between relevance and diversity (0-1)
        
        Returns:
            List of diverse, relevant documents
        """
        top_k = top_k or settings.TOP_K
        
        # Get initial pool of candidates
        candidates = self.chroma_manager.search(
            query=query,
            top_k=top_k * 3
        )
        
        if not candidates:
            return []
        
        # MMR algorithm
        selected = []
        remaining = candidates.copy()
        
        # Select first (most relevant) document
        selected.append(remaining.pop(0))
        
        # Select subsequent documents balancing relevance and diversity
        while len(selected) < top_k and remaining:
            best_score = float('-inf')
            best_idx = 0
            
            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate['similarity']
                
                # Diversity score (distance from already selected)
                max_similarity_to_selected = max(
                    self._compute_similarity(candidate, sel)
                    for sel in selected
                )
                diversity = 1 - max_similarity_to_selected
                
                # Combined MMR score
                mmr_score = (diversity_factor * relevance + 
                           (1 - diversity_factor) * diversity)
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _compute_similarity(self, doc1: Dict, doc2: Dict) -> float:
        """Compute similarity between two document results."""
        # Simple approach: use content similarity
        # In production, you might use embeddings directly
        content1 = doc1['content']
        content2 = doc2['content']
        
        # Jaccard similarity on words
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def hybrid_search(self,
                     query: str,
                     top_k: Optional[int] = None,
                     semantic_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic search (0-1)
        
        Returns:
            Combined results from semantic and keyword search
        """
        top_k = top_k or settings.TOP_K
        
        # Semantic search
        semantic_results = self.retrieve(query=query, top_k=top_k * 2)
        
        # Simple keyword search (in production, use BM25 or similar)
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and rerank
        combined_scores = {}
        
        for idx, result in enumerate(semantic_results):
            doc_id = result['id']
            score = result['similarity'] * semantic_weight
            combined_scores[doc_id] = {
                'score': score,
                'result': result
            }
        
        for idx, result in enumerate(keyword_results):
            doc_id = result['id']
            keyword_score = (1 - idx / len(keyword_results)) * (1 - semantic_weight)
            
            if doc_id in combined_scores:
                combined_scores[doc_id]['score'] += keyword_score
            else:
                combined_scores[doc_id] = {
                    'score': keyword_score,
                    'result': result
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return [r['result'] for r in sorted_results[:top_k]]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Simple keyword-based search."""
        # Get all documents and do simple keyword matching
        # In production, use proper BM25 implementation
        query_terms = query.lower().split()
        
        results = self.chroma_manager.search(
            query=query,
            top_k=top_k * 2
        )
        
        # Score by keyword overlap
        for result in results:
            content_terms = result['content'].lower().split()
            overlap = len(set(query_terms) & set(content_terms))
            result['keyword_score'] = overlap / len(query_terms) if query_terms else 0
        
        # Sort by keyword score
        results.sort(key=lambda x: x['keyword_score'], reverse=True)
        
        return results[:top_k]