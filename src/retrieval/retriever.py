from typing import List, Dict, Optional
from src.vector_store.chroma_manager import ChromaDBManager
from src.retrieval.reranker import Reranker
from src.retrieval.query_processor import QueryOptimizer, Gatekeeper, Auditor
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class Retriever:
    """Advanced document retrieval with multi-step pipeline."""
    
    def __init__(self, 
                 chroma_manager: ChromaDBManager,
                 use_reranking: bool = None,
                 use_query_optimization: bool = True,
                 use_gatekeeper: bool = False):
        """
        Initialize retriever with advanced features.
        
        Args:
            chroma_manager: ChromaDBManager instance
            use_reranking: Whether to use reranking (default from settings)
            use_query_optimization: Whether to optimize queries before search
            use_gatekeeper: Whether to validate query clarity
        """
        self.chroma_manager = chroma_manager
        self.use_reranking = use_reranking if use_reranking is not None else settings.USE_RERANKING
        
        # Initialize advanced components
        # Pass embedding manager into QueryOptimizer for verification to prevent query drift
        self.query_optimizer = QueryOptimizer(embedding_manager=self.chroma_manager.embedding_manager) if use_query_optimization else None
        self.gatekeeper = Gatekeeper() if use_gatekeeper else None
        self.auditor = Auditor()
        
        if self.use_reranking:
            self.reranker = Reranker()
            logger.info("Retriever initialized with reranking")
        else:
            self.reranker = None
            logger.info("Retriever initialized without reranking")
        
        if self.query_optimizer and self.query_optimizer.enabled:
            logger.info("Query optimization enabled")
        if self.gatekeeper and self.gatekeeper.enabled:
            logger.info("Gatekeeper enabled")
    
    def retrieve(self, 
                query: str,
                top_k: Optional[int] = None,
                filters: Optional[Dict] = None,
                similarity_threshold: Optional[float] = None,
                validate_results: bool = False) -> Dict[str, any]:
        """
        Advanced multi-step retrieval with optimization and validation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
            similarity_threshold: Minimum similarity score
            validate_results: Whether to run auditor validation
        
        Returns:
            Dict with 'results', 'metadata', and optionally 'validation'
        """
        top_k = top_k or settings.TOP_K
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        metadata = {
            'original_query': query,
            'optimized_query': None,
            'gatekeeper_check': None,
            'validation': None
        }
        
        # Step 1: Gatekeeper check (optional)
        if self.gatekeeper and self.gatekeeper.enabled:
            clarity_check = self.gatekeeper.check_query_clarity(query)
            metadata['gatekeeper_check'] = clarity_check
            
            if not clarity_check['is_clear']:
                logger.warning(f"Query needs clarification: {query}")
                return {
                    'results': [],
                    'metadata': metadata,
                    'clarification_needed': True,
                    'clarification': clarity_check['clarification']
                }
        
        # Step 2: Query optimization
        search_query = query
        if self.query_optimizer and self.query_optimizer.enabled:
            search_query = self.query_optimizer.optimize_query(query)
            metadata['optimized_query'] = search_query
            logger.info(f"Optimized: '{query}' -> '{search_query}'")
        
        # Step 3: Initial broad retrieval
        initial_k = top_k * 3 if self.use_reranking else top_k * 2
        
        results = self.chroma_manager.search(
            query=search_query,
            top_k=initial_k,
            filter_dict=filters
        )
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity'] >= similarity_threshold
        ]
        
        logger.info(f"Retrieved {len(filtered_results)} documents above threshold {similarity_threshold}")
        
        # Step 4: Reranking (if enabled)
        if self.use_reranking and filtered_results:
            reranked_results = self.reranker.rerank(
                query=search_query,
                documents=filtered_results,
                top_k=top_k
            )
            final_results = reranked_results[:top_k]
        else:
            final_results = filtered_results[:top_k]
        
        # Step 5: Validation (optional)
        if validate_results and self.auditor and self.auditor.enabled:
            validation = self.auditor.validate_results(query, final_results)
            metadata['validation'] = validation
            
            if not validation['is_valid']:
                logger.warning(f"Results failed validation: {validation['issues']}")
        
        return {
            'results': final_results,
            'metadata': metadata,
            'clarification_needed': False
        }
    
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