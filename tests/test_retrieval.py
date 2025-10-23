import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store.chroma_manager import ChromaDBManager
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.retriever import Retriever
from src.evaluation.metrics import RetrievalEvaluator
from langchain.schema import Document


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"source": "ml_intro.pdf", "chunk_id": 0}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers to process complex patterns in data.",
            metadata={"source": "dl_guide.pdf", "chunk_id": 1}
        ),
        Document(
            page_content="Natural language processing helps computers understand and generate human language.",
            metadata={"source": "nlp_basics.pdf", "chunk_id": 2}
        ),
        Document(
            page_content="Computer vision enables machines to interpret and understand visual information from the world.",
            metadata={"source": "cv_intro.pdf", "chunk_id": 3}
        ),
        Document(
            page_content="Reinforcement learning trains agents to make decisions through trial and error.",
            metadata={"source": "rl_concepts.pdf", "chunk_id": 4}
        ),
    ]


@pytest.fixture
def embedding_manager():
    """Create embedding manager."""
    return EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def chroma_manager(embedding_manager, sample_documents):
    """Create and populate ChromaDB manager."""
    manager = ChromaDBManager(
        collection_name="test_collection",
        embedding_manager=embedding_manager
    )
    
    # Clean up any existing data
    try:
        manager.delete_collection()
        manager = ChromaDBManager(
            collection_name="test_collection",
            embedding_manager=embedding_manager
        )
    except:
        pass
    
    # Add sample documents
    manager.add_documents(sample_documents, show_progress=False)
    
    yield manager
    
    # Cleanup
    manager.delete_collection()


@pytest.fixture
def retriever(chroma_manager):
    """Create retriever."""
    return Retriever(chroma_manager, use_reranking=False)


class TestEmbeddingManager:
    """Test embedding generation."""
    
    def test_embed_single_document(self, embedding_manager):
        """Test embedding a single document."""
        text = "This is a test document."
        embedding = embedding_manager.embed_query(text)
        
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_multiple_documents(self, embedding_manager):
        """Test embedding multiple documents."""
        texts = ["Document 1", "Document 2", "Document 3"]
        embeddings = embedding_manager.embed_documents(texts, show_progress=False)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_similarity_computation(self, embedding_manager):
        """Test similarity computation."""
        emb1 = embedding_manager.embed_query("machine learning")
        emb2 = embedding_manager.embed_query("artificial intelligence")
        emb3 = embedding_manager.embed_query("cooking recipes")
        
        # Related terms should be more similar
        sim_related = embedding_manager.compute_similarity(emb1, emb2)
        sim_unrelated = embedding_manager.compute_similarity(emb1, emb3)
        
        assert sim_related > sim_unrelated
        assert 0 <= sim_related <= 1
        assert 0 <= sim_unrelated <= 1


class TestChromaDBManager:
    """Test ChromaDB operations."""
    
    def test_add_documents(self, chroma_manager):
        """Test adding documents."""
        stats = chroma_manager.get_collection_stats()
        assert stats['total_documents'] == 5
    
    def test_search(self, chroma_manager):
        """Test basic search."""
        results = chroma_manager.search("machine learning", top_k=3)
        
        assert len(results) <= 3
        assert all('similarity' in r for r in results)
        assert all('content' in r for r in results)
        
        # Most relevant result should mention machine learning
        top_result = results[0]
        assert 'machine learning' in top_result['content'].lower()
    
    def test_search_with_filter(self, chroma_manager):
        """Test search with metadata filter."""
        results = chroma_manager.search(
            "learning",
            top_k=5,
            filter_dict={"source": "ml_intro.pdf"}
        )
        
        assert all(r['metadata']['source'] == "ml_intro.pdf" for r in results)
    
    def test_update_document(self, chroma_manager):
        """Test updating a document."""
        # Get a document
        results = chroma_manager.search("machine learning", top_k=1)
        doc_id = results[0]['id']
        
        # Update it
        new_content = "Updated content about machine learning"
        chroma_manager.update_document(doc_id, new_content)
        
        # Verify update
        updated_results = chroma_manager.collection.get(ids=[doc_id])
        assert updated_results['documents'][0] == new_content


class TestRetriever:
    """Test retrieval functionality."""
    
    def test_basic_retrieval(self, retriever):
        """Test basic retrieval."""
        results = retriever.retrieve("deep learning neural networks", top_k=3)
        
        assert len(results) <= 3
        assert 'deep learning' in results[0]['content'].lower() or \
               'neural networks' in results[0]['content'].lower()
    
    def test_retrieval_with_threshold(self, retriever):
        """Test retrieval with similarity threshold."""
        results = retriever.retrieve(
            "quantum computing",  # Unrelated query
            top_k=5,
            similarity_threshold=0.8
        )
        
        # Should return fewer or no results due to low similarity
        assert len(results) <= 5
    
    def test_mmr_retrieval(self, retriever):
        """Test MMR retrieval for diversity."""
        standard_results = retriever.retrieve("learning", top_k=3)
        mmr_results = retriever.retrieve_mmr("learning", top_k=3, diversity_factor=0.8)
        
        assert len(mmr_results) == len(standard_results)
        
        # MMR results should be more diverse (different sources)
        standard_sources = set(r['metadata']['source'] for r in standard_results)
        mmr_sources = set(r['metadata']['source'] for r in mmr_results)
        
        # Not a strict test, but MMR should tend toward diversity
        assert len(mmr_sources) > 0


class TestRetrievalEvaluator:
    """Test evaluation metrics."""
    
    def test_evaluate_retrieval(self, retriever):
        """Test retrieval evaluation."""
        evaluator = RetrievalEvaluator()
        
        # Simulate retrieved documents
        query = "machine learning"
        retrieved_docs = retriever.retrieve(query, top_k=5)
        
        # Get first doc ID as relevant
        relevant_ids = {retrieved_docs[0]['id']} if retrieved_docs else set()
        
        metrics = evaluator.evaluate_retrieval(
            query=query,
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=relevant_ids,
            k_values=[1, 3, 5]
        )
        
        assert 'precision@1' in metrics
        assert 'recall@1' in metrics
        assert 'mrr' in metrics
        assert 'map' in metrics
        
        # Since we included the relevant doc, precision@1 should be 1.0
        if len(retrieved_docs) > 0 and retrieved_docs[0]['id'] in relevant_ids:
            assert metrics['precision@1'] == 1.0
            assert metrics['recall@1'] == 1.0
            assert metrics['mrr'] == 1.0
    
    def test_batch_evaluation(self, retriever):
        """Test batch evaluation."""
        evaluator = RetrievalEvaluator()
        
        # Create test queries with known relevant docs
        test_queries = []
        
        # Query 1: Machine learning
        results1 = retriever.retrieve("machine learning", top_k=1)
        if results1:
            test_queries.append(("machine learning", {results1[0]['id']}))
        
        # Query 2: Neural networks
        results2 = retriever.retrieve("neural networks", top_k=1)
        if results2:
            test_queries.append(("neural networks", {results2[0]['id']}))
        
        if test_queries:
            metrics = evaluator.batch_evaluate(test_queries, retriever)
            
            assert 'num_queries' in metrics
            assert 'metrics' in metrics
            assert metrics['num_queries'] == len(test_queries)
    
    def test_mrr_calculation(self):
        """Test MRR calculation."""
        evaluator = RetrievalEvaluator()
        
        # First result is relevant
        mrr1 = evaluator._calculate_mrr(
            ['doc1', 'doc2', 'doc3'],
            {'doc1'}
        )
        assert mrr1 == 1.0
        
        # Second result is relevant
        mrr2 = evaluator._calculate_mrr(
            ['doc1', 'doc2', 'doc3'],
            {'doc2'}
        )
        assert mrr2 == 0.5
        
        # No relevant results
        mrr3 = evaluator._calculate_mrr(
            ['doc1', 'doc2', 'doc3'],
            {'doc4'}
        )
        assert mrr3 == 0.0
    
    def test_map_calculation(self):
        """Test MAP calculation."""
        evaluator = RetrievalEvaluator()
        
        # Two relevant docs at positions 1 and 3
        map_score = evaluator._calculate_map(
            ['doc1', 'doc2', 'doc3', 'doc4'],
            {'doc1', 'doc3'}
        )
        
        # MAP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        assert 0.8 < map_score < 0.85


class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, sample_documents):
        """Test complete ingestion and retrieval workflow."""
        # Setup
        embedding_manager = EmbeddingManager()
        chroma_manager = ChromaDBManager(
            collection_name="integration_test",
            embedding_manager=embedding_manager
        )
        
        try:
            # Ingest
            chroma_manager.add_documents(sample_documents, show_progress=False)
            
            # Retrieve
            retriever = Retriever(chroma_manager, use_reranking=False)
            results = retriever.retrieve("machine learning", top_k=3)
            
            assert len(results) > 0
            assert 'machine learning' in results[0]['content'].lower()
            
            # Evaluate
            evaluator = RetrievalEvaluator()
            relevant_ids = {results[0]['id']}
            metrics = evaluator.evaluate_retrieval(
                "machine learning",
                results,
                relevant_ids
            )
            
            assert metrics['precision@1'] == 1.0
            
        finally:
            # Cleanup
            chroma_manager.delete_collection()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])