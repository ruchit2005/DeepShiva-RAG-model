import logging
from pathlib import Path
from typing import Optional, List
import argparse

from config.settings import settings
from src.document_processor.gdrive_client import GoogleDriveClient
from src.document_processor.loader import DocumentLoader
from src.document_processor.chunker import OptimizedChunker
from src.document_processor.enrichment_manager import EnrichmentManager
from src.embeddings.embedding_manager import EmbeddingManager
from src.vector_store.chroma_manager import ChromaDBManager
from src.retrieval.retriever import Retriever
from src.evaluation.metrics import RetrievalEvaluator, TestDatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGAgent:
    """Main RAG Agent orchestrator."""
    
    def __init__(self):
        """Initialize RAG Agent components."""
        logger.info("Initializing RAG Agent...")
        
        self.gdrive_client = None
        self.embedding_manager = EmbeddingManager()
        self.enrichment_manager = EnrichmentManager(use_enrichment=settings.USE_ENRICHMENT)
        self.chroma_manager = ChromaDBManager(embedding_manager=self.embedding_manager)
        self.retriever = Retriever(
            self.chroma_manager,
            use_query_optimization=settings.USE_QUERY_OPTIMIZATION,
            use_gatekeeper=settings.USE_GATEKEEPER
        )
        self.evaluator = RetrievalEvaluator()
        
        logger.info("RAG Agent initialized successfully")
    
    def setup_google_drive(self) -> GoogleDriveClient:
        """Initialize Google Drive client."""
        if not self.gdrive_client:
            self.gdrive_client = GoogleDriveClient()
        return self.gdrive_client
    
    def ingest_from_google_drive(self, folder_id: Optional[str] = None) -> int:
        """
        Download and ingest documents from Google Drive.
        
        Args:
            folder_id: Specific Google Drive folder ID (None for all accessible files)
        
        Returns:
            Number of documents ingested
        """
        logger.info("Starting Google Drive ingestion...")
        
        # Setup Google Drive client
        gdrive = self.setup_google_drive()
        
        # Download files
        downloaded_files = gdrive.download_all_files(folder_id)
        logger.info(f"Downloaded {len(downloaded_files)} files")
        
        # Process and ingest
        total_ingested = self.ingest_local_documents(settings.RAW_DATA_DIR)
        
        return total_ingested
    
    def ingest_local_documents(self, directory: Path) -> int:
        """
        Ingest documents from local directory.
        
        Args:
            directory: Path to directory containing documents
        
        Returns:
            Number of chunks ingested
        """
        logger.info(f"Loading documents from {directory}")
        
        # Load documents
        documents = DocumentLoader.load_documents_from_directory(directory)
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found to ingest")
            return 0
        
        # Chunk documents
        chunker = OptimizedChunker(strategy=settings.CHUNKING_STRATEGY)
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks using {settings.CHUNKING_STRATEGY} strategy")
        
        # Enrich chunks with metadata (if enabled)
        if settings.USE_ENRICHMENT:
            logger.info("Enriching chunks with LLM-generated metadata...")
            chunks = self.enrichment_manager.enrich_chunks_batch(chunks, show_progress=True)
        
        # Add to vector store
        num_added = self.chroma_manager.add_documents(chunks)
        logger.info(f"Added {num_added} chunks to ChromaDB")
        
        return num_added
    
    def search(self, 
               query: str,
               top_k: int = None,
               use_reranking: bool = None,
               validate_results: bool = None) -> dict:
        """
        Advanced search with multi-step pipeline.
        
        Args:
            query: Search query
            top_k: Number of results
            use_reranking: Override default reranking setting
            validate_results: Whether to validate results quality
        
        Returns:
            Dict with results, metadata, and any clarifications needed
        """
        if use_reranking is not None:
            self.retriever.use_reranking = use_reranking
            if use_reranking and not self.retriever.reranker:
                from src.retrieval.reranker import Reranker
                self.retriever.reranker = Reranker()
        
        validate = validate_results if validate_results is not None else settings.VALIDATE_RESULTS
        
        response = self.retriever.retrieve(
            query=query, 
            top_k=top_k,
            validate_results=validate
        )
        
        results = response.get('results', [])
        logger.info(f"Found {len(results)} results for query: '{query}'")
        
        # Check if clarification is needed
        if response.get('clarification_needed'):
            logger.info(f"Clarification needed: {response.get('clarification')}")
        
        # Check validation results
        if validate and 'validation' in response.get('metadata', {}):
            validation = response['metadata']['validation']
            if validation and not validation['is_valid']:
                logger.warning(f"Results validation issues: {validation['issues']}")
        
        return response
    
    def evaluate_system(self, 
                       test_queries: Optional[List] = None,
                       num_test_queries: int = 50) -> dict:
        """
        Evaluate retrieval performance.
        
        Args:
            test_queries: List of (query, relevant_doc_ids) tuples
            num_test_queries: Number of synthetic queries to generate if test_queries not provided
        
        Returns:
            Evaluation metrics
        """
        logger.info("Starting system evaluation...")
        
        if not test_queries:
            # Generate synthetic test queries
            logger.info(f"Generating {num_test_queries} synthetic test queries...")
            
            # Get sample documents
            collection = self.chroma_manager.collection
            sample = collection.get(limit=min(100, collection.count()))
            
            documents = [
                {'id': id_, 'content': doc}
                for id_, doc in zip(sample['ids'], sample['documents'])
            ]
            
            test_queries = TestDatasetGenerator.generate_test_queries(
                documents, 
                num_queries=num_test_queries
            )
        
        # Run evaluation
        metrics = self.evaluator.batch_evaluate(
            test_queries,
            self.retriever
        )
        
        # Generate report
        report = self.evaluator.generate_report(metrics)
        print("\n" + report)
        
        # Save results
        results_path = settings.BASE_DIR / "evaluation_results.json"
        self.evaluator.export_results(results_path)
        
        return metrics
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        stats = self.chroma_manager.get_collection_stats()
        logger.info(f"System stats: {stats}")
        return stats
    
    def reset_database(self):
        """Delete all documents and reset the database."""
        logger.warning("Resetting database...")
        self.chroma_manager.delete_collection()
        self.chroma_manager = ChromaDBManager(embedding_manager=self.embedding_manager)
        self.retriever = Retriever(self.chroma_manager)
        logger.info("Database reset complete")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="RAG Agent with Google Drive Integration")
    parser.add_argument('command', choices=[
        'ingest-gdrive', 'ingest-local', 'search', 'evaluate', 'stats', 'reset'
    ], help='Command to execute')
    parser.add_argument('--folder-id', help='Google Drive folder ID')
    parser.add_argument('--directory', help='Local directory path')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    parser.add_argument('--no-rerank', action='store_true', help='Disable reranking')
    parser.add_argument('--num-test-queries', type=int, default=50, 
                       help='Number of test queries for evaluation')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RAGAgent()
    
    # Execute command
    if args.command == 'ingest-gdrive':
        num_docs = agent.ingest_from_google_drive(args.folder_id)
        print(f"\n✅ Ingested {num_docs} document chunks from Google Drive")
    
    elif args.command == 'ingest-local':
        directory = Path(args.directory) if args.directory else settings.RAW_DATA_DIR
        num_docs = agent.ingest_local_documents(directory)
        print(f"\n✅ Ingested {num_docs} document chunks from {directory}")
    
    elif args.command == 'search':
        if not args.query:
            print("❌ Error: --query is required for search command")
            return
        
        response = agent.search(
            query=args.query,
            top_k=args.top_k,
            use_reranking=not args.no_rerank
        )
        
        # Check if clarification is needed
        if response.get('clarification_needed'):
            print(f"\n❓ Clarification Needed:\n{response.get('clarification')}")
            return
        
        results = response.get('results', [])
        metadata = response.get('metadata', {})
        
        print(f"\n🔍 Search Results for: '{args.query}'\n")
        print("=" * 80)
        
        # Show query optimization if used
        if metadata.get('optimized_query'):
            print(f"📝 Optimized Query: {metadata['optimized_query']}\n")
            print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i} (Similarity: {result['similarity']:.4f})")
            print(f"Source: {result['metadata'].get('source', 'Unknown')}")
            print(f"Content: {result['content'][:200]}...")
            if 'rerank_score' in result:
                print(f"Rerank Score: {result['rerank_score']:.4f}")
            print("-" * 80)
        
        # Show validation results if enabled
        if metadata.get('validation'):
            validation = metadata['validation']
            status = "✅ VALID" if validation['is_valid'] else "⚠️ NEEDS REVIEW"
            print(f"\n{status} | Confidence: {validation['confidence']:.2f}")
            if validation['issues']:
                print(f"Issues: {', '.join(validation['issues'])}")
    
    elif args.command == 'evaluate':
        metrics = agent.evaluate_system(num_test_queries=args.num_test_queries)
        print("\n✅ Evaluation complete. Results saved to evaluation_results.json")
    
    elif args.command == 'stats':
        stats = agent.get_stats()
        print("\n📊 System Statistics")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 60)
    
    elif args.command == 'reset':
        confirm = input("⚠️  Are you sure you want to reset the database? (yes/no): ")
        if confirm.lower() == 'yes':
            agent.reset_database()
            print("\n✅ Database reset complete")
        else:
            print("\n❌ Reset cancelled")


if __name__ == "__main__":
    main()