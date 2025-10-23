from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.schema import Document
from config.settings import settings
from src.embeddings.embedding_manager import EmbeddingManager
import logging
from tqdm import tqdm
import uuid

logger = logging.getLogger(__name__)

class ChromaDBManager:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self, collection_name: Optional[str] = None,
                 embedding_manager: Optional[EmbeddingManager] = None):
        """
        Initialize ChromaDB manager.
        
        Args:
            collection_name: Name of the ChromaDB collection
            embedding_manager: EmbeddingManager instance
        """
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_manager = embedding_manager or EmbeddingManager()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_DIR),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        logger.info(f"ChromaDB collection '{self.collection_name}' initialized")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We handle embeddings manually
            )
            logger.info(f"Loaded existing collection with {collection.count()} documents")
            return collection
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": settings.CHROMA_DISTANCE_METRIC,
                    "hnsw:construction_ef": 200,  # Higher = better recall, slower indexing
                    "hnsw:search_ef": 100,  # Higher = better recall, slower search
                    "hnsw:M": 16  # Number of connections, higher = better recall, more memory
                }
            )
            logger.info(f"Created new collection '{self.collection_name}'")
            return collection
    
    def add_documents(self, documents: List[Document], 
                     batch_size: int = 100,
                     show_progress: bool = True) -> int:
        """
        Add documents to ChromaDB with embeddings.
        
        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process at once
            show_progress: Show progress bar
        
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No documents to add")
            return 0
        
        total_added = 0
        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]
        
        iterator = tqdm(batches, desc="Adding documents to ChromaDB") if show_progress else batches
        
        for batch in iterator:
            try:
                # Extract texts and metadata
                texts = [doc.page_content for doc in batch]
                metadatas = [self._prepare_metadata(doc.metadata) for doc in batch]
                
                # Generate embeddings
                embeddings = self.embedding_manager.embed_documents(texts, show_progress=False)
                
                # Generate unique IDs
                ids = [str(uuid.uuid4()) for _ in batch]
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(batch)
                
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
        
        logger.info(f"Added {total_added} documents to ChromaDB")
        return total_added
    
    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """
        Prepare metadata for ChromaDB (must be JSON serializable).
        
        Args:
            metadata: Original metadata dictionary
        
        Returns:
            Cleaned metadata dictionary
        """
        cleaned = {}
        for key, value in metadata.items():
            # Convert non-serializable types
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif value is None:
                cleaned[key] = ""
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def search(self, query: str, 
               top_k: Optional[int] = None,
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Metadata filters (e.g., {"file_type": ".pdf"})
        
        Returns:
            List of result dictionaries with content, metadata, and similarity
        """
        top_k = top_k or settings.TOP_K
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_query(query)
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Convert distance to similarity (for cosine: similarity = 1 - distance)
            distance = results['distances'][0][i]
            similarity = 1 - distance if settings.CHROMA_DISTANCE_METRIC == "cosine" else distance
            
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity': similarity,
                'distance': distance
            })
        
        return formatted_results
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        # Sample a few documents to get average metadata
        sample_size = min(10, count)
        if sample_size > 0:
            sample = self.collection.get(limit=sample_size)
            avg_length = sum(len(doc) for doc in sample['documents']) / sample_size
        else:
            avg_length = 0
        
        return {
            'collection_name': self.collection_name,
            'total_documents': count,
            'average_document_length': avg_length,
            'embedding_dimension': self.embedding_manager.get_embedding_dimension(),
            'distance_metric': settings.CHROMA_DISTANCE_METRIC
        }
    
    def update_document(self, doc_id: str, 
                       new_content: str,
                       new_metadata: Optional[Dict] = None):
        """
        Update an existing document.
        
        Args:
            doc_id: Document ID to update
            new_content: New document content
            new_metadata: New metadata (optional)
        """
        # Generate new embedding
        new_embedding = self.embedding_manager.embed_query(new_content)
        
        update_dict = {
            'ids': [doc_id],
            'embeddings': [new_embedding],
            'documents': [new_content]
        }
        
        if new_metadata:
            update_dict['metadatas'] = [self._prepare_metadata(new_metadata)]
        
        self.collection.update(**update_dict)
        logger.info(f"Updated document {doc_id}")
    
    def delete_documents(self, doc_ids: List[str]):
        """Delete documents by IDs."""
        self.collection.delete(ids=doc_ids)
        logger.info(f"Deleted {len(doc_ids)} documents")