from typing import List
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import settings
import tiktoken
import logging

logger = logging.getLogger(__name__)

class OptimizedChunker:
    """Advanced document chunking with multiple strategies."""
    
    def __init__(self, strategy: str = None):
        """
        Initialize chunker with specified strategy.
        
        Args:
            strategy: Chunking strategy - "fixed", "recursive", or "semantic"
        """
        self.strategy = strategy or settings.CHUNKING_STRATEGY
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents using the configured strategy.
        
        Args:
            documents: List of Document objects to chunk
        
        Returns:
            List of chunked Document objects
        """
        if self.strategy == "fixed":
            return self._fixed_size_chunking(documents)
        elif self.strategy == "recursive":
            return self._recursive_chunking(documents)
        elif self.strategy == "semantic":
            return self._semantic_chunking(documents)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using recursive")
            return self._recursive_chunking(documents)
    
    def _fixed_size_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Fixed-size chunking based on token count.
        Fast but may split semantic units.
        """
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Fixed chunking: {len(documents)} docs -> {len(chunks)} chunks")
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_strategy'] = 'fixed'
            chunk.metadata['token_count'] = self._count_tokens(chunk.page_content)
        
        return chunks
    
    def _recursive_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Recursive chunking that respects document structure.
        Best for most use cases - balances speed and quality.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,  # Approximate character count
            chunk_overlap=self.chunk_overlap * 4,
            length_function=self._count_tokens,
            separators=[
                "\n\n\n",  # Page breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                ", ",      # Clauses
                " ",       # Words
                ""         # Characters
            ],
            is_separator_regex=False,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Recursive chunking: {len(documents)} docs -> {len(chunks)} chunks")
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_strategy'] = 'recursive'
            chunk.metadata['token_count'] = self._count_tokens(chunk.page_content)
        
        return chunks
    
    def _semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """
        Semantic chunking based on embedding similarity.
        Highest quality but slower - groups semantically related content.
        """
        try:
            # Use same embeddings as main system for consistency
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            text_splitter = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
                breakpoint_threshold_amount=95,  # Higher = fewer, larger chunks
            )
            
            chunks = []
            for doc in documents:
                doc_chunks = text_splitter.create_documents([doc.page_content])
                # Preserve original metadata
                for chunk in doc_chunks:
                    chunk.metadata.update(doc.metadata)
                chunks.extend(doc_chunks)
            
            logger.info(f"Semantic chunking: {len(documents)} docs -> {len(chunks)} chunks")
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_strategy'] = 'semantic'
                chunk.metadata['token_count'] = self._count_tokens(chunk.page_content)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Semantic chunking failed: {e}, falling back to recursive")
            return self._recursive_chunking(documents)
    
    def optimize_chunk_size(self, documents: List[Document], 
                           test_queries: List[str] = None) -> dict:
        """
        Test different chunk sizes and return optimal configuration.
        
        Args:
            documents: Sample documents to test
            test_queries: Sample queries for retrieval testing
        
        Returns:
            Dictionary with optimal chunk_size and chunk_overlap
        """
        test_sizes = [256, 512, 1024]
        test_overlaps = [25, 50, 100]
        
        results = []
        
        for size in test_sizes:
            for overlap in test_overlaps:
                if overlap >= size:
                    continue
                
                # Create chunks
                temp_chunker = OptimizedChunker("recursive")
                temp_chunker.chunk_size = size
                temp_chunker.chunk_overlap = overlap
                
                chunks = temp_chunker.chunk_documents(documents[:5])  # Test on sample
                
                # Metrics
                avg_tokens = sum(c.metadata['token_count'] for c in chunks) / len(chunks)
                total_chunks = len(chunks)
                
                results.append({
                    'chunk_size': size,
                    'chunk_overlap': overlap,
                    'total_chunks': total_chunks,
                    'avg_tokens': avg_tokens,
                    'score': total_chunks / avg_tokens  # Lower is better (fewer, denser chunks)
                })
        
        # Find optimal configuration
        optimal = min(results, key=lambda x: x['score'])
        
        logger.info(f"Optimal chunk configuration: {optimal}")
        return optimal