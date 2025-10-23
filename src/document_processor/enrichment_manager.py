from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from config.settings import settings
import logging
import json

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """Structured metadata for document chunks."""
    
    summary: str = Field(
        description="A concise 2-3 sentence summary of the chunk's main content"
    )
    keywords: List[str] = Field(
        description="5-7 key terms or phrases that best represent this chunk"
    )
    hypothetical_questions: List[str] = Field(
        description="3-5 questions that this chunk could answer"
    )
    table_summary: Optional[str] = Field(
        default=None,
        description="For table content: A natural language description of what the table shows"
    )
    content_type: str = Field(
        description="Type of content: 'text', 'table', 'list', or 'mixed'"
    )
    relevance_score: float = Field(
        default=1.0,
        description="Estimated importance/relevance score (0-1)"
    )


class EnrichmentManager:
    """Manages LLM-based enrichment of document chunks."""
    
    def __init__(self, use_enrichment: bool = True):
        """
        Initialize enrichment manager.
        
        Args:
            use_enrichment: Whether to use LLM enrichment (requires API key)
        """
        self.use_enrichment = use_enrichment and settings.OPENAI_API_KEY is not None
        self.llm = None
        
        if self.use_enrichment:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=settings.OPENAI_API_KEY
                ).with_structured_output(ChunkMetadata)
                logger.info("EnrichmentManager initialized with LLM support")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM for enrichment: {e}")
                self.use_enrichment = False
                self.llm = None
        else:
            logger.info("EnrichmentManager initialized without LLM (will use basic metadata)")
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content in the chunk."""
        text_lower = text.lower()
        
        # Check for table indicators
        has_pipes = '|' in text
        has_table_keywords = any(kw in text_lower for kw in ['table', 'column', 'row'])
        
        # Check for list indicators
        has_bullets = any(marker in text for marker in ['•', '▪', '◦', '-', '*'])
        has_numbers = any(f"{i}." in text or f"({i})" in text for i in range(1, 10))
        
        if (has_pipes or has_table_keywords) and len(text.split('\n')) > 3:
            return 'table'
        elif (has_bullets or has_numbers) and len(text.split('\n')) > 5:
            return 'list'
        elif any([has_pipes, has_bullets, has_numbers]):
            return 'mixed'
        else:
            return 'text'
    
    def _generate_enrichment_prompt(self, chunk_text: str, content_type: str) -> str:
        """Generate appropriate prompt based on content type."""
        
        base_instructions = """Analyze the following document chunk and provide structured metadata."""
        
        if content_type == 'table':
            specific_instructions = """
This chunk contains tabular data. Pay special attention to:
- What metrics or data points are shown
- What time periods or categories are covered
- Key trends or patterns visible in the data
- Provide a detailed table_summary that explains the table in natural language
"""
        elif content_type == 'list':
            specific_instructions = """
This chunk contains a list or enumeration. Focus on:
- What items or concepts are being listed
- The organizing principle or category
- Key themes across the list items
"""
        else:
            specific_instructions = """
This chunk contains narrative text. Focus on:
- Main concepts or arguments presented
- Key facts or claims
- Context and implications
"""
        
        return f"""{base_instructions}

{specific_instructions}

Document Chunk:
\"\"\"
{chunk_text[:2000]}  
\"\"\"

Provide:
1. A concise 2-3 sentence summary
2. 5-7 relevant keywords
3. 3-5 questions this chunk could answer
4. Content type classification
5. (If table) A natural language summary of the table's data
"""
    
    def enrich_chunk(self, chunk: Document) -> Document:
        """
        Enrich a single chunk with LLM-generated metadata.
        
        Args:
            chunk: Document to enrich
        
        Returns:
            Enriched Document with additional metadata
        """
        chunk_text = chunk.page_content
        content_type = self._detect_content_type(chunk_text)
        
        if self.use_enrichment:
            try:
                prompt = self._generate_enrichment_prompt(chunk_text, content_type)
                
                # Get structured metadata from LLM
                metadata: ChunkMetadata = self.llm.invoke(prompt)
                
                # Add enriched metadata to chunk
                chunk.metadata.update({
                    'summary': metadata.summary,
                    'keywords': metadata.keywords,
                    'hypothetical_questions': metadata.hypothetical_questions,
                    'table_summary': metadata.table_summary,
                    'content_type': metadata.content_type,
                    'relevance_score': metadata.relevance_score,
                    'enriched': True
                })
                
                logger.debug(f"Enriched chunk with {len(metadata.keywords)} keywords")
                
            except Exception as e:
                logger.warning(f"Enrichment failed for chunk: {e}")
                # Fall back to basic metadata
                chunk.metadata.update(self._create_basic_metadata(chunk_text, content_type))
        else:
            # Use basic metadata without LLM
            chunk.metadata.update(self._create_basic_metadata(chunk_text, content_type))
        
        return chunk
    
    def _create_basic_metadata(self, text: str, content_type: str) -> Dict[str, Any]:
        """Create basic metadata without LLM (fallback)."""
        # Extract simple keywords (most common significant words)
        words = text.lower().split()
        word_freq = {}
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                     'been', 'this', 'that', 'these', 'those', 'it', 'its'}
        
        for word in words:
            if len(word) > 3 and word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:7]
        keywords = [word for word, _ in keywords]
        
        # Basic summary (first 200 chars)
        summary = text[:200].strip() + "..."
        
        return {
            'summary': summary,
            'keywords': keywords,
            'hypothetical_questions': [],
            'table_summary': None,
            'content_type': content_type,
            'relevance_score': 1.0,
            'enriched': False
        }
    
    def enrich_chunks_batch(self, chunks: List[Document], 
                           show_progress: bool = True) -> List[Document]:
        """
        Enrich multiple chunks with metadata.
        
        Args:
            chunks: List of Documents to enrich
            show_progress: Show progress bar
        
        Returns:
            List of enriched Documents
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(chunks, desc="Enriching chunks")
            except ImportError:
                iterator = chunks
                logger.info(f"Enriching {len(chunks)} chunks...")
        else:
            iterator = chunks
        
        enriched_chunks = []
        for chunk in iterator:
            enriched_chunk = self.enrich_chunk(chunk)
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f"Enriched {len(enriched_chunks)} chunks")
        return enriched_chunks
    
    def create_embedding_text(self, chunk: Document) -> str:
        """
        Create optimized text for embedding that includes enriched metadata.
        
        Args:
            chunk: Document with metadata
        
        Returns:
            Combined text for embedding
        """
        parts = []
        
        # Add summary if available
        if 'summary' in chunk.metadata:
            parts.append(f"Summary: {chunk.metadata['summary']}")
        
        # Add keywords
        if 'keywords' in chunk.metadata:
            keywords_str = ", ".join(chunk.metadata['keywords'])
            parts.append(f"Keywords: {keywords_str}")
        
        # Add table summary for tables
        if chunk.metadata.get('content_type') == 'table' and chunk.metadata.get('table_summary'):
            parts.append(f"Table Description: {chunk.metadata['table_summary']}")
        
        # Add original content (truncated if too long)
        content = chunk.page_content
        if len(content) > 1000:
            content = content[:1000] + "..."
        parts.append(f"Content: {content}")
        
        return "\n".join(parts)
