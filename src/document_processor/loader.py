from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading documents from various file formats."""
    
    LOADER_MAP = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.txt': TextLoader,
        '.md': TextLoader,
    }
    
    @classmethod
    def load_document(cls, file_path: Path) -> List[Document]:
        """
        Load a document and return LangChain Document objects.
        
        Args:
            file_path: Path to the document
        
        Returns:
            List of Document objects with content and metadata
        """
        extension = file_path.suffix.lower()
        
        if extension not in cls.LOADER_MAP:
            logger.warning(f"Unsupported file type: {extension}")
            return []
        
        try:
            loader_class = cls.LOADER_MAP[extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = str(file_path)
                doc.metadata['file_name'] = file_path.name
                doc.metadata['file_type'] = extension
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []
    
    @classmethod
    def load_documents_from_directory(cls, directory: Path) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
        
        Returns:
            List of all loaded Document objects
        """
        all_documents = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in cls.LOADER_MAP:
                documents = cls.load_document(file_path)
                all_documents.extend(documents)
        
        logger.info(f"Loaded {len(all_documents)} total documents from {directory}")
        return all_documents