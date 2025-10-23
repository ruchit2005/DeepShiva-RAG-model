# DEEP-SHIV: Advanced RAG System with Google Drive Integration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-ready Retrieval-Augmented Generation (RAG) system with Google Drive integration, featuring advanced document processing, semantic search, and comprehensive evaluation metrics.

## 🌟 Features

- **📁 Google Drive Integration** - OAuth2 authentication with automatic document syncing
- **📄 Multi-Format Support** - PDF, DOCX, PPTX, XLSX, TXT, Markdown
- **🧠 Flexible Embeddings** - HuggingFace (free), OpenAI, Cohere support
- **🔍 Advanced Retrieval** - MMR diversity, hybrid search, cross-encoder reranking
- **📊 Comprehensive Evaluation** - Precision@K, Recall@K, MRR, MAP, NDCG metrics
- **⚡ ChromaDB Vector Store** - Fast, efficient similarity search
- **🎯 CLI Interface** - Easy-to-use command-line tools

## 🏗️ Architecture

```
DEEP-SHIV/
├── config/              # Configuration and credentials
│   ├── credentials.json # Google OAuth credentials (gitignored)
│   └── settings.py      # Application settings
├── data/
│   ├── raw/            # Downloaded documents
│   ├── processed/      # Processed documents
│   └── chroma_db/      # Vector database
├── src/
│   ├── document_processor/
│   │   ├── gdrive_client.py   # Google Drive API client
│   │   ├── loader.py          # Multi-format document loader
│   │   └── chunker.py         # Text chunking strategies
│   ├── embeddings/
│   │   └── embedding_manager.py  # Embedding generation
│   ├── vector_store/
│   │   └── chroma_manager.py     # ChromaDB operations
│   ├── retrieval/
│   │   ├── retriever.py          # Retrieval strategies
│   │   └── reranker.py           # Cross-encoder reranking
│   └── evaluation/
│       └── metrics.py            # Evaluation metrics
├── tests/              # Unit tests
├── main.py            # CLI entry point
└── requirements.txt   # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DEEP-SHIV.git
cd DEEP-SHIV

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

#### Create `.env` file:

```env
# Embedding Model (free option, no API key needed)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=50
CHUNKING_STRATEGY=recursive

# Retrieval Configuration
TOP_K=5
SIMILARITY_THRESHOLD=0.3
USE_RERANKING=True

# Optional: API Keys for paid models
# OPENAI_API_KEY=your_key_here
# COHERE_API_KEY=your_key_here
```

#### Set up Google Drive (Optional):

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Desktop app)
5. Download credentials as `config/credentials.json`

### 3. Usage

#### Ingest Documents from Google Drive

```bash
# Get folder ID from Google Drive URL:
# https://drive.google.com/drive/folders/YOUR_FOLDER_ID

python main.py ingest-gdrive --folder-id "YOUR_FOLDER_ID"
```

#### Ingest Local Documents

```bash
python main.py ingest-local --directory ./data/raw
```

#### Search Documents

```bash
# Basic search
python main.py search --query "What is Atisara?" --top-k 5

# Search without reranking (faster)
python main.py search --query "treatment guidelines" --top-k 10 --no-rerank
```

#### Evaluate System Performance

```bash
python main.py evaluate --num-test-queries 100
```

#### View System Statistics

```bash
python main.py stats
```

#### Reset Database

```bash
python main.py reset
```

## 📊 Search Results Example

```
🔍 Search Results for: 'what is Atisara'

📄 Result 1 (Similarity: 0.5503)
Source: ASTG_Book.pdf
Content: INTRODUCTION Atisara is an acute gastrointestinal disorder...
Rerank Score: 7.8921
```

## 🎛️ Configuration Options

### Embedding Models

**Free (HuggingFace):**
- `sentence-transformers/all-MiniLM-L6-v2` (default, 384 dims)
- `BAAI/bge-small-en-v1.5` (384 dims)
- `sentence-transformers/all-mpnet-base-v2` (768 dims)

**Paid:**
- `text-embedding-3-small` (OpenAI)
- `embed-english-v3.0` (Cohere)

### Chunking Strategies

- **recursive** (default) - Splits on paragraphs, then sentences
- **fixed** - Fixed-size chunks with overlap
- **semantic** - Meaning-based splitting (requires more processing)

### Retrieval Modes

1. **Basic Retrieval** - Cosine similarity search
2. **MMR (Maximal Marginal Relevance)** - Balances relevance and diversity
3. **Hybrid Search** - Combines semantic + keyword search
4. **Reranking** - Cross-encoder refinement (recommended)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_embeddings.py

# Run with coverage
pytest --cov=src tests/
```

## 📈 Evaluation Metrics

The system provides comprehensive retrieval evaluation:

- **Precision@K** - Relevance of top-K results
- **Recall@K** - Coverage of relevant documents
- **MRR (Mean Reciprocal Rank)** - Position of first relevant result
- **MAP (Mean Average Precision)** - Overall precision across queries
- **NDCG (Normalized Discounted Cumulative Gain)** - Ranked relevance quality
- **Hit Rate** - Percentage of queries with at least one relevant result

## 🛠️ Advanced Usage

### Custom Similarity Threshold

Lower threshold for more results (may include less relevant):
```bash
# In .env file
SIMILARITY_THRESHOLD=0.2
```

Higher threshold for precision (may miss some results):
```bash
SIMILARITY_THRESHOLD=0.5
```

### Batch Processing

```python
from src.document_processor.loader import DocumentLoader
from src.document_processor.chunker import OptimizedChunker

# Load and process documents
loader = DocumentLoader()
docs = loader.load_directory("./data/raw")

chunker = OptimizedChunker(strategy="recursive")
chunks = chunker.chunk_documents(docs)
```

### Programmatic Search

```python
from main import RAGAgent

# Initialize agent
agent = RAGAgent()

# Search
results = agent.search(
    query="What are the symptoms?",
    top_k=10,
    use_reranking=True
)

for result in results:
    print(f"Score: {result['similarity']}")
    print(f"Content: {result['content']}")
```

## 🔧 Troubleshooting

### Common Issues

**1. No search results found:**
- Lower `SIMILARITY_THRESHOLD` in `.env` (try 0.2 or 0.3)
- Clear Python cache: `Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force`
- Ensure documents are properly ingested: `python main.py stats`

**2. Google Drive authentication fails:**
- Delete `config/token.json` and re-authenticate
- Verify `config/credentials.json` is valid JSON
- Check OAuth scopes include `drive.readonly`

**3. Out of memory errors:**
- Reduce `BATCH_SIZE` in settings.py
- Use smaller embedding model
- Process documents in smaller batches

**4. Slow performance:**
- Use GPU for embeddings (change `device: 'cuda'` in embedding_manager.py)
- Reduce reranking candidates
- Use smaller, faster embedding model

## 📚 Dependencies

Core libraries:
- `langchain` & `langchain-community` - Document processing
- `sentence-transformers` - Embeddings and reranking
- `chromadb` - Vector database
- `google-api-python-client` - Google Drive integration
- `pypdf`, `python-docx`, `python-pptx` - Document loaders

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- Vector store by [ChromaDB](https://www.trychroma.com/)
- Document processing by Google Drive API

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Made with ❤️ for advanced document retrieval**
