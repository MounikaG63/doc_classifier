# Weaviate Document Classifier Guide

## Overview
s
- Legal: Lease Agreements, Sale Deeds, Title Deeds
- Business: Invoices, Purchase Orders, Balance Sheets, P&L Statements
- Employment: Appointment Letters, Experience Letters, Relieving Letters

## Installation & Setup

### 1. Install Weaviate

#### Option A: Quick Start Scripts (Recommended)

**Windows:**
```bash 
The Weaviate Document Classifier is an advanced document classification system that combines:
- **Weaviate Vector Database** for semantic similarity search
- **Hybrid Classification** using vector similarity + keyword matching
- **Multiple OCR Engines** (Tesseract, EasyOCR, PaddleOCR, Surya)
- **Consensus Detection** to optimize classification accuracy

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   OCR Engine     │    │   Weaviate      │
│   Upload        │───▶│   (4 options)    │───▶│   Vector DB     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Final         │◀───│   Hybrid         │◀───│   Vector        │
│   Result        │    │   Classifier     │    │   Search        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Keyword        │    │   Consensus     │
                       │   Matching       │    │   Detection     │
                       └──────────────────┘    └─────────────────┘
```

## Features

### 1. Weaviate Integration
- **Vector Storage**: Documents stored as embeddings in Weaviate
- **Semantic Search**: Find similar documents using vector similarity
- **Scalable**: Handles large document collections efficiently
- **Schema Management**: Automatic schema creation and management

### 2. Hybrid Classification
- **Vector Similarity**: Primary classification method using embeddings
- **Keyword Matching**: Fallback using document-specific keywords
- **Consensus Detection**: Skip keyword matching when 3+ of top 5 results agree
- **Combined Scoring**: Intelligent combination of vector and keyword scores

### 3. OCR Engine Support
- **Tesseract**: Fast, reliable OCR
- **EasyOCR**: Balanced accuracy and speed
- **PaddleOCR**: High accuracy, production-ready
- **Surya**: Advanced OCR with layout understanding

### 4. Document Types Supported
- Identity Documents: Aadhaar, PAN, Passport, Driving License, Voter ID
- Financial: Bank Statements, Salary Slips, Tax Returns, GST Returns
- Utility Bills: Electricity, Water, Ga
# Run the provided batch script
start_weaviate.bat
```

**Linux/Mac:**
```bash
# Run the provided shell script
chmod +x start_weaviate.sh
./start_weaviate.sh
```

#### Option B: Manual Docker Command

**Windows:**
```bash
docker run -d ^
  --name weaviate-classifier ^
  -p 8080:8080 ^
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true ^
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate ^
  -e DEFAULT_VECTORIZER_MODULE=none ^
  -e ENABLE_MODULES= ^
  -e CLUSTER_HOSTNAME=node1 ^
  -v "%cd%\weaviate_data:/var/lib/weaviate" ^
  semitechnologies/weaviate:latest
```

**Linux/Mac:**
```bash
docker run -d \
  --name weaviate-classifier \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e ENABLE_MODULES= \
  -e CLUSTER_HOSTNAME=node1 \
  -v "$(pwd)/weaviate_data:/var/lib/weaviate" \
  semitechnologies/weaviate:latest
```

#### Option C: Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: semitechnologies/weaviate:latest
    ports:
    - 8080:8080
    - 50051:50051
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      ENABLE_MODULES: ''
      CLUSTER_HOSTNAME: 'node1'
```

Run: `docker-compose up -d`

#### Important Docker Configuration Notes

**Key Environment Variables:**
- `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`: Allows access without authentication
- `PERSISTENCE_DATA_PATH=/var/lib/weaviate`: Enables data persistence
- `DEFAULT_VECTORIZER_MODULE=none`: We provide our own embeddings
- `ENABLE_MODULES=`: Disables unnecessary modules for better performance

**Data Persistence:**
- The `-v` flag mounts a local `weaviate_data` directory
- This ensures your data survives container restarts
- **Critical**: Without this, you'll lose all data when the container stops

**Container Management:**
```bash
# Check if Weaviate is running
docker ps | grep weaviate

# View logs
docker logs weaviate-classifier

# Stop Weaviate
docker stop weaviate-classifier

# Restart Weaviate
docker start weaviate-classifier

# Remove container (data persists in weaviate_data folder)
docker rm weaviate-classifier
```

### 2. Install Python Dependencies
```bash
pip install weaviate-client sentence-transformers
```

### 3. Configure Environment Variables (Optional)
```bash
export WEAVIATE_URL=http://localhost:8080
export WEAVIATE_API_KEY=your_api_key  # For cloud instances
```

## Usage

### 1. Start Weaviate
Ensure Weaviate is running on `http://localhost:8080`

### 2. Test Connection
```bash
python test_weaviate_classifier.py
```

### 3. Migrate from ChromaDB (Optional)
If you have existing ChromaDB data:
```bash
python migrate_chromadb_to_weaviate.py
```

### 4. Start Django Server
```bash
python manage.py runserver
```

### 5. Access Web Interface
Visit: `http://127.0.0.1:8000/weaviate/`

## API Endpoints

### Classification
```http
POST /weaviate/classify/
Content-Type: multipart/form-data

file: <document_file>
ocr_engine: tesseract|easyocr|paddleocr|surya
```

### Database Management
```http
# Get statistics
GET /weaviate/stats/

# Health check
GET /weaviate/health/

# Rebuild database
POST /weaviate/rebuild/
Content-Type: application/json
{
  "ocr_engine": "paddleocr"
}

# Add new files
POST /weaviate/add-files/
Content-Type: application/json
{
  "ocr_engine": "paddleocr"
}
```

## Classification Logic

### 1. Vector Similarity Search
```python
# Get top 5 similar documents
vector_results = weaviate_store.search_similar(text, limit=5)
```

### 2. Consensus Detection
```python
# Check if 3+ of top 5 results are same document type
type_counts = Counter([result["label"] for result in vector_results])
most_common_type, count = type_counts.most_common(1)[0]

if count >= 3:
    # Use consensus result, skip keyword matching
    return consensus_result
```

### 3. Keyword Matching (if no consensus)
```python
# Apply keyword rules to unique document types
for doc_type in unique_types:
    keyword_score = calculate_keyword_score(text, doc_type)

# Use keyword result if score > 30%, otherwise fall back to vector
```

### 4. Response Format
```json
{
  "document_type": "electricity_bill",
  "confidence": 87.5,
  "method": "keyword",
  "vector_score": 45.2,
  "keyword_score": 87.5,
  "details": "Strong keyword match (87.5%)",
  "processing_time": 2.3,
  "top_3_matches": [
    {
      "rank": 1,
      "type": "electricity_bill",
      "vector_score": 45.2,
      "keyword_score": 87.5,
      "combined_score": 87.5
    }
  ]
}
```

## Performance Optimization

### 1. Weaviate Configuration
- **Memory**: Allocate sufficient RAM for vector storage
- **CPU**: Multi-core processing for better performance
- **Disk**: SSD recommended for faster I/O

### 2. Batch Operations
```python
# Use batch uploads for better performance
weaviate_store.add_documents_batch(documents)
```

### 3. OCR Engine Selection
- **Tesseract**: Fastest, good for clear text
- **PaddleOCR**: Best accuracy/speed balance
- **EasyOCR**: Good for handwritten text
- **Surya**: Best for complex layouts

## Troubleshooting

### Common Issues

#### 1. Weaviate Connection Failed
```bash
# Check if Weaviate is running
curl http://localhost:8080/v1/meta

# Start Weaviate
docker run -p 8080:8080 semitechnologies/weaviate:latest
```

#### 2. Schema Creation Failed
```python
# Manually recreate schema
weaviate_store.clear_all()  # This recreates schema
```

#### 3. Low Classification Accuracy
- Rebuild database with better OCR engine
- Add more training samples
- Check keyword rules for document types

#### 4. Slow Performance
- Use batch operations for bulk uploads
- Optimize Weaviate configuration
- Consider using faster OCR engine

### Debug Mode
Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison: ChromaDB vs Weaviate

| Feature | ChromaDB | Weaviate |
|---------|----------|----------|
| **Setup** | Simple file-based | Requires server |
| **Scalability** | Limited | Highly scalable |
| **Performance** | Good for small datasets | Excellent for large datasets |
| **Features** | Basic vector search | Advanced vector database |
| **Production** | Development/testing | Production-ready |
| **Memory Usage** | Lower | Higher |
| **Query Speed** | Fast for small data | Consistently fast |
| **Backup** | File copy | Database backup tools |

## Migration Guide

### From ChromaDB to Weaviate
1. **Backup ChromaDB**: Copy the `db/` directory
2. **Start Weaviate**: Ensure server is running
3. **Run Migration**: `python migrate_chromadb_to_weaviate.py`
4. **Verify Results**: Check document counts and test searches
5. **Update Application**: Switch to Weaviate endpoints

### Rollback Plan
If migration fails:
1. Keep ChromaDB data intact
2. Use original classifier endpoints
3. Debug Weaviate issues
4. Retry migration when ready

## Best Practices

### 1. Data Management
- Regular backups of Weaviate data
- Monitor database size and performance
- Clean up duplicate or low-quality documents

### 2. Classification Accuracy
- Use consistent OCR engine for training and inference
- Regularly update keyword rules
- Add new document samples to improve coverage

### 3. Performance
- Use batch operations for bulk data
- Monitor Weaviate resource usage
- Optimize schema for your use case

### 4. Monitoring
- Set up health checks
- Monitor classification accuracy
- Track processing times

## Future Enhancements

### Planned Features
1. **Multi-language Support**: Extend to non-English documents
2. **Custom Embeddings**: Train domain-specific embeddings
3. **Active Learning**: Improve classification with user feedback
4. **Distributed Setup**: Multi-node Weaviate cluster
5. **Advanced Analytics**: Classification confidence trends
6. **API Rate Limiting**: Production-ready API controls

### Integration Options
- **REST API**: Standalone classification service
- **GraphQL**: Advanced querying capabilities
- **Webhooks**: Real-time classification notifications
- **Batch Processing**: Large-scale document processing

## Support

For issues and questions:
1. Check this guide first
2. Review error logs
3. Test with `test_weaviate_classifier.py`
4. Check Weaviate documentation
5. Verify OCR engine installation

## License

This project is part of the RAG Document Classifier system.