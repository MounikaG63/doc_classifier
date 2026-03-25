"""
Weaviate Client for Document Classification
Handles vector storage and similarity search using Weaviate
"""

import weaviate
import weaviate.classes as wvc
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class WeaviateDocumentStore:
    """Weaviate client for document classification"""
    
    def __init__(self, 
                 url: str = "http://localhost:8080",
                 api_key: str = None,
                 openai_api_key: str = None):
        """
        Initialize Weaviate client
        
        Args:
            url: Weaviate instance URL
            api_key: Weaviate API key (for cloud)
            openai_api_key: OpenAI API key (if using OpenAI embeddings)
        """
        self.url = url
        self.api_key = api_key
        
        # Initialize client (v4 API)
        if api_key:
            self.client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key)
            )
        else:
            self.client = weaviate.connect_to_local(host="localhost", port=8080)
        
        # Initialize sentence transformer for embeddings
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Collection name
        self.collection_name = "DocumentText"
        
        # Create collection if it doesn't exist
        self._create_collection()
    
    def _create_collection(self):
        """Create Weaviate collection for documents"""
        try:
            # Check if collection exists
            if self.client.collections.exists(self.collection_name):
                logger.info(f"Collection already exists: {self.collection_name}")
                self.collection = self.client.collections.get(self.collection_name)
                return
            
            # Create collection with schema
            self.collection = self.client.collections.create(
                name=self.collection_name,
                description="Document text for classification",
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # We provide our own vectors
                properties=[
                    wvc.config.Property(
                        name="text",
                        data_type=wvc.config.DataType.TEXT,
                        description="OCR extracted text from document"
                    ),
                    wvc.config.Property(
                        name="label",
                        data_type=wvc.config.DataType.TEXT,
                        description="Document type label"
                    ),
                    wvc.config.Property(
                        name="filename",
                        data_type=wvc.config.DataType.TEXT,
                        description="Original filename"
                    ),
                    wvc.config.Property(
                        name="text_length",
                        data_type=wvc.config.DataType.INT,
                        description="Length of extracted text"
                    ),
                    wvc.config.Property(
                        name="ocr_engine",
                        data_type=wvc.config.DataType.TEXT,
                        description="OCR engine used for extraction"
                    )
                ]
            )
            logger.info(f"Created Weaviate collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_document(self, text: str, label: str, filename: str = "", ocr_engine: str = ""):
        """
        Add a single document to Weaviate
        
        Args:
            text: OCR extracted text
            label: Document type label
            filename: Original filename
            ocr_engine: OCR engine used
        """
        try:
            # Generate embedding
            vector = self.embedder.encode([text])[0].tolist()
            
            # Prepare data object
            data_object = {
                "text": text,
                "label": label,
                "filename": filename,
                "text_length": len(text),
                "ocr_engine": ocr_engine
            }
            
            # Add to Weaviate
            result = self.collection.data.insert(
                properties=data_object,
                vector=vector
            )
            
            logger.info(f"Added document: {filename} -> {label}")
            return result
            
        except Exception as e:
            logger.error(f"Error adding document {filename}: {e}")
            raise
    
    def add_documents_batch(self, documents: List[Dict[str, str]]):
        """
        Add multiple documents in batch
        
        Args:
            documents: List of dicts with keys: text, label, filename, ocr_engine
        """
        try:
            with self.collection.batch.dynamic() as batch:
                for doc in documents:
                    # Generate embedding
                    vector = self.embedder.encode([doc["text"]])[0].tolist()
                    
                    # Prepare data object
                    data_object = {
                        "text": doc["text"],
                        "label": doc["label"],
                        "filename": doc.get("filename", ""),
                        "text_length": len(doc["text"]),
                        "ocr_engine": doc.get("ocr_engine", "")
                    }
                    
                    batch.add_object(
                        properties=data_object,
                        vector=vector
                    )
            
            logger.info(f"Added {len(documents)} documents in batch")
            
        except Exception as e:
            logger.error(f"Error in batch upload: {e}")
            raise
    
    def search_similar(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_text: Text to search for
            limit: Number of results to return
            
        Returns:
            List of similar documents with metadata and distances
        """
        try:
            # Generate query vector
            query_vector = self.embedder.encode([query_text])[0].tolist()
            
            # Search in Weaviate
            response = self.collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            
            # Format results similar to ChromaDB
            formatted_results = []
            for obj in response.objects:
                formatted_results.append({
                    "text": obj.properties["text"],
                    "label": obj.properties["label"],
                    "filename": obj.properties["filename"],
                    "text_length": obj.properties["text_length"],
                    "ocr_engine": obj.properties["ocr_engine"],
                    "distance": obj.metadata.distance,
                    "similarity": 1 - obj.metadata.distance
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Get total count
            total_count = self.collection.aggregate.over_all(total_count=True).total_count
            
            # Get count by label
            by_label = {}
            response = self.collection.aggregate.over_all(
                group_by="label"
            )
            
            if hasattr(response, 'groups') and response.groups:
                for group in response.groups:
                    label = group.grouped_by.value
                    count = group.total_count
                    by_label[label] = count
            
            return {
                "total": total_count,
                "by_type": by_label
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total": 0, "by_type": {}}
    
    def clear_all(self):
        """Clear all documents from the database"""
        try:
            self.client.collections.delete(self.collection_name)
            self._create_collection()
            logger.info("Cleared all documents from Weaviate")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Weaviate is healthy"""
        try:
            return self.client.is_ready()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def close(self):
        """Close the Weaviate connection"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


# Global instance
_weaviate_store = None

def get_weaviate_store() -> WeaviateDocumentStore:
    """Get global Weaviate store instance"""
    global _weaviate_store
    
    if _weaviate_store is None:
        # Configuration from environment or defaults
        url = os.environ.get('WEAVIATE_URL', 'http://localhost:8080')
        api_key = os.environ.get('WEAVIATE_API_KEY')
        
        _weaviate_store = WeaviateDocumentStore(
            url=url,
            api_key=api_key
        )
    
    return _weaviate_store