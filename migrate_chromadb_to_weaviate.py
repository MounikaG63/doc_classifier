#!/usr/bin/env python3
"""
Migration script to transfer data from ChromaDB to Weaviate
"""

import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
django.setup()

import chromadb
from weaviate_classifier.weaviate_client import get_weaviate_store


def migrate_data():
    """Migrate data from ChromaDB to Weaviate"""
    print("🔄 CHROMADB TO WEAVIATE MIGRATION")
    print("=" * 60)
    
    try:
        # Initialize ChromaDB
        print("📂 Connecting to ChromaDB...")
        chroma_client = chromadb.PersistentClient(path="db")
        chroma_collection = chroma_client.get_collection("text_docs")
        
        # Get all documents from ChromaDB
        print("📊 Fetching documents from ChromaDB...")
        all_docs = chroma_collection.get()
        
        if not all_docs["documents"]:
            print("❌ No documents found in ChromaDB!")
            return False
        
        total_docs = len(all_docs["documents"])
        print(f"📋 Found {total_docs} documents in ChromaDB")
        
        # Initialize Weaviate
        print("🔗 Connecting to Weaviate...")
        weaviate_store = get_weaviate_store()
        
        # Check Weaviate health
        if not weaviate_store.health_check():
            print("❌ Weaviate is not healthy! Please start Weaviate server with persistence:")
            print("Windows:")
            print("docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -v %cd%/weaviate_data:/var/lib/weaviate semitechnologies/weaviate:latest")
            print("Linux/Mac:")
            print("docker run -d -p 8080:8080 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -v $(pwd)/weaviate_data:/var/lib/weaviate semitechnologies/weaviate:latest")
            return False
        
        print("✅ Weaviate connection successful")
        
        # Clear existing Weaviate data
        print("🗑️ Clearing existing Weaviate data...")
        weaviate_store.clear_all()
        
        # Prepare documents for batch upload
        print("📦 Preparing documents for migration...")
        documents_to_migrate = []
        
        for i, (doc_text, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
            if not doc_text or not metadata:
                continue
            
            # Extract filename from ID (format: label_filename)
            doc_id = all_docs["ids"][i]
            filename = doc_id.split("_", 1)[1] if "_" in doc_id else doc_id
            
            document = {
                "text": doc_text,
                "label": metadata["label"],
                "filename": filename,
                "ocr_engine": metadata.get("ocr_engine", "unknown")
            }
            documents_to_migrate.append(document)
        
        print(f"📋 Prepared {len(documents_to_migrate)} documents for migration")
        
        # Batch upload to Weaviate
        print("⬆️ Uploading documents to Weaviate...")
        weaviate_store.add_documents_batch(documents_to_migrate)
        
        # Verify migration
        print("✅ Verifying migration...")
        weaviate_stats = weaviate_store.get_stats()
        
        print(f"📊 Migration Results:")
        print(f"   ChromaDB documents: {total_docs}")
        print(f"   Weaviate documents: {weaviate_stats['total']}")
        print(f"   Success rate: {(weaviate_stats['total'] / total_docs) * 100:.1f}%")
        
        if weaviate_stats['by_type']:
            print(f"\n📋 Document types in Weaviate:")
            for doc_type, count in sorted(weaviate_stats['by_type'].items()):
                print(f"   - {doc_type}: {count}")
        
        if weaviate_stats['total'] == total_docs:
            print("\n🎉 Migration completed successfully!")
        else:
            print(f"\n⚠️ Migration partially successful. {total_docs - weaviate_stats['total']} documents may have failed.")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_search_results():
    """Compare search results between ChromaDB and Weaviate"""
    print("\n" + "=" * 60)
    print("🔍 SEARCH COMPARISON TEST")
    print("=" * 60)
    
    try:
        # Initialize both databases
        chroma_client = chromadb.PersistentClient(path="db")
        chroma_collection = chroma_client.get_collection("text_docs")
        weaviate_store = get_weaviate_store()
        
        # Test query
        test_query = "electricity bill consumer number meter reading units consumed"
        print(f"Test Query: {test_query}")
        
        # Search in ChromaDB
        print("\n📊 ChromaDB Results:")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = embedder.encode([test_query])[0].tolist()
        
        chroma_results = chroma_collection.query(query_embeddings=[query_vec], n_results=3)
        
        if chroma_results["metadatas"] and chroma_results["metadatas"][0]:
            for i, (metadata, distance) in enumerate(zip(chroma_results["metadatas"][0], chroma_results["distances"][0])):
                similarity = (1 - distance) * 100
                print(f"   {i+1}. {metadata['label']} - {similarity:.2f}%")
        else:
            print("   No results found")
        
        # Search in Weaviate
        print("\n🔗 Weaviate Results:")
        weaviate_results = weaviate_store.search_similar(test_query, limit=3)
        
        if weaviate_results:
            for i, result in enumerate(weaviate_results):
                similarity = result["similarity"] * 100
                print(f"   {i+1}. {result['label']} - {similarity:.2f}%")
        else:
            print("   No results found")
        
        return True
        
    except Exception as e:
        print(f"❌ Search comparison failed: {e}")
        return False


def main():
    """Main migration function"""
    print("🚀 CHROMADB TO WEAVIATE MIGRATION TOOL")
    print("=" * 60)
    
    # Check if ChromaDB exists
    if not os.path.exists("db"):
        print("❌ ChromaDB not found! Please ensure the 'db' directory exists.")
        return
    
    # Perform migration
    if not migrate_data():
        print("\n❌ Migration failed!")
        return
    
    # Compare search results
    if not compare_search_results():
        print("\n⚠️ Search comparison failed, but migration may have succeeded.")
    
    print("\n" + "=" * 60)
    print("✅ MIGRATION COMPLETED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the Weaviate classifier: python test_weaviate_classifier.py")
    print("2. Start Django server: python manage.py runserver")
    print("3. Visit: http://127.0.0.1:8000/weaviate/")


if __name__ == "__main__":
    main()