#!/usr/bin/env python
"""
Script to rebuild the database using PaddleOCR
Run this after fixing the PaddleOCR extraction
"""
import os
import sys
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
django.setup()

from classifier.views import (
    extract_text_paddleocr, get_resources, SAMPLES_PATH, DB_PATH
)
import chromadb

def rebuild_with_paddleocr():
    print("🔄 Rebuilding database with PaddleOCR...")
    
    # Get resources
    collection, embedder, easy_ocr, paddle_ocr, surya_ocr = get_resources()
    
    # Delete old collection
    try:
        chromadb.PersistentClient(path=DB_PATH).delete_collection("text_docs")
        print("✓ Deleted old collection")
    except:
        pass
    
    # Recreate collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection("text_docs")
    print("✓ Created new collection")
    
    added = 0
    total = 0
    failed = 0
    
    for label in sorted(os.listdir(SAMPLES_PATH)):
        folder = os.path.join(SAMPLES_PATH, label)
        if not os.path.isdir(folder):
            continue
        
        print(f"\n📁 Processing {label}...")
        files = os.listdir(folder)
        
        for f in files:
            total += 1
            path = os.path.join(folder, f)
            
            try:
                text = extract_text_paddleocr(path, paddle_ocr)
                
                if not text.strip() or text.startswith("ERROR"):
                    print(f"  ✗ {f}: {text[:50]}")
                    failed += 1
                    continue
                
                emb = embedder.encode([text])[0].tolist()
                file_id = f"{label}_{f}"
                collection.add(
                    ids=[file_id], 
                    embeddings=[emb],
                    metadatas=[{"label": label, "ocr_engine": "paddleocr"}], 
                    documents=[text]
                )
                added += 1
                print(f"  ✓ {f} ({len(text)} chars)")
                
            except Exception as e:
                print(f"  ✗ {f}: {str(e)[:50]}")
                failed += 1
    
    print(f"\n{'='*50}")
    print(f"✓ Database rebuild complete!")
    print(f"  Total files: {total}")
    print(f"  Added: {added}")
    print(f"  Failed: {failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    rebuild_with_paddleocr()
