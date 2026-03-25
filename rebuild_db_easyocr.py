#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to rebuild the database using EasyOCR (most stable)
This ensures consistent embeddings for classification
"""
import os
import sys
import django

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
django.setup()

from classifier.views import (
    extract_text_easyocr, SAMPLES_PATH, DB_PATH
)
import chromadb

def rebuild_with_easyocr():
    print("[*] Rebuilding database with EasyOCR...")
    
    # Delete old database completely
    import shutil
    if os.path.exists(DB_PATH):
        try:
            shutil.rmtree(DB_PATH)
            print("[+] Deleted old database directory")
        except Exception as e:
            print(f"[!] Could not delete old DB: {e}")
    
    # Create fresh client and collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma_client.get_or_create_collection("text_docs")
    print("[+] Created new collection")
    
    # Get resources (embedder and easy_ocr)
    from sentence_transformers import SentenceTransformer
    import easyocr
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    easy_ocr = easyocr.Reader(["en"], gpu=False)
    print("[+] Loaded embedder and EasyOCR")
    
    added = 0
    total = 0
    failed = 0
    
    for label in sorted(os.listdir(SAMPLES_PATH)):
        folder = os.path.join(SAMPLES_PATH, label)
        if not os.path.isdir(folder):
            continue
        
        print(f"\n[*] Processing {label}...")
        files = os.listdir(folder)
        
        for f in files:
            total += 1
            path = os.path.join(folder, f)
            
            try:
                text = extract_text_easyocr(path, easy_ocr)
                
                if not text.strip() or text.startswith("ERROR"):
                    print(f"  [-] {f}: {text[:50]}")
                    failed += 1
                    continue
                
                emb = embedder.encode([text])[0].tolist()
                file_id = f"{label}_{f}"
                
                # Add to collection with error handling
                try:
                    collection.add(
                        ids=[file_id], 
                        embeddings=[emb],
                        metadatas=[{"label": label, "ocr_engine": "easyocr"}], 
                        documents=[text]
                    )
                    added += 1
                    print(f"  [+] {f} ({len(text)} chars)")
                except Exception as add_err:
                    print(f"  [-] {f}: Add error - {str(add_err)[:50]}")
                    failed += 1
                
            except Exception as e:
                print(f"  [-] {f}: {str(e)[:50]}")
                failed += 1
    
    print(f"\n{'='*50}")
    print(f"[+] Database rebuild complete!")
    print(f"  Total files: {total}")
    print(f"  Added: {added}")
    print(f"  Failed: {failed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    rebuild_with_easyocr()
