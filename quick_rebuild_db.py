#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import django
import shutil

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
django.setup()

from classifier.views import extract_text_easyocr, SAMPLES_PATH, DB_PATH
import chromadb
from sentence_transformers import SentenceTransformer
import easyocr

# Delete old DB
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print("Deleted old database")

# Create new DB
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection("text_docs")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
easy_ocr = easyocr.Reader(["en"], gpu=False)

print("Starting rebuild...")
added = 0
failed = 0

for label in sorted(os.listdir(SAMPLES_PATH)):
    folder = os.path.join(SAMPLES_PATH, label)
    if not os.path.isdir(folder):
        continue
    
    print(f"Processing {label}...", end=" ", flush=True)
    count = 0
    
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            text = extract_text_easyocr(path, easy_ocr)
            if text.strip() and not text.startswith("ERROR"):
                emb = embedder.encode([text])[0].tolist()
                collection.add(
                    ids=[f"{label}_{f}"],
                    embeddings=[emb],
                    metadatas=[{"label": label, "ocr_engine": "easyocr"}],
                    documents=[text]
                )
                added += 1
                count += 1
        except:
            failed += 1
    
    print(f"OK ({count} files)")

print(f"\nDone! Added: {added}, Failed: {failed}")
