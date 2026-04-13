import os
import sys
import json
import time
import chromadb
from sentence_transformers import SentenceTransformer
import concurrent.futures
from tqdm import tqdm

# Import OCR functions from Django views
# Setting up Django environment to allow imports
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
import django
django.setup()

from classifier.views import (
    extract_text_surya,
    extract_text_paddleocr,
    extract_text_easyocr,
    extract_text_tesseract,
    extract_text_rapidocr,
    get_resources,
    DB_PATH,
    SAMPLES_PATH
)

def process_single_file(args):
    """Worker function for multiprocessing"""
    label, filename, ocr_engine = args
    folder = os.path.join(SAMPLES_PATH, label)
    path = os.path.join(folder, filename)
    
    # We need to get resources inside the worker or passed carefully
    # Re-initializing resources in each process is safest for OCR engines
    # but we only do it once per process if we use an initializer.
    global _local_resources
    if '_local_resources' not in globals():
        _local_resources = get_resources()
    
    collection, embedder, easy_ocr, paddle_ocr, surya_ocr, rapid_ocr = _local_resources
    
    try:
        if ocr_engine == "paddleocr":
            text = extract_text_paddleocr(path, paddle_ocr)
        elif ocr_engine == "tesseract":
            text = extract_text_tesseract(path)
        elif ocr_engine == "surya":
            text = extract_text_surya(path, surya_ocr)
        elif ocr_engine == "rapidocr":
            text = extract_text_rapidocr(path, rapid_ocr)
        else:
            text = extract_text_easyocr(path, easy_ocr)
            
        if not text.strip() or text.startswith("ERROR"):
            return None
            
        emb = embedder.encode([text])[0].tolist()
        doc_id = f"{label}_{filename}"
        
        return {
            "id": doc_id,
            "embedding": emb,
            "metadata": {"label": label, "ocr_engine": ocr_engine},
            "document": text
        }
    except Exception as e:
        # Use a more visible error message
        import traceback
        error_msg = f"{str(e)}"
        print(f"\n❌ Error processing {filename}: {error_msg}")
        return None

def main():
    print("🚀 Starting Optimized Incremental Rebuild Script")
    
    # Set environment variables for better CPU performance with parallel OCR
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    ocr_engine = "paddleocr"
    if len(sys.argv) > 1:
        ocr_engine = sys.argv[1].lower()
    
    print(f"🛠️ Selected OCR Engine: {ocr_engine.upper()}")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection("text_docs")
    
    # Get existing IDs to enable incremental updates
    print("🔍 Checking existing documents in database...")
    # Get total count first to know how many IDs to fetch
    total_in_db = collection.count()
    existing_data = collection.get(limit=max(100, total_in_db + 100))
    existing_ids = set(existing_data.get("ids", []))
    print(f"📦 Found {len(existing_ids)} documents already in database.")
    
    # Collect all files that need processing
    tasks = []
    skipped = 0
    for label in os.listdir(SAMPLES_PATH):
        folder = os.path.join(SAMPLES_PATH, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            doc_id = f"{label}_{f}"
            if doc_id in existing_ids:
                skipped += 1
                continue
            tasks.append((label, f, ocr_engine))
    
    total_new = len(tasks)
    print(f"📂 Found {total_new} new documents to index. (Skipped {skipped} existing)")
    
    if total_new == 0:
        print("✅ Database is already up to date!")
        return
    
    results_added = 0
    max_workers = 2 
    print(f"⚡ Processing {total_new} files with {max_workers} parallel workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_new, desc="Updating DB"):
            res = future.result()
            if res:
                collection.add(
                    ids=[res["id"]],
                    embeddings=[res["embedding"]],
                    metadatas=[res["metadata"]],
                    documents=[res["document"]]
                )
                results_added += 1
                
    print(f"\n✅ Update Complete!")
    print(f"📈 New Documents Indexed: {results_added}/{total_new}")
    print(f"📊 Total Documents in DB: {collection.count()}")
    
    # Trigger migration to Weaviate if changes were made
    if results_added > 0:
        print("\n🔄 Detected changes. Triggering migration to Weaviate...")
        try:
            import subprocess
            subprocess.run([sys.executable, "migrate_chromadb_to_weaviate.py"], check=True)
        except Exception as e:
            print(f"⚠️ Error during auto-migration: {e}")
            print("Run 'python migrate_chromadb_to_weaviate.py' manually.")

if __name__ == "__main__":
    main()
