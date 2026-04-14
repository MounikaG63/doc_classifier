from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from rapidocr_onnxruntime import RapidOCR
from PIL import Image
import fitz
import time
import zipfile
import tempfile
import shutil
import threading

DB_PATH = "db"
SAMPLES_PATH = "samples"

chroma_client = None
collection = None
embedder = None
rapid_ocr = None
resource_lock = threading.Lock()


def get_resources():
    global chroma_client, collection, embedder, rapid_ocr
    with resource_lock:
        if chroma_client is None:
            chroma_client = chromadb.PersistentClient(path=DB_PATH)
            collection = chroma_client.get_or_create_collection("text_docs")
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            
            import sys
            try:
                print("Attempting to initialize RapidOCR...", file=sys.stderr)
                rapid_ocr = RapidOCR()
                print("✓ RapidOCR initialized successfully", file=sys.stderr)
            except Exception as e:
                print(f"✗ Failed to initialize RapidOCR: {e}", file=sys.stderr)
                rapid_ocr = None
                
    return collection, embedder, rapid_ocr


def pdf_to_images(pdf_path, return_pil=False):
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        if return_pil:
            try:
                from PIL import Image as PILImage
                import io
                img_data = pix.tobytes("png")
                images.append(PILImage.open(io.BytesIO(img_data)).convert("RGB"))
            except Exception as e:
                print(f"Error converting page {i} to PIL: {e}")
        else:
            out_path = f"{pdf_path}_page_{i}.png"
            pix.save(out_path)
            images.append(out_path)
    doc.close()
    return images


def extract_text_from_pdf_direct(path):
    try:
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += " " + page.get_text()
        doc.close()
        return full_text.strip()
    except:
        return ""


def extract_text_rapidocr(path, rapid_ocr):
    """Extract text using RapidOCR with layout-aware sorting"""
    try:
        if rapid_ocr is None:
            return "ERROR: RapidOCR not initialized"
        if path.lower().endswith(".pdf"):
            direct_text = extract_text_from_pdf_direct(path)
            if direct_text and len(direct_text) > 50:
                return direct_text
            pages = pdf_to_images(path)
            full_text = ""
            for img in pages:
                results, _ = rapid_ocr(img)
                if results:
                    # Sort results by y then x
                    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
                    full_text += " " + " ".join([line[1] for line in results])
                if os.path.exists(img):
                    os.remove(img)
            return full_text.strip()
        results, _ = rapid_ocr(path)
        if results:
            # Sort results by y then x
            results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
            return " ".join([line[1] for line in results])
        return "ERROR: No text found in image"
    except Exception as e:
        return f"ERROR: {str(e)}"


def index(request):
    return render(request, 'classifier/index.html')


@csrf_exempt
def classify_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    ext = file.name.split('.')[-1].lower()
    temp_path = f"temp_upload.{ext}"
    
    results = {}
    try:
        with open(temp_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        collection, embedder, rapid_ocr = get_resources()
        start_time = time.time()
        
        # Extract text using RapidOCR only
        extracted_text = extract_text_rapidocr(temp_path, rapid_ocr)
        engine_name = "RapidOCR"
        
        processing_time = time.time() - start_time
        if extracted_text.strip() and not extracted_text.startswith("ERROR"):
            text_vec = embedder.encode([extracted_text])[0].tolist()
            try:
                # Simple vector similarity search
                query_res = collection.query(query_embeddings=[text_vec], n_results=5)
                
                if query_res["metadatas"] and query_res["metadatas"][0]:
                    best_match = query_res["metadatas"][0][0]
                    best_distance = query_res["distances"][0][0]
                    confidence = (1 - best_distance) * 100
                    
                    top_3_matches = []
                    for i in range(min(3, len(query_res["metadatas"][0]))):
                        doc_type = query_res["metadatas"][0][i]["label"]
                        vector_score = (1 - query_res["distances"][0][i]) * 100
                        
                        top_3_matches.append({
                            "rank": i + 1,
                            "type": doc_type,
                            "vector_score": round(vector_score, 2),
                            "confidence": round(vector_score, 2)
                        })
                    
                    results = {
                        "ocr_engine": engine_name,
                        "document_type": best_match["label"],
                        "confidence": round(confidence, 2),
                        "classification_method": "vector_similarity",
                        "ocr_text": extracted_text[:1000],
                        "processing_time": round(processing_time, 2),
                        "top_3_matches": top_3_matches
                    }
                else:
                    results = {
                        "ocr_engine": engine_name,
                        "document_type": "unknown",
                        "reason": "no_match_in_database"
                    }
            except Exception as e:
                results = {
                    "ocr_engine": engine_name,
                    "document_type": "unknown",
                    "reason": f"vector_search_error: {str(e)}"
                }
        else:
            try:
                from .gemini_classifier import classify_with_gemini
                gemini_result = classify_with_gemini(temp_path)
                if "error" not in gemini_result:
                    results = {
                        "ocr_engine": f"{engine_name} + Gemini (fallback)",
                        "document_type": gemini_result.get("doc_type", "unknown"),
                        "category": gemini_result.get("category", ""),
                        "sub_category": gemini_result.get("sub_category", ""),
                        "confidence": gemini_result.get("confidence", ""),
                        "processing_time": round(processing_time, 2),
                        "gemini_used": True,
                        "reason": "OCR failed (or text too short), classified using Gemini Vision"
                    }
                else:
                    results = {
                        "ocr_engine": engine_name,
                        "document_type": "unknown",
                        "reason": f"ocr_failed and gemini_error: {gemini_result.get('error', 'Unknown')}"
                    }
            except Exception as e:
                results = {
                    "ocr_engine": engine_name,
                    "document_type": "unknown",
                    "reason": f"ocr_failed: {extracted_text[:100]}, gemini_error: {str(e)}"
                }
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JsonResponse(results)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        import traceback
        return JsonResponse({'error': f'Server error: {str(e)}', 'traceback': traceback.format_exc()[:500]}, status=500)


@csrf_exempt
def classify_zip(request):
    """Handle ZIP file uploads containing multiple documents using RapidOCR"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    if not file.name.lower().endswith('.zip'):
        return JsonResponse({'error': 'File must be a ZIP archive'}, status=400)
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    for chunk in file.chunks():
        temp_zip.write(chunk)
    temp_zip.close()
    
    temp_dir = tempfile.mkdtemp()
    collection, embedder, rapid_ocr = get_resources()
    all_results = []
    supported_extensions = {'.jpg', '.jpeg', '.png', '.pdf'}
    
    try:
        with zipfile.ZipFile(temp_zip.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for root, dirs, files in os.walk(temp_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported_extensions:
                    continue
                
                file_path = os.path.join(root, filename)
                start_time = time.time()
                
                extracted_text = extract_text_rapidocr(file_path, rapid_ocr)
                engine_name = "RapidOCR"
                
                processing_time = time.time() - start_time
                result = {"filename": filename, "ocr_engine": engine_name}
                
                if extracted_text.strip() and not extracted_text.startswith("ERROR"):
                    text_vec = embedder.encode([extracted_text])[0].tolist()
                    try:
                        query_res = collection.query(query_embeddings=[text_vec], n_results=3)
                        if query_res["metadatas"] and query_res["metadatas"][0]:
                            top = query_res["metadatas"][0][0]
                            dist = query_res["distances"][0][0]
                            similarity_score = round((1 - dist) * 100, 2)
                            if similarity_score < 0:
                                result.update({"document_type": "Miscellaneous", "similarity_score": similarity_score,
                                               "processing_time": round(processing_time, 2)})
                            else:
                                result.update({
                                    "document_type": top["label"], "similarity_score": similarity_score,
                                    "processing_time": round(processing_time, 2),
                                    "top_3_matches": [
                                        {"type": query_res["metadatas"][0][i]["label"],
                                         "score": round((1 - query_res["distances"][0][i]) * 100, 2)}
                                        for i in range(min(3, len(query_res["metadatas"][0])))
                                    ]
                                })
                        else:
                            result.update({"document_type": "unknown", "reason": "no_match"})
                    except Exception as e:
                        result.update({"document_type": "unknown", "reason": f"query_error: {str(e)}"})
                else:
                    try:
                        from .gemini_classifier import classify_with_gemini
                        gemini_result = classify_with_gemini(file_path)
                        if "error" not in gemini_result:
                            result.update({
                                "ocr_engine": f"{engine_name} + Gemini (fallback)",
                                "document_type": gemini_result.get("doc_type", "unknown"),
                                "category": gemini_result.get("category", ""),
                                "confidence": gemini_result.get("confidence", ""),
                                "processing_time": round(processing_time, 2), "gemini_used": True
                            })
                        else:
                            result.update({"document_type": "unknown",
                                           "reason": f"ocr_failed, gemini_error: {gemini_result.get('error', '')}"})
                    except Exception as e:
                        result.update({"document_type": "unknown", "reason": f"ocr_failed: {str(e)}"})
                
                all_results.append(result)
        
        return JsonResponse({"success": True, "total_files": len(all_results), "results": all_results})
    except zipfile.BadZipFile:
        return JsonResponse({'error': 'Invalid ZIP file'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Processing error: {str(e)}'}, status=500)
    finally:
        os.unlink(temp_zip.name)
        shutil.rmtree(temp_dir, ignore_errors=True)


@csrf_exempt
def rebuild_database(request):
    """Rebuild database using RapidOCR"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    global chroma_client, collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        chroma_client.delete_collection("text_docs")
    except:
        pass
    
    collection = chroma_client.get_or_create_collection("text_docs")
    collection, embedder, rapid_ocr = get_resources()
    
    added = 0
    total = 0
    
    for label in os.listdir(SAMPLES_PATH):
        folder = os.path.join(SAMPLES_PATH, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            total += 1
            path = os.path.join(folder, f)
            text = extract_text_rapidocr(path, rapid_ocr)
            
            if not text.strip() or text.startswith("ERROR"):
                continue
            
            emb = embedder.encode([text])[0].tolist()
            file_id = f"{label}_{f}"
            collection.add(ids=[file_id], embeddings=[emb],
                           metadatas=[{"label": label, "ocr_engine": "rapidocr"}], documents=[text])
            added += 1
    
    return JsonResponse({'success': True, 'added': added, 'total': total})


@csrf_exempt
def add_new_files(request):
    """Add new files to database using RapidOCR"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    collection, embedder, rapid_ocr = get_resources()
    existing_ids = set(collection.get().get("ids", []))
    added = 0
    
    for label in os.listdir(SAMPLES_PATH):
        folder = os.path.join(SAMPLES_PATH, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            doc_id = f"{label}_{f}"
            if doc_id in existing_ids:
                continue
            path = os.path.join(folder, f)
            text = extract_text_rapidocr(path, rapid_ocr)
            
            if not text.strip() or text.startswith("ERROR"):
                continue
            
            emb = embedder.encode([text])[0].tolist()
            collection.add(ids=[doc_id], embeddings=[emb],
                           metadatas=[{"label": label, "ocr_engine": "rapidocr"}], documents=[text])
            added += 1
    
    return JsonResponse({'success': True, 'added': added})


def get_db_stats(request):
    collection, _, _ = get_resources()
    try:
        doc_count = collection.count()
        all_docs = collection.get()
        type_counts = {}
        for meta in all_docs["metadatas"]:
            label = meta["label"]
            type_counts[label] = type_counts.get(label, 0) + 1
        return JsonResponse({'total': doc_count, 'by_type': type_counts})
    except:
        return JsonResponse({'total': 0, 'by_type': {}})
