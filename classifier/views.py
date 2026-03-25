from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import easyocr
from paddleocr import PaddleOCR
from PIL import Image
import fitz
import time
import zipfile
import tempfile
import shutil
# Removed hybrid classifier - using simple vector similarity

DB_PATH = "db"
SAMPLES_PATH = "samples"

chroma_client = None
collection = None
embedder = None
easy_ocr = None
surya_ocr = None


def get_resources():
    global chroma_client, collection, embedder, easy_ocr, paddle_ocr, surya_ocr
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_or_create_collection("text_docs")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        easy_ocr = easyocr.Reader(["en"], gpu=False)
        import sys
        try:
            print("Attempting to initialize PaddleOCR...", file=sys.stderr)
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
            print("✓ PaddleOCR initialized successfully", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"✗ Failed to initialize PaddleOCR: {e}", file=sys.stderr)
            paddle_ocr = None
        try:
            print("Attempting to initialize Surya OCR...", file=sys.stderr)
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            foundation = FoundationPredictor()
            detection = DetectionPredictor()  # No argument needed
            recognition = RecognitionPredictor(foundation)
            surya_ocr = {
                'foundation': foundation,
                'recognition': recognition,
                'detection': detection
            }
            print("✓ Surya OCR initialized successfully", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"✗ Failed to initialize Surya OCR: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            surya_ocr = None
    return collection, embedder, easy_ocr, paddle_ocr, surya_ocr


def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        out_path = f"{pdf_path}_page_{i}.png"
        pix.save(out_path)
        image_paths.append(out_path)
    return image_paths


def preprocess_image(path):
    try:
        img = Image.open(path).convert("RGB")
        max_size = 2000
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(path)
    except:
        pass


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


def extract_text_paddleocr(path, paddle_ocr):
    try:
        if path.lower().endswith(".pdf"):
            direct_text = extract_text_from_pdf_direct(path)
            if direct_text and len(direct_text) > 50:
                return direct_text
        result = paddle_ocr.predict(path)
        if not result:
            return "ERROR: PaddleOCR returned empty result"
        all_texts = []
        for page_result in result:
            if hasattr(page_result, 'rec_texts'):
                all_texts.extend(page_result.rec_texts)
            elif isinstance(page_result, dict) and 'rec_texts' in page_result:
                all_texts.extend(page_result['rec_texts'])
        return " ".join(all_texts) if all_texts else "ERROR: No text extracted"
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_text_tesseract(path):
    try:
        import pytesseract
        if path.lower().endswith(".pdf"):
            direct_text = extract_text_from_pdf_direct(path)
            if direct_text and len(direct_text) > 50:
                return direct_text
            pages = pdf_to_images(path)
            full_text = ""
            for img in pages:
                try:
                    image = Image.open(img)
                    full_text += " " + pytesseract.image_to_string(image)
                except Exception as e:
                    full_text += f" [Error: {str(e)}]"
                os.remove(img)
            return full_text.strip() if full_text.strip() else "ERROR: No text extracted from PDF"
        image = Image.open(path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else "ERROR: No text found in image"
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_text_easyocr(path, easy_ocr):
    if path.lower().endswith(".pdf"):
        direct_text = extract_text_from_pdf_direct(path)
        if direct_text and len(direct_text) > 50:
            return direct_text
        pages = pdf_to_images(path)
        full_text = ""
        for img in pages:
            preprocess_image(img)
            raw = easy_ocr.readtext(img, detail=0)
            full_text += " " + " ".join(raw)
            os.remove(img)
        return full_text.strip()
    try:
        preprocess_image(path)
        raw = easy_ocr.readtext(path, detail=0)
        return " ".join(raw)
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_text_surya(path, surya_ocr):
    try:
        if surya_ocr is None:
            return "ERROR: Surya OCR not initialized"
        from PIL import Image as PILImage
        if path.lower().endswith(".pdf"):
            direct_text = extract_text_from_pdf_direct(path)
            if direct_text and len(direct_text) > 50:
                return direct_text
            pages = pdf_to_images(path)
            full_text = ""
            for img_path in pages:
                try:
                    img = PILImage.open(img_path).convert("RGB")
                    # Pass det_predictor to recognition - it handles detection internally
                    rec_results = surya_ocr['recognition'](
                        [img],
                        det_predictor=surya_ocr['detection']
                    )
                    for page_result in rec_results:
                        if hasattr(page_result, 'text_lines'):
                            for line in page_result.text_lines:
                                full_text += " " + line.text
                        elif hasattr(page_result, 'text'):
                            full_text += " " + page_result.text
                except Exception as e:
                    full_text += f" [Error: {str(e)}]"
                os.remove(img_path)
            return full_text.strip() if full_text.strip() else "ERROR: No text extracted from PDF"
        img = PILImage.open(path).convert("RGB")
        # Pass det_predictor to recognition - it handles detection internally
        rec_results = surya_ocr['recognition'](
            [img],
            det_predictor=surya_ocr['detection']
        )
        full_text = ""
        for page_result in rec_results:
            if hasattr(page_result, 'text_lines'):
                for line in page_result.text_lines:
                    full_text += " " + line.text
            elif hasattr(page_result, 'text'):
                full_text += " " + page_result.text
        return full_text.strip() if full_text.strip() else "ERROR: No text found in image"
    except Exception as e:
        import traceback
        return f"ERROR: {str(e)} | {traceback.format_exc()[:200]}"


def index(request):
    return render(request, 'classifier/index.html')


@csrf_exempt
@csrf_exempt
def classify_document(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    ocr_engine = request.POST.get('ocr_engine', 'tesseract').lower()
    file = request.FILES['file']
    ext = file.name.split('.')[-1].lower()
    temp_path = f"temp_upload.{ext}"
    
    with open(temp_path, 'wb+') as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    collection, embedder, easy_ocr, paddle_ocr, surya_ocr = get_resources()
    start_time = time.time()
    
    if ocr_engine == 'tesseract':
        extracted_text = extract_text_tesseract(temp_path)
        engine_name = "Tesseract"
    elif ocr_engine == 'easyocr':
        extracted_text = extract_text_easyocr(temp_path, easy_ocr)
        engine_name = "EasyOCR"
    elif ocr_engine == 'paddleocr':
        extracted_text = extract_text_paddleocr(temp_path, paddle_ocr)
        engine_name = "PaddleOCR"
    elif ocr_engine == 'surya':
        extracted_text = extract_text_surya(temp_path, surya_ocr)
        engine_name = "Surya"
    else:
        os.remove(temp_path)
        return JsonResponse({'error': 'Invalid OCR engine'}, status=400)
    
    processing_time = time.time() - start_time
    results = {}
    
    if extracted_text.strip() and not extracted_text.startswith("ERROR"):
        text_vec = embedder.encode([extracted_text])[0].tolist()
        try:
            # Simple vector similarity search
            query_res = collection.query(query_embeddings=[text_vec], n_results=5)
            
            if query_res["metadatas"] and query_res["metadatas"][0]:
                # Get the best match (highest similarity)
                best_match = query_res["metadatas"][0][0]
                best_distance = query_res["distances"][0][0]
                confidence = (1 - best_distance) * 100  # Convert distance to confidence percentage
                
                # Prepare top 3 matches for display
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
                
                # Prepare response
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
                    "reason": "OCR failed, classified using Gemini Vision"
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
    
    os.remove(temp_path)
    return JsonResponse(results)


@csrf_exempt
def classify_zip(request):
    """Handle ZIP file uploads containing multiple documents"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    file = request.FILES['file']
    if not file.name.lower().endswith('.zip'):
        return JsonResponse({'error': 'File must be a ZIP archive'}, status=400)
    
    ocr_engine = request.POST.get('ocr_engine', 'tesseract').lower()
    
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    for chunk in file.chunks():
        temp_zip.write(chunk)
    temp_zip.close()
    
    temp_dir = tempfile.mkdtemp()
    collection, embedder, easy_ocr, paddle_ocr, surya_ocr = get_resources()
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
                
                if ocr_engine == 'tesseract':
                    extracted_text = extract_text_tesseract(file_path)
                    engine_name = "Tesseract"
                elif ocr_engine == 'easyocr':
                    extracted_text = extract_text_easyocr(file_path, easy_ocr)
                    engine_name = "EasyOCR"
                elif ocr_engine == 'paddleocr':
                    extracted_text = extract_text_paddleocr(file_path, paddle_ocr)
                    engine_name = "PaddleOCR"
                elif ocr_engine == 'surya':
                    extracted_text = extract_text_surya(file_path, surya_ocr)
                    engine_name = "Surya"
                else:
                    continue
                
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
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    data = json.loads(request.body)
    ocr_engine = data.get('ocr_engine', 'easyocr')
    
    global chroma_client, collection
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        chroma_client.delete_collection("text_docs")
    except:
        pass
    
    collection = chroma_client.get_or_create_collection("text_docs")
    _, embedder, easy_ocr, paddle_ocr, surya_ocr = get_resources()
    
    added = 0
    total = 0
    
    for label in os.listdir(SAMPLES_PATH):
        folder = os.path.join(SAMPLES_PATH, label)
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            total += 1
            path = os.path.join(folder, f)
            if ocr_engine == "paddleocr":
                text = extract_text_paddleocr(path, paddle_ocr)
            elif ocr_engine == "tesseract":
                text = extract_text_tesseract(path)
            elif ocr_engine == "surya":
                text = extract_text_surya(path, surya_ocr)
            else:
                text = extract_text_easyocr(path, easy_ocr)
            
            if not text.strip() or text.startswith("ERROR"):
                continue
            
            emb = embedder.encode([text])[0].tolist()
            file_id = f"{label}_{f}"
            collection.add(ids=[file_id], embeddings=[emb],
                           metadatas=[{"label": label, "ocr_engine": ocr_engine}], documents=[text])
            added += 1
    
    return JsonResponse({'success': True, 'added': added, 'total': total})


@csrf_exempt
def add_new_files(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    data = json.loads(request.body)
    ocr_engine = data.get('ocr_engine', 'easyocr')
    
    collection, embedder, easy_ocr, paddle_ocr, surya_ocr = get_resources()
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
            if ocr_engine == "paddleocr":
                text = extract_text_paddleocr(path, paddle_ocr)
            elif ocr_engine == "tesseract":
                text = extract_text_tesseract(path)
            elif ocr_engine == "surya":
                text = extract_text_surya(path, surya_ocr)
            else:
                text = extract_text_easyocr(path, easy_ocr)
            
            if not text.strip() or text.startswith("ERROR"):
                continue
            
            emb = embedder.encode([text])[0].tolist()
            collection.add(ids=[doc_id], embeddings=[emb],
                           metadatas=[{"label": label, "ocr_engine": ocr_engine}], documents=[text])
            added += 1
    
    return JsonResponse({'success': True, 'added': added})


def get_db_stats(request):
    collection, _, _, _, _ = get_resources()
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
