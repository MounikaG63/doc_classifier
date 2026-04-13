from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json
import time
import zipfile
import tempfile
import shutil
from PIL import Image
import fitz
import easyocr
from paddleocr import PaddleOCR
from rapidocr_onnxruntime import RapidOCR

import threading
from .weaviate_client import get_weaviate_store
from .hybrid_classifier import WeaviateHybridClassifier

# Global OCR instances
easy_ocr = None
paddle_ocr = None
surya_ocr = None
rapid_ocr = None
hybrid_classifier = None
resource_lock = threading.Lock()
paddle_lock = threading.Lock()


def get_resources():
    """Initialize OCR engines and hybrid classifier"""
    global easy_ocr, paddle_ocr, surya_ocr, rapid_ocr, hybrid_classifier
    with resource_lock:
        if easy_ocr is None:
            easy_ocr = easyocr.Reader(["en"], gpu=False)
        
        if paddle_ocr is None:
            try:
                print("Attempting to initialize PaddleOCR...")
                paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                print("✓ PaddleOCR initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize PaddleOCR: {e}")
                paddle_ocr = None
        
        if rapid_ocr is None:
            try:
                print("Attempting to initialize RapidOCR...")
                rapid_ocr = RapidOCR()
                print("✓ RapidOCR initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize RapidOCR: {e}")
                rapid_ocr = None
    
    if surya_ocr is None:
        try:
            print("Attempting to initialize Surya OCR...")
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            detection = DetectionPredictor()
            recognition = RecognitionPredictor()
            surya_ocr = {
                'recognition': recognition,
                'detection': detection
            }
            print("✓ Surya OCR initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Surya OCR: {e}")
            surya_ocr = None
    
    if hybrid_classifier is None:
        try:
            weaviate_store = get_weaviate_store()
            hybrid_classifier = WeaviateHybridClassifier(weaviate_store)
            print("✓ Weaviate Hybrid Classifier initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Hybrid Classifier: {e}")
            hybrid_classifier = None
    
    return easy_ocr, paddle_ocr, surya_ocr, rapid_ocr, hybrid_classifier


def extract_text_from_pdf_direct(path):
    """Extract text directly from PDF if it's text-based"""
    try:
        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += " " + page.get_text()
        doc.close()
        return full_text.strip()
    except:
        return ""


def pdf_to_images(pdf_path):
    """Convert PDF pages to images"""
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        out_path = f"{pdf_path}_page_{i}.png"
        pix.save(out_path)
        image_paths.append(out_path)
    return image_paths


def preprocess_image(path):
    """Preprocess image for better OCR results"""
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


def extract_text_paddleocr(path, paddle_ocr):
    """Extract text using PaddleOCR"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".pdf"]:
            return f"ERROR: Unsupported file type for PaddleOCR: {ext}"
        if ext == ".pdf":
            direct_text = extract_text_from_pdf_direct(path)
            if direct_text and len(direct_text) > 50:
                return direct_text
        
        with paddle_lock:
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
    """Extract text using Tesseract"""
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
    """Extract text using EasyOCR"""
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
    """Extract text using Surya OCR"""
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
    """Main page for Weaviate classifier"""
    return render(request, 'weaviate_classifier/index.html')


@csrf_exempt
def classify_document(request):
    """Classify a single document using Weaviate + hybrid approach"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)
    
    ocr_engine = request.POST.get('ocr_engine', 'tesseract').lower()
    file = request.FILES['file']
    ext = file.name.split('.')[-1].lower()
    temp_path = f"temp_upload.{ext}"
    
    results = {}
    try:
        # Save uploaded file
        with open(temp_path, 'wb+') as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        easy_ocr, paddle_ocr, surya_ocr, rapid_ocr, hybrid_classifier = get_resources()
        start_time = time.time()
        
        # Extract text using selected OCR engine
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
        elif ocr_engine == 'rapidocr':
            extracted_text = extract_text_rapidocr(temp_path, rapid_ocr)
            engine_name = "RapidOCR"
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JsonResponse({'error': 'Invalid OCR engine'}, status=400)
        
        processing_time = time.time() - start_time
        
        # Classify using hybrid approach
        if extracted_text.strip() and not extracted_text.startswith("ERROR") and hybrid_classifier:
            try:
                classification_result = hybrid_classifier.classify(extracted_text)
                
                results = {
                    "ocr_engine": engine_name,
                    "document_type": classification_result["document_type"],
                    "confidence": classification_result["confidence"],
                    "classification_method": classification_result["method"],
                    "vector_score": classification_result.get("vector_score", 0),
                    "keyword_score": classification_result.get("keyword_score", 0),
                    "details": classification_result.get("details", ""),
                    "ocr_text": extracted_text[:2000],
                    "processing_time": round(processing_time, 2),
                    "top_3_matches": classification_result.get("top_3_matches", [])
                }
            except Exception as e:
                results = {
                    "ocr_engine": engine_name,
                    "document_type": "unknown",
                    "confidence": 0.0,
                    "ocr_text": extracted_text[:2000],
                    "reason": f"classification_error: {str(e)}",
                    "processing_time": round(processing_time, 2)
                }
        else:
            # Fallback to Gemini if OCR fails
            try:
                from classifier.gemini_classifier import classify_with_gemini
                gemini_result = classify_with_gemini(temp_path)
                if "error" not in gemini_result:
                    results = {
                        "ocr_engine": f"{engine_name} + Gemini (fallback)",
                        "document_type": gemini_result.get("doc_type", "unknown"),
                        "category": gemini_result.get("category", ""),
                        "sub_category": gemini_result.get("sub_category", ""),
                        "ocr_text": extracted_text[:2000],
                        "confidence": gemini_result.get("confidence", ""),
                        "processing_time": round(processing_time, 2),
                        "gemini_used": True,
                        "reason": "OCR failed, classified using Gemini Vision"
                    }
                else:
                    results = {
                        "ocr_engine": engine_name,
                        "document_type": "unknown",
                        "confidence": 0.0,
                        "reason": f"ocr_failed and gemini_error: {gemini_result.get('error', 'Unknown')}",
                        "processing_time": round(processing_time, 2)
                    }
            except Exception as e:
                results = {
                    "ocr_engine": engine_name,
                    "document_type": "unknown",
                    "confidence": 0.0,
                    "reason": f"ocr_failed: {extracted_text[:100]}, gemini_error: {str(e)}",
                    "processing_time": round(processing_time, 2)
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
def rebuild_database(request):
    """Rebuild Weaviate database from samples"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    data = json.loads(request.body)
    ocr_engine = data.get('ocr_engine', 'easyocr')
    
    easy_ocr, paddle_ocr, surya_ocr, rapid_ocr, hybrid_classifier = get_resources()
    
    if not hybrid_classifier:
        return JsonResponse({'error': 'Hybrid classifier not initialized'}, status=500)
    
    # Clear existing data
    try:
        hybrid_classifier.weaviate_store.clear_all()
    except Exception as e:
        return JsonResponse({'error': f'Failed to clear database: {str(e)}'}, status=500)
    
    added = 0
    total = 0
    samples_path = "samples"
    
    for label in os.listdir(samples_path):
        folder = os.path.join(samples_path, label)
        if not os.path.isdir(folder):
            continue
        
        for f in os.listdir(folder):
            total += 1
            path = os.path.join(folder, f)
            
            # Extract text using selected OCR engine
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
                continue
            
            try:
                hybrid_classifier.weaviate_store.add_document(
                    text=text,
                    label=label,
                    filename=f,
                    ocr_engine=ocr_engine
                )
                added += 1
            except Exception as e:
                print(f"Error adding document {f}: {e}")
                continue
    
    return JsonResponse({'success': True, 'added': added, 'total': total})


@csrf_exempt
def add_new_files(request):
    """Add new files to Weaviate database"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    data = json.loads(request.body)
    ocr_engine = data.get('ocr_engine', 'easyocr')
    
    easy_ocr, paddle_ocr, surya_ocr, rapid_ocr, hybrid_classifier = get_resources()
    
    if not hybrid_classifier:
        return JsonResponse({'error': 'Hybrid classifier not initialized'}, status=500)
    
    # Get existing filenames to avoid duplicates
    try:
        stats = hybrid_classifier.weaviate_store.get_stats()
        # For simplicity, we'll add all files (Weaviate handles duplicates differently)
        existing_files = set()
    except:
        existing_files = set()
    
    added = 0
    samples_path = "samples"
    
    for label in os.listdir(samples_path):
        folder = os.path.join(samples_path, label)
        if not os.path.isdir(folder):
            continue
        
        for f in os.listdir(folder):
            file_key = f"{label}_{f}"
            if file_key in existing_files:
                continue
            
            path = os.path.join(folder, f)
            
            # Extract text using selected OCR engine
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
                continue
            
            try:
                hybrid_classifier.weaviate_store.add_document(
                    text=text,
                    label=label,
                    filename=f,
                    ocr_engine=ocr_engine
                )
                added += 1
            except Exception as e:
                print(f"Error adding document {f}: {e}")
                continue
    
    return JsonResponse({'success': True, 'added': added})


def get_db_stats(request):
    """Get Weaviate database statistics"""
    try:
        weaviate_store = get_weaviate_store()
        if weaviate_store is None:
            return JsonResponse({'error': 'Weaviate store not initialized'}, status=503)
        stats = weaviate_store.get_stats()
        return JsonResponse(stats)
    except Exception as e:
        return JsonResponse({'error': f'Failed to get stats: {str(e)}'}, status=500)


def health_check(request):
    """Check Weaviate connection health"""
    try:
        weaviate_store = get_weaviate_store()
        if weaviate_store is None:
            return JsonResponse({
                'healthy': False,
                'message': 'Weaviate store not initialized (check connection/configuration)'
            }, status=503)
        is_healthy = weaviate_store.health_check()
        return JsonResponse({
            'healthy': is_healthy,
            'message': 'Weaviate is running' if is_healthy else 'Weaviate is not accessible'
        })
    except Exception as e:
        return JsonResponse({
            'healthy': False,
            'message': f'Health check failed: {str(e)}'
        }, status=500)