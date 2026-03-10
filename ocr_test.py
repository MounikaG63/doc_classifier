from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import pytesseract
from PIL import Image
import cv2
import numpy as np
import zipfile
import tempfile
import shutil
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

app = Flask(__name__, template_folder='classifier/templates', static_folder='classifier/templates')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'temp_uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OCR models globally (only once)
paddle_ocr = None

def init_paddle_ocr():
    global paddle_ocr
    if paddle_ocr is None:
        print("Initializing PaddleOCR...")
        try:
            paddle_ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            print("✓ PaddleOCR initialized")
        except Exception as e:
            print(f"✗ PaddleOCR failed: {e}")
            import traceback
            traceback.print_exc()
    return paddle_ocr

# Initialize on first request
@app.before_request
def before_request():
    global paddle_ocr
    if paddle_ocr is None:
        init_paddle_ocr()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf', 'zip'}
IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_images_from_zip(zip_path):
    """Extract images from ZIP file"""
    images = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                ext = file_info.filename.rsplit('.', 1)[1].lower() if '.' in file_info.filename else ''
                if ext in IMAGE_EXTENSIONS:
                    with zip_ref.open(file_info) as file:
                        img = Image.open(file)
                        images.append(img.copy())
    except Exception as e:
        raise Exception(f"Error extracting ZIP: {str(e)}")
    return images

def extract_images_from_pdf(pdf_path):
    """Extract images from PDF file using PyMuPDF"""
    try:
        images = []
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # Render page to image with higher zoom for better OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # 3x zoom for better quality
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        pdf_document.close()
        return images
    except Exception as e:
        raise Exception(f"Error converting PDF: {str(e)}")

def extract_text_from_pdf_direct(pdf_path):
    """Extract text directly from PDF using PyMuPDF (fastest method)"""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += " " + page.get_text()
        doc.close()
        return full_text.strip() if full_text.strip() else None
    except:
        return None

def get_images_from_file(file_path):
    """Get list of PIL Images from file (image, PDF, or ZIP)"""
    ext = file_path.rsplit('.', 1)[1].lower()
    
    if ext in IMAGE_EXTENSIONS:
        return [Image.open(file_path)]
    elif ext == 'pdf':
        return extract_images_from_pdf(file_path)
    elif ext == 'zip':
        return extract_images_from_zip(file_path)
    else:
        raise Exception(f"Unsupported file type: {ext}")

def ocr_tesseract(image_path):
    """Extract text using Tesseract OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error with Tesseract: {str(e)}"

def ocr_tesseract_from_pil(pil_image):
    """Extract text from PIL Image using Tesseract"""
    try:
        text = pytesseract.image_to_string(pil_image)
        return text
    except Exception as e:
        return f"Error with Tesseract: {str(e)}"

def ocr_paddle(image_path):
    """Extract text using PaddleOCR"""
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(image_path)
        text = '\n'.join([line[0][1][0] for line in result if result and line])
        return text
    except Exception as e:
        return f"Error with PaddleOCR: {str(e)}"

def ocr_paddle_from_pil(pil_image):
    """Extract text from PIL Image using PaddleOCR"""
    try:
        if not paddle_ocr:
            return "Error with PaddleOCR: Not initialized"
        
        # Save PIL image temporarily
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pil_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = paddle_ocr.predict(tmp_path)
            
            # Extract text from result using predict method
            all_texts = []
            if result:
                for page_result in result:
                    if hasattr(page_result, 'rec_texts'):
                        all_texts.extend(page_result.rec_texts)
                    elif isinstance(page_result, dict) and 'rec_texts' in page_result:
                        all_texts.extend(page_result['rec_texts'])
            
            text = ' '.join(all_texts) if all_texts else "No text extracted"
            return text
        finally:
            # Delete temp file after processing
            try:
                os.unlink(tmp_path)
            except:
                pass
    except Exception as e:
        return f"Error with PaddleOCR: {str(e)}"

def ocr_surya(image_path):
    """Extract text using Surya OCR"""
    try:
        from surya.ocr import run_ocr
        from surya.model_init import initialize_model_dir
        from PIL import Image
        
        model_dir = initialize_model_dir()
        image = Image.open(image_path)
        results = run_ocr([image], languages=['en'], model_dir=model_dir)
        text = results[0].text
        return text
    except Exception as e:
        return f"Error with Surya: {str(e)}"

@app.route('/')
def index():
    return render_template('classifier/ocr_test.html')

@app.route('/extract', methods=['POST'])
def extract():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    ocr_model = request.form.get('ocr_model')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp, tiff, pdf, zip'}), 400
    
    if not ocr_model or ocr_model not in ['tesseract', 'paddle', 'surya']:
        return jsonify({'error': 'Invalid OCR model selected'}), 400
    
    temp_dir = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # For PDFs, try direct text extraction first (much faster)
        if filepath.lower().endswith('.pdf'):
            direct_text = extract_text_from_pdf_direct(filepath)
            if direct_text and len(direct_text) > 50:
                os.remove(filepath)
                end_time = time.time()
                processing_time = round(end_time - start_time, 2)
                return jsonify({
                    'text': f"--- Page/Image 1 ---\n{direct_text}",
                    'model': f"{ocr_model} (PDF Direct)",
                    'pages': 1,
                    'processing_time': processing_time
                })
        
        # Get images from file
        images = get_images_from_file(filepath)
        
        if not images:
            return jsonify({'error': 'No images found in file'}), 400
        
        # Extract text from all images
        all_text = []
        for idx, img in enumerate(images):
            text = ""
            try:
                if ocr_model == 'tesseract':
                    text = ocr_tesseract_from_pil(img)
                elif ocr_model == 'paddle':
                    text = ocr_paddle_from_pil(img)
                elif ocr_model == 'surya':
                    # For Surya, save temporarily
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img.save(tmp.name)
                        text = ocr_surya(tmp.name)
                        os.unlink(tmp.name)
                
                # Only add if text was extracted
                if text and not text.startswith('Error') and text.strip():
                    all_text.append(f"--- Page/Image {idx + 1} ---\n{text}")
                else:
                    all_text.append(f"--- Page/Image {idx + 1} ---\nNo text extracted")
            except Exception as e:
                all_text.append(f"--- Page/Image {idx + 1} ---\nError: {str(e)}")
        
        # Clean up uploaded file
        os.remove(filepath)
        
        combined_text = '\n\n'.join(all_text) if all_text else 'No text extracted from any page'
        
        # Calculate processing time
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        return jsonify({
            'text': combined_text, 
            'model': ocr_model, 
            'pages': len(images),
            'processing_time': processing_time
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    finally:
        # Cleanup temp directory if created
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
