import os
import sys

# Setup Django
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ocr_project.settings')
import django
django.setup()

from classifier.views import get_resources, extract_text_rapidocr

def test():
    print("Testing RapidOCR integration...")
    resources = get_resources()
    # collection, embedder, easy_ocr, paddle_ocr, surya_ocr, rapid_ocr
    rapid_ocr = resources[-1]
    
    if rapid_ocr is None:
        print("❌ RapidOCR failed to initialize!")
        return

    print("✅ RapidOCR initialized successfully")
    
    # Try to extract text from a sample if available
    sample_dir = "samples"
    if os.path.exists(sample_dir):
        for label in os.listdir(sample_dir):
            folder = os.path.join(sample_dir, label)
            if os.path.isdir(folder):
                files = os.listdir(folder)
                if files:
                    test_file = os.path.join(folder, files[0])
                    print(f"Testing OCR on: {test_file}")
                    text = extract_text_rapidocr(test_file, rapid_ocr)
                    if not text.startswith("ERROR"):
                        print(f"✅ Extracted text (first 100 chars): {text[:100]}...")
                    else:
                        print(f"❌ Extraction error: {text}")
                    break
    else:
        print("⚠️ No samples folder found for testing extraction.")

if __name__ == "__main__":
    test()
