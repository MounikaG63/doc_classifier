from rapidocr_onnxruntime import RapidOCR
import os
import sys

def test(sample_file=None):
    print("Testing RapidOCR standalone...")
    try:
        engine = RapidOCR()
        print("✅ RapidOCR initialized successfully")
        
        if not sample_file:
            sample_file = "samples/aadhar/AAdhar.jpg"
            
        if not os.path.exists(sample_file):
            print(f"⚠️ Sample file {sample_file} not found, searching...")
            if os.path.exists("samples"):
                for root, dirs, files in os.walk("samples"):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            sample_file = os.path.join(root, f)
                            break
                    if sample_file: break
        
        if os.path.exists(sample_file):
            print(f"Testing OCR on: {sample_file}")
            result, elapse = engine(sample_file)
            if result:
                texts = [line[1] for line in result]
                print(f"✅ Extracted text count: {len(texts)}")
                print(f"✅ Extracted text: {' '.join(texts)[:200]}...")
                print(f"⏱️ Elapse: {elapse}")
            else:
                print("❌ No text found")
        else:
            print("❌ No suitable sample file found")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    test(path)
