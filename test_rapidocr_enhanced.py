from rapidocr_onnxruntime import RapidOCR
import os
import sys

def test_enhanced():
    print("Testing Enhanced RapidOCR with sorting...")
    try:
        engine = RapidOCR()
        print("✅ RapidOCR initialized")
        
        # Find a sample image
        sample_file = None
        if os.path.exists("samples"):
            for root, dirs, files in os.walk("samples"):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        sample_file = os.path.join(root, f)
                        break
                if sample_file: break
        
        if sample_file:
            print(f"Testing on: {sample_file}")
            results, _ = engine(sample_file)
            if results:
                print(f"Original order (first 3): {[res[1] for res in results[:3]]}")
                
                # Apply my sorting logic
                results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
                
                print(f"Sorted order (first 3): {[res[1] for res in results[:3]]}")
                print(f"✅ Sorting successful")
            else:
                print("❌ No text found in sample")
        else:
            print("❌ No sample image found")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_enhanced()
