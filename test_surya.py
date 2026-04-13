import sys
import os

def test_surya():
    try:
        print("Attempting to import surya components...")
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        
        print("✓ Imports successful")
        
        print("Attempting to initialize FoundationPredictor...")
        foundation = FoundationPredictor()
        print("✓ FoundationPredictor initialized")
        
        print("Attempting to initialize DetectionPredictor...")
        detection = DetectionPredictor()
        print("✓ DetectionPredictor initialized")
        
        print("Attempting to initialize RecognitionPredictor...")
        recognition = RecognitionPredictor(foundation)
        print("✓ RecognitionPredictor initialized")
        
        print("Surya initialization successful!")
    except Exception as e:
        print(f"✗ Surya initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_surya()
