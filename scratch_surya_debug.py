try:
    print("Attempting to import surya components (new pattern)...")
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    
    print("✓ Imports successful")
    
    print("Attempting to initialize DetectionPredictor...")
    # Using small setup or just checking init
    detection = DetectionPredictor()
    print("✓ DetectionPredictor initialized")
    
    print("Attempting to initialize RecognitionPredictor...")
    # In 0.14.6 RecognitionPredictor is initialized without foundation model
    recognition = RecognitionPredictor()
    print("✓ RecognitionPredictor initialized")
    
    print("Surya 0.14.6+ initialization successful!")
except Exception as e:
    print(f"✗ Surya initialization failed: {e}")
    import traceback
    traceback.print_exc()
