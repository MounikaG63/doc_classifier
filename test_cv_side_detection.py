"""
Test script for OpenCV-based document side detection
Test single file or batch from directories
"""

from classifier.document_side_detector_cv import DocumentSideDetectorCV
import os
import sys

def test_single_file(image_path, doc_type):
    """Test side detection on a single file"""
    detector = DocumentSideDetectorCV()
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    print("=" * 80)
    print("OPENCV-BASED DOCUMENT SIDE DETECTION TEST")
    print("=" * 80)
    print(f"\n📄 File: {image_path}")
    print(f"   Document Type: {doc_type.upper()}")
    
    result = detector.detect_side(image_path, doc_type)
    
    print(f"\n   Detected Side: {result['side'].upper()}")
    print(f"   Confidence: {result['confidence']}%")
    
    if 'error' not in result['details']:
        details = result['details']
        
        # Handle combined view
        if result['side'] == 'combined':
            print(f"\n   Analysis Details:")
            print(f"     • {details.get('message', 'Combined view detected')}")
        else:
            print(f"\n   Analysis Details:")
            print(f"     • Has Photo: {details['has_photo']}")
            print(f"     • Text Density: {details['text_density']}%")
            print(f"     • Text Concentration: {details['text_concentration']}")
            print(f"     • Color Variance: {details['color_variance']}")
            print(f"     • Mostly White: {details['is_mostly_white']}")
            print(f"     • Edge Density: {details['edge_density']}%")
            print(f"     • Edge Concentration: {details['edge_concentration']}")
            print(f"\n   Scoring:")
            print(f"     • Front Score: {details['front_score']}")
            print(f"     • Back Score: {details['back_score']}")
    else:
        print(f"   Error: {result['details']['error']}")
    
    # Confidence indicator
    if result['confidence'] >= 70:
        indicator = "✅ HIGH CONFIDENCE"
    elif result['confidence'] >= 55:
        indicator = "⚠️  MEDIUM CONFIDENCE"
    else:
        indicator = "❌ LOW CONFIDENCE"
    
    print(f"\n   Confidence Level: {indicator}")
    print("=" * 80)


def test_batch_from_directories():
    """Test batch from sample directories"""
    detector = DocumentSideDetectorCV()
    
    sample_dirs = {
        'aadhar': 'samples/aadhar',
        'pan_card_individual': 'samples/pan_card_individual',
        'voter_id': 'samples/voter_id',
        'driving_license': 'samples/driving_license'
    }
    
    print("=" * 80)
    print("OPENCV-BASED DOCUMENT SIDE DETECTION TEST - BATCH MODE")
    print("=" * 80)
    
    for doc_type, sample_path in sample_dirs.items():
        if not os.path.exists(sample_path):
            print(f"\n⚠️  {doc_type.upper()}: Sample directory not found ({sample_path})")
            continue
        
        files = os.listdir(sample_path)
        if not files:
            print(f"\n⚠️  {doc_type.upper()}: No files in directory")
            continue
        
        # Test first 2 files from each category
        for i, filename in enumerate(files[:2]):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf')):
                continue
            
            file_path = os.path.join(sample_path, filename)
            
            print(f"\n📄 {doc_type.upper()} - {filename}")
            print(f"   Path: {file_path}")
            
            result = detector.detect_side(file_path, doc_type)
            
            print(f"   Detected Side: {result['side'].upper()}")
            print(f"   Confidence: {result['confidence']}%")
            
            if 'error' not in result['details']:
                details = result['details']
                print(f"   Details:")
                print(f"     - Has Photo: {details['has_photo']}")
                print(f"     - Text Density: {details['text_density']}%")
                print(f"     - Text Concentration: {details['text_concentration']}")
                print(f"     - Color Variance: {details['color_variance']}")
                print(f"     - Mostly White: {details['is_mostly_white']}")
                print(f"     - Edge Density: {details['edge_density']}%")
                print(f"     - Front Score: {details['front_score']} | Back Score: {details['back_score']}")
            else:
                print(f"   Error: {result['details']['error']}")
            
            # Confidence indicator
            if result['confidence'] >= 70:
                indicator = "✅ HIGH"
            elif result['confidence'] >= 55:
                indicator = "⚠️  MEDIUM"
            else:
                indicator = "❌ LOW"
            print(f"   Confidence Level: {indicator}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # Single file mode
        image_path = sys.argv[1]
        doc_type = sys.argv[2].lower()
        
        valid_types = ['aadhar', 'pan', 'voter_id', 'driving_license']
        if doc_type not in valid_types:
            print(f"❌ Invalid doc_type: {doc_type}")
            print(f"   Valid types: {', '.join(valid_types)}")
            sys.exit(1)
        
        test_single_file(image_path, doc_type)
    else:
        # Batch mode from directories
        print("Usage for single file:")
        print('  python test_cv_side_detection.py "path/to/image.jpg" "aadhar"')
        print("\nRunning batch test from sample directories...\n")
        test_batch_from_directories()
