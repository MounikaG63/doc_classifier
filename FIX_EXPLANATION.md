# Document Classification Issues - Root Cause & Fix

## Problem Summary
1. **PaddleOCR showing EasyOCR results** - The UI shows "EasyOCR" label even when PaddleOCR is selected
2. **Tax documents classified as PAN Card** - Tax-related documents are being misclassified as individual PAN cards

## Root Causes

### Issue 1: PaddleOCR Text Extraction Bug
The `extract_text_paddleocr()` function had incorrect logic:
- Used wrong method names (`predict()` instead of `ocr()`)
- Used deprecated parameters (`cls=True` not supported in newer versions)
- Incorrect result parsing

**Fixed in:** `classifier/views.py` - Updated extraction logic to use correct PaddleOCR API

### Issue 2: Database Embedding Mismatch
The **real issue** causing misclassification:
- Database was built with **EasyOCR** text embeddings
- When you use **PaddleOCR**, it extracts different text
- Different text → different embeddings → poor matching against database
- Result: Tax documents match poorly and fall back to closest match (PAN Card)

**Example:**
```
EasyOCR extracts: "PAN CARD 123456789 JOHN DOE"
PaddleOCR extracts: "PAN CARD 123456789 JOHN DOE" (slightly different formatting/spacing)
Different embeddings → Different similarity scores → Wrong classification
```

## Solution

### Step 1: Rebuild Database with Consistent OCR Engine
Choose ONE OCR engine and rebuild the entire database with it:

**Option A: Use EasyOCR (Recommended - Most Stable)**
```bash
python rebuild_db_easyocr.py
```

**Option B: Use PaddleOCR (If you fix Windows compatibility)**
```bash
python rebuild_db_paddleocr.py
```

### Step 2: Use Same Engine for Classification
After rebuilding, always use the same OCR engine for classification:
- If database built with EasyOCR → Always select "EasyOCR" in UI
- If database built with PaddleOCR → Always select "PaddleOCR" in UI

## Why This Matters

The classification system works by:
1. Extract text from document using OCR engine
2. Convert text to embedding vector
3. Compare embedding against database embeddings
4. Find closest match

**If OCR engines extract different text → embeddings differ → matches fail**

## Technical Details

### PaddleOCR Result Format
```python
result = paddle_ocr.ocr(image_path)
# Returns: [[[x1,y1,x2,y2,x3,y3,x4,y4], [text, confidence]], ...]]
# Access text: result[line_idx][word_idx][1]
```

### EasyOCR Result Format
```python
result = easy_ocr.readtext(image_path, detail=0)
# Returns: [text1, text2, text3, ...]
# Access text: result[idx]
```

## Files Modified
- `classifier/views.py`:
  - Fixed `extract_text_paddleocr()` function
  - Fixed `get_resources()` PaddleOCR initialization
  - Added better logging to `classify_document()`

## Next Steps
1. Run `python rebuild_db_easyocr.py` to rebuild database
2. Test classification with EasyOCR selected
3. Verify tax documents are now classified correctly
4. (Optional) If you want to use PaddleOCR, fix Windows compatibility and rebuild with it

## Testing
After rebuild, test with:
- Tax-related document → Should classify as tax document, not PAN Card
- PAN Card → Should classify as PAN Card
- Other documents → Should classify correctly

## Notes
- PaddleOCR has Windows compatibility issues (OneDNN backend)
- EasyOCR is more stable and works reliably on Windows
- The database rebuild takes time (processes all sample files)
- Once rebuilt, classification will be accurate and consistent
