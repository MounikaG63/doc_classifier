"""
Gemini Vision API for document classification
Used as fallback when OCR fails or returns empty text
"""
import os
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# System prompt for classification
SYSTEM_PROMPT = """
You are a strict document classification engine. You classify documents into predefined categories.

### Document Categories:

1. **Stakeholder Documents - Address Proof**
   - PAN Card (individual), Passport, Aadhar Card, Voter ID, Driving Licence
   - Utility Bills: Electricity Bill, Water Bill, Gas Bill, Internet Bill, Telephone Bill
   - Bank Statements, Rent Agreement, OCI, Property Tax Receipts

2. **Entity Documents**
   - Cancelled Cheque, Company PAN, Company TAN, GST Registration
   - MOA & AOA, COI (Certificate of Incorporation)

3. **Entity Documents - Address Proof**
   - Electricity Bill, Water Bill, Gas Bill (for company)
   - Bank Statements, Rent Agreement, Property Tax Receipts

4. **HR Documents**
   - Appointment Letter, Offer Letter, Payroll Sheet
   - Employee Master, HR Policies

5. **Finance Docs**
   - Profit and Loss Statement, Balance Sheet
   - Cash Flow Statement, MIS Report

6. **Tax Filings**
   - GST Returns (GSTR-1, GSTR-2B, GSTR-3B)
   - ITR, TDS Returns, Form 26AS

7. **Compliance Docs**
   - PF Filings, ESI Filings, PT Filings
   - Board Resolutions, AGM Filings

### Classification Rules:
- Identify the PRIMARY document type from the image
- For utility bills: Check company name, account number, usage details
- For ID cards: Check name, ID number, photo, government logos
- For bank documents: Check bank name, account details

### Output Format:
Return ONLY valid JSON:
{
    "category": "main category name",
    "sub_category": "sub category if applicable, else empty string",
    "doc_type": "specific document type",
    "confidence": "high/medium/low"
}
"""


def classify_with_gemini(image_path):
    """
    Classify document using Gemini Vision API
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict with classification result or error
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not found in .env file"}
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Determine mime type
        ext = image_path.lower().split('.')[-1]
        mime_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp',
            'pdf': 'application/pdf'
        }
        mime_type = mime_types.get(ext, 'image/jpeg')
        
        # Create model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create image part
        image_part = {
            "mime_type": mime_type,
            "data": image_data
        }
        
        # Generate classification
        prompt = f"""{SYSTEM_PROMPT}

Analyze this document image and classify it. Return ONLY valid JSON."""
        
        response = model.generate_content([prompt, image_part])
        
        # Parse response
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        try:
            result = json.loads(response_text)
            result['source'] = 'gemini'
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw response
            return {
                "category": "Unknown",
                "doc_type": response_text[:100],
                "confidence": "low",
                "source": "gemini",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}


def classify_with_gemini_text(extracted_text):
    """
    Classify document using Gemini based on extracted text
    
    Args:
        extracted_text: Text extracted from document
        
    Returns:
        dict with classification result or error
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not found in .env file"}
    
    try:
        import google.generativeai as genai
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""{SYSTEM_PROMPT}

Analyze this document text and classify it:

DOCUMENT TEXT:
{extracted_text[:3000]}

Return ONLY valid JSON."""
        
        response = model.generate_content(prompt)
        
        # Parse response
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        try:
            result = json.loads(response_text)
            result['source'] = 'gemini'
            return result
        except json.JSONDecodeError:
            return {
                "category": "Unknown",
                "doc_type": response_text[:100],
                "confidence": "low",
                "source": "gemini",
                "raw_response": response_text
            }
            
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}
