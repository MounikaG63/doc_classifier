"""
Document Side Detection Script
Detects front/back side of Aadhar, PAN, Voter ID, and Driving License
Based on OCR text analysis with confidence scoring
Supports: English, Hindi, Telugu
"""

import re
from typing import Dict, Tuple

class DocumentSideDetector:
    """Detect document side (front/back) based on OCR text"""
    
    def __init__(self):
        # Keywords for each document type and side
        self.keywords = {
            'aadhar': {
                'front': {
                    'en': ['aadhar', 'uid', 'dob', 'date of birth', 'name', 'gender', 'photo'],
                    'hi': ['आधार', 'यूआईडी', 'जन्म', 'नाम', 'लिंग', 'फोटो'],
                    'te': ['ఆధార్', 'జన్మ', 'పేరు', 'ఫోటో']
                },
                'back': {
                    'en': ['address', 'postal', 'pin code', 'state', 'district', 'village', 'street'],
                    'hi': ['पता', 'पिन', 'राज्य', 'जिला', 'गांव', 'सड़क'],
                    'te': ['చిరునామా', 'పిన్', 'రాష్ట్రం', 'జిల్లా', 'గ్రామం']
                }
            },
            'pan': {
                'front': {
                    'en': ['pan', 'income tax', 'name', 'father', 'dob', 'photo'],
                    'hi': ['पैन', 'आयकर', 'नाम', 'पिता', 'जन्म', 'फोटो'],
                    'te': ['పాన్', 'ఆదాయ', 'పేరు', 'తండ్రి', 'జన్మ']
                },
                'back': {
                    'en': ['signature', 'sign', 'issued', 'valid', 'authority'],
                    'hi': ['हस्ताक्षर', 'जारी', 'वैध', 'प्राधिकार'],
                    'te': ['సంతకం', 'జారీ', 'చెల్లుబాటు', 'అధికారం']
                }
            },
            'voter_id': {
                'front': {
                    'en': ['voter', 'election', 'name', 'father', 'dob', 'photo', 'epic'],
                    'hi': ['मतदाता', 'चुनाव', 'नाम', 'पिता', 'जन्म', 'फोटो'],
                    'te': ['ఓటర్', 'ఎన్నికలు', 'పేరు', 'తండ్రి', 'జన్మ', 'ఫోటో']
                },
                'back': {
                    'en': ['address', 'constituency', 'postal', 'pin', 'state', 'district'],
                    'hi': ['पता', 'निर्वाचन क्षेत्र', 'पिन', 'राज्य', 'जिला'],
                    'te': ['చిరునామా', 'నియోజకవర్గం', 'పిన్', 'రాష్ట్రం', 'జిల్లా']
                }
            },
            'driving_license': {
                'front': {
                    'en': ['dl no', 'driving license', 'name', 'dob', 'address', 'photo', 'validity'],
                    'hi': ['डीएल', 'ड्राइविंग लाइसेंस', 'नाम', 'जन्म', 'पता', 'फोटो', 'वैधता'],
                    'te': ['డీఎల్', 'డ్రైవింగ్ లైసెన్స్', 'పేరు', 'జన్మ', 'చిరునామా', 'ఫోటో']
                },
                'back': {
                    'en': ['endorsement', 'vehicle class', 'cov', 'restrictions', 'signature'],
                    'hi': ['समर्थन', 'वाहन वर्ग', 'प्रतिबंध', 'हस्ताक्षर'],
                    'te': ['ఆమోదం', 'వాహన తరగతి', 'నిషేధాలు', 'సంతకం']
                }
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detect language from text (en, hi, te)"""
        text_lower = text.lower()
        
        # Count script indicators
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', text))
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', text))
        
        if telugu_chars > hindi_chars and telugu_chars > 5:
            return 'te'
        elif hindi_chars > 5:
            return 'hi'
        else:
            return 'en'
    
    def calculate_side_score(self, text: str, doc_type: str, side: str, language: str) -> float:
        """Calculate confidence score for a specific side"""
        text_lower = text.lower()
        keywords = self.keywords[doc_type][side].get(language, [])
        
        if not keywords:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
        
        # Score: percentage of keywords found
        score = (matches / len(keywords)) * 100
        return score
    
    def detect_side(self, text: str, doc_type: str) -> Dict:
        """
        Detect document side
        
        Args:
            text: OCR extracted text
            doc_type: 'aadhar', 'pan', 'voter_id', or 'driving_license'
        
        Returns:
            {
                'side': 'front' or 'back' or 'unknown',
                'confidence': 0-100,
                'front_score': 0-100,
                'back_score': 0-100,
                'language': 'en', 'hi', or 'te',
                'details': explanation
            }
        """
        if not text or not text.strip():
            return {
                'side': 'unknown',
                'confidence': 0,
                'front_score': 0,
                'back_score': 0,
                'language': 'unknown',
                'details': 'No text extracted'
            }
        
        # Detect language
        language = self.detect_language(text)
        
        # Calculate scores for both sides
        front_score = self.calculate_side_score(text, doc_type, 'front', language)
        back_score = self.calculate_side_score(text, doc_type, 'back', language)
        
        # Determine side
        if front_score > back_score:
            side = 'front'
            confidence = front_score
        elif back_score > front_score:
            side = 'back'
            confidence = back_score
        else:
            side = 'unknown'
            confidence = 0
        
        # Determine if confidence is high enough (threshold: 30%)
        if confidence < 30:
            side = 'unknown'
        
        return {
            'side': side,
            'confidence': round(confidence, 2),
            'front_score': round(front_score, 2),
            'back_score': round(back_score, 2),
            'language': language,
            'details': f"Front: {front_score:.1f}%, Back: {back_score:.1f}%"
        }


# Test cases
def test_detector():
    """Test the detector with sample texts"""
    detector = DocumentSideDetector()
    
    test_cases = [
        # Aadhar tests
        {
            'name': 'Aadhar Front (English)',
            'doc_type': 'aadhar',
            'text': 'AADHAR UID 123456789012 Name: John Doe DOB: 01/01/1990 Gender: Male Photo Present'
        },
        {
            'name': 'Aadhar Back (English)',
            'doc_type': 'aadhar',
            'text': 'Address: 123 Main Street, Village: Springfield, District: County, State: State, Pin Code: 123456'
        },
        {
            'name': 'Aadhar Front (Hindi)',
            'doc_type': 'aadhar',
            'text': 'आधार यूआईडी 123456789012 नाम: राज कुमार जन्म: 01/01/1990 लिंग: पुरुष फोटो'
        },
        {
            'name': 'Aadhar Back (Hindi)',
            'doc_type': 'aadhar',
            'text': 'पता: 123 मुख्य सड़क, गांव: स्प्रिंगफील्ड, जिला: काउंटी, राज्य: स्टेट, पिन कोड: 123456'
        },
        
        # PAN tests
        {
            'name': 'PAN Front (English)',
            'doc_type': 'pan',
            'text': 'PAN Income Tax Name: Rajesh Kumar Father: Kumar Singh DOB: 15/06/1985 Photo'
        },
        {
            'name': 'PAN Back (English)',
            'doc_type': 'pan',
            'text': 'Signature Authority Issued Valid Till 2030'
        },
        
        # Voter ID tests
        {
            'name': 'Voter ID Front (English)',
            'doc_type': 'voter_id',
            'text': 'VOTER ID Election Commission Name: Priya Sharma Father: Sharma Singh DOB: 20/03/1992 EPIC Photo'
        },
        {
            'name': 'Voter ID Back (English)',
            'doc_type': 'voter_id',
            'text': 'Address: 456 Oak Lane Constituency: Central District Pin: 654321 State: Maharashtra'
        },
        {
            'name': 'Voter ID Front (Telugu)',
            'doc_type': 'voter_id',
            'text': 'ఓటర్ ఎన్నికలు పేరు: రాజేష్ తండ్రి: సింగ్ జన్మ: 20/03/1992 ఫోటో'
        },
        
        # Driving License tests
        {
            'name': 'Driving License Front (English)',
            'doc_type': 'driving_license',
            'text': 'DL NO DL-0123456789 Driving License Name: Amit Patel DOB: 10/05/1988 Address: 789 Pine Road Photo Validity: 2025'
        },
        {
            'name': 'Driving License Back (English)',
            'doc_type': 'driving_license',
            'text': 'Endorsement Vehicle Class: LMV COV Restrictions Signature Authority'
        },
        {
            'name': 'Driving License Front (Hindi)',
            'doc_type': 'driving_license',
            'text': 'डीएल नंबर ड्राइविंग लाइसेंस नाम: अमित पटेल जन्म: 10/05/1988 पता: 789 पाइन रोड फोटो वैधता: 2025'
        },
    ]
    
    print("=" * 80)
    print("DOCUMENT SIDE DETECTION TEST")
    print("=" * 80)
    
    for test in test_cases:
        result = detector.detect_side(test['text'], test['doc_type'])
        
        print(f"\n📄 {test['name']}")
        print(f"   Document Type: {test['doc_type'].upper()}")
        print(f"   Detected Side: {result['side'].upper()}")
        print(f"   Confidence: {result['confidence']}%")
        print(f"   Language: {result['language'].upper()}")
        print(f"   Details: {result['details']}")
        print(f"   Front Score: {result['front_score']}% | Back Score: {result['back_score']}%")
        
        # Confidence indicator
        if result['confidence'] >= 70:
            indicator = "✅ HIGH"
        elif result['confidence'] >= 40:
            indicator = "⚠️  MEDIUM"
        else:
            indicator = "❌ LOW"
        print(f"   Confidence Level: {indicator}")


if __name__ == "__main__":
    test_detector()
