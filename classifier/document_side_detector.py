"""
Document Side Detection Module
Detects front/back side of Aadhar, PAN, Voter ID, and Driving License
"""

import re
from typing import Dict


class DocumentSideDetector:
    """Detect document side (front/back) based on OCR text"""
    
    def __init__(self):
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
                'language': 'en', 'hi', or 'te'
            }
        """
        if not text or not text.strip():
            return {
                'side': 'unknown',
                'confidence': 0,
                'language': 'unknown'
            }
        
        language = self.detect_language(text)
        
        front_score = self.calculate_side_score(text, doc_type, 'front', language)
        back_score = self.calculate_side_score(text, doc_type, 'back', language)
        
        if front_score > back_score:
            side = 'front'
            confidence = front_score
        elif back_score > front_score:
            side = 'back'
            confidence = back_score
        else:
            side = 'unknown'
            confidence = 0
        
        if confidence < 30:
            side = 'unknown'
        
        return {
            'side': side,
            'confidence': round(confidence, 2),
            'language': language
        }
