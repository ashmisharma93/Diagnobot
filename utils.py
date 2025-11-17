# improved_utils.py - Better medical text preprocessing

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure downloads
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"NLTK download error: {e}")

_stopwords = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()

# CRITICAL: Medical terms to ALWAYS keep
MEDICAL_KEEP_WORDS = {
    # Symptoms
    'pain', 'ache', 'fever', 'cough', 'cold', 'hot', 'blood', 'dizzy',
    'nausea', 'vomit', 'tired', 'weak', 'fatigue', 'swelling', 'rash',
    'itch', 'sore', 'burning', 'tingling', 'numbness',
    
    # Severity
    'severe', 'mild', 'moderate', 'chronic', 'acute', 'sudden', 'sharp',
    'dull', 'constant', 'intermittent', 'extreme', 'intense',
    
    # Duration
    'day', 'days', 'week', 'weeks', 'month', 'months', 'hour', 'hours',
    'since', 'ago', 'started', 'began',
    
    # Negation (IMPORTANT!)
    'not', 'no', 'never', 'without', 'cant', 'cannot', 'wont',
    
    # Body parts
    'head', 'chest', 'stomach', 'back', 'neck', 'throat', 'eye', 'ear',
    'nose', 'mouth', 'arm', 'leg', 'hand', 'foot', 'skin',
    
    # Important modifiers
    'left', 'right', 'upper', 'lower', 'front', 'back', 'side', 'both'
}

# Remove these from stopwords
_stopwords = _stopwords - MEDICAL_KEEP_WORDS

# Medical compound phrases to preserve
MEDICAL_PHRASES = {
    "chest pain": "chestpain",
    "shortness of breath": "shortnessbreath",
    "difficulty breathing": "difficultybreathing",
    "rapid heartbeat": "rapidheartbeat",
    "heart palpitation": "heartpalpitation",
    "high fever": "highfever",
    "low fever": "lowfever",
    "severe headache": "severeheadache",
    "stomach pain": "stomachpain",
    "abdominal pain": "abdominalpain",
    "back pain": "backpain",
    "joint pain": "jointpain",
    "muscle pain": "musclepain",
    "sore throat": "sorethroat",
    "runny nose": "runnynose",
    "stuffy nose": "stuffynose",
    "blurred vision": "blurredvision",
    "double vision": "doublevision",
    "weight loss": "weightloss",
    "weight gain": "weightgain",
    "loss of appetite": "lossappetite",
    "difficulty swallowing": "difficultyswallowing",
    "blood pressure": "bloodpressure",
    "blood sugar": "bloodsugar",
    "urinary tract": "urinarytract",
    "bowel movement": "bowelmovement",
    "night sweats": "nightsweats",
    "cold sweat": "coldsweat",
    "rapid pulse": "rapidpulse",
    "irregular heartbeat": "irregularheartbeat",
    "chest tightness": "chesttightness",
    "crushing sensation": "crushingsensation",
    "radiating pain": "radiatingpain",
    "burning sensation": "burningsensation",
    "pins and needles": "pinsneedles"
}

def preprocess_text(text: str) -> str:
    """
    Enhanced medical text preprocessing
    Preserves important medical context
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # IMPORTANT: Preserve medical compound phrases
    phrase_map = {}
    for phrase, replacement in MEDICAL_PHRASES.items():
        if phrase in text:
            text = text.replace(phrase, replacement)
            phrase_map[replacement] = phrase
    
    # Preserve number + unit patterns (e.g., "3 days", "10/10 pain")
    text = re.sub(r'(\d+)\s*(day|days|week|weeks|month|year)', r'\1\2', text)
    text = re.sub(r'(\d+)/10', r'\1outof10', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = text.split()
    
    # Keep medical terms and filter stopwords
    filtered_tokens = []
    for token in tokens:
        # Keep if:
        # 1. Not a stopword, OR
        # 2. Is a medical term, OR
        # 3. Is a number, OR
        # 4. Length > 2
        if (token not in _stopwords or 
            token in MEDICAL_KEEP_WORDS or
            token.isdigit() or
            any(token.startswith(prefix) for prefix in ['high', 'low', 'severe', 'mild'])):
            if len(token) > 1:  # Keep tokens longer than 1 char
                filtered_tokens.append(token)
    
    # Lemmatize (but preserve medical compounds)
    lemmatized = []
    for token in filtered_tokens:
        if token in phrase_map:
            # Restore compound phrase
            lemmatized.append(phrase_map[token])
        else:
            lemmatized.append(_lemmatizer.lemmatize(token))
    
    return " ".join(lemmatized)


def extract_duration(text: str) -> Dict:
    """Extract time duration from text"""
    patterns = [
        r'(\d+)\s*(day|days)',
        r'(\d+)\s*(week|weeks)',
        r'(\d+)\s*(month|months)',
        r'(\d+)\s*(hour|hours)',
        r'(\d+)\s*(year|years)',
        r'since\s+(\d+)\s*(day|days|week|weeks)',
        r'for\s+(\d+)\s*(day|days|week|weeks)'
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return {
                'found': True,
                'value': int(match.group(1)),
                'unit': match.group(2),
                'text': match.group(0)
            }
    
    return {'found': False}


def extract_severity(text: str) -> Dict:
    """Extract severity level from text"""
    severity_map = {
        'mild': 1,
        'slight': 1,
        'little': 1,
        'moderate': 2,
        'medium': 2,
        'severe': 3,
        'intense': 3,
        'sharp': 3,
        'extreme': 4,
        'unbearable': 4,
        'worst': 4,
        'crushing': 4
    }
    
    text_lower = text.lower()
    
    # Check for severity words
    for word, score in severity_map.items():
        if word in text_lower:
            return {
                'found': True,
                'level': word,
                'score': score
            }
    
    # Check for numeric scale (1-10 or X/10)
    scale_patterns = [
        r'(\d+)\s*(?:out of|/)\s*10',
        r'(\d+)/10',
        r'scale\s+(?:of\s+)?(\d+)'
    ]
    
    for pattern in scale_patterns:
        match = re.search(pattern, text_lower)
        if match:
            score = int(match.group(1))
            normalized = min(score / 2.5, 4)  # Convert to 1-4 scale
            
            level_name = "mild" if score <= 3 else "moderate" if score <= 6 else "severe" if score <= 8 else "extreme"
            
            return {
                'found': True,
                'level': f"{score}/10 ({level_name})",
                'score': normalized,
                'numeric_value': score
            }
    
    return {'found': False}


def extract_location(text: str) -> List[str]:
    """Extract body part/location mentions"""
    body_parts = {
        'head': ['head', 'skull'],
        'face': ['face', 'facial'],
        'eye': ['eye', 'eyes', 'vision'],
        'ear': ['ear', 'ears', 'hearing'],
        'nose': ['nose', 'nasal'],
        'mouth': ['mouth', 'oral'],
        'throat': ['throat', 'pharynx'],
        'neck': ['neck', 'cervical'],
        'chest': ['chest', 'thorax', 'breast'],
        'heart': ['heart', 'cardiac'],
        'lung': ['lung', 'lungs', 'pulmonary'],
        'stomach': ['stomach', 'gastric', 'belly'],
        'abdomen': ['abdomen', 'abdominal'],
        'back': ['back', 'spine', 'spinal'],
        'shoulder': ['shoulder', 'shoulders'],
        'arm': ['arm', 'arms'],
        'elbow': ['elbow', 'elbows'],
        'wrist': ['wrist', 'wrists'],
        'hand': ['hand', 'hands'],
        'finger': ['finger', 'fingers'],
        'hip': ['hip', 'hips'],
        'leg': ['leg', 'legs'],
        'knee': ['knee', 'knees'],
        'ankle': ['ankle', 'ankles'],
        'foot': ['foot', 'feet'],
        'toe': ['toe', 'toes'],
        'skin': ['skin', 'dermal']
    }
    
    text_lower = text.lower()
    found_parts = []
    
    for body_part, keywords in body_parts.items():
        if any(keyword in text_lower for keyword in keywords):
            found_parts.append(body_part)
    
    return found_parts


def validate_medical_input(text: str) -> Dict:
    """Comprehensive validation of medical input"""
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'metadata': {}
    }
    
    if not text or not text.strip():
        result['valid'] = False
        result['errors'].append("Please describe your symptoms")
        return result
    
    text = text.strip()
    
    if len(text) < 5:
        result['valid'] = False
        result['errors'].append("Please provide more details (at least 5 characters)")
        return result
    
    if len(text) > 1000:
        result['warnings'].append("Input truncated to 1000 characters")
        text = text[:1000]
    
    # Check for spam patterns
    spam_patterns = [
        r'http[s]?://',
        r'www\.',
        r'buy\s+now',
        r'click\s+here',
        r'\$\d+'
    ]
    
    for pattern in spam_patterns:
        if re.search(pattern, text.lower()):
            result['valid'] = False
            result['errors'].append("Invalid input detected")
            return result
    
    # Check for medical content
    medical_keywords = [
        'pain', 'ache', 'fever', 'cough', 'cold', 'hurt', 'sore',
        'sick', 'ill', 'dizzy', 'nausea', 'vomit', 'tired', 'weak',
        'swelling', 'rash', 'itch', 'bleeding', 'breath'
    ]
    
    has_medical = any(kw in text.lower() for kw in medical_keywords)
    if not has_medical:
        result['warnings'].append("Please include specific symptoms")
    
    # Extract metadata
    duration = extract_duration(text)
    if duration['found']:
        result['metadata']['duration'] = duration
    
    severity = extract_severity(text)
    if severity['found']:
        result['metadata']['severity'] = severity
    
    location = extract_location(text)
    if location:
        result['metadata']['body_parts'] = location
    
    return result


# Test
if __name__ == "__main__":
    test_cases = [
        "I have severe chest pain for 3 days",
        "dizzy weak tired most of the time",
        "shortness of breath and rapid heartbeat",
        "stomach pain with nausea 8/10",
        "headache behind eyes since yesterday"
    ]
    
    print("Testing improved preprocessing:\n")
    for test in test_cases:
        print(f"Input: {test}")
        print(f"Processed: {preprocess_text(test)}")
        
        validation = validate_medical_input(test)
        if validation['metadata']:
            print(f"Metadata: {validation['metadata']}")
        print("-" * 60)