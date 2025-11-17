# # FINAL_FIX_diagnose_api.py - Actually accurate this time!

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from utils import preprocess_text
# from sklearn.metrics.pairwise import cosine_similarity
# import datetime
# import logging
# from typing import Dict, List, Tuple
# import re

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# MODELS_DIR = "models"

# # Load models (we'll use them as fallback only)
# try:
#     nb = joblib.load(os.path.join(MODELS_DIR, "diagnobot_nb.pkl"))
#     lr = joblib.load(os.path.join(MODELS_DIR, "diagnobot_lr.pkl"))
#     svc = joblib.load(os.path.join(MODELS_DIR, "diagnobot_svc_calibrated.pkl"))
#     le = joblib.load(os.path.join(MODELS_DIR, "diagnobot_label_encoder.pkl"))
#     embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
#     centroids_path = os.path.join(MODELS_DIR, "diagnobot_class_centroids.pkl")
#     class_centroids = joblib.load(centroids_path) if os.path.exists(centroids_path) else {}
    
#     logger.info("Models loaded")
# except Exception as e:
#     logger.error(f"Model loading failed: {e}")
#     nb = lr = svc = le = embedder = class_centroids = None

# # Import disease info
# try:
#     from disease_info import disease_info
#     logger.info(f"Loaded {len(disease_info)} diseases from disease_info.py")
# except:
#     disease_info = {}
#     logger.warning("disease_info.py not found")

# # ==============================================================================
# # COMPREHENSIVE SYMPTOM DATABASE - The Core Intelligence
# # ==============================================================================

# DIAGNOSIS_DATABASE = {
#     # Gastrointestinal
#     "gastroenteritis": {
#         "keywords": ["diarrhea", "vomiting", "stomach pain", "nausea", "abdominal cramp"],
#         "min_matches": 2,
#         "disease": "Gastroenteritis (Stomach Flu)",
#         "confidence": 0.85,
#         "specialist": "Gastroenterologist",
#         "precautions": [
#             "Drink ORS (Oral Rehydration Solution) frequently",
#             "Eat bland foods (BRAT: Banana, Rice, Applesauce, Toast)",
#             "Avoid dairy, spicy, and oily foods for 2-3 days",
#             "Rest adequately",
#             "Seek care if symptoms persist >3 days"
#         ],
#         "description": "Inflammation of stomach and intestines, usually due to viral infection or food contamination.",
#         "urgency": "Medium"
#     },
    
#     "appendicitis": {
#         "keywords": ["lower right abdominal pain", "right lower abdomen", "sharp abdominal pain lower right"],
#         "min_matches": 1,
#         "disease": "Possible Appendicitis",
#         "confidence": 0.70,
#         "specialist": "General Surgeon",
#         "precautions": [
#             "URGENT: Go to Emergency Department within 2-4 hours",
#             "Do NOT eat or drink anything",
#             "Do NOT take pain medication (can mask symptoms)",
#             "This requires immediate medical evaluation"
#         ],
#         "description": "Inflammation of the appendix. Requires urgent evaluation as it can rupture.",
#         "urgency": "High"
#     },
    
#     "constipation": {
#         "keywords": ["constipation", "difficulty passing stool", "hard stool", "bloating"],
#         "min_matches": 1,
#         "disease": "Constipation",
#         "confidence": 0.88,
#         "specialist": "Gastroenterologist",
#         "precautions": [
#             "Increase water intake (8-10 glasses/day)",
#             "Eat high-fiber foods (fruits, vegetables, whole grains)",
#             "Exercise regularly (30 min walk)",
#             "Consider mild laxative if severe",
#             "Consult doctor if persists >1 week"
#         ],
#         "description": "Difficulty passing stools or infrequent bowel movements.",
#         "urgency": "Low"
#     },
    
#     "acid_reflux": {
#         "keywords": ["heartburn", "burning chest", "acid reflux", "regurgitation", "upper abdominal pain"],
#         "min_matches": 1,
#         "disease": "Acid Reflux (GERD)",
#         "confidence": 0.82,
#         "specialist": "Gastroenterologist",
#         "precautions": [
#             "Avoid spicy, oily, and acidic foods",
#             "Don't lie down for 2-3 hours after eating",
#             "Elevate head of bed",
#             "Take antacids as needed",
#             "Eat smaller, more frequent meals"
#         ],
#         "description": "Stomach acid flows back into esophagus, causing burning sensation.",
#         "urgency": "Low"
#     },
    
#     # Respiratory
#     "flu": {
#         "keywords": ["fever", "body ache", "sore throat", "fatigue", "chills", "headache"],
#         "min_matches": 3,
#         "disease": "Influenza (Flu)",
#         "confidence": 0.87,
#         "specialist": "General Physician",
#         "precautions": [
#             "Rest for 5-7 days",
#             "Drink plenty of fluids (8-10 glasses/day)",
#             "Take paracetamol for fever (650mg every 6 hours)",
#             "Antiviral medication if within 48 hours of onset",
#             "Avoid contact with others"
#         ],
#         "description": "Viral respiratory infection causing fever, body aches, and fatigue. Self-limiting, resolves in 5-7 days.",
#         "urgency": "Medium"
#     },
    
#     "common_cold": {
#         "keywords": ["runny nose", "sneezing", "nasal congestion", "mild fever", "sore throat"],
#         "min_matches": 2,
#         "disease": "Common Cold",
#         "confidence": 0.90,
#         "specialist": "General Physician",
#         "precautions": [
#             "Rest and stay warm",
#             "Vitamin C supplements (500mg/day)",
#             "Steam inhalation 2-3 times/day",
#             "Warm fluids (soup, tea with honey)",
#             "Avoid cold environments"
#         ],
#         "description": "Viral upper respiratory infection. Self-limiting, resolves in 5-7 days.",
#         "urgency": "Low"
#     },
    
#     "pneumonia": {
#         "keywords": ["difficulty breathing", "chest pain", "cough", "fever", "shortness of breath"],
#         "min_matches": 3,
#         "disease": "Pneumonia (Suspected)",
#         "confidence": 0.75,
#         "specialist": "Pulmonologist",
#         "precautions": [
#             "URGENT: Seek medical care within 4-6 hours",
#             "Chest X-ray required",
#             "Antibiotics will be prescribed",
#             "Monitor oxygen levels",
#             "Complete full course of medication"
#         ],
#         "description": "Infection causing inflammation in lungs. Requires antibiotics.",
#         "urgency": "High"
#     },
    
#     # Urinary
#     "uti": {
#         "keywords": ["burning urination", "frequent urination", "cloudy urine", "blood in urine", "pelvic pain"],
#         "min_matches": 2,
#         "disease": "Urinary Tract Infection (UTI)",
#         "confidence": 0.88,
#         "specialist": "Urologist",
#         "precautions": [
#             "Drink 3-4 liters water daily",
#             "Cranberry juice (unsweetened)",
#             "Urine culture test required",
#             "Complete full course of antibiotics",
#             "Maintain personal hygiene"
#         ],
#         "description": "Bacterial infection of urinary tract. Requires antibiotics.",
#         "urgency": "Medium"
#     },
    
#     # Neurological
#     "migraine": {
#         "keywords": ["severe headache", "one sided headache", "nausea", "sensitivity to light", "visual disturbances"],
#         "min_matches": 2,
#         "disease": "Migraine",
#         "confidence": 0.83,
#         "specialist": "Neurologist",
#         "precautions": [
#             "Rest in dark, quiet room",
#             "Cold compress on forehead",
#             "Avoid triggers (bright light, loud noise, stress)",
#             "Pain medication as prescribed",
#             "Keep headache diary to identify triggers"
#         ],
#         "description": "Neurological condition causing intense headaches with nausea and light sensitivity.",
#         "urgency": "Medium"
#     },
    
#     "tension_headache": {
#         "keywords": ["headache", "pressure head", "tight band head", "stress headache"],
#         "min_matches": 1,
#         "disease": "Tension Headache",
#         "confidence": 0.85,
#         "specialist": "General Physician",
#         "precautions": [
#             "Take paracetamol or ibuprofen",
#             "Rest in quiet place",
#             "Massage temples and neck",
#             "Stress management",
#             "Adequate sleep (7-8 hours)"
#         ],
#         "description": "Common headache caused by muscle tension, often due to stress.",
#         "urgency": "Low"
#     },
    
#     # Musculoskeletal
#     "back_pain": {
#         "keywords": ["back pain", "lower back pain", "spinal pain"],
#         "min_matches": 1,
#         "disease": "Non-specific Back Pain",
#         "confidence": 0.80,
#         "specialist": "Orthopedist",
#         "precautions": [
#             "Apply hot/cold compress",
#             "Gentle stretching exercises",
#             "Maintain good posture",
#             "Sleep on firm mattress",
#             "Avoid heavy lifting"
#         ],
#         "description": "Pain in back muscles or spine, often due to strain or poor posture.",
#         "urgency": "Low"
#     },
    
#     # Skin
#     "allergic_reaction": {
#         "keywords": ["rash", "itching", "hives", "skin redness", "swelling"],
#         "min_matches": 2,
#         "disease": "Allergic Reaction",
#         "confidence": 0.85,
#         "specialist": "Dermatologist",
#         "precautions": [
#             "Identify and avoid allergen",
#             "Take antihistamine (cetirizine 10mg)",
#             "Apply calamine lotion",
#             "Cold compress on affected area",
#             "Seek ER if breathing difficulty"
#         ],
#         "description": "Immune system reaction to allergen causing skin symptoms.",
#         "urgency": "Low"
#     },
    
#     # General
#     "viral_fever": {
#         "keywords": ["fever", "body ache", "weakness", "fatigue"],
#         "min_matches": 2,
#         "disease": "Viral Fever",
#         "confidence": 0.82,
#         "specialist": "General Physician",
#         "precautions": [
#             "Rest adequately",
#             "Hydrate (water, ORS, coconut water)",
#             "Paracetamol for fever (650mg every 6 hours)",
#             "Light, easily digestible food",
#             "Monitor temperature 3 times daily"
#         ],
#         "description": "Common viral infection causing fever and body aches. Self-limiting within 3-5 days.",
#         "urgency": "Medium"
#     },
    
#     "food_poisoning": {
#         "keywords": ["vomiting", "diarrhea", "stomach cramps", "nausea", "after eating"],
#         "min_matches": 2,
#         "disease": "Food Poisoning",
#         "confidence": 0.83,
#         "specialist": "Gastroenterologist",
#         "precautions": [
#             "Drink ORS frequently",
#             "Avoid solid food for 4-6 hours",
#             "Then start with bland foods",
#             "Rest",
#             "Seek care if symptoms severe or >24 hours"
#         ],
#         "description": "Illness from consuming contaminated food or water.",
#         "urgency": "Medium"
#     },
    
#     "dehydration": {
#         "keywords": ["dry mouth", "dizziness", "dark urine", "weakness", "fatigue", "thirst"],
#         "min_matches": 2,
#         "disease": "Dehydration",
#         "confidence": 0.85,
#         "specialist": "General Physician",
#         "precautions": [
#             "Drink water immediately (small sips)",
#             "ORS solution",
#             "Coconut water",
#             "Avoid caffeine and alcohol",
#             "Seek ER if severe (unable to keep fluids down)"
#         ],
#         "description": "Lack of adequate fluids in body.",
#         "urgency": "Medium"
#     },
    
#     "anxiety": {
#         "keywords": ["anxiety", "panic", "rapid heartbeat", "sweating", "nervousness", "worry"],
#         "min_matches": 2,
#         "disease": "Anxiety",
#         "confidence": 0.78,
#         "specialist": "Psychiatrist / Psychologist",
#         "precautions": [
#             "Deep breathing exercises",
#             "Meditation and mindfulness",
#             "Regular exercise",
#             "Adequate sleep",
#             "Consider therapy/counseling"
#         ],
#         "description": "Mental health condition causing excessive worry and physical symptoms.",
#         "urgency": "Low"
#     }
# }

# # ==============================================================================
# # EMERGENCY DETECTION
# # ==============================================================================

# EMERGENCY_PATTERNS = {
#     "heart_attack": {
#         "keywords": ["crushing chest pain", "severe chest pain", "chest pressure", "left arm pain", "jaw pain"],
#         "min_matches": 2
#     },
#     "stroke": {
#         "keywords": ["facial drooping", "slurred speech", "one side weakness", "sudden weakness", "face droop"],
#         "min_matches": 2
#     },
#     "severe_bleeding": {
#         "keywords": ["vomiting blood", "coughing blood", "heavy bleeding", "large blood loss"],
#         "min_matches": 1
#     },
#     "anaphylaxis": {
#         "keywords": ["difficulty breathing", "throat swelling", "tongue swelling", "severe allergic"],
#         "min_matches": 1
#     }
# }

# def detect_emergency(text: str) -> Dict:
#     """Detect life-threatening emergencies"""
#     text_lower = text.lower()
    
#     for emergency_type, pattern in EMERGENCY_PATTERNS.items():
#         matches = sum(1 for kw in pattern["keywords"] if kw in text_lower)
        
#         if matches >= pattern["min_matches"]:
#             return {
#                 "emergency": True,
#                 "type": emergency_type.upper(),
#                 "matched": [kw for kw in pattern["keywords"] if kw in text_lower]
#             }
    
#     return {"emergency": False}

# # ==============================================================================
# # SMART DIAGNOSIS ENGINE
# # ==============================================================================

# def diagnose(user_input: str, follow_up_answers: List[str] = None) -> dict:
#     """
#     PRIMARY DIAGNOSIS FUNCTION
    
#     Priority:
#     1. Emergency detection
#     2. Pattern matching from database
#     3. ML fallback (rarely used)
#     """
#     try:
#         # Validation
#         if not user_input or len(user_input.strip()) < 3:
#             return {"error": "Please describe your symptoms"}
        
#         # Combine all input
#         combined = user_input.lower()
#         if follow_up_answers:
#             combined += " " + " ".join([str(a).lower() for a in follow_up_answers])
        
#         logger.info(f"Diagnosing: {combined}")
        
#         # Step 1: Check for emergencies
#         emergency = detect_emergency(combined)
#         if emergency["emergency"]:
#             logger.critical(f"EMERGENCY: {emergency['type']}")
#             return {
#                 "emergency": True,
#                 "predicted_disease": f"🚨 {emergency['type']} SUSPECTED",
#                 "confidence": 0.95,
#                 "urgency": "CRITICAL",
#                 "referral": get_emergency_response(),
#                 "precautions": [
#                     "CALL 102/108/112 IMMEDIATELY",
#                     "Go to Emergency Department NOW",
#                     "Do NOT wait or delay",
#                     "Time is critical"
#                 ],
#                 "specialist": "Emergency Medicine",
#                 "matched_symptoms": emergency["matched"]
#             }
        
#         # Step 2: Pattern matching from database
#         best_match = None
#         best_score = 0
        
#         for pattern_id, pattern in DIAGNOSIS_DATABASE.items():
#             keywords = pattern["keywords"]
#             min_matches = pattern["min_matches"]
            
#             # Count matches
#             matches = sum(1 for kw in keywords if kw in combined)
            
#             if matches >= min_matches:
#                 # Calculate score (higher = better match)
#                 score = matches / len(keywords)
                
#                 # Boost score for exact phrase matches
#                 for kw in keywords:
#                     if kw in combined and len(kw) > 5:  # Longer phrases get bonus
#                         score += 0.1
                
#                 if score > best_score:
#                     best_score = score
#                     best_match = {
#                         **pattern,
#                         "match_score": score,
#                         "matched_keywords": [kw for kw in keywords if kw in combined],
#                         "pattern_id": pattern_id
#                     }
        
#         # If we have a good match, use it
#         if best_match and best_score >= 0.15:  # Threshold: 15% of keywords match
#             logger.info(f"Pattern match: {best_match['disease']} (score: {best_score:.2f})")
            
#             # Calculate urgency
#             urgency = best_match.get("urgency", "Medium")
            
#             # Adjust confidence based on match quality
#             confidence = best_match["confidence"]
#             if best_score >= 0.5:  # >50% keywords matched
#                 confidence = min(confidence * 1.1, 0.95)
#             elif best_score >= 0.3:  # 30-50% matched
#                 confidence = confidence
#             else:  # <30% matched
#                 confidence = confidence * 0.9
            
#             return {
#                 "predicted_disease": best_match["disease"],
#                 "confidence": round(confidence, 2),
#                 "confidence_explanation": f"Pattern matching (matched {len(best_match['matched_keywords'])} symptoms)",
#                 "explanation": best_match["description"],
#                 "precautions": best_match["precautions"],
#                 "specialist": best_match["specialist"],
#                 "urgency": urgency,
#                 "urgency_score": calculate_urgency_score(combined, urgency),
#                 "referral": get_referral(urgency, confidence),
#                 "matched_symptoms": best_match["matched_keywords"],
#                 "diagnosis_method": "pattern_matching",
#                 "emergency": False
#             }
        
#         # Step 3: ML fallback (if no pattern match)
#         logger.info("No strong pattern match, using ML model fallback")
        
#         if svc is None:
#             return {
#                 "error": "Unable to diagnose. Please provide more specific symptoms.",
#                 "suggestion": "Try describing: location of pain, severity, duration, other symptoms"
#             }
        
#         cleaned = preprocess_text(combined)
#         X_emb = embedder.encode([cleaned])
        
#         probs_svc = svc.predict_proba(X_emb)[0]
#         probs_lr = lr.predict_proba(X_emb)[0]
#         probs_nb = nb.predict_proba(np.abs(X_emb))[0]
        
#         ensemble = np.average([probs_svc, probs_lr, probs_nb], 
#                              axis=0, weights=[0.5, 0.3, 0.2])
        
#         top_idx = np.argsort(ensemble)[-3:][::-1]
#         top_diseases = le.inverse_transform(top_idx)
#         top_conf = ensemble[top_idx]
        
#         ml_disease = str(top_diseases[0])
#         ml_conf = float(top_conf[0])
        
#         # Get info from disease_info.py if available
#         disease_data = disease_info.get(ml_disease, {})
        
#         return {
#             "predicted_disease": ml_disease,
#             "confidence": round(ml_conf * 0.85, 2),  # Reduce confidence for ML-only
#             "confidence_explanation": "ML prediction - Consider consulting doctor for confirmation",
#             "explanation": disease_data.get("description", f"{ml_disease} - Medical evaluation recommended"),
#             "precautions": disease_data.get("precautions", [
#                 "Consult a healthcare provider for proper diagnosis",
#                 "Monitor your symptoms closely",
#                 "Rest adequately",
#                 "Stay hydrated"
#             ]),
#             "specialist": disease_data.get("specialist", "General Physician"),
#             "urgency": "Medium",
#             "urgency_score": 5,
#             "referral": get_referral("Medium", ml_conf),
#             "diagnosis_method": "ml_fallback",
#             "alternatives": [
#                 {"disease": str(d), "confidence": round(float(c) * 0.85, 2)}
#                 for d, c in zip(top_diseases[1:], top_conf[1:])
#             ],
#             "emergency": False,
#             "note": "Low confidence - symptoms not clearly matched. Please consult doctor."
#         }
        
#     except Exception as e:
#         logger.error(f"Diagnosis error: {str(e)}", exc_info=True)
#         return {
#             "error": "An error occurred. Please rephrase your symptoms.",
#             "suggestion": "Be specific: What hurts? Where? How severe? How long?"
#         }

# def calculate_urgency_score(text: str, urgency: str) -> int:
#     """Calculate numeric urgency score"""
#     urgency_map = {"CRITICAL": 15, "High": 10, "Medium": 5, "Low": 2}
#     return urgency_map.get(urgency, 5)

# def get_referral(urgency: str, confidence: float) -> str:
#     """Generate referral text"""
    
#     if urgency == "CRITICAL":
#         return """
# 🚨 **EMERGENCY - ACT NOW**

# CALL 102/108/112 IMMEDIATELY
# Go to Emergency Department
# Do NOT delay - Time critical
# """
    
#     elif urgency == "High":
#         return """
# ⚠️ **URGENT CARE NEEDED**

# Seek medical attention within 2-6 hours:
# • Emergency Department
# • District Hospital
# • Private Hospital Emergency

# Call 102/108 if symptoms worsen
# """
    
#     elif urgency == "Medium":
#         timeline = "24 hours" if confidence >= 0.75 else "24-48 hours"
#         return f"""
# 📋 **CONSULT DOCTOR WITHIN {timeline.upper()}**

# Options:
# • Primary Health Centre (PHC) - Free
# • Private GP clinic (₹300-800)
# • Telemedicine (Practo, 1mg)

# **Seek immediate care if symptoms worsen**
# """
    
#     else:
#         return """
# 💡 **SELF-CARE WITH MONITORING**

# • Rest and hydrate well
# • Monitor symptoms for 2-3 days
# • Try home remedies if appropriate
# • Consult doctor if:
#   - Symptoms persist >3 days
#   - Symptoms worsen
#   - New symptoms develop
# """

# def get_emergency_response() -> str:
#     """Emergency response text"""
#     return """
# 🚨 **MEDICAL EMERGENCY**

# **IMMEDIATE ACTIONS:**

# 1. **CALL NOW:**
#    • Ambulance: 102 / 108
#    • Emergency: 112

# 2. **DO NOT WAIT:**
#    • Go to nearest Emergency Department
#    • Do NOT drive yourself
#    • Alert family/friends

# 3. **STAY CALM:**
#    • Note time symptoms started
#    • Bring any medications you take
#    • Someone should accompany you

# ---

# ⚠️ This is time-critical. Act immediately.
# """

# def detect_body_system(symptoms: str) -> str:
#     """Detect body system"""
#     s = symptoms.lower()
    
#     if any(x in s for x in ["abdomen", "stomach", "nausea", "vomit", "diarrhea"]):
#         return "digestive"
#     elif any(x in s for x in ["cough", "breath", "throat", "chest"]):
#         return "respiratory"
#     elif any(x in s for x in ["urine", "bladder"]):
#         return "urinary"
#     elif any(x in s for x in ["headache", "dizzy"]):
#         return "neurological"
#     else:
#         return "general"

# def log_feedback(symptoms, predicted, confidence, feedback, correction=None, urgency=None):
#     """Log user feedback"""
#     try:
#         os.makedirs("logs", exist_ok=True)
#         log_file = "logs/feedback_logs.csv"
        
#         entry = {
#             "timestamp": datetime.datetime.now().isoformat(),
#             "symptoms": symptoms,
#             "predicted_disease": predicted,
#             "confidence": confidence,
#             "user_feedback": feedback,
#             "user_correction": correction,
#             "urgency": urgency
#         }
        
#         df = pd.DataFrame([entry])
#         if os.path.exists(log_file):
#             df.to_csv(log_file, mode="a", index=False, header=False)
#         else:
#             df.to_csv(log_file, index=False, header=True)
            
#         logger.info(f"Feedback logged: {feedback}")
#     except Exception as e:
#         logger.error(f"Error logging feedback: {str(e)}")

# # Test
# if __name__ == "__main__":
#     tests = [
#         "pain in lower abdomen severity 5",
#         "high fever sore throat body ache",
#         "vomiting diarrhea stomach cramps",
#         "burning urination frequent urination",
#         "severe headache nausea sensitivity to light"
#     ]
    
#     print("\n" + "="*70)
#     print("TESTING DIAGNOSIS SYSTEM")
#     print("="*70 + "\n")
    
#     for test in tests:
#         print(f"Input: {test}")
#         result = diagnose(test)
        
#         if "error" in result:
#             print(f"Error: {result['error']}")
#         else:
#             print(f"Disease: {result['predicted_disease']}")
#             print(f"Confidence: {result['confidence']:.0%}")
#             print(f"Method: {result.get('diagnosis_method', 'unknown')}")
#             print(f"Urgency: {result['urgency']}")
#         print("-"*70 + "\n")

# enhanced_diagnose_api.py - Significantly improved diagnosis accuracy

import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import preprocess_text
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import logging
from typing import Dict, List, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models"

# Load models
try:
    nb = joblib.load(os.path.join(MODELS_DIR, "diagnobot_nb.pkl"))
    lr = joblib.load(os.path.join(MODELS_DIR, "diagnobot_lr.pkl"))
    svc = joblib.load(os.path.join(MODELS_DIR, "diagnobot_svc_calibrated.pkl"))
    le = joblib.load(os.path.join(MODELS_DIR, "diagnobot_label_encoder.pkl"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    centroids_path = os.path.join(MODELS_DIR, "diagnobot_class_centroids.pkl")
    class_centroids = joblib.load(centroids_path) if os.path.exists(centroids_path) else {}
    
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    nb = lr = svc = le = embedder = class_centroids = None

# Import disease info
try:
    from disease_info import disease_info
    logger.info(f"Loaded {len(disease_info)} diseases from disease_info.py")
except:
    disease_info = {}
    logger.warning("disease_info.py not found")

# ==============================================================================
# EXPANDED SYMPTOM DATABASE - Much more comprehensive
# ==============================================================================

DIAGNOSIS_DATABASE = {
    # === GASTROINTESTINAL ===
    "gastroenteritis": {
        "keywords": ["diarrhea", "loose stool", "watery stool", "vomiting", "stomach pain", 
                    "abdominal pain", "nausea", "cramping", "stomach cramp"],
        "min_matches": 2,
        "disease": "Gastroenteritis (Stomach Flu)",
        "confidence": 0.88,
        "specialist": "Gastroenterologist",
        "precautions": [
            "Drink ORS (Oral Rehydration Solution) every 2 hours",
            "BRAT diet: Banana, Rice, Applesauce, Toast",
            "Avoid dairy, spicy food, caffeine for 48 hours",
            "Rest adequately - stay home from work/school",
            "See doctor if: blood in stool, fever >102°F, severe dehydration"
        ],
        "description": "Inflammation of stomach and intestines, usually viral or from contaminated food/water. Self-limiting in 24-72 hours.",
        "urgency": "Medium",
        "causes": ["Viral infection (norovirus, rotavirus)", "Bacterial contamination", "Food poisoning"]
    },
    
    "food_poisoning": {
        "keywords": ["vomiting", "nausea", "diarrhea", "ate", "food", "after eating", 
                    "stomach upset", "bad food", "spoiled"],
        "min_matches": 2,
        "disease": "Food Poisoning",
        "confidence": 0.86,
        "specialist": "Gastroenterologist",
        "precautions": [
            "Stop eating solid food for 4-6 hours",
            "Sip ORS or clear fluids frequently",
            "When improving: start bland foods (crackers, toast)",
            "Avoid anti-diarrhea medication unless prescribed",
            "URGENT care if: bloody stool, high fever, severe weakness"
        ],
        "description": "Illness from consuming contaminated food. Usually resolves in 24-48 hours.",
        "urgency": "Medium",
        "causes": ["Bacteria (Salmonella, E.coli)", "Toxins in food", "Parasites"]
    },
    
    "acid_reflux_gerd": {
        "keywords": ["heartburn", "burning chest", "acid", "reflux", "regurgitation", 
                    "chest burning", "sour taste", "throat burning"],
        "min_matches": 1,
        "disease": "Acid Reflux (GERD)",
        "confidence": 0.85,
        "specialist": "Gastroenterologist",
        "precautions": [
            "Avoid: spicy food, citrus, tomatoes, chocolate, caffeine",
            "Don't lie down for 3 hours after eating",
            "Elevate head of bed 6 inches",
            "Eat smaller, more frequent meals",
            "Take antacids (omeprazole, pantoprazole) if prescribed"
        ],
        "description": "Stomach acid flows back into esophagus causing burning. Chronic condition manageable with lifestyle changes.",
        "urgency": "Low",
        "causes": ["Weak lower esophageal sphincter", "Hiatal hernia", "Obesity", "Certain foods"]
    },
    
    "constipation": {
        "keywords": ["constipation", "hard stool", "difficult stool", "cant poop", 
                    "no bowel", "bloating", "straining"],
        "min_matches": 1,
        "disease": "Constipation",
        "confidence": 0.90,
        "specialist": "Gastroenterologist",
        "precautions": [
            "Drink 8-10 glasses water daily",
            "High fiber: fruits, vegetables, whole grains, oats",
            "Exercise 30 minutes daily (even walking helps)",
            "Prune juice or psyllium husk (Isabgol)",
            "If severe: mild laxative (milk of magnesia)"
        ],
        "description": "Difficulty passing stools or infrequent bowel movements. Usually diet/lifestyle related.",
        "urgency": "Low",
        "causes": ["Low fiber diet", "Dehydration", "Lack of exercise", "Medications"]
    },
    
    "appendicitis": {
        "keywords": ["right lower abdomen pain", "lower right pain", "sharp abdomen", 
                    "mcburney point", "right side pain", "appendix"],
        "min_matches": 1,
        "disease": "⚠️ Possible Appendicitis",
        "confidence": 0.75,
        "specialist": "General Surgeon (Emergency)",
        "precautions": [
            "🚨 GO TO EMERGENCY ROOM IMMEDIATELY",
            "Do NOT eat or drink anything",
            "Do NOT take pain medication (masks symptoms)",
            "Do NOT apply heat to abdomen",
            "Time is critical - appendix can rupture"
        ],
        "description": "Inflammation of appendix. Medical emergency requiring surgery. Can rupture if untreated.",
        "urgency": "High",
        "causes": ["Blockage of appendix opening", "Infection"]
    },
    
    # === RESPIRATORY ===
    "common_cold": {
        "keywords": ["runny nose", "stuffy nose", "sneezing", "sore throat", "congestion",
                    "nasal", "nose blocked", "throat scratchy", "mild fever"],
        "min_matches": 2,
        "disease": "Common Cold",
        "confidence": 0.92,
        "specialist": "General Physician",
        "precautions": [
            "Rest 7-8 hours sleep",
            "Hot tea with honey and ginger",
            "Steam inhalation 3 times daily",
            "Vitamin C 500mg daily",
            "Saltwater gargle for sore throat",
            "Resolves in 5-7 days naturally"
        ],
        "description": "Viral upper respiratory infection. Self-limiting, very common.",
        "urgency": "Low",
        "causes": ["Rhinovirus", "Other respiratory viruses", "Spread by droplets"]
    },
    
    "flu_influenza": {
        "keywords": ["fever", "high fever", "body ache", "chills", "fatigue", "exhausted",
                    "muscle ache", "weakness", "headache", "sore throat"],
        "min_matches": 3,
        "disease": "Influenza (Flu)",
        "confidence": 0.89,
        "specialist": "General Physician",
        "precautions": [
            "Complete bed rest 5-7 days",
            "Paracetamol 650mg every 6 hours for fever",
            "Drink 10-12 glasses fluids (water, soup, ORS)",
            "Antiviral (Oseltamivir) if within 48 hours of onset",
            "Isolate to avoid spreading",
            "See doctor if: breathing difficulty, chest pain, confusion"
        ],
        "description": "Viral respiratory infection more severe than cold. Resolves in 5-10 days but exhaustion can last 2 weeks.",
        "urgency": "Medium",
        "causes": ["Influenza virus A, B, C", "Highly contagious airborne spread"]
    },
    
    "bronchitis": {
        "keywords": ["cough", "chest congestion", "mucus", "phlegm", "wheezing",
                    "chest tight", "productive cough", "rattling chest"],
        "min_matches": 2,
        "disease": "Acute Bronchitis",
        "confidence": 0.84,
        "specialist": "Pulmonologist",
        "precautions": [
            "Rest and hydrate well",
            "Honey for cough relief (1 tbsp)",
            "Steam inhalation to loosen mucus",
            "Avoid smoke, dust, pollution",
            "Cough suppressant at night if needed",
            "Antibiotics ONLY if bacterial (doctor will decide)"
        ],
        "description": "Inflammation of bronchial tubes. Usually viral, resolves in 2-3 weeks.",
        "urgency": "Medium",
        "causes": ["Viral infection", "Sometimes bacterial", "Irritants (smoke)"]
    },
    
    "pneumonia": {
        "keywords": ["difficulty breathing", "shortness breath", "chest pain", "cough",
                    "fever", "breathing hard", "gasping", "rapid breathing"],
        "min_matches": 3,
        "disease": "⚠️ Pneumonia (Suspected)",
        "confidence": 0.78,
        "specialist": "Pulmonologist",
        "precautions": [
            "🚨 SEEK MEDICAL CARE WITHIN 6 HOURS",
            "Chest X-ray required for diagnosis",
            "Antibiotics will be prescribed",
            "Hospitalization may be needed",
            "Monitor oxygen levels",
            "Complete full antibiotic course"
        ],
        "description": "Lung infection causing inflammation. Requires antibiotics. Can be serious, especially in elderly/children.",
        "urgency": "High",
        "causes": ["Bacterial infection (Streptococcus)", "Viral", "Rarely fungal"]
    },
    
    "asthma_attack": {
        "keywords": ["wheezing", "cant breathe", "tight chest", "gasping", "inhaler",
                    "asthma", "breathing attack", "chest tight"],
        "min_matches": 2,
        "disease": "Asthma Attack/Exacerbation",
        "confidence": 0.87,
        "specialist": "Pulmonologist",
        "precautions": [
            "Use rescue inhaler (salbutamol) immediately",
            "Sit upright, don't lie down",
            "Slow deep breaths",
            "If no improvement in 15 min: GO TO ER",
            "Avoid triggers: dust, smoke, cold air, allergens",
            "Follow up with doctor to adjust maintenance medication"
        ],
        "description": "Airways narrow making breathing difficult. Chronic condition requiring management.",
        "urgency": "High",
        "causes": ["Allergens", "Cold air", "Exercise", "Respiratory infections", "Stress"]
    },
    
    # === URINARY ===
    "uti": {
        "keywords": ["burning urination", "painful urination", "frequent urination",
                    "urge to urinate", "cloudy urine", "blood urine", "pee burns"],
        "min_matches": 2,
        "disease": "Urinary Tract Infection (UTI)",
        "confidence": 0.90,
        "specialist": "Urologist",
        "precautions": [
            "Drink 3-4 liters water daily (flush bacteria)",
            "Cranberry juice (unsweetened) helpful",
            "Urine culture test required",
            "Antibiotics: complete full course (3-7 days)",
            "Maintain hygiene",
            "Avoid: caffeine, alcohol, spicy food during infection"
        ],
        "description": "Bacterial infection of urinary tract. Very common, especially in women. Easily treatable.",
        "urgency": "Medium",
        "causes": ["E. coli bacteria", "Poor hygiene", "Holding urine too long", "Sexual activity"]
    },
    
    # === NEUROLOGICAL ===
    "migraine": {
        "keywords": ["severe headache", "one side headache", "pulsing headache", "nausea",
                    "light sensitive", "sound sensitive", "visual aura", "throbbing head"],
        "min_matches": 2,
        "disease": "Migraine",
        "confidence": 0.86,
        "specialist": "Neurologist",
        "precautions": [
            "Rest in dark, quiet room immediately",
            "Cold compress on forehead/neck",
            "Pain medication: sumatriptan or prescribed migraine medication",
            "Avoid triggers: bright light, loud noise, certain foods",
            "Track triggers in diary",
            "Preventive medication if >4 episodes/month"
        ],
        "description": "Neurological condition causing severe headaches. Can last 4-72 hours. Manageable with medication.",
        "urgency": "Medium",
        "causes": ["Genetic predisposition", "Triggers vary: stress, foods, hormones, weather"]
    },
    
    "tension_headache": {
        "keywords": ["headache", "head pressure", "tight band", "dull head", "stress headache",
                    "temple pain", "scalp tender"],
        "min_matches": 1,
        "disease": "Tension Headache",
        "confidence": 0.88,
        "specialist": "General Physician",
        "precautions": [
            "Paracetamol or ibuprofen",
            "Rest in quiet place",
            "Massage temples, neck, shoulders",
            "Hot compress on neck/shoulders",
            "Stress management: deep breathing, meditation",
            "Improve posture, take breaks from screens"
        ],
        "description": "Most common headache type. Caused by muscle tension from stress or poor posture.",
        "urgency": "Low",
        "causes": ["Stress", "Anxiety", "Poor posture", "Eye strain", "Lack of sleep"]
    },
    
    # === SKIN ===
    "allergic_reaction": {
        "keywords": ["rash", "itching", "hives", "skin redness", "swelling", "itchy",
                    "red bumps", "welts", "skin irritation"],
        "min_matches": 2,
        "disease": "Allergic Reaction",
        "confidence": 0.87,
        "specialist": "Dermatologist / Allergist",
        "precautions": [
            "Identify and avoid allergen",
            "Antihistamine: cetirizine 10mg or loratadine 10mg",
            "Calamine lotion on affected area",
            "Cold compress to reduce swelling",
            "Don't scratch (causes infection)",
            "🚨 ER if: throat swelling, breathing difficulty (anaphylaxis)"
        ],
        "description": "Immune response to allergen. Usually mild but can be severe (anaphylaxis).",
        "urgency": "Low",
        "causes": ["Food allergens", "Insect bites", "Medications", "Pollen", "Cosmetics"]
    },
    
    # === GENERAL ===
    "viral_fever": {
        "keywords": ["fever", "temperature", "body ache", "weakness", "tired", "chills",
                    "not feeling well", "feverish"],
        "min_matches": 2,
        "disease": "Viral Fever",
        "confidence": 0.84,
        "specialist": "General Physician",
        "precautions": [
            "Rest completely",
            "Paracetamol 650mg every 6 hours (not more than 4 times/day)",
            "Sponge bath with lukewarm water if fever >102°F",
            "Fluids: water, ORS, coconut water, soups",
            "Light, easily digestible food",
            "Monitor temperature every 4 hours",
            "See doctor if: fever >3 days, >103°F, seizures, rash"
        ],
        "description": "Common viral infection. Self-limiting, resolves in 3-5 days.",
        "urgency": "Medium",
        "causes": ["Various viruses", "Dengue, chikungunya in endemic areas", "Seasonal infections"]
    },
    
    "dehydration": {
        "keywords": ["dry mouth", "thirsty", "dark urine", "dizzy", "lightheaded",
                    "no urine", "headache", "weak", "fatigue"],
        "min_matches": 2,
        "disease": "Dehydration",
        "confidence": 0.89,
        "specialist": "General Physician",
        "precautions": [
            "Drink water immediately (small frequent sips)",
            "ORS solution (best for rehydration)",
            "Coconut water, lemon water with salt/sugar",
            "Avoid: caffeine, alcohol (worsen dehydration)",
            "Rest in cool environment",
            "🚨 ER if: can't keep fluids down, very dark urine, confusion, rapid heartbeat"
        ],
        "description": "Insufficient body fluids. Can range from mild to severe.",
        "urgency": "Medium",
        "causes": ["Diarrhea/vomiting", "Excessive sweating", "Insufficient water intake", "Diabetes"]
    },
    
    "anxiety_panic": {
        "keywords": ["anxiety", "panic", "rapid heartbeat", "sweating", "nervous",
                    "racing heart", "worry", "chest tight", "cant breathe", "fear"],
        "min_matches": 2,
        "disease": "Anxiety / Panic Attack",
        "confidence": 0.81,
        "specialist": "Psychiatrist / Psychologist",
        "precautions": [
            "Deep breathing: inhale 4 counts, hold 4, exhale 6",
            "Grounding technique: name 5 things you see, 4 you touch, 3 you hear",
            "Sit down in calm environment",
            "Remind yourself it will pass (panic peaks at 10 min)",
            "Regular: exercise, meditation, adequate sleep",
            "Consider therapy (CBT very effective)",
            "Medication if severe (consult psychiatrist)"
        ],
        "description": "Mental health condition causing excessive worry and physical symptoms. Very treatable.",
        "urgency": "Low",
        "causes": ["Stress", "Genetics", "Brain chemistry", "Trauma", "Life circumstances"]
    },
    
    # === DIABETES-RELATED ===
    "hyperglycemia": {
        "keywords": ["high sugar", "excessive thirst", "frequent urination", "blurred vision",
                    "sugar high", "diabetes", "glucose high"],
        "min_matches": 2,
        "disease": "Hyperglycemia (High Blood Sugar)",
        "confidence": 0.83,
        "specialist": "Endocrinologist",
        "precautions": [
            "Check blood sugar immediately",
            "Drink water (helps flush sugar)",
            "Take prescribed diabetes medication/insulin",
            "Light exercise if sugar <300 mg/dL",
            "Avoid: sugary foods, refined carbs",
            "🚨 ER if: sugar >400, vomiting, confusion, fruity breath"
        ],
        "description": "Elevated blood glucose. Requires management to avoid complications.",
        "urgency": "Medium",
        "causes": ["Missed diabetes medication", "Excessive carb intake", "Illness/infection", "Stress"]
    },
    
    # === MUSCULOSKELETAL ===
    "back_pain": {
        "keywords": ["back pain", "lower back", "spine", "back hurts", "back ache",
                    "lumbar pain"],
        "min_matches": 1,
        "disease": "Non-specific Back Pain",
        "confidence": 0.82,
        "specialist": "Orthopedist / Physiotherapist",
        "precautions": [
            "Apply hot compress 15-20 min, 3 times daily",
            "Gentle stretching exercises",
            "Maintain good posture when sitting",
            "Sleep on firm mattress, pillow between knees if side sleeping",
            "Avoid heavy lifting",
            "Pain medication: ibuprofen or paracetamol",
            "Physical therapy if chronic"
        ],
        "description": "Very common. Usually muscle strain from poor posture or overexertion. Most resolve in 2-4 weeks.",
        "urgency": "Low",
        "causes": ["Muscle strain", "Poor posture", "Heavy lifting", "Prolonged sitting", "Injury"]
    },
}

# ==============================================================================
# EMERGENCY PATTERNS - Expanded
# ==============================================================================

EMERGENCY_PATTERNS = {
    "heart_attack": {
        "keywords": ["crushing chest pain", "severe chest pain", "chest pressure", "left arm pain",
                    "jaw pain", "chest squeezing", "elephant chest", "heart attack"],
        "min_matches": 2
    },
    "stroke": {
        "keywords": ["facial drooping", "face droop", "slurred speech", "one side weakness",
                    "sudden weakness", "arm weakness one side", "cant speak properly"],
        "min_matches": 2
    },
    "severe_bleeding": {
        "keywords": ["vomiting blood", "coughing blood", "heavy bleeding", "large blood loss",
                    "bleeding wont stop", "hemorrhage"],
        "min_matches": 1
    },
    "anaphylaxis": {
        "keywords": ["cant breathe", "throat swelling", "tongue swelling", "severe allergic",
                    "throat closing", "lips swelling", "breathing stopped"],
        "min_matches": 1
    },
    "severe_asthma": {
        "keywords": ["cant breathe", "gasping", "blue lips", "inhaler not working",
                    "turning blue", "severe wheezing"],
        "min_matches": 2
    },
    "diabetic_emergency": {
        "keywords": ["blood sugar very high", "blood sugar very low", "unconscious diabetic",
                    "sugar over 400", "sugar under 50", "diabetic coma"],
        "min_matches": 1
    }
}

def detect_emergency(text: str) -> Dict:
    """Detect life-threatening emergencies"""
    text_lower = text.lower()
    
    for emergency_type, pattern in EMERGENCY_PATTERNS.items():
        matches = sum(1 for kw in pattern["keywords"] if kw in text_lower)
        
        if matches >= pattern["min_matches"]:
            return {
                "emergency": True,
                "type": emergency_type.upper().replace("_", " "),
                "matched": [kw for kw in pattern["keywords"] if kw in text_lower]
            }
    
    return {"emergency": False}

# ==============================================================================
# SMART DIAGNOSIS ENGINE - Enhanced
# ==============================================================================

def diagnose(user_input: str, follow_up_answers: List[str] = None) -> dict:
    """
    Enhanced diagnosis with better accuracy
    """
    try:
        # Validation
        if not user_input or len(user_input.strip()) < 3:
            return {"error": "Please describe your symptoms in more detail"}
        
        # Combine all input
        combined = user_input.lower()
        if follow_up_answers:
            combined += " " + " ".join([str(a).lower() for a in follow_up_answers])
        
        logger.info(f"Diagnosing: {combined[:100]}...")
        
        # Step 1: Emergency detection
        emergency = detect_emergency(combined)
        if emergency["emergency"]:
            logger.critical(f"EMERGENCY: {emergency['type']}")
            return {
                "emergency": True,
                "predicted_disease": f"🚨 {emergency['type']} SUSPECTED",
                "confidence": 0.95,
                "urgency": "CRITICAL",
                "referral": get_emergency_response(),
                "precautions": [
                    "CALL 102/108/112 IMMEDIATELY",
                    "Go to Emergency Department NOW",
                    "Do NOT wait or delay",
                    "Time is critical - every minute matters"
                ],
                "specialist": "Emergency Medicine",
                "matched_symptoms": emergency["matched"]
            }
        
        # Step 2: Pattern matching (IMPROVED THRESHOLD)
        best_match = None
        best_score = 0
        
        for pattern_id, pattern in DIAGNOSIS_DATABASE.items():
            keywords = pattern["keywords"]
            min_matches = pattern["min_matches"]
            
            # Count matches
            matches = 0
            matched_keywords = []
            for kw in keywords:
                if kw in combined:
                    matches += 1
                    matched_keywords.append(kw)
            
            if matches >= min_matches:
                # Improved scoring
                score = matches / len(keywords)
                
                # Bonus for exact phrase matches
                for kw in keywords:
                    if kw in combined and len(kw) > 5:
                        score += 0.15
                
                # Bonus for multiple matches
                if matches >= 3:
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        **pattern,
                        "match_score": score,
                        "matched_keywords": matched_keywords,
                        "pattern_id": pattern_id,
                        "total_matches": matches
                    }
        
        # IMPROVED THRESHOLD: 20% instead of 15%
        if best_match and best_score >= 0.20:
            logger.info(f"Pattern match: {best_match['disease']} (score: {best_score:.2f}, matches: {best_match['total_matches']})")
            
            confidence = best_match["confidence"]
            
            # Confidence adjustment based on match quality
            if best_score >= 0.6:  # Excellent match
                confidence = min(confidence * 1.15, 0.95)
            elif best_score >= 0.4:  # Good match
                confidence = min(confidence * 1.05, 0.92)
            elif best_score < 0.25:  # Weak match
                confidence = confidence * 0.9
            
            urgency = best_match.get("urgency", "Medium")
            
            return {
                "predicted_disease": best_match["disease"],
                "confidence": round(confidence, 2),
                "confidence_explanation": f"Matched {best_match['total_matches']} key symptoms with {confidence:.0%} confidence",
                "explanation": best_match["description"],
                "precautions": best_match["precautions"],
                "specialist": best_match["specialist"],
                "urgency": urgency,
                "urgency_score": calculate_urgency_score(combined, urgency),
                "referral": get_referral(urgency, confidence),
                "matched_symptoms": best_match["matched_keywords"],
                "causes": best_match.get("causes", []),
                "diagnosis_method": "pattern_matching",
                "emergency": False,
                "match_quality": "excellent" if best_score >= 0.5 else "good" if best_score >= 0.3 else "moderate"
            }
        
        # Step 3: ML fallback with improved handling
        logger.info("Pattern matching insufficient, using ML model")
        
        if svc is None:
            return {
                "error": "Unable to diagnose with confidence. Please describe symptoms more specifically.",
                "suggestion": "Try including: exact location, severity (1-10), duration, what makes it better/worse"
            }
        
        cleaned = preprocess_text(combined)
        X_emb = embedder.encode([cleaned])
        
        probs_svc = svc.predict_proba(X_emb)[0]
        probs_lr = lr.predict_proba(X_emb)[0]
        probs_nb = nb.predict_proba(np.abs(X_emb))[0]
        
        # Weighted ensemble (favor SVC)
        ensemble = np.average([probs_svc, probs_lr, probs_nb], 
                             axis=0, weights=[0.6, 0.25, 0.15])
        
        top_idx = np.argsort(ensemble)[-3:][::-1]
        top_diseases = le.inverse_transform(top_idx)
        top_conf = ensemble[top_idx]
        
        ml_disease = str(top_diseases[0])
        ml_conf = float(top_conf[0])
        
        # Get disease info
        disease_data = disease_info.get(ml_disease, {})
        
        # Only use ML if confidence is reasonable
        if ml_conf < 0.3:
            return {
                "error": "Unable to diagnose with confidence based on provided symptoms.",
                "suggestion": "Please provide more specific details:\n- Exact location of symptoms\n- How severe (1-10 scale)\n- How long you've had them\n- What makes it better or worse",
                "note": "Consider consulting a doctor for proper examination"
            }
        
        return {
            "predicted_disease": ml_disease,
            "confidence": round(ml_conf * 0.80, 2),  # Slightly reduce ML confidence
            "confidence_explanation": "AI prediction - recommend doctor consultation for confirmation",
            "explanation": disease_data.get("description", f"{ml_disease} - Medical evaluation recommended"),
            "precautions": disease_data.get("precautions", [
                "Consult a healthcare provider for proper diagnosis",
                "Monitor your symptoms closely",
                "Rest adequately and stay hydrated",
                "Seek immediate care if symptoms worsen"
            ]),
            "specialist": disease_data.get("specialist", "General Physician"),
            "urgency": "Medium",
            "urgency_score": 5,
            "referral": get_referral("Medium", ml_conf),
            "diagnosis_method": "ml_fallback",
            "alternatives": [
                {"disease": str(d), "confidence": round(float(c) * 0.80, 2)}
                for d, c in zip(top_diseases[1:], top_conf[1:])
            ],
            "emergency": False,
            "note": "Moderate confidence - symptoms match but doctor consultation strongly recommended"
        }
        
    except Exception as e:
        logger.error(f"Diagnosis error: {str(e)}", exc_info=True)
        return {
            "error": "Unable to process symptoms. Please try rephrasing.",
            "suggestion": "Example: 'I have severe stomach pain and nausea for 2 days'"
        }

def calculate_urgency_score(text: str, urgency: str) -> int:
    """Calculate numeric urgency score"""
    urgency_map = {"CRITICAL": 15, "High": 10, "Medium": 5, "Low": 2}
    base_score = urgency_map.get(urgency, 5)
    
    # Increase urgency for concerning keywords
    concerning = ["severe", "intense", "unbearable", "worst", "cant", "difficulty breathing", "chest pain"]
    if any(word in text.lower() for word in concerning):
        base_score = min(base_score + 3, 15)
    
    return base_score

def get_referral(urgency: str, confidence: float) -> str:
    """Generate referral text"""
    
    if urgency == "CRITICAL":
        return """
🚨 **MEDICAL EMERGENCY**

**IMMEDIATE ACTIONS:**
1. CALL 102/108/112 NOW
2. Go to nearest Emergency Department
3. Do NOT drive yourself
4. Time is critical

This is a life-threatening situation requiring immediate medical attention.
"""
    
    elif urgency == "High":
        timeline = "2-6 hours" if confidence >= 0.75 else "within 12 hours"
        return f"""
⚠️ **URGENT MEDICAL CARE NEEDED**

**Seek medical attention {timeline}:**
• Emergency Department or Urgent Care
• District Hospital
• Private Hospital Emergency

**Call 102/108 if symptoms worsen**

This condition requires prompt medical evaluation.
"""
    
    elif urgency == "Medium":
        timeline = "24 hours" if confidence >= 0.80 else "24-48 hours"
        return f"""
📋 **CONSULT DOCTOR WITHIN {timeline.upper()}**

**Recommended options:**
• Primary Health Centre (PHC) - Free/low cost
• Private clinic (₹300-800)
• Telemedicine consultation (Practo, 1mg, Tata 1mg)

**Seek immediate care if:**
- Symptoms worsen significantly
- New concerning symptoms develop
- No improvement after self-care

Consider booking appointment today.
"""
    
    else:  # Low urgency
        return """
💡 **SELF-CARE WITH MONITORING**

**Home management appropriate for now:**
• Rest and stay well hydrated
• Follow recommended precautions
• Monitor symptoms for 2-3 days
• Track any changes

**Consult doctor if:**
- Symptoms persist beyond 3 days
- Symptoms worsen
- New symptoms develop
- You're concerned about progression

Most cases resolve with proper self-care.
"""

def get_emergency_response() -> str:
    """Emergency response text"""
    return """
🚨 **MEDICAL EMERGENCY - ACT NOW**

**IMMEDIATE ACTIONS:**

1. **CALL EMERGENCY SERVICES:**
   📞 Ambulance: 102 / 108
   📞 Emergency: 112
   📞 For heart attack/stroke: Request priority ambulance

2. **DO NOT WAIT:**
   • Go to nearest Emergency Department immediately
   • Do NOT drive yourself if possible
   • Alert family member or friend

3. **WHILE WAITING:**
   • Stay calm, sit or lie down
   • Note when symptoms started
   • Bring list of any medications you take
   • Someone should accompany you

4. **IMPORTANT:**
   • Do NOT eat or drink anything
   • Do NOT take any medication without medical advice
   • Keep phone accessible

---

⚠️ **This is time-critical. Every minute matters.**

**Common Emergency Hospitals in India:**
- AIIMS (if available in your city)
- Government District Hospital
- Apollo Hospitals
- Fortis Healthcare
- Max Healthcare
- Manipal Hospitals
"""

def detect_body_system(symptoms: str) -> str:
    """Detect affected body system"""
    s = symptoms.lower()
    
    if any(x in s for x in ["abdomen", "stomach", "nausea", "vomit", "diarrhea", "constipation", "bowel"]):
        return "digestive"
    elif any(x in s for x in ["cough", "breath", "throat", "chest", "lung", "wheezing"]):
        return "respiratory"
    elif any(x in s for x in ["urine", "urination", "bladder", "kidney", "pee"]):
        return "urinary"
    elif any(x in s for x in ["headache", "dizzy", "migraine", "neurological"]):
        return "neurological"
    elif any(x in s for x in ["heart", "chest pain", "palpitation"]):
        return "cardiovascular"
    elif any(x in s for x in ["joint", "muscle", "back pain", "neck", "bone"]):
        return "musculoskeletal"
    elif any(x in s for x in ["rash", "itching", "skin", "hives"]):
        return "dermatological"
    else:
        return "general"

def log_feedback(symptoms, predicted, confidence, feedback, correction=None, urgency=None):
    """Log user feedback for improvement"""
    try:
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/feedback_logs.csv"
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "symptoms": symptoms[:200],
            "predicted_disease": predicted,
            "confidence": confidence,
            "user_feedback": feedback,
            "user_correction": correction,
            "urgency": urgency
        }
        
        df = pd.DataFrame([entry])
        if os.path.exists(log_file):
            df.to_csv(log_file, mode="a", index=False, header=False)
        else:
            df.to_csv(log_file, index=False, header=True)
            
        logger.info(f"Feedback logged: {feedback}")
    except Exception as e:
        logger.error(f"Error logging feedback: {str(e)}")

# Quick test
if __name__ == "__main__":
    test_cases = [
        "I have severe stomach pain and diarrhea for 2 days",
        "high fever with body ache and weakness",
        "burning sensation when urinating",
        "severe headache on one side with nausea and light sensitivity",
        "runny nose sneezing and sore throat",
        "crushing chest pain radiating to left arm",
        "difficulty breathing and wheezing",
        "rash and itching after eating shrimp"
    ]
    
    print("\n" + "="*70)
    print("TESTING ENHANCED DIAGNOSIS SYSTEM")
    print("="*70 + "\n")
    
    for test in test_cases:
        print(f"Input: {test}")
        result = diagnose(test)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Disease: {result['predicted_disease']}")
            print(f"   Confidence: {result['confidence']:.0%}")
            print(f"   Method: {result.get('diagnosis_method', 'unknown')}")
            print(f"   Urgency: {result['urgency']}")
            if 'matched_symptoms' in result:
                print(f"   Matched: {', '.join(result['matched_symptoms'][:3])}")
        print("-"*70 + "\n")