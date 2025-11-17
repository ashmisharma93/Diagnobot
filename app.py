# # app_enhanced.py — Intelligent Diagnobot with Medical Reasoning
# import streamlit as st
# import random
# from diagnose_api import diagnose, log_feedback, detect_body_system
# from typing import List, Dict, Tuple
# import re

# # ----------------------------------------
# # Page Configuration
# # ----------------------------------------
# st.set_page_config(page_title="Diagnobot", page_icon="🩺", layout="centered")

# # Custom CSS for better UI
# st.markdown("""
# <style>
# .stChatMessage {
#     padding: 1rem;
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
# }
# .confidence-high { color: #28a745; }
# .confidence-medium { color: #ffc107; }
# .confidence-low { color: #dc3545; }
# .urgency-badge {
#     display: inline-block;
#     padding: 0.25rem 0.75rem;
#     border-radius: 1rem;
#     font-weight: bold;
#     margin: 0.25rem;
# }
# .urgency-high { background-color: #ff4444; color: white; }
# .urgency-medium { background-color: #ffbb33; color: black; }
# .urgency-low { background-color: #00c851; color: white; }
# </style>
# """, unsafe_allow_html=True)

# st.title("🩺 Diagnobot: Your Health Chatbot")
# st.markdown("> ⚠️ **Medical Disclaimer**: This AI provides preliminary health insights only. Always consult a licensed healthcare professional for diagnosis and treatment.")

# # ----------------------------------------
# # Enhanced Medical Knowledge Base
# # ----------------------------------------
# SYMPTOM_PATTERNS = {
#     "headache_fever": {
#         "primary": ["headache", "fever"],
#         "likely_conditions": [
#             ("Common Cold", 0.35, ["Rest", "Hydration", "OTC pain relievers"]),
#             ("Flu (Influenza)", 0.30, ["Antiviral medication", "Rest", "Fluids"]),
#             ("Sinusitis", 0.20, ["Decongestants", "Warm compress", "Steam inhalation"]),
#             ("Tension Headache", 0.15, ["Stress management", "Pain relievers", "Rest"])
#         ],
#         "specialist": "General Physician",
#         "urgency": "Medium"
#     },
#     "headache_severe": {
#         "primary": ["severe headache", "sudden headache", "worst headache"],
#         "likely_conditions": [
#             ("Migraine", 0.40, ["Prescription medication", "Dark quiet room", "Cold compress"]),
#             ("Cluster Headache", 0.25, ["Oxygen therapy", "Sumatriptan", "Specialist consultation"]),
#             ("Possible Serious Condition", 0.20, ["URGENT: Seek emergency care if sudden/severe"])
#         ],
#         "specialist": "Neurologist",
#         "urgency": "High"
#     },
#     "cough_fever": {
#         "primary": ["cough", "fever"],
#         "likely_conditions": [
#             ("Upper Respiratory Infection", 0.40, ["Rest", "Fluids", "Cough suppressants"]),
#             ("Bronchitis", 0.30, ["Bronchodilators", "Rest", "Avoid irritants"]),
#             ("Pneumonia", 0.20, ["Antibiotics", "Chest X-ray", "Medical evaluation"])
#         ],
#         "specialist": "Pulmonologist",
#         "urgency": "Medium"
#     },
#     "stomach_pain": {
#         "primary": ["stomach pain", "abdominal pain"],
#         "likely_conditions": [
#             ("Gastritis", 0.35, ["Antacids", "Avoid spicy food", "Small meals"]),
#             ("Food Poisoning", 0.25, ["Hydration", "Bland diet", "Rest"]),
#             ("Gastroenteritis", 0.20, ["Electrolytes", "Clear liquids", "Rest"])
#         ],
#         "specialist": "Gastroenterologist",
#         "urgency": "Medium"
#     }
# }

# # ----------------------------------------
# # Intelligent Follow-up Generator
# # ----------------------------------------
# class SmartFollowupEngine:
#     """Generates contextually relevant follow-up questions"""
    
#     @staticmethod
#     def generate_followups(symptoms: str, system: str) -> List[Tuple[str, str]]:
#         """Generate smart follow-ups based on initial symptoms"""
#         questions = []
#         symptoms_lower = symptoms.lower()
        
#         # Headache-specific
#         if "headache" in symptoms_lower:
#             if "severe" not in symptoms_lower:
#                 questions.append(("severity", "On a scale of 1-10, how severe is the headache?"))
#             questions.append(("duration", "How long have you had this headache? (hours/days)"))
#             questions.append(("location", "Where exactly is the pain? (front, sides, back, all over)"))
#             if "fever" not in symptoms_lower:
#                 questions.append(("fever", "Do you have a fever or feel feverish?"))
#             questions.append(("nausea", "Do you feel nauseous or sensitive to light?"))
        
#         # Fever-specific
#         elif "fever" in symptoms_lower:
#             questions.append(("temp", "What is your temperature? (or approximately how high?)"))
#             questions.append(("duration", "How many days have you had the fever?"))
#             questions.append(("other", "Any other symptoms? (body ache, chills, sore throat)"))
        
#         # Cough-specific
#         elif "cough" in symptoms_lower:
#             questions.append(("type", "Is the cough dry or with mucus?"))
#             questions.append(("duration", "How long have you had the cough?"))
#             questions.append(("breathing", "Any difficulty breathing or chest pain?"))
        
#         # Stomach-specific
#         elif any(x in symptoms_lower for x in ["stomach", "abdomen", "belly"]):
#             questions.append(("location", "Where exactly is the pain? (upper, lower, right, left)"))
#             questions.append(("severity", "Rate the pain severity (1-10)"))
#             questions.append(("other", "Any vomiting, diarrhea, or nausea?"))
        
#         # Generic system-based questions
#         if system == "respiratory" and not any("breath" in q[1].lower() for q in questions):
#             questions.append(("breathing", "Any difficulty breathing or chest tightness?"))
        
#         return questions[:3]  # Limit to 3 follow-ups

# # ----------------------------------------
# # Enhanced Diagnosis Logic
# # ----------------------------------------
# class IntelligentDiagnosis:
#     """Enhanced diagnosis with pattern matching and reasoning"""
    
#     @staticmethod
#     def analyze_symptoms(symptoms: str, answers: List[str]) -> Dict:
#         """Perform intelligent symptom analysis"""
#         combined = (symptoms + " " + " ".join(answers)).lower()
        
#         # Pattern matching
#         best_match = None
#         best_score = 0
        
#         for pattern_name, pattern_data in SYMPTOM_PATTERNS.items():
#             score = 0
#             for primary_symptom in pattern_data["primary"]:
#                 if primary_symptom in combined:
#                     score += 1
            
#             # Normalize score
#             score = score / len(pattern_data["primary"])
            
#             if score > best_score:
#                 best_score = score
#                 best_match = pattern_data
        
#         # If pattern match found
#         if best_match and best_score > 0.5:
#             return {
#                 "matched": True,
#                 "conditions": best_match["likely_conditions"],
#                 "specialist": best_match["specialist"],
#                 "urgency": best_match["urgency"],
#                 "confidence_boost": 0.2
#             }
        
#         return {"matched": False}
    
#     @staticmethod
#     def explain_reasoning(disease: str, symptoms: str, confidence: float) -> str:
#         """Generate medical reasoning explanation"""
#         reasoning = []
        
#         reasoning.append(f"**Analysis Process:**")
#         reasoning.append(f"1. Analyzed symptom pattern: '{symptoms[:50]}...'")
        
#         if confidence >= 0.7:
#             reasoning.append(f"2. Strong correlation found between symptoms and {disease}")
#             reasoning.append(f"3. Key indicators match clinical presentation")
#         elif confidence >= 0.5:
#             reasoning.append(f"2. Moderate correlation with {disease}")
#             reasoning.append(f"3. Some symptoms align with typical presentation")
#         else:
#             reasoning.append(f"2. Weak correlation - symptoms may indicate {disease} or related conditions")
#             reasoning.append(f"3. Further medical evaluation needed")
        
#         return "\n".join(reasoning)

# # ----------------------------------------
# # Session Initialization
# # ----------------------------------------
# def reset_chat():
#     st.session_state.chat_history = [
#         ("bot", "👋 Hello! I'm Diagnobot Pro, your intelligent health assistant.\n\n"
#                 "I use advanced AI to analyze symptoms, but remember:\n"
#                 "- I provide preliminary insights only\n"
#                 "- Always consult a doctor for diagnosis\n"
#                 "- For emergencies, call your local emergency number\n\n"
#                 "**How can I help you today?**")
#     ]
#     st.session_state.state = "greeting"
#     st.session_state.symptoms = ""
#     st.session_state.system = None
#     st.session_state.followups = []
#     st.session_state.answers = []
#     st.session_state.last_result = None
#     st.session_state.severity_info = {}

# if "chat_history" not in st.session_state:
#     reset_chat()

# # Sidebar
# st.sidebar.header("🔧 Settings")
# if st.sidebar.button("🔄 New Consultation"):
#     reset_chat()
#     st.rerun()

# st.sidebar.markdown("---")
# st.sidebar.markdown("### 📊 Session Info")
# st.sidebar.info(f"Messages: {len(st.session_state.chat_history)}")

# # Emergency hotline - India specific
# st.sidebar.markdown("---")
# st.sidebar.error("""
# 🚨 **Emergency Services (India)**

# **Ambulance:** 102 / 108  
# **Medical Emergency:** 112  
# **COVID-19 Helpline:** 1075  

# **For Mental Health:**  
# NIMHANS Helpline: 080-46110007
# """)

# st.sidebar.markdown("---")
# st.sidebar.info("""
# 🏥 **Nearby Facilities**
# - Primary Health Centre (PHC)
# - Community Health Centre (CHC)
# - District Hospital
# - Private Hospitals (Apollo, Fortis, Max, etc.)
# """)

# # ----------------------------------------
# # Intent Detection
# # ----------------------------------------
# YES_SET = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "correct", "right"}
# NO_SET = {"no", "n", "nah", "nope", "not really", "negative"}

# def detect_intent(text: str) -> str:
#     t = text.lower().strip()
#     if any(w in t for w in ["hi", "hello", "hey", "greetings"]): 
#         return "greeting"
#     if any(w in t for w in ["pain", "fever", "cough", "cold", "nausea", "vomit", "rash", 
#                              "breath", "itch", "urine", "sore", "headache", "dizzy", "tired",
#                              "stomach", "chest", "throat", "sick", "ill", "hurt"]):
#         return "symptom"
#     if t in YES_SET: 
#         return "yes"
#     if t in NO_SET: 
#         return "no"
#     if any(w in t for w in ["thanks", "thank", "appreciate"]):
#         return "thanks"
#     if any(w in t for w in ["bye", "goodbye", "exit", "quit"]):
#         return "goodbye"
#     # Check for numeric severity (1-10 scale)
#     if re.search(r'\b([1-9]|10)\b', t):
#         return "severity"
#     return "unknown"

# # ----------------------------------------
# # Main Chat Flow
# # ----------------------------------------
# user_input = st.chat_input("Type your message here...")

# if user_input:
#     st.session_state.chat_history.append(("user", user_input))
#     intent = detect_intent(user_input)
#     state = st.session_state.state

#     # Greeting
#     if intent == "greeting" and state == "greeting":
#         st.session_state.chat_history.append(
#             ("bot", "👋 Please describe your symptoms in detail.\n\n"
#                     "For example: 'I have a severe headache and mild fever for 2 days'\n\n"
#                     "The more specific you are, the better I can assist you.")
#         )
#         st.session_state.state = "symptom_query"

#     # Initial Symptom Collection
#     elif (intent == "symptom" or state == "symptom_query") and state != "followup":
#         st.session_state.symptoms = user_input.strip()
#         st.session_state.system = detect_body_system(user_input)
        
#         # Generate smart follow-ups
#         engine = SmartFollowupEngine()
#         st.session_state.followups = engine.generate_followups(user_input, st.session_state.system)
#         st.session_state.answers = [user_input.lower()]

#         if st.session_state.followups:
#             _, q = st.session_state.followups.pop(0)
#             st.session_state.chat_history.append(("bot", f"📋 {q}"))
#             st.session_state.state = "followup"
#         else:
#             st.session_state.state = "diagnose"

#     # Follow-up Phase
#     elif state == "followup":
#         st.session_state.answers.append(user_input.lower())
        
#         # Extract severity if mentioned
#         severity_match = re.search(r'\b([1-9]|10)\b', user_input)
#         if severity_match:
#             st.session_state.severity_info['score'] = int(severity_match.group(1))

#         if st.session_state.followups:
#             _, q = st.session_state.followups.pop(0)
#             st.session_state.chat_history.append(("bot", f"📋 {q}"))
#         else:
#             st.session_state.state = "diagnose"

#     # Enhanced Diagnosis
#     if st.session_state.state == "diagnose":
#         combined = st.session_state.symptoms + " " + " ".join(st.session_state.answers)
        
#         # Get diagnosis (includes emergency detection)
#         ml_result = diagnose(combined)
        
#         # Check if emergency detected
#         if ml_result.get("emergency", False):
#             # CRITICAL EMERGENCY RESPONSE
#             st.session_state.chat_history.append(("bot", ml_result.get("referral", "")))
            
#             # Add emergency contact reminder
#             st.session_state.chat_history.append(("bot", 
#                 "🚨 **REMEMBER: Call 102/108/112 immediately!**\n\n"
#                 "This is a medical emergency that requires immediate professional care. "
#                 "Do not rely on this AI assessment alone."
#             ))
            
#             st.session_state.last_result = ml_result
#             st.session_state.state = "await_feedback"
#             st.rerun()
        
#         # Smart pattern analysis for non-emergencies
#         smart_result = IntelligentDiagnosis.analyze_symptoms(
#             st.session_state.symptoms, 
#             st.session_state.answers
#         )

#         if "error" in ml_result:
#             st.session_state.chat_history.append(("bot", f"⚠️ {ml_result['error']}"))
#         else:
#             # Combine smart pattern matching with ML results
#             if smart_result["matched"]:
#                 # Use pattern-based diagnosis
#                 conditions = smart_result["conditions"]
#                 primary_disease, primary_conf, precautions = conditions[0]
#                 specialist = smart_result["specialist"]
#                 urgency = smart_result["urgency"]
                
#                 conf_display = f"{primary_conf:.1%}"
                
#                 # Build response
#                 response = f"### 🩺 Analysis Complete\n\n"
#                 response += f"**Most Likely Condition:** {primary_disease}\n\n"
                
#                 # Confidence indicator
#                 if primary_conf >= 0.7:
#                     conf_class = "confidence-high"
#                     conf_text = "High Confidence"
#                 elif primary_conf >= 0.5:
#                     conf_class = "confidence-medium"
#                     conf_text = "Moderate Confidence"
#                 else:
#                     conf_class = "confidence-low"
#                     conf_text = "Low Confidence"
                
#                 response += f"**Confidence:** <span class='{conf_class}'>{conf_display} ({conf_text})</span>\n\n"
                
#                 # Urgency badge
#                 urgency_class = f"urgency-{urgency.lower()}"
#                 urgency_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[urgency]
#                 response += f"{urgency_emoji} <span class='urgency-badge {urgency_class}'>Urgency: {urgency}</span>\n\n"
                
#                 # Other possible conditions
#                 if len(conditions) > 1:
#                     response += f"**Other Possibilities:**\n"
#                     for disease, conf, _ in conditions[1:3]:
#                         response += f"- {disease} ({conf:.1%})\n"
#                     response += "\n"
                
#                 # Recommended actions
#                 response += f"**Recommended Actions:**\n"
#                 for i, prec in enumerate(precautions, 1):
#                     response += f"{i}. {prec}\n"
#                 response += f"\n**Specialist:** {specialist}\n\n"
                
#                 # Medical reasoning
#                 reasoning = IntelligentDiagnosis.explain_reasoning(
#                     primary_disease, 
#                     st.session_state.symptoms,
#                     primary_conf
#                 )
#                 response += f"\n---\n\n{reasoning}\n\n"
                
#                 # Disclaimer
#                 response += "---\n\n⚠️ **Important:** This is a preliminary assessment. Please consult a healthcare professional for proper diagnosis and treatment.\n\n"
#                 response += "**Was this analysis helpful?** (Yes/No)"
                
#                 st.session_state.chat_history.append(("bot", response))
                
#                 # Store for feedback
#                 st.session_state.last_result = {
#                     "predicted_disease": primary_disease,
#                     "confidence": primary_conf
#                 }
                
#             else:
#                 # Use ML model result with enhancement
#                 conf = ml_result.get("confidence", 0)
#                 disease = ml_result["predicted_disease"]
#                 urgency = ml_result.get("urgency", "Medium")
                
#                 # Enhanced display (similar to above)
#                 response = f"### 🩺 AI Analysis Complete\n\n"
#                 response += f"**Predicted Condition:** {disease}\n\n"
                
#                 if conf >= 0.6:
#                     conf_class = "confidence-high"
#                     message = "The symptoms you described align with"
#                 elif conf >= 0.4:
#                     conf_class = "confidence-medium"
#                     message = "Your symptoms suggest a possibility of"
#                 else:
#                     conf_class = "confidence-low"
#                     message = "Based on limited information, this might be related to"
                
#                 response += f"{message} **{disease}**\n\n"
#                 response += f"**Confidence:** <span class='{conf_class}'>{conf:.1%}</span>\n\n"
                
#                 # Add rest of ML result details
#                 urgency_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[urgency]
#                 response += f"{urgency_emoji} **Urgency:** {urgency}\n\n"
                
#                 if "precautions" in ml_result:
#                     response += "**Recommended Precautions:**\n"
#                     for prec in ml_result["precautions"][:4]:
#                         response += f"- {prec}\n"
                
#                 response += f"\n**Specialist:** {ml_result.get('specialist', 'General Physician')}\n\n"
#                 response += "---\n\n**Was this helpful?** (Yes/No)"
                
#                 st.session_state.chat_history.append(("bot", response))
#                 st.session_state.last_result = ml_result

#             st.session_state.state = "await_feedback"

#     # Feedback
#     elif state == "await_feedback" and intent in {"yes", "no"}:
#         res = st.session_state.last_result
#         if res:
#             log_feedback(
#                 st.session_state.symptoms, 
#                 res["predicted_disease"], 
#                 res["confidence"], 
#                 intent
#             )
            
#             if intent == "yes":
#                 msg = "✅ **Thank you for your feedback!**\n\n"
#                 msg += "Remember to:\n"
#                 msg += "- Follow the recommended precautions\n"
#                 msg += "- Consult a healthcare professional\n"
#                 msg += "- Monitor your symptoms\n\n"
#                 msg += "Take care! 💙"
#             else:
#                 msg = "📝 **Thank you for the feedback.**\n\n"
#                 msg += "I recommend:\n"
#                 msg += "- Consulting a doctor in person\n"
#                 msg += "- Getting proper diagnostic tests\n"
#                 msg += "- Seeking a second opinion if needed\n\n"
#                 msg += "Your health is important! 🏥"
            
#             st.session_state.chat_history.append(("bot", msg))
        
#         st.session_state.state = "greeting"

#     elif intent == "thanks":
#         st.session_state.chat_history.append(
#             ("bot", "😊 You're welcome! Stay healthy and take care!")
#         )
#     elif intent == "goodbye":
#         st.session_state.chat_history.append(
#             ("bot", "👋 Goodbye! Remember to consult a doctor if symptoms persist. Stay safe!")
#         )
#     else:
#         if st.session_state.state not in ["followup", "await_feedback"]:
#             st.session_state.chat_history.append(
#                 ("bot", "💬 Please describe your symptoms, or type 'new' to start over.")
#             )

# # ----------------------------------------
# # Display Chat History
# # ----------------------------------------
# for role, msg in st.session_state.chat_history:
#     with st.chat_message("user" if role == "user" else "assistant"):
#         st.markdown(msg, unsafe_allow_html=True)

# # Footer
# st.markdown("---")
# st.caption("Diagnobot Pro v2.0 | AI-Powered Health Assistant | For Educational Purposes Only")

# enhanced_app.py - Improved Diagnobot with better accuracy
import streamlit as st
import random
from diagnose_api import diagnose, log_feedback, detect_body_system
from typing import List, Dict, Tuple
import re

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(page_title="Diagnobot Pro", page_icon="🩺", layout="centered")

# Custom CSS
st.markdown("""
<style>
.stChatMessage {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.confidence-high { color: #28a745; font-weight: bold; }
.confidence-medium { color: #ffc107; font-weight: bold; }
.confidence-low { color: #dc3545; font-weight: bold; }
.urgency-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-weight: bold;
    margin: 0.25rem;
}
.urgency-high { background-color: #ff4444; color: white; }
.urgency-medium { background-color: #ffbb33; color: black; }
.urgency-low { background-color: #00c851; color: white; }
.symptom-box {
    background-color: #f0f7ff;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #0066cc;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diagnobot Pro: Your Health Assistant")
st.markdown("> ⚠️ **Medical Disclaimer**: This AI provides preliminary health insights only. Always consult a licensed healthcare professional for diagnosis and treatment.")

# ----------------------------------------
# Enhanced Follow-up Engine
# ----------------------------------------
class SmartFollowupEngine:
    """Generates comprehensive follow-up questions"""
    
    @staticmethod
    def generate_followups(symptoms: str, system: str) -> List[Tuple[str, str]]:
        """Generate contextual follow-up questions"""
        questions = []
        symptoms_lower = symptoms.lower()
        
        # === PAIN-RELATED ===
        if "pain" in symptoms_lower or "ache" in symptoms_lower:
            if "severe" not in symptoms_lower and "mild" not in symptoms_lower:
                questions.append(("severity", "How would you rate the pain severity on a scale of 1-10?\n(1=mild, 5=moderate, 10=worst pain ever)"))
            
            if not any(x in symptoms_lower for x in ["day", "days", "week", "hour", "since"]):
                questions.append(("duration", "How long have you had this pain?\n(hours/days/weeks)"))
            
            if not any(x in symptoms_lower for x in ["sharp", "dull", "burning", "throbbing", "stabbing"]):
                questions.append(("pain_type", "What type of pain is it?\n(sharp/dull/burning/throbbing/cramping)"))
            
            questions.append(("triggers", "Does anything make the pain worse or better?\n(movement/rest/eating/position)"))
        
        # === HEADACHE-SPECIFIC ===
        if "headache" in symptoms_lower or "head" in symptoms_lower:
            if "location" not in [q[0] for q in questions]:
                questions.append(("location", "Where exactly is the headache?\n(front/temples/one side/back/all over)"))
            
            if "nausea" not in symptoms_lower and "vomit" not in symptoms_lower:
                questions.append(("nausea", "Do you have nausea, vomiting, or sensitivity to light/sound?"))
            
            if "sudden" not in symptoms_lower:
                questions.append(("onset", "Did the headache come on suddenly or gradually?"))
            
            questions.append(("frequency", "Is this your first headache like this, or do you get them regularly?"))
        
        # === FEVER-RELATED ===
        if "fever" in symptoms_lower or "temperature" in symptoms_lower:
            if not re.search(r'\d+', symptoms_lower):
                questions.append(("temp", "What is your temperature?\n(or estimate: low-grade 99-100°F, moderate 101-102°F, high >102°F)"))
            
            if not any(x in symptoms_lower for x in ["day", "days", "hours"]):
                questions.append(("duration", "How many days/hours have you had the fever?"))
            
            questions.append(("pattern", "Is the fever constant or does it come and go?"))
            
            if "chills" not in symptoms_lower and "shiver" not in symptoms_lower:
                questions.append(("chills", "Do you have chills, sweating, or body aches?"))
        
        # === COUGH-RELATED ===
        if "cough" in symptoms_lower:
            if "dry" not in symptoms_lower and "mucus" not in symptoms_lower and "phlegm" not in symptoms_lower:
                questions.append(("cough_type", "Is the cough dry or are you coughing up mucus/phlegm?"))
            
            if "duration" not in [q[0] for q in questions]:
                questions.append(("duration", "How long have you had the cough?"))
            
            if "breath" not in symptoms_lower:
                questions.append(("breathing", "Any difficulty breathing, wheezing, or chest tightness?"))
            
            questions.append(("timing", "Is the cough worse at night, morning, or constant throughout the day?"))
        
        # === DIGESTIVE ===
        if any(x in symptoms_lower for x in ["stomach", "abdomen", "belly", "gut"]):
            if not any(x in symptoms_lower for x in ["upper", "lower", "right", "left", "center"]):
                questions.append(("location", "Where exactly is the pain?\n(upper/lower abdomen, right/left/center)"))
            
            if "vomit" not in symptoms_lower and "nausea" not in symptoms_lower:
                questions.append(("nausea", "Do you have nausea, vomiting, or loss of appetite?"))
            
            if "diarrhea" not in symptoms_lower and "constipation" not in symptoms_lower:
                questions.append(("bowel", "Any changes in bowel movements?\n(diarrhea/constipation/normal)"))
            
            questions.append(("eating", "Does eating make it better or worse? When did you last eat?"))
        
        # === URINARY ===
        if any(x in symptoms_lower for x in ["urine", "urination", "pee", "bladder"]):
            if "burn" not in symptoms_lower:
                questions.append(("burning", "Is there burning or pain when urinating?"))
            
            if "frequent" not in symptoms_lower:
                questions.append(("frequency", "Are you urinating more frequently than usual? How often?"))
            
            if "blood" not in symptoms_lower:
                questions.append(("blood", "Any blood in urine or unusual color/smell?"))
            
            questions.append(("urgency", "Do you feel a sudden urgent need to urinate?"))
        
        # === RESPIRATORY ===
        if any(x in symptoms_lower for x in ["breath", "breathing", "chest", "wheeze"]):
            if "difficult" not in symptoms_lower and "short" not in symptoms_lower:
                questions.append(("breathing", "How severe is the breathing difficulty?\n(mild/moderate/severe/can't complete sentences)"))
            
            if "wheeze" not in symptoms_lower:
                questions.append(("wheezing", "Any wheezing sounds when breathing?"))
            
            questions.append(("position", "Is breathing easier sitting up or lying down?"))
            
            if "history" not in [q[0] for q in questions]:
                questions.append(("history", "Do you have asthma or any lung conditions?"))
        
        # === SKIN ===
        if any(x in symptoms_lower for x in ["rash", "itch", "hives", "skin"]):
            questions.append(("appearance", "Describe the rash:\n(red bumps/flat patches/blisters/hives)"))
            
            if "duration" not in [q[0] for q in questions]:
                questions.append(("duration", "When did the rash appear?"))
            
            questions.append(("trigger", "Did you eat anything new, use new products, or get bitten by insects?"))
            
            if "spread" not in symptoms_lower:
                questions.append(("spreading", "Is the rash spreading or staying in one area?"))
        
        # === GENERAL CONTEXT ===
        if len(questions) < 3:
            # Add general questions if not enough specific ones
            if "medication" not in symptoms_lower:
                questions.append(("medication", "Are you currently taking any medications?"))
            
            if "medical" not in symptoms_lower and "history" not in symptoms_lower:
                questions.append(("history", "Do you have any medical conditions?\n(diabetes/blood pressure/asthma/etc.)"))
            
            questions.append(("other_symptoms", "Any other symptoms we haven't discussed?"))
        
        # Limit to 4 most relevant questions
        return questions[:4]

# ----------------------------------------
# Session Initialization
# ----------------------------------------
def reset_chat():
    st.session_state.chat_history = [
        ("bot", "👋 Hello! I'm **Diagnobot Pro**, your intelligent health assistant.\n\n"
                "I'll help analyze your symptoms, but remember:\n"
                "- ✅ I provide preliminary insights\n"
                "- ⚠️ Always consult a doctor for diagnosis\n"
                "- 🚨 For emergencies, call 102/108/112\n\n"
                "**Please describe your symptoms in detail.**\n"
                "Example: *'I have severe stomach pain and diarrhea for 2 days'*")
    ]
    st.session_state.state = "greeting"
    st.session_state.symptoms = ""
    st.session_state.system = None
    st.session_state.followups = []
    st.session_state.answers = []
    st.session_state.last_result = None
    st.session_state.severity_info = {}
    st.session_state.followup_count = 0

if "chat_history" not in st.session_state:
    reset_chat()

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.header("🔧 Controls")
if st.sidebar.button("🔄 New Consultation"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Info")
st.sidebar.info(f"Messages: {len(st.session_state.chat_history)}\nState: {st.session_state.state}")

# Emergency contacts
st.sidebar.markdown("---")
st.sidebar.error("""
🚨 **Emergency Services (India)**

**Ambulance:** 102 / 108  
**Emergency:** 112  
**COVID-19:** 1075  

**Mental Health:**  
NIMHANS: 080-46110007  
Vandrevala: 1860-2662-345
""")

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Tips for Better Diagnosis**

• Describe location specifically
• Rate severity (1-10)
• Mention duration
• List all symptoms
• Include triggers/relievers
""")

# ----------------------------------------
# Intent Detection
# ----------------------------------------
YES_SET = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "correct", "right", "yup"}
NO_SET = {"no", "n", "nah", "nope", "not really", "negative", "no way"}

def detect_intent(text: str) -> str:
    """Enhanced intent detection"""
    t = text.lower().strip()
    
    # Greeting
    if any(w in t for w in ["hi", "hello", "hey", "greetings", "good morning", "good evening"]):
        return "greeting"
    
    # Symptoms
    symptom_keywords = [
        "pain", "ache", "fever", "cough", "cold", "nausea", "vomit", "rash", 
        "breath", "itch", "urine", "sore", "headache", "dizzy", "tired", "weak",
        "stomach", "chest", "throat", "sick", "ill", "hurt", "burning", "swelling",
        "diarrhea", "constipation", "bleeding", "injury", "fracture", "sprain"
    ]
    if any(w in t for w in symptom_keywords):
        return "symptom"
    
    # Yes/No
    if t in YES_SET or t.startswith("yes"):
        return "yes"
    if t in NO_SET or t.startswith("no"):
        return "no"
    
    # Numeric responses (severity ratings)
    if re.match(r'^\d+(/10)?$', t) or re.match(r'^\d+\s*(out of)?\s*10$', t):
        return "severity"
    
    # Thanks
    if any(w in t for w in ["thanks", "thank", "appreciate", "helpful"]):
        return "thanks"
    
    # Goodbye
    if any(w in t for w in ["bye", "goodbye", "exit", "quit", "stop"]):
        return "goodbye"
    
    return "unknown"

# ----------------------------------------
# Main Chat Logic
# ----------------------------------------
user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    intent = detect_intent(user_input)
    state = st.session_state.state

    # === GREETING ===
    if intent == "greeting" and state == "greeting":
        st.session_state.chat_history.append(
            ("bot", "👋 Hello! Please describe your symptoms in detail.\n\n"
                    "**For best results, include:**\n"
                    "• 📍 Location (where it hurts)\n"
                    "• 📊 Severity (how bad is it, 1-10)\n"
                    "• ⏱️ Duration (how long you've had it)\n"
                    "• 📋 All symptoms you're experiencing\n\n"
                    "Example: *'Severe headache on right side (8/10) for 6 hours with nausea'*")
        )
        st.session_state.state = "symptom_query"

    # === INITIAL SYMPTOM COLLECTION ===
    elif (intent == "symptom" or state == "symptom_query") and state != "followup":
        # Validate input quality
        if len(user_input.strip()) < 10:
            st.session_state.chat_history.append(
                ("bot", "⚠️ Please provide more details about your symptoms.\n\n"
                        "Try to include:\n"
                        "• Where exactly do you feel discomfort?\n"
                        "• How severe is it?\n"
                        "• When did it start?\n\n"
                        "The more specific you are, the better I can help!")
            )
        else:
            st.session_state.symptoms = user_input.strip()
            st.session_state.system = detect_body_system(user_input)
            st.session_state.answers = [user_input.lower()]
            
            # Generate smart follow-ups
            engine = SmartFollowupEngine()
            st.session_state.followups = engine.generate_followups(user_input, st.session_state.system)
            st.session_state.followup_count = 0
            
            if st.session_state.followups:
                _, q = st.session_state.followups.pop(0)
                st.session_state.followup_count += 1
                st.session_state.chat_history.append(
                    ("bot", f"📋 **Follow-up Question {st.session_state.followup_count}:**\n\n{q}")
                )
                st.session_state.state = "followup"
            else:
                st.session_state.state = "diagnose"

    # === FOLLOW-UP PHASE ===
    elif state == "followup":
        st.session_state.answers.append(user_input.lower())
        
        # Extract severity if mentioned
        severity_match = re.search(r'\b([1-9]|10)\b', user_input)
        if severity_match:
            st.session_state.severity_info['score'] = int(severity_match.group(1))
        
        # Ask next follow-up or proceed to diagnosis
        if st.session_state.followups and st.session_state.followup_count < 4:
            _, q = st.session_state.followups.pop(0)
            st.session_state.followup_count += 1
            st.session_state.chat_history.append(
                ("bot", f"📋 **Follow-up Question {st.session_state.followup_count}:**\n\n{q}")
            )
        else:
            # Enough info collected, proceed to diagnosis
            st.session_state.chat_history.append(
                ("bot", "✅ Thank you for the detailed information. Let me analyze your symptoms...\n\n"
                        "*Analyzing pattern matching and medical database...*")
            )
            st.session_state.state = "diagnose"

    # === DIAGNOSIS ===
    if st.session_state.state == "diagnose":
        combined = st.session_state.symptoms + " " + " ".join(st.session_state.answers)
        
        # Get diagnosis
        ml_result = diagnose(combined)
        
        # === EMERGENCY HANDLING ===
        if ml_result.get("emergency", False):
            st.session_state.chat_history.append(("bot", ml_result.get("referral", "")))
            st.session_state.chat_history.append(("bot", 
                "🚨 **REMEMBER: CALL 102/108/112 IMMEDIATELY!**\n\n"
                "This is a medical emergency requiring immediate professional care."
            ))
            st.session_state.last_result = ml_result
            st.session_state.state = "await_feedback"
            st.rerun()
        
        # === NORMAL DIAGNOSIS DISPLAY ===
        if "error" in ml_result:
            st.session_state.chat_history.append(("bot", f"⚠️ {ml_result['error']}\n\n{ml_result.get('suggestion', '')}"))
            st.session_state.state = "greeting"
        else:
            # Build comprehensive response
            disease = ml_result["predicted_disease"]
            conf = ml_result.get("confidence", 0)
            urgency = ml_result.get("urgency", "Medium")
            
            response = f"### 🩺 Diagnosis Complete\n\n"
            response += f"**Condition:** {disease}\n\n"
            
            # Confidence display
            if conf >= 0.75:
                conf_class = "confidence-high"
                conf_emoji = "✅"
            elif conf >= 0.55:
                conf_class = "confidence-medium"
                conf_emoji = "⚠️"
            else:
                conf_class = "confidence-low"
                conf_emoji = "❓"
            
            response += f"{conf_emoji} **Confidence:** <span class='{conf_class}'>{conf:.0%}</span>\n"
            response += f"*{ml_result.get('confidence_explanation', '')}*\n\n"
            
            # Urgency
            urgency_emoji = {"CRITICAL": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
            urgency_class = f"urgency-{urgency.lower()}"
            response += f"{urgency_emoji.get(urgency, '🟡')} <span class='urgency-badge {urgency_class}'>Urgency: {urgency}</span>\n\n"
            
            # What is it?
            if "explanation" in ml_result:
                response += f"**📖 What is it?**\n{ml_result['explanation']}\n\n"
            
            # Causes
            if "causes" in ml_result and ml_result["causes"]:
                response += f"**🔍 Common Causes:**\n"
                for cause in ml_result["causes"][:3]:
                    response += f"• {cause}\n"
                response += "\n"
            
            # Precautions
            if "precautions" in ml_result:
                response += f"**✅ Recommended Actions:**\n"
                for i, prec in enumerate(ml_result["precautions"][:5], 1):
                    response += f"{i}. {prec}\n"
                response += "\n"
            
            # Specialist
            response += f"**👨‍⚕️ Specialist to Consult:** {ml_result.get('specialist', 'General Physician')}\n\n"
            
            # Matched symptoms
            if "matched_symptoms" in ml_result and ml_result["matched_symptoms"]:
                response += f"**🎯 Matched Symptoms:** {', '.join(ml_result['matched_symptoms'][:5])}\n\n"
            
            # Referral guidance
            if "referral" in ml_result:
                response += f"{ml_result['referral']}\n"
            
            # Alternatives
            if "alternatives" in ml_result and ml_result["alternatives"]:
                response += f"**💡 Other Possibilities:**\n"
                for alt in ml_result["alternatives"][:2]:
                    response += f"• {alt['disease']} ({alt['confidence']:.0%})\n"
                response += "\n"
            
            # Footer
            response += "---\n\n"
            response += "⚠️ **Important:** This is a preliminary assessment. Please consult a healthcare professional for proper diagnosis.\n\n"
            response += "**Was this helpful?** (Yes/No)"
            
            st.session_state.chat_history.append(("bot", response))
            st.session_state.last_result = ml_result
            st.session_state.state = "await_feedback"

    # === FEEDBACK ===
    elif state == "await_feedback" and intent in {"yes", "no"}:
        res = st.session_state.last_result
        if res:
            log_feedback(
                st.session_state.symptoms, 
                res["predicted_disease"], 
                res["confidence"], 
                intent
            )
            
            if intent == "yes":
                msg = "✅ **Glad I could help!**\n\n"
                msg += "**Next Steps:**\n"
                msg += "• Follow the recommended precautions\n"
                msg += "• Monitor your symptoms\n"
                msg += "• Consult a doctor if symptoms worsen\n"
                msg += "• Keep track of any changes\n\n"
                msg += "Take care and feel better soon! 💙\n\n"
                msg += "*Type 'new' to start another consultation*"
            else:
                msg = "📝 **Thank you for the feedback.**\n\n"
                msg += "**I recommend:**\n"
                msg += "• See a doctor in person for examination\n"
                msg += "• Get proper diagnostic tests if needed\n"
                msg += "• Seek a second opinion if concerned\n\n"
                msg += "Your health matters! 🏥\n\n"
                msg += "*Type 'new' to start another consultation*"
            
            st.session_state.chat_history.append(("bot", msg))
        
        st.session_state.state = "greeting"

    # === OTHER RESPONSES ===
    elif intent == "thanks":
        st.session_state.chat_history.append(
            ("bot", "😊 You're welcome! Stay healthy and take care!\n\n"
                    "*Type 'new' to start another consultation*")
        )
    elif intent == "goodbye":
        st.session_state.chat_history.append(
            ("bot", "👋 Goodbye! Remember:\n"
                    "• Consult a doctor if symptoms persist\n"
                    "• Call emergency services if needed\n"
                    "• Take care of yourself!\n\n"
                    "Stay safe and healthy! 💙")
        )
    else:
        if st.session_state.state not in ["followup", "await_feedback"]:
            st.session_state.chat_history.append(
                ("bot", "💬 I'm not sure I understood that.\n\n"
                        "• To describe symptoms, be specific\n"
                        "• To start over, type 'new'\n"
                        "• For help, just ask!\n\n"
                        "What would you like to do?")
            )

# ----------------------------------------
# Display Chat
# ----------------------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Diagnobot Pro v3.0 | Enhanced AI Health Assistant | For Educational Purposes Only | Always Consult a Doctor")