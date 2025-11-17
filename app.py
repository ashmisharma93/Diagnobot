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
# ----------------------------------------cls

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