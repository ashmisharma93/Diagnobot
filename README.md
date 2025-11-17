# 🩺 Diagnobot — AI-Powered Symptom-Based Health Chatbot

Diagnobot is an intelligent medical assistant that analyzes user symptoms,
detects emergencies, predicts possible conditions, suggests precautions, and
recommends the right specialist.  
This system uses a hybrid AI architecture combining:

- ✔ NLP-based medical text preprocessing  
- ✔ Rule-based pattern matching  
- ✔ Emergency detection engine  
- ✔ Ensemble ML model (SVC + Logistic Regression + Naive Bayes)  
- ✔ Streamlit Chat UI  
- ✔ Smart follow-up question system  

---

## 🚀 Features

### Hybrid Diagnosis Engine
- Emergency detection (Heart attack, Stroke, Anaphylaxis, etc.)
- Pattern-based reasoning with medical knowledge base
- ML fallback using SentenceTransformer embeddings

### 💬 Multi-turn Chatbot Interface
- Built with Streamlit
- Detects symptom intent, severity, duration
- Asks follow-up questions for better accuracy
- Displays urgency badges & confidence levels

###  Intelligent NLP Pipeline
- Medical phrase preservation (e.g., chest pain → chestpain)
- Custom stopword filtering
- Lemmatization
- Severity, duration & location extraction

---

## 📂 Project Structure
```
Diagnobot/
│
├── app.py               # Streamlit chatbot UI
├── diagnose_api.py      # Core diagnosis engine
├── utils.py             # NLP preprocessors & validators
├── disease_info.py      # Medical metadata (symptoms/precautions)
├── Train_Model.ipynb    # Notebook for ML model training
├── requirements.txt     # Dependencies
│
├── models/              # (Optional) Download externally
│ └── README.md          # Instructions for model downloads
│
└── README.md            # Project documentation

```


## 🧪 Installation & Usage

### 1️⃣ Clone the repository
```
git clone https://github.com/ashmisharma93/Diagnobot.git
cd Diagnobot
```

### 2️⃣ Create a virtual environment
```
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Download trained ML models  
*(If not included in repo — recommended)*  
Place them inside the **/models** folder.

### 5️⃣ Run the Streamlit app
```
streamlit run app.py
```

---

## 📥 Dataset
The dataset used for training is stored locally and is **not included** in the GitHub repository  
to keep the repo lightweight and respect data license guidelines.
Dataset used: Disease-Symptom Dataset
Source: [Dataset Source (Kaggle)]([https://www.kaggle.com/your-dataset-link](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset))

---

## 🛑 Important Note
Diagnobot Pro is **not a medical diagnostic tool**.  
It provides preliminary analysis only.  
For emergencies or serious symptoms, consult a certified medical professional.

---

## 🧑‍💻 Author
**Ashmita Sharma**  
B.Tech — Artificial Intelligence & Machine Learning  
Delhi Technical Campus, Greater Noida  

---


