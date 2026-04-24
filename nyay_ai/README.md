# 🏛️ Nyay AI — न्याय AI
## Multilingual Legal Rights Assistant for Rural India

A complete AI-powered legal chatbot for rural India, supporting Hindi, Marathi, and English.

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies
```bash
cd nyay_ai
pip install flask requests numpy scikit-learn
```

### Step 2: Set Your Gemini API Key
```bash
# Get free API key from: https://makersuite.google.com/app/apikey

# On Linux/Mac:
export GEMINI_API_KEY="your-api-key-here"

# On Windows (Command Prompt):
set GEMINI_API_KEY=your-api-key-here

# On Windows (PowerShell):
$env:GEMINI_API_KEY="your-api-key-here"
```

### Step 3: Run the App
```bash
python app.py
```

### Step 4: Open Browser
Go to: **http://127.0.0.1:5000**

---

## 📁 Project Structure
```
nyay_ai/
├── app.py              # Flask backend (main server)
├── rag.py              # RAG pipeline (law retrieval)
├── llm.py              # Gemini AI integration
├── utils.py            # Language detection & helpers
├── requirements.txt    # Python dependencies
├── data/
│   └── laws.txt        # Legal dataset (BNS, BNSS, POSH, Labour)
└── templates/
    └── index.html      # Frontend UI
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 AI Chatbot | Ask legal questions in Hindi, Marathi, or English |
| 🔍 RAG Pipeline | Retrieves relevant law sections (BNS, BNSS, POSH) |
| 🌐 Language Detection | Auto-detects Hindi/Marathi/English |
| 🚨 Emergency Mode | Immediate rights for arrest/harassment situations |
| 🧾 FIR Generator | Ready-to-use FIR template based on your incident |
| 🗺️ Legal Help Finder | Shows nearby legal aid centers & helplines |
| 🎤 Voice Input | Speak questions in Hindi/Marathi (Web Speech API) |
| 🔊 Voice Output | Hear responses read aloud |
| 📋 Document Checklist | Required documents for your case type |
| 📵 Offline Mode | Works without internet using local Q&A database |

---

## 🎯 Demo Queries to Test

1. **मला चोरी झाली आहे, FIR कशी दाखल करावी?** (Marathi - Theft)
2. **पती मला मारहाण करतो, मला काय करता येईल?** (Marathi - Domestic Violence)
3. **कामाच्या ठिकाणी माझ्यावर लैंगिक छळ होत आहे** (Marathi - POSH)
4. **मुझे पुलिस ने गिरफ्तार किया, मेरे क्या अधिकार हैं?** (Hindi - Arrest Rights)
5. **My employer is not paying minimum wages** (English - Labour)

---

## 📞 Emergency Helplines (Built-in)
- 🚔 **Police**: 100
- 👩 **Women Helpline**: 181
- ⚖️ **NALSA Legal Aid**: 15100
- 🚑 **Ambulance**: 108
- 👶 **Child Helpline**: 1098
- 🏛️ **NCW Helpline**: 7827-170-170

---

## 🔧 Optional: Enhanced RAG with Sentence Transformers

For better multilingual retrieval:
```bash
pip install sentence-transformers faiss-cpu
```
The app will automatically use better embeddings when these are installed.

---

## 📖 Legal Data Covered
- **BNS (Bharatiya Nyaya Sanhita) 2023** — Criminal law
- **BNSS (Bharatiya Nagarik Suraksha Sanhita) 2023** — Criminal procedure
- **POSH Act 2013** — Workplace sexual harassment
- **Labour Laws** — Minimum wages, EPF, maternity, gratuity
- **Domestic Violence Act 2005**
- **RTI Act 2005**

---

## ⚠️ Disclaimer
Nyay AI provides general legal information for educational purposes only. 
It is not a substitute for professional legal advice. 
For specific legal matters, please consult a qualified lawyer.
