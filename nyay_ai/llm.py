# llm.py - Google Gemini API Integration
# Handles legal responses using google-generativeai SDK (google.generativeai).
# Falls back to offline keyword-based answers if Gemini is unavailable.

import json
import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Check google-genai SDK availability once at startup
try:
    import google.genai as _genai
    _GENAI_OK = True
except Exception as _genai_err:
    _GENAI_OK = False
    print(f"[LLM] google-genai not available ({_genai_err}). Running in offline-only mode.")

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_GEMINI_MODEL = "gemini-flash-latest"       # primary model
FALLBACK_GEMINI_MODEL = "gemini-2.0-flash-lite"  # lighter model, different quota bucket
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2.5
CALL_DELAY_SECONDS = 0.3

# ── Circuit Breaker ───────────────────────────────────────────────────────────
# When all models fail (quota/404), skip Gemini for CIRCUIT_BREAKER_SECONDS
CIRCUIT_BREAKER_SECONDS = 300   # 5 minutes
_circuit_open_until: float = 0.0  # epoch timestamp; 0 = closed (Gemini enabled)


def _circuit_is_open() -> bool:
    """Return True if Gemini calls should be skipped right now."""
    return time.time() < _circuit_open_until


def _trip_circuit(seconds: float = CIRCUIT_BREAKER_SECONDS) -> None:
    """Open the circuit breaker for `seconds` seconds."""
    global _circuit_open_until
    _circuit_open_until = time.time() + seconds
    mins = int(seconds // 60)
    print(f"[LLM] Circuit breaker OPEN — skipping Gemini for {mins} min (quota/model exhausted)")
LAWS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'laws.txt')

# ── Offline Knowledge Base ───────────────────────────────────────────────────
# Used when Gemini API is unavailable. Keywords are multilingual (En/Hi/Mr/Tamil/Telugu/Bengali).
OFFLINE_QA = {
    'arrest': {
        'keywords': [
            'arrested', 'arrest', 'गिरफ्तार', 'गिरफ़्तारी', 'अटक', 'अटकणे',
            'custody', 'हिरासत', 'कोठडी', 'கைது', 'అరెస్ట్', 'গ্রেফতার'
        ],
        'answer_en': (
            "Your Rights on Arrest (BNSS Section 37 & 43):\n"
            "1. Right to know the reason for arrest\n"
            "2. Right to remain silent – you cannot be forced to confess\n"
            "3. Right to inform a family member or friend\n"
            "4. Right to consult a lawyer of your choice (free if you cannot afford one)\n"
            "5. Must be produced before a Magistrate within 24 hours\n"
            "6. Right to medical examination if injured\n"
            "7. Women cannot be arrested after sunset / before sunrise without female officer\n"
            "📞 Free Legal Aid: 15100 | Police: 100"
        ),
        'answer_hi': (
            "गिरफ्तारी पर आपके अधिकार (BNSS धारा 37 और 43):\n"
            "1. गिरफ्तारी का कारण जानने का अधिकार\n"
            "2. चुप रहने का अधिकार – जबरदस्ती बयान नहीं लिया जा सकता\n"
            "3. परिवार को सूचित करने का अधिकार\n"
            "4. वकील से मिलने का अधिकार (अगर फीस नहीं दे सकते तो मुफ्त)\n"
            "5. 24 घंटे के अंदर मजिस्ट्रेट के सामने पेश करना जरूरी\n"
            "6. चोट लगी हो तो मेडिकल जांच का अधिकार\n"
            "7. महिला को सूर्यास्त के बाद / सूर्योदय से पहले बिना महिला अधिकारी के नहीं पकड़ सकते\n"
            "📞 मुफ्त कानूनी सहायता: 15100 | पुलिस: 100"
        ),
        'answer_mr': (
            "अटक झाल्यावर तुमचे हक्क (BNSS कलम 37 व 43):\n"
            "1. अटकेचे कारण जाणून घेण्याचा हक्क\n"
            "2. शांत राहण्याचा हक्क – जबरदस्तीने कबुली घेता येत नाही\n"
            "3. कुटुंबाला कळवण्याचा हक्क\n"
            "4. वकिलाशी बोलण्याचा हक्क (फी नसेल तर मोफत)\n"
            "5. 24 तासांत दंडाधिकाऱ्यापुढे सादर करणे बंधनकारक\n"
            "6. दुखापत असल्यास वैद्यकीय तपासणीचा हक्क\n"
            "7. सूर्यास्तानंतर / सूर्योदयापूर्वी महिला पोलिसाशिवाय महिलेला अटक नाही\n"
            "📞 मोफत कायदेशीर मदत: 15100 | पोलीस: 100"
        ),
    },
    'domestic_violence': {
        'keywords': [
            'domestic violence', 'घरगुती हिंसा', 'घरेलू हिंसा', 'पती मारतो',
            'husband beats', 'పతి కొడతాడు', 'husband hits', 'दहेज', 'dowry',
            'हुंडा', 'குடும்ப வன்முறை', 'গার্হস্থ্য সহিংসতা', 'cruelty by husband'
        ],
        'answer_en': (
            "Protection under Domestic Violence Act 2005 & BNS Section 85:\n"
            "1. File FIR at the nearest police station\n"
            "2. Call Women Helpline: 181 (free, 24×7)\n"
            "3. Approach a Magistrate directly with help from a Protection Officer\n"
            "4. You can get a Protection Order, Residence Order, and Monetary Relief\n"
            "5. Husband or in-laws cannot evict you from the home without a court order\n"
            "6. Dowry harassment: BNS Section 84 (Dowry Death) & Section 85 (Cruelty)\n"
            "📞 Women Helpline: 181 | Police: 100 | Legal Aid: 15100"
        ),
        'answer_hi': (
            "घरेलू हिंसा अधिनियम 2005 और BNS धारा 85 के अंतर्गत सुरक्षा:\n"
            "1. नजदीकी पुलिस थाने में FIR दर्ज करें\n"
            "2. महिला हेल्पलाइन: 181 पर कॉल करें (मुफ्त, 24×7)\n"
            "3. सीधे मजिस्ट्रेट के पास जाएं (संरक्षण अधिकारी मदद करेंगे)\n"
            "4. संरक्षण आदेश, निवास आदेश और आर्थिक राहत मिल सकती है\n"
            "5. बिना अदालत के आदेश के पति या ससुराल वाले घर से नहीं निकाल सकते\n"
            "6. दहेज उत्पीड़न: BNS धारा 84 (दहेज मृत्यु) और धारा 85 (क्रूरता)\n"
            "📞 महिला हेल्पलाइन: 181 | पुलिस: 100 | कानूनी सहायता: 15100"
        ),
        'answer_mr': (
            "घरगुती हिंसाचार कायदा 2005 आणि BNS कलम 85 अंतर्गत संरक्षण:\n"
            "1. जवळच्या पोलीस ठाण्यात FIR दाखल करा\n"
            "2. महिला हेल्पलाइन: 181 वर कॉल करा (मोफत, 24×7)\n"
            "3. संरक्षण अधिकाऱ्याच्या मदतीने थेट दंडाधिकाऱ्याकडे जा\n"
            "4. संरक्षण आदेश, निवास आदेश आणि आर्थिक दिलासा मिळू शकतो\n"
            "5. न्यायालयाच्या आदेशाशिवाय कोणीही घरातून काढू शकत नाही\n"
            "6. हुंडा छळ: BNS कलम 84 (हुंडाबळी) आणि कलम 85 (क्रूरता)\n"
            "📞 महिला हेल्पलाइन: 181 | पोलीस: 100 | मोफत मदत: 15100"
        ),
    },
    'theft': {
        'keywords': [
            'theft', 'stolen', 'चोरी', 'rob', 'लूट', 'चोरटे', 'robbery',
            'snatch', 'झपटमारी', 'கொள்ளை', 'దొంగతనం', 'চুরি'
        ],
        'answer_en': (
            "Theft – BNS Section 303 | Robbery – BNS Section 309:\n"
            "1. File FIR immediately at the nearest police station (it is FREE)\n"
            "2. Police cannot refuse FIR – if they do, contact SP or file before Magistrate\n"
            "3. Preserve all evidence (CCTV footage, photos, receipts)\n"
            "4. For snatch theft, note the description of the accused\n"
            "5. You can file a Zero FIR at any station regardless of jurisdiction\n"
            "📞 Police: 100 | Legal Aid: 15100"
        ),
        'answer_hi': (
            "चोरी – BNS धारा 303 | लूट – BNS धारा 309:\n"
            "1. तुरंत नजदीकी थाने में FIR दर्ज करें (बिल्कुल मुफ्त)\n"
            "2. FIR से मना नहीं कर सकते – मना करें तो SP या मजिस्ट्रेट से मिलें\n"
            "3. सभी सबूत सुरक्षित रखें (CCTV, फोटो, रसीद)\n"
            "4. झपटमारी हो तो आरोपी का हुलिया नोट करें\n"
            "5. कहीं भी किसी भी थाने में Zero FIR दर्ज करा सकते हैं\n"
            "📞 पुलिस: 100 | कानूनी सहायता: 15100"
        ),
        'answer_mr': (
            "चोरी – BNS कलम 303 | दरोडा – BNS कलम 309:\n"
            "1. लगेच जवळच्या पोलीस ठाण्यात FIR दाखल करा (मोफत आहे)\n"
            "2. FIR नाकारता येत नाही – नाकारल्यास SP किंवा दंडाधिकाऱ्याकडे जा\n"
            "3. सर्व पुरावे जपवा (CCTV, फोटो, पावत्या)\n"
            "4. दरोड्याचे प्रकरण असल्यास आरोपीचे वर्णन नोंदवा\n"
            "5. कोणत्याही पोलीस ठाण्यात Zero FIR दाखल करता येते\n"
            "📞 पोलीस: 100 | मोफत मदत: 15100"
        ),
    },
    'harassment': {
        'keywords': [
            'harassment', 'उत्पीड़न', 'छळ', 'sexual harassment', 'लैंगिक छळ',
            'workplace harassment', 'कामाच्या ठिकाणी', 'POSH', 'ICC',
            'யौன் துன்புறுத்தல்', 'లైంగిక వేధింపు', 'যৌন হয়রানি'
        ],
        'answer_en': (
            "POSH Act 2013 (Sexual Harassment at Workplace):\n"
            "1. File written complaint with Internal Complaints Committee (ICC) within 3 months\n"
            "2. If ICC does not exist, approach Local Complaints Committee (LCC) at district level\n"
            "3. ICC must complete inquiry within 90 days\n"
            "4. All proceedings are CONFIDENTIAL\n"
            "5. Relief: written apology, suspension, termination, compensation\n"
            "6. Complaint can be filed even if you have left the organisation\n"
            "📞 Women Helpline: 181 | Police: 100 | Legal Aid: 15100"
        ),
        'answer_hi': (
            "POSH अधिनियम 2013 (कार्यस्थल पर यौन उत्पीड़न):\n"
            "1. घटना के 3 महीने के अंदर ICC को लिखित शिकायत दें\n"
            "2. ICC न हो तो जिला स्तर पर LCC से संपर्क करें\n"
            "3. ICC को 90 दिन में जांच पूरी करनी होगी\n"
            "4. सभी कार्यवाही गोपनीय रहती है\n"
            "5. राहत: लिखित माफी, निलंबन, बर्खास्तगी, मुआवजा\n"
            "6. संस्था छोड़ने के बाद भी शिकायत दर्ज कर सकते हैं\n"
            "📞 महिला हेल्पलाइन: 181 | पुलिस: 100 | कानूनी सहायता: 15100"
        ),
        'answer_mr': (
            "POSH कायदा 2013 (कामाच्या ठिकाणी लैंगिक छळ):\n"
            "1. घटनेच्या 3 महिन्यांत ICC ला लेखी तक्रार द्या\n"
            "2. ICC नसल्यास जिल्हा स्तरावरील LCC कडे जा\n"
            "3. ICC ने 90 दिवसांत चौकशी पूर्ण करणे बंधनकारक\n"
            "4. सर्व कार्यवाही गोपनीय ठेवली जाते\n"
            "5. दिलासा: लेखी माफी, निलंबन, बडतर्फी, नुकसानभरपाई\n"
            "6. संस्था सोडल्यानंतरही तक्रार दाखल करता येते\n"
            "📞 महिला हेल्पलाइन: 181 | पोलीस: 100 | मोफत मदत: 15100"
        ),
    },
    'labour': {
        'keywords': [
            'wages', 'salary', 'minimum wage', 'वेतन', 'मजदूरी', 'मजुरी',
            'employer', 'मालक', 'job', 'employment', 'नौकरी', 'काम',
            'கூலி', 'వేతనం', 'মজুরি', 'gratuity', 'EPF', 'provident fund',
            'overtime', 'ओवरटाइम'
        ],
        'answer_en': (
            "Your Labour Rights:\n"
            "1. Minimum Wages Act 1948: employer must pay state-fixed minimum wage\n"
            "2. Non-payment → file complaint with Labour Commissioner (your district)\n"
            "3. Equal pay for equal work – Equal Remuneration Act 1976\n"
            "4. Gratuity after 5 years of service – Payment of Gratuity Act 1972\n"
            "5. EPF: both employee & employer contribute 12% of basic salary\n"
            "6. Maternity leave: 26 weeks for first two children\n"
            "7. Overtime must be paid at double the ordinary rate\n"
            "📞 Labour Commissioner | Legal Aid: 15100"
        ),
        'answer_hi': (
            "आपके श्रम अधिकार:\n"
            "1. न्यूनतम वेतन अधिनियम 1948: नियोक्ता को राज्य-निर्धारित न्यूनतम वेतन देना अनिवार्य\n"
            "2. वेतन न दे तो जिला श्रम आयुक्त के पास शिकायत करें\n"
            "3. समान काम के लिए समान वेतन – समान पारिश्रमिक अधिनियम 1976\n"
            "4. 5 साल की सेवा के बाद ग्रेच्युटी – भुगतान ग्रेच्युटी अधिनियम 1972\n"
            "5. EPF: कर्मचारी और नियोक्ता दोनों मूल वेतन का 12% जमा करते हैं\n"
            "6. मातृत्व अवकाश: पहले दो बच्चों के लिए 26 हफ्ते\n"
            "7. ओवरटाइम की दोगुनी दर से भुगतान होना चाहिए\n"
            "📞 जिला श्रम आयुक्त | कानूनी सहायता: 15100"
        ),
        'answer_mr': (
            "तुमचे कामगार हक्क:\n"
            "1. किमान वेतन कायदा 1948: मालकाने राज्य-निर्धारित किमान वेतन द्यायलाच हवे\n"
            "2. वेतन न दिल्यास जिल्हा कामगार आयुक्तांकडे तक्रार करा\n"
            "3. समान कामासाठी समान वेतन – समान मोबदला कायदा 1976\n"
            "4. 5 वर्षे सेवेनंतर ग्रॅच्युइटी – ग्रॅच्युइटी अदायगी कायदा 1972\n"
            "5. EPF: कर्मचारी व मालक दोघेही मूळ वेतनाचे 12% जमा करतात\n"
            "6. प्रसूती रजा: पहिल्या दोन अपत्यांसाठी 26 आठवडे\n"
            "7. ओव्हरटाईमसाठी दुप्पट दराने पगार मिळायला हवा\n"
            "📞 जिल्हा कामगार आयुक्त | मोफत मदत: 15100"
        ),
    },
    'rti': {
        'keywords': [
            'RTI', 'right to information', 'माहितीचा अधिकार', 'सूचना का अधिकार',
            'सरकारी माहिती', 'government information', 'PIO', 'public information',
            'তথ্যের অধিকার', 'సమాచార హక్కు', 'தகவல் உரிமை'
        ],
        'answer_en': (
            "Right to Information Act 2005:\n"
            "1. Any citizen can file RTI to any Government department\n"
            "2. Write application in Hindi, English or local language\n"
            "3. Fee: Rs. 10 (Central Govt); BPL card holders are EXEMPT\n"
            "4. PIO must reply within 30 days (48 hours for life & liberty matters)\n"
            "5. No reply = First Appeal to Appellate Authority within 30 days\n"
            "6. Second Appeal to State/Central Information Commission within 90 days\n"
            "7. Penalty on PIO for delay: Rs. 250/day up to Rs. 25,000\n"
            "📞 Legal Aid: 15100"
        ),
        'answer_hi': (
            "सूचना का अधिकार अधिनियम 2005:\n"
            "1. कोई भी नागरिक किसी भी सरकारी विभाग को RTI लगा सकता है\n"
            "2. हिंदी, अंग्रेजी या स्थानीय भाषा में आवेदन करें\n"
            "3. शुल्क: Rs. 10 (केंद्र सरकार); BPL कार्डधारक को छूट\n"
            "4. PIO को 30 दिन में (जीवन/स्वतंत्रता मामलों में 48 घंटे) जवाब देना होगा\n"
            "5. जवाब न मिले तो 30 दिन में प्रथम अपील करें\n"
            "6. 90 दिन में CIC/SIC को द्वितीय अपील\n"
            "7. देरी पर PIO पर जुर्माना: Rs. 250/दिन (अधिकतम Rs. 25,000)\n"
            "📞 कानूनी सहायता: 15100"
        ),
        'answer_mr': (
            "माहितीचा अधिकार कायदा 2005:\n"
            "1. कोणताही नागरिक कोणत्याही सरकारी विभागाकडे RTI दाखल करू शकतो\n"
            "2. हिंदी, इंग्रजी किंवा स्थानिक भाषेत अर्ज करा\n"
            "3. शुल्क: Rs. 10 (केंद्र सरकार); BPL कार्डधारक सूट\n"
            "4. PIO ने 30 दिवसांत (जीवन/स्वातंत्र्य प्रकरणात 48 तास) उत्तर द्यायला हवे\n"
            "5. उत्तर नाही तर 30 दिवसांत प्रथम अपील करा\n"
            "6. 90 दिवसांत CIC/SIC कडे द्वितीय अपील\n"
            "7. उशिरास PIO वर दंड: Rs. 250/दिवस (कमाल Rs. 25,000)\n"
            "📞 मोफत मदत: 15100"
        ),
    },
}

# ── Helper: detect language from response ───────────────────────────────────
def _pick_offline_answer(qa_entry: dict, language: str) -> str:
    if language == 'marathi':
        return qa_entry.get('answer_mr', qa_entry.get('answer_en', ''))
    if language == 'hindi':
        return qa_entry.get('answer_hi', qa_entry.get('answer_en', ''))
    return qa_entry.get('answer_en', '')


# ── Offline Fallback Logic ───────────────────────────────────────────────────
def offline_answer(query: str, language: str = 'english') -> dict | None:
    """Return an answer from the local knowledge base without any API call."""
    query_lower = query.lower()
    for category, data in OFFLINE_QA.items():
        if any(kw in query_lower for kw in data['keywords']):
            answer_text = _pick_offline_answer(data, language)
            
            # Format offline answers based on language
            if language == 'hindi':
                formatted_answer = f"✔ कानून: BNS (अपराध से संबंधित)\n✔ आपका अधिकार: संविधान के अनुसार\n✔ आपको क्या करना चाहिए:\n{answer_text}\n✔ पुलिस का कर्तव्य: FIR दर्ज करना अनिवार्य है\n✔ समर्थन: NALSA द्वारा मुफ्त कानूनी सहायता - 15100"
            elif language == 'marathi':
                formatted_answer = f"✔ कायदा: BNS (गुन्ह्याशी संबंधित)\n✔ आपला हक्क: संविधान अनुसार\n✔ आपण काय करायला हवे:\n{answer_text}\n✔ पोलिसांचे कर्तव्य: FIR दाखल करणे बंधनकारक आहे\n✔ मदत: NALSA द्वारे मोफत कायदेशीर सहायता - 15100"
            else:
                formatted_answer = answer_text
            
            return {
                'answer': f"📴 [Offline Mode]\n\n{formatted_answer}",
                'relevant_sections': _extract_sections_from_file(query),
                'next_steps': [
                    'Police: 100' if language == 'english' else ('पुलिस: 100' if language == 'hindi' else 'पोलीस: 100'),
                    'Women Helpline: 181' if language == 'english' else ('महिला हेल्पलाइन: 181' if language == 'hindi' else 'महिला हेल्पलाइन: 181'),
                    'Free Legal Aid - NALSA: 15100' if language == 'english' else ('मुफ्त कानूनी सहायता - NALSA: 15100' if language == 'hindi' else 'मोफत कायदेशीर सहायता - NALSA: 15100'),
                ],
                'important_rights': [
                    'FIR is FREE – no one can refuse it' if language == 'english' else ('FIR बिल्कुल मुफ्त है' if language == 'hindi' else 'FIR पूर्णपणे मोफत आहे'),
                    'Free lawyer if you cannot afford one' if language == 'english' else ('अगर आप वकील की फीस नहीं दे सकते तो मुफ्त वकील' if language == 'hindi' else 'वकिलाची फी देता न येल तर मोफत वकील'),
                    'Magistrate within 24 hours of arrest' if language == 'english' else ('गिरफ्तारी के 24 घंटे में मजिस्ट्रेट के सामने' if language == 'hindi' else 'अटकेच्या 24 तासांत दंडाधिकाऱ्यापुढे'),
                ],
                'emergency_contacts': ['Police: 100', 'NALSA: 15100', 'Women: 181'],
            }
    return None


def _extract_sections_from_file(query: str, max_results: int = 3) -> list[str]:
    """Quick keyword scan of laws.txt to find relevant section headings."""
    if not os.path.exists(LAWS_FILE):
        return []
    try:
        with open(LAWS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        query_words = set(query.lower().split())
        sections = re.findall(r'(Section \d+[A-Za-z]* - [^\n]+)', content)
        hits = []
        for sec in sections:
            if any(w in sec.lower() for w in query_words):
                hits.append(sec)
        return hits[:max_results] or sections[:max_results]
    except Exception:
        return []


# ── Gemini SDK helpers ───────────────────────────────────────────────────────
def get_api_key() -> str:
    return os.environ.get('GEMINI_API_KEY', '').strip()


def get_model_name() -> str:
    model = os.environ.get('GEMINI_MODEL', DEFAULT_GEMINI_MODEL).strip()
    return model if model else DEFAULT_GEMINI_MODEL


def _call_gemini(prompt: str, max_output_tokens: int = 1024) -> str:
    """
    Call Gemini API using the google-genai SDK (google.genai).
    Raises immediately if package is missing; uses circuit breaker for quota errors.
    """
    if not _GENAI_OK:
        raise ImportError('google-genai is not installed')

    import google.genai as genai
    from google.genai import types as genai_types

    api_key = get_api_key()
    if not api_key:
        raise ValueError('GEMINI_API_KEY not set')

    client = genai.Client(api_key=api_key)
    model_name = get_model_name()
    tried_fallback = False
    retry_count = 0

    while True:
        try:
            time.sleep(CALL_DELAY_SECONDS)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = response.text
            if not text:
                raise ValueError('Empty response from Gemini')
            return text

        except Exception as err:
            err_str = str(err).lower()

            # ── Exhausted daily/total quota → try fallback model once, then fail fast ──
            is_quota_dead = 'limit: 0' in err_str or 'per_day' in err_str or 'perday' in err_str
            is_429 = '429' in err_str or 'resource_exhausted' in err_str

            if is_429 and is_quota_dead and not tried_fallback:
                model_name = FALLBACK_GEMINI_MODEL
                tried_fallback = True
                continue

            # ── Model 404 → try fallback model once ──
            if ('404' in err_str or 'not found' in err_str) and not tried_fallback:
                model_name = FALLBACK_GEMINI_MODEL
                tried_fallback = True
                continue

            # ── Transient per-minute rate limit or server overload (503) → short retry ──
            is_transient = (is_429 and not is_quota_dead) or ('503' in err_str) or ('unavailable' in err_str)
            if is_transient and retry_count < MAX_RETRIES:
                retry_count += 1
                time.sleep(RETRY_DELAY_SECONDS * retry_count)
                continue

            # Everything else (or fallback also failed) → raise immediately
            raise


def _clean_json(text: str) -> str:
    """Extract and clean JSON from potential markdown fences or surrounding text."""
    text = text.strip()
    
    # 1. Handle common markdown fences
    for prefix in ('```json', '```'):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()
    
    # 2. If it still doesn't look like JSON (starts with {), try to find the first { and last }
    if not text.startswith('{'):
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
            
    return text.strip()


# ── Public API ───────────────────────────────────────────────────────────────

def generate_legal_response(
    user_query: str,
    retrieved_laws: list,
    language: str = 'hindi',
    query_type: str = 'general',
) -> dict:
    """
    Generate a legal response.
    Tries Gemini first; falls back to offline knowledge base on any failure.
    """
    api_key = get_api_key()

    # ── Circuit breaker / No SDK / No key → offline immediately ──────────────
    if _circuit_is_open() or not _GENAI_OK or not api_key:
        if _circuit_is_open():
            print("[LLM] Circuit breaker open — using offline fallback")
        offline = offline_answer(user_query, language)
        return offline if offline else _fallback_response(user_query, language)

    # ── Build prompt ─────────────────────────────────────────────────────────
    law_context = '\n\n'.join(retrieved_laws[:5]) if retrieved_laws else 'General Indian law'

    if language == 'marathi':
        lang_instruction = (
            "मराठी भाषेत सोप्या शब्दांत उत्तर द्या. "
            "ग्रामीण लोकांना समजेल अशा साध्या मराठीत."
        )
        response_format_instruction = """
        Response Format (मराठी):
        ✔ कायदा: BNS XXX (गुन्हाचे नाव)
        ✔ आपला हक्क: संविधान अनुच्छेद XXX – अधिकारचे वर्णन
        ✔ आपण काय करायला हवे: (मराठीत 3-4 बिंदू)
        ✔ पोलिसांचे कर्तव्य: (मराठीत कर्तव्य)
        ✔ मदत: (मराठीत समर्थन संसाधने)
        """
    elif language == 'hindi':
        lang_instruction = (
            "हिंदी भाषा में सरल शब्दों में जवाब दें। "
            "ग्रामीण लोगों को समझ आए ऐसी सरल हिंदी में।"
        )
        response_format_instruction = """
        Response Format (हिंदी):
        ✔ कानून: BNS XXX (अपराध का नाम)
        ✔ आपका अधिकार: संविधान अनुच्छेद XXX – अधिकार का विवरण
        ✔ आपको क्या करना चाहिए: (हिंदी में 3-4 कदम)
        ✔ पुलिस का कर्तव्य: (हिंदी में कर्तव्य)
        ✔ समर्थन: (हिंदी में समर्थन संसाधन)
        """
    elif language == 'tamil':
        lang_instruction = "தமிழில் எளிய வார்த்தைகளில் பதிலளிக்கவும்."
        response_format_instruction = "Response Format (Tamil): Answer in structured sections with BNS code, Rights, What to do, Police Duty, and Support in Tamil."
    elif language == 'telugu':
        lang_instruction = "తెలుగులో సరళమైన పదాలలో సమాధానం ఇవ్వండి."
        response_format_instruction = "Response Format (Telugu): Answer in structured sections with BNS code, Rights, What to do, Police Duty, and Support in Telugu."
    elif language == 'bengali':
        lang_instruction = "বাংলায় সহজ ভাষায় উত্তর দিন।"
        response_format_instruction = "Response Format (Bengali): Answer in structured sections with BNS code, Rights, What to do, Police Duty, and Support in Bengali."
    else:
        lang_instruction = "Answer in simple English."
        response_format_instruction = """
        Response Format (English):
        ✔ Law: BNS XXX (Crime Name)
        ✔ Your Right: Article XXX – Right Description
        ✔ What You Should Do: (3-4 steps)
        ✔ Police Duty: (Police duty details)
        ✔ Support: (Support resources)
        """

    prompt = f"""You are Nyay AI, a legal rights assistant helping rural Indian citizens understand their rights.

User Question: {user_query}

Relevant Indian Laws:
{law_context}

Language Instruction: {lang_instruction}

{response_format_instruction}

Respond ONLY with a valid JSON object (no markdown fences) in this exact format:
{{
  "answer": "Formatted response with sections - must be in {language} language:\n✔ कायदा/कानून/Law: BNS XXX (Crime)\n✔ आपला/आपका/Your Right: ...\n✔ आपण/आपको/You Should: ...\n✔ पोलिसांचे/पुलिस/Police Duty: ...\n✔ मदत/समर्थन/Support: ...",
  "relevant_sections": ["Section XXX - description", "Section YYY - description"],
  "next_steps": ["Step 1", "Step 2", "Step 3"],
  "important_rights": ["Right 1", "Right 2", "Right 3"],
  "emergency_contacts": ["Police: 100", "Women Helpline: 181", "Legal Aid: 15100"]
}}

Critical Rules:
1. Answer MUST be in {language} language - DO NOT mix languages
2. Use simple language rural citizens understand
3. Mention specific BNS/BNSS/POSH/Labour law section numbers
4. Format answer with 5 sections marked with ✔
5. Be compassionate and supportive
6. Always mention NALSA free legal aid: 15100"""

    # ── Try Gemini ───────────────────────────────────────────────────────────
    try:
        raw = _call_gemini(prompt, max_output_tokens=1500)
        cleaned = _clean_json(raw)
        
        try:
            # First attempt: standard JSON parse
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Second attempt: try to fix common JSON errors (trailing commas, unescaped newlines)
            try:
                # Remove trailing commas
                fixed = re.sub(r',\s*([\]}])', r'\1', cleaned)
                # Handle potential unescaped newlines in JSON strings (risky but better than failing)
                fixed = fixed.replace('\n', '\\n').replace('\r', '\\r')
                # If we replaced \n inside { }, we might have double escaped legitimate ones
                fixed = fixed.replace('\\n\\n', '\\n') 
                return json.loads(fixed)
            except:
                # Third attempt: regex extract specifically for the 'answer' field
                answer_match = re.search(r'"answer":\s*"((?:\\.|[^"\\])*)"', raw)
                if answer_match:
                    ans_text = answer_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                    return {
                        'answer': ans_text,
                        'relevant_sections': extract_sections_from_text(raw),
                        'next_steps': ['Check with local police station', 'Consult NALSA free legal aid: 15100'],
                        'important_rights': ['Right to free legal aid', 'Right to a copy of FIR'],
                        'emergency_contacts': ['Police: 100', 'NALSA: 15100']
                    }
                
                # Final fallback: return the raw text but stripped of JSON delimiters if they are at the ends
                final_ans = raw.strip()
                if final_ans.startswith('{') and final_ans.endswith('}'):
                    # If it looks like JSON but we couldn't parse it, try to find the "answer" content without JSON markup
                    match = re.search(r'"answer":\s*"(.*?)"', final_ans, re.DOTALL)
                    if match:
                        final_ans = match.group(1)
                
                return {
                    'answer': final_ans,
                    'relevant_sections': extract_sections_from_text(raw),
                    'next_steps': ['Police: 100', 'Legal Aid: 15100', 'Women Helpline: 181'],
                    'important_rights': [],
                    'emergency_contacts': ['Police: 100', 'NALSA: 15100', 'Women Helpline: 181'],
                }

    except Exception as err:
        print(f"[LLM] Gemini failed ({err}), switching to offline fallback")
        # If the error is quota/model-not-found, trip the circuit breaker
        err_str = str(err).lower()
        if '429' in err_str or 'quota' in err_str or '404' in err_str or 'not found' in err_str or 'resource_exhausted' in err_str:
            _trip_circuit(CIRCUIT_BREAKER_SECONDS)
        offline = offline_answer(user_query, language)
        return offline if offline else _fallback_response(user_query, language)


def generate_fir_format(user_query: str, language: str = 'hindi') -> str:
    """
    Generate a ready-to-use FIR draft using Gemini.
    Falls back to a local template if Gemini is unavailable.
    """
    api_key = get_api_key()
    if not api_key:
        return _fallback_fir(user_query, language)

    if language == 'marathi':
        lang_instruction = "FIR मराठी भाषेत तयार करा"
        fmt = "Marathi"
    elif language == 'tamil':
        lang_instruction = "FIR தமிழில் தயாரிக்கவும்"
        fmt = "Tamil"
    elif language == 'telugu':
        lang_instruction = "FIR తెలుగులో తయారు చేయండి"
        fmt = "Telugu"
    elif language == 'bengali':
        lang_instruction = "FIR বাংলায় তৈরি করুন"
        fmt = "Bengali"
    else:
        lang_instruction = "FIR हिंदी भाषा में तैयार करें"
        fmt = "Hindi"

    prompt = f"""Generate a professional FIR (First Information Report) draft for the following incident.

Incident: {user_query}
Language: {fmt}

Include all these fields:
- To: (Officer In-Charge, Police Station)
- Subject:
- Complainant Details: Name, Age, Address, Phone [use placeholders]
- Date & Time of Incident:
- Place of Incident:
- Description of Incident: (3-4 sentences based on the query)
- Names of Accused (if known):
- Witnesses (if any):
- Evidence Available:
- Relief Sought:
- Declaration & Signature

{lang_instruction}. Use [YOUR NAME], [DATE], [ADDRESS] as placeholders where needed.
Keep the language simple and formal."""

    try:
        return _call_gemini(prompt, max_output_tokens=800)
    except Exception as err:
        print(f"[LLM] FIR generation failed ({err}), using template")
        return _fallback_fir(user_query, language)


def extract_sections_from_text(text: str) -> list[str]:
    """Extract law section references from free text."""
    sections = re.findall(
        r'(?:Section|धारा|कलम|ధారా|பிரிவு)\s+\d+[A-Za-z]*(?:\s+[A-Za-z]+)*',
        text,
    )
    return list(dict.fromkeys(sections))[:5]  # deduplicate, keep order


# ── Fallback templates ───────────────────────────────────────────────────────

def _fallback_response(query: str, language: str) -> dict:
    """Generic fallback when both Gemini and offline KB fail."""
    if language == 'marathi':
        return {
            'answer': (
                'आपला प्रश्न मिळाला. सध्या AI सेवा उपलब्ध नाही. '
                'कृपया खालील हेल्पलाइन क्रमांकावर संपर्क करा.'
            ),
            'relevant_sections': [
                'भारतीय न्याय संहिता (BNS) 2023',
                'भारतीय नागरिक सुरक्षा संहिता (BNSS) 2023',
            ],
            'next_steps': [
                'पोलीस: 100',
                'महिला हेल्पलाइन: 181',
                'मोफत कायदेशीर मदत NALSA: 15100',
                'जवळच्या पोलीस ठाण्यात जा',
            ],
            'important_rights': [
                'मोफत कायदेशीर मदत मिळण्याचा हक्क',
                '24 तासांत दंडाधिकाऱ्यापुढे सादर करणे बंधनकारक',
                'FIR मोफत दाखल होते',
            ],
            'emergency_contacts': ['Police: 100', 'Women Helpline: 181', 'NALSA: 15100'],
        }
    elif language == 'hindi':
        return {
            'answer': (
                'आपका प्रश्न प्राप्त हुआ। अभी AI सेवा उपलब्ध नहीं है। '
                'कृपया नीचे दिए गए हेल्पलाइन नंबर पर संपर्क करें।'
            ),
            'relevant_sections': [
                'भारतीय न्याय संहिता (BNS) 2023',
                'भारतीय नागरिक सुरक्षा संहिता (BNSS) 2023',
            ],
            'next_steps': [
                'पुलिस: 100',
                'महिला हेल्पलाइन: 181',
                'मुफ्त कानूनी सहायता NALSA: 15100',
                'नजदीकी पुलिस स्टेशन जाएं',
            ],
            'important_rights': [
                'मुफ्त कानूनी सहायता पाने का अधिकार',
                '24 घंटे में मजिस्ट्रेट के सामने पेश करना जरूरी',
                'FIR मुफ्त दर्ज होती है',
            ],
            'emergency_contacts': ['Police: 100', 'Women Helpline: 181', 'NALSA: 15100'],
        }
    else:
        return {
            'answer': (
                'Your query was received. The AI service is temporarily unavailable. '
                'Please contact the helplines below.'
            ),
            'relevant_sections': [
                'Bharatiya Nyaya Sanhita (BNS) 2023',
                'Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023',
            ],
            'next_steps': [
                'Police: 100',
                'Women Helpline: 181',
                'Free Legal Aid NALSA: 15100',
                'Visit nearest police station',
            ],
            'important_rights': [
                'Right to free legal aid',
                'Must be produced before Magistrate within 24 hours',
                'FIR is free of cost',
            ],
            'emergency_contacts': ['Police: 100', 'Women Helpline: 181', 'NALSA: 15100'],
        }


def _fallback_fir(query: str, language: str) -> str:
    today = datetime.now().strftime('%d/%m/%Y')

    if language == 'marathi':
        return f"""प्रति,
पोलीस निरीक्षक,
[पोलीस ठाण्याचे नाव], [जिल्हा]

विषय: [गुन्ह्याचे स्वरूप] – FIR नोंदणीसाठी अर्ज

महोदय/महोदया,

मी [आपले नाव], वय [वय], रा. [पूर्ण पत्ता], मोबाइल [फोन नंबर], आपणास खालील घटनेबद्दल तक्रार करत आहे:

घटनेची तारीख व वेळ: [तारीख], [वेळ]
घटनेचे ठिकाण: [पत्ता]

घटनेचे वर्णन:
{query}

आरोपीचे नाव/वर्णन: [आरोपीचे नाव किंवा वर्णन]
साक्षीदार: [साक्षीदाराचे नाव व पत्ता]
उपलब्ध पुरावे: [फोटो/व्हिडिओ/दस्तावेज]

विनंती: सदर घटनेची FIR नोंदवून योग्य कायदेशीर कारवाई करण्यात यावी.

दिनांक: {today}
अर्जदाराची सही: [सही]
नाव: [आपले नाव]"""

    elif language == 'tamil':
        return f"""செவிப்பதிவு அருள்,
காவல் அதிகாரி,
[காவல் நிலையம்], [மாவட்டம்]

பொருள்: [குற்றத்தின் தன்மை] – FIR பதிவு கோரிக்கை

ஐயா/அம்மா,

நான் [உங்கள் பெயர்], வயது [வயது], முகவரி [முழு முகவரி], மொபைல் [தொலைபேசி], கீழ்வரும் சம்பவம் குறித்து புகார் அளிக்கிறேன்:

சம்பவ தேதி மற்றும் நேரம்: [தேதி], [நேரம்]
சம்பவ இடம்: [முகவரி]

சம்பவ விவரம்:
{query}

குற்றவாளி பெயர்/விவரம்: [பெயர் அல்லது விவரம்]
சாட்சிகள்: [பெயர் மற்றும் முகவரி]
ஆதாரங்கள்: [புகைப்படம்/வீடியோ/ஆவணங்கள்]

கோரிக்கை: மேற்கண்ட சம்பவம் குறித்து FIR பதிவு செய்து உரிய நடவடிக்கை எடுக்கவும்.

தேதி: {today}
மனுதாரர் கையொப்பம்: [கையொப்பம்]
பெயர்: [உங்கள் பெயர்]"""

    else:  # Hindi (default)
        return f"""सेवा में,
थानाध्यक्ष महोदय/महोदया,
[पुलिस थाना], [जिला]

विषय: [अपराध का प्रकार] – FIR दर्ज करने हेतु प्रार्थना पत्र

महोदय/महोदया,

मैं [आपका नाम], उम्र [उम्र], निवासी [पूरा पता], मोबाइल [फोन नंबर], निम्नलिखित घटना के संबंध में शिकायत करना चाहता/चाहती हूँ:

घटना की तारीख एवं समय: [तारीख], [समय]
घटना का स्थान: [पता]

घटना का विवरण:
{query}

आरोपी का नाम/विवरण: [नाम या हुलिया]
गवाह: [गवाह का नाम और पता]
उपलब्ध साक्ष्य: [फोटो/वीडियो/दस्तावेज]

निवेदन: उक्त घटना की FIR दर्ज कर उचित कानूनी कार्रवाई की जाए।

दिनांक: {today}
प्रार्थी के हस्ताक्षर: [हस्ताक्षर]
नाव: [आपका नाम]"""
