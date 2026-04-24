# utils.py - Utility functions for language detection and helpers

import re

def detect_language(text):
    """
    Detect the language of the input text.
    Returns 'hindi', 'marathi', or 'english'
    """
    # Devanagari Unicode range: \u0900-\u097F
    devanagari_chars = len(re.findall(r'[\u0900-\u097F]', text))
    total_chars = len(text.strip())
    
    if total_chars == 0:
        return 'english'
    
    # If more than 20% characters are Devanagari, it's Hindi or Marathi
    if devanagari_chars / total_chars > 0.2:
        # Common Marathi-specific words
        marathi_words = [
            'आहे', 'नाही', 'केले', 'करणे', 'मला', 'तुम्ही', 'आम्ही',
            'त्यांनी', 'काय', 'कसे', 'कुठे', 'माझा', 'माझी', 'आपले',
            'झाले', 'होते', 'असेल', 'जाणे', 'येणे', 'सांगा', 'द्या'
        ]
        # Common Hindi-specific words
        hindi_words = [
            'है', 'हैं', 'था', 'थी', 'हूँ', 'हो', 'मैं', 'तुम',
            'आप', 'वह', 'यह', 'कया', 'कैसे', 'कहाँ', 'मुझे', 'हमें',
            'करना', 'जाना', 'आना', 'बताओ', 'दो', 'लो'
        ]
        
        marathi_count = sum(1 for word in marathi_words if word in text)
        hindi_count = sum(1 for word in hindi_words if word in text)
        
        if marathi_count > hindi_count:
            return 'marathi'
        else:
            return 'hindi'
    
    return 'english'


def is_emergency_query(text):
    """Check if the query is an emergency situation"""
    emergency_keywords = [
        # English
        'arrested', 'arrest', 'police', 'detention', 'custody', 'jail',
        'threat', 'attack', 'violence', 'harassment', 'rape', 'assault',
        'emergency', 'help', 'danger', 'kidnap', 'abduct',
        # Hindi
        'गिरफ्तार', 'पुलिस', 'हिरासत', 'जेल', 'धमकी', 'हमला',
        'उत्पीड़न', 'बलात्कार', 'खतरा', 'मदद', 'अपहरण',
        # Marathi
        'अटक', 'पोलीस', 'कोठडी', 'धमकी', 'हल्ला', 'छळ', 'बलात्कार',
        'धोका', 'मदत', 'अपहरण'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in emergency_keywords)


def is_crime_query(text):
    """Check if query involves a crime (for FIR generation)"""
    crime_keywords = [
        # English
        'theft', 'stolen', 'rob', 'robbery', 'fraud', 'cheat', 'cheated',
        'harassment', 'rape', 'assault', 'murder', 'kidnap', 'extortion',
        'domestic violence', 'dowry', 'bribe', 'corruption', 'accident',
        # Hindi
        'चोरी', 'लूट', 'धोखा', 'धोखाधड़ी', 'उत्पीड़न', 'बलात्कार',
        'हत्या', 'अपहरण', 'जबरदस्ती', 'दहेज', 'रिश्वत', 'भ्रष्टाचार',
        'हिंसा', 'मारपीट', 'दुर्घटना',
        # Marathi
        'चोरी', 'लूट', 'फसवणूक', 'छळ', 'बलात्कार', 'खून',
        'अपहरण', 'जबरदस्ती', 'हुंडा', 'लाच', 'भ्रष्टाचार', 'हिंसा'
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in crime_keywords)


def get_document_checklist(query_type):
    """Return document checklist based on query type"""
    checklists = {
        'fir': {
            'title': 'FIR दाखल करण्यासाठी / FIR दर्ज करने के लिए आवश्यक दस्तावेज',
            'documents': [
                'आधार कार्ड / Aadhaar Card',
                'घटनेचे लेखी वर्णन / Written description of incident',
                'साक्षीदारांची नावे आणि पत्ते / Names and addresses of witnesses',
                'पुरावे (फोटो, व्हिडिओ) / Evidence (photos, videos) if available',
                'वैद्यकीय अहवाल (जर लागू असेल) / Medical report (if applicable)'
            ]
        },
        'domestic_violence': {
            'title': 'घरगुती हिंसाचार / Domestic Violence - आवश्यक दस्तावेज',
            'documents': [
                'आधार कार्ड',
                'विवाह प्रमाणपत्र / Marriage Certificate',
                'वैद्यकीय अहवाल / Medical Reports',
                'जखमांचे फोटो / Photos of injuries',
                'साक्षीदार / Witnesses',
                'बँक खात्याचा तपशील / Bank account details'
            ]
        },
        'labour': {
            'title': 'कामगार तक्रार / Labour Complaint - आवश्यक दस्तावेज',
            'documents': [
                'नियुक्ती पत्र / Appointment Letter',
                'वेतन स्लिप / Salary Slips (last 3 months)',
                'ओळखपत्र / Identity Proof',
                'बँक खाते विवरण / Bank Statement',
                'ESI/PF Card (if applicable)',
                'Service Certificate'
            ]
        },
        'rti': {
            'title': 'RTI अर्ज / RTI Application - आवश्यक दस्तावेज',
            'documents': [
                'रु. १० चा पोस्टल ऑर्डर / Rs. 10 Postal Order (Central Govt.)',
                'आधार कार्ड झेरॉक्स / Aadhaar Card Xerox',
                'BPL कार्ड (शुल्क माफीसाठी) / BPL Card (for fee waiver)',
                'अर्जाची प्रत / Copy of application'
            ]
        },
        'posh': {
            'title': 'POSH - कार्यस्थळी लैंगिक छळ / Workplace Harassment - आवश्यक दस्तावेज',
            'documents': [
                'घटनेचे लेखी वर्णन / Written account of incidents (with dates)',
                'ईमेल / संदेश / Email or message evidence',
                'साक्षीदारांची माहिती / Witness information',
                'कंपनी ओळखपत्र / Company ID',
                'नियुक्ती पत्र / Appointment letter'
            ]
        }
    }
    return checklists.get(query_type, checklists['fir'])


def detect_query_category(text):
    """Detect which category the query belongs to for document checklist"""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ['domestic', 'घरगुती', 'पती', 'husband', 'dowry', 'दहेज', 'हुंडा']):
        return 'domestic_violence'
    elif any(w in text_lower for w in ['rti', 'right to information', 'माहिती']):
        return 'rti'
    elif any(w in text_lower for w in ['posh', 'workplace', 'office', 'कार्यस्थळ', 'कार्यालय', 'sexual harassment']):
        return 'posh'
    elif any(w in text_lower for w in ['labour', 'worker', 'salary', 'wage', 'कामगार', 'वेतन', 'नोकरी']):
        return 'labour'
    else:
        return 'fir'


# Mock legal aid centers data
LEGAL_AID_CENTERS = [
    {
        'name': 'District Legal Services Authority (DLSA)',
        'type': 'Legal Aid',
        'distance': '0.5 km',
        'address': 'District Court Complex',
        'phone': '15100',
        'helpline': 'NALSA Helpline: 15100'
    },
    {
        'name': 'Mahila Police Station / महिला पोलीस ठाणे',
        'type': 'Police',
        'distance': '1.2 km',
        'address': 'Near City Center',
        'phone': '100',
        'helpline': 'Police: 100'
    },
    {
        'name': 'District Legal Aid Clinic',
        'type': 'Legal Aid',
        'distance': '2.0 km',
        'address': 'District Collectorate',
        'phone': '1800-233-4422',
        'helpline': 'Free Legal Aid'
    },
    {
        'name': 'Women Helpline / महिला हेल्पलाइन',
        'type': 'Helpline',
        'distance': '0 km',
        'address': 'Available 24x7',
        'phone': '181',
        'helpline': 'Women Helpline: 181'
    },
    {
        'name': 'National Commission for Women Helpline',
        'type': 'Helpline',
        'distance': '0 km',
        'address': 'Available 24x7',
        'phone': '7827-170-170',
        'helpline': 'NCW Helpline: 7827-170-170'
    }
]

EMERGENCY_RIGHTS = [
    "🔴 आपल्याला शांत राहण्याचा अधिकार आहे - You have the RIGHT TO REMAIN SILENT",
    "⚖️ आपल्याला वकील मिळण्याचा अधिकार आहे - You have the RIGHT TO A LAWYER (Free if you can't afford)",
    "⏰ तुम्हाला 24 तासांत दंडाधिकाऱ्यापुढे हजर करणे अनिवार्य आहे - Must be produced before Magistrate within 24 HOURS",
    "📋 अटकेचे कारण जाणून घेण्याचा अधिकार आहे - RIGHT TO KNOW REASON for arrest",
    "👨‍👩‍👧 कुटुंबाला सूचित करण्याचा अधिकार आहे - RIGHT TO INFORM your family",
    "🏥 वैद्यकीय तपासणीचा अधिकार आहे - RIGHT TO MEDICAL EXAMINATION",
    "🚺 महिलेला सूर्यास्तानंतर अटक होत नाही - Women CANNOT be arrested after SUNSET (except special cases)",
    "📄 जामीनपात्र गुन्ह्यात जामीनाचा अधिकार आहे - RIGHT TO BAIL in bailable offences",
    "❌ जबरदस्ती कबुली देण्याची गरज नाही - You CANNOT be forced to confess",
    "🆓 मोफत कायदेशीर मदत मिळण्याचा अधिकार आहे - FREE LEGAL AID is your right: Call 15100"
]
