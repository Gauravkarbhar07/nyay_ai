# app.py - Main Flask Application for Nyay AI
# Legal Rights Assistant for Rural India

from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime

# Import our custom modules
from utils import (
    detect_language, is_emergency_query, is_crime_query,
    get_document_checklist, detect_query_category,
    LEGAL_AID_CENTERS, EMERGENCY_RIGHTS
)
from llm import generate_legal_response, generate_fir_format

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'nyay-ai-secret-2024')

# Initialize RAG system
rag_initialized = False
def init_rag():
    global rag_initialized
    if not rag_initialized:
        try:
            from rag import initialize, retrieve_relevant_laws
            initialize()
            rag_initialized = True
            print("[APP] RAG system initialized successfully")
        except Exception as e:
            print(f"[APP] RAG initialization warning: {e}")

# Offline fallback Q&A database (for when internet is unavailable)
OFFLINE_QA = {
    'arrest': {
        'keywords': ['arrested', 'arrest', 'गिरफ्तार', 'अटक', 'पुलिस', 'पोलीस', 'custody', 'हिरासत', 'कोठडी'],
        'answer': 'गिरफ्तारी पर आपके अधिकार (अटक झाल्यावर आपले हक्क): 1) शांत राहण्याचा अधिकार (Right to remain silent) 2) वकील मिळण्याचा अधिकार (Right to lawyer) 3) 24 तासांत दंडाधिकाऱ्यापुढे हजर करणे (Magistrate within 24 hours) 4) कुटुंबाला सूचित करणे (Inform family) 5) मोफत कायदेशीर मदत: 15100'
    },
    'domestic_violence': {
        'keywords': ['domestic violence', 'घरगुती हिंसा', 'पती मारतो', 'husband beats', 'दहेज', 'dowry', 'हुंडा'],
        'answer': 'घरगुती हिंसाचार कायदा 2005 अंतर्गत: 1) जवळच्या पोलीस ठाण्यात FIR दाखल करा 2) महिला हेल्पलाइन 181 वर कॉल करा 3) दंडाधिकाऱ्याकडे तक्रार करा 4) संरक्षण आदेश मिळवा 5) मोफत कायदेशीर मदत: 15100'
    },
    'theft': {
        'keywords': ['theft', 'stolen', 'चोरी', 'rob', 'लूट'],
        'answer': 'चोरी झाल्यावर: 1) लगेच पोलीस ठाण्यात FIR दाखल करा (BNS कलम 303) 2) FIR मोफत आहे, नकार देता येत नाही 3) चोरीचे पुरावे जपवा 4) पोलीस: 100'
    },
    'harassment': {
        'keywords': ['harassment', 'उत्पीड़न', 'छळ', 'sexual harassment', 'लैंगिक छळ'],
        'answer': 'POSH कायदा 2013 अंतर्गत: 1) ICC कडे 3 महिन्यांत तक्रार करा 2) ICC नसल्यास LCC कडे जा 3) पोलीस: 100, महिला हेल्पलाइन: 181 4) गोपनीयता राखली जाते'
    }
}


@app.route('/')
def index():
    """Main page - serve the chatbot interface"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint.
    Process user query and return legal response.
    """
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        
        if not user_query:
            return jsonify({'error': 'कृपया प्रश्न टाईप करा / Please enter a question'}), 400
        
        # Step 1: Detect language
        language = detect_language(user_query)
        print(f"[APP] Query: {user_query[:50]}... | Language: {language}")
        
        # Step 2: Check if offline mode requested or API key missing
        if not os.environ.get('GEMINI_API_KEY'):
            offline_response = check_offline_mode(user_query, language)
            if offline_response:
                return jsonify({
                    'response': offline_response,
                    'language': language,
                    'mode': 'offline',
                    'is_crime': is_crime_query(user_query),
                    'query_category': detect_query_category(user_query),
                    'retrieved_sections': 0,
                    'similarity_scores': [],
                    'mapping': None
                })
        
        # Step 3: Retrieve relevant law sections using RAG with scores
        retrieved_laws = []
        retrieved_with_scores = []
        try:
            from rag import retrieve_relevant_laws_with_scores, get_bns_constitution_mapping
            retrieved_with_scores = retrieve_relevant_laws_with_scores(user_query, top_k=4)
            retrieved_laws = [item[0] for item in retrieved_with_scores]
            print(f"[APP] Retrieved {len(retrieved_with_scores)} law sections with scores")
        except Exception as e:
            print(f"[APP] RAG retrieval with scores warning: {e}")
            try:
                from rag import retrieve_relevant_laws
                retrieved_laws = retrieve_relevant_laws(user_query, top_k=4)
                retrieved_with_scores = [(law, 0.0, i) for i, law in enumerate(retrieved_laws)]
            except Exception as e2:
                print(f"[APP] RAG retrieval fallback warning: {e2}")
                retrieved_laws = []
        
        # Step 4: Detect query category
        query_type = detect_query_category(user_query)
        
        # Step 5: Generate response using Gemini
        response = generate_legal_response(user_query, retrieved_laws, language, query_type)
        
        # Step 6: Extract mapping information
        mapping_data = None
        try:
            from rag import get_bns_constitution_mapping
            mapping_data = get_bns_constitution_mapping(user_query)
        except Exception as e:
            print(f"[APP] Mapping retrieval warning: {e}")
        
        # Step 7: Prepare similarity scores (R values)
        similarity_scores = []
        for law_text, score, idx in retrieved_with_scores:
            similarity_scores.append({
                'index': idx,
                'r_value': round(score, 4),  # Cosine similarity score
                'text_preview': law_text[:100] + '...' if len(law_text) > 100 else law_text
            })
        
        # Step 8: Add additional metadata
        result = {
            'response': response,
            'language': language,
            'mode': 'online',
            'is_crime': is_crime_query(user_query),
            'query_category': query_type,
            'retrieved_sections': len(retrieved_laws),
            'similarity_scores': similarity_scores,  # R values
            'mapping': mapping_data  # BNS-Constitution mapping
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"[APP] Chat error: {e}")
        # Try to detect language from query, default to hindi
        try:
            error_query = data.get('query', '') if 'data' in locals() else ''
            error_language = detect_language(error_query) if error_query else 'hindi'
        except:
            error_language = 'hindi'
        
        # Return offline fallback as a proper 200 so the frontend renders it correctly
        return jsonify({
            'response': get_error_fallback(error_language),
            'language': error_language,
            'mode': 'offline',
            'is_crime': False,
            'query_category': 'general',
            'retrieved_sections': 0,
            'similarity_scores': [],
            'mapping': None
        })


@app.route('/api/emergency', methods=['GET'])
def emergency():
    """Return emergency legal rights"""
    return jsonify({
        'rights': EMERGENCY_RIGHTS,
        'contacts': [
            {'name': 'Police', 'number': '100', 'icon': '🚔'},
            {'name': 'Women Helpline', 'number': '181', 'icon': '👩'},
            {'name': 'NALSA Legal Aid', 'number': '15100', 'icon': '⚖️'},
            {'name': 'Ambulance', 'number': '108', 'icon': '🚑'},
            {'name': 'Child Helpline', 'number': '1098', 'icon': '👶'},
            {'name': 'NCW Helpline', 'number': '7827-170-170', 'icon': '🏛️'}
        ]
    })


@app.route('/api/generate-fir', methods=['POST'])
def generate_fir():
    """Generate FIR format based on user's incident description"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        language = data.get('language', 'hindi')
        
        if not query:
            return jsonify({'error': 'घटनेचे वर्णन द्या / Please describe the incident'}), 400
        
        fir_text = generate_fir_format(query, language)
        
        return jsonify({
            'fir': fir_text,
            'note': 'हे FIR टेम्पलेट आहे. पोलीस ठाण्यात जाऊन अधिकृत FIR दाखल करा. / This is an FIR template. Please file official FIR at police station.',
            'steps': [
                '1. FIR प्रिंट करा किंवा लिहा / Print or write the FIR',
                '2. जवळच्या पोलीस ठाण्यात जा / Go to nearest police station',
                '3. FIR मोफत आहे / FIR is free of cost',
                '4. FIR ची प्रत मागा / Ask for FIR copy',
                '5. FIR नाकारली तर SP ला संपर्क करा / If refused, contact SP'
            ]
        })
        
    except Exception as e:
        print(f"[APP] FIR generation error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/legal-help', methods=['GET'])
def legal_help():
    """Return nearest legal aid centers (mock data)"""
    lat = request.args.get('lat', '19.9975')  # Default: Nashik
    lng = request.args.get('lng', '73.7898')
    
    return jsonify({
        'centers': LEGAL_AID_CENTERS,
        'note': 'जवळचे कायदेशीर सहाय्य केंद्र / Nearest Legal Aid Centers',
        'helplines': {
            'NALSA': '15100',
            'Police': '100',
            'Women': '181',
            'Child': '1098'
        }
    })


@app.route('/api/document-checklist', methods=['POST'])
def document_checklist():
    """Generate document checklist based on query"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        category = detect_query_category(query)
        checklist = get_document_checklist(category)
        
        return jsonify({
            'checklist': checklist,
            'category': category
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo-queries', methods=['GET'])
def demo_queries():
    """Return demo queries for testing"""
    return jsonify({
        'queries': [
            {
                'text': 'मला चोरी झाली आहे, FIR कशी दाखल करावी?',
                'language': 'marathi',
                'category': 'theft'
            },
            {
                'text': 'पती मला मारहाण करतो, मला काय करता येईल?',
                'language': 'marathi',
                'category': 'domestic_violence'
            },
            {
                'text': 'कामाच्या ठिकाणी माझ्यावर लैंगिक छळ होत आहे',
                'language': 'marathi',
                'category': 'posh'
            },
            {
                'text': 'मुझे पुलिस ने गिरफ्तार किया, मेरे क्या अधिकार हैं?',
                'language': 'hindi',
                'category': 'arrest'
            },
            {
                'text': 'My employer is not paying minimum wages, what should I do?',
                'language': 'english',
                'category': 'labour'
            }
        ]
    })


def check_offline_mode(query, language):
    """Check if query can be answered from offline database"""
    from llm import offline_answer
    offline_resp = offline_answer(query, language)
    return offline_resp


def get_error_fallback(language='hindi'):
    """Return a helpful fallback response"""
    if language == 'marathi':
        answer_text = 'सध्या सेवा उपलब्ध नाही. कृपया खाली दिलेल्या हेल्पलाइन क्रमांकावर संपर्क करा.'
    elif language == 'hindi':
        answer_text = 'सध्या सेवा उपलब्ध नाही. कृपया खाली दिलेल्या हेल्पलाइन क्रमांकावर संपर्क करा.'
    else:
        answer_text = 'Service temporarily unavailable. Please contact the helplines below.'
    
    return {
        'answer': answer_text,
        'relevant_sections': [],
        'next_steps': ['पोलीस: 100' if language != 'english' else 'Police: 100', 
                       'महिला हेल्पलाइन: 181' if language != 'english' else 'Women Helpline: 181', 
                       'NALSA: 15100'],
        'important_rights': [],
        'emergency_contacts': ['Police: 100', 'NALSA: 15100', 'Women Helpline: 181']
    }


# ========== BNS TO CONSTITUTION MAPPING ENDPOINTS ==========

def load_bns_constitution_data():
    """Load BNS-Constitution mapping from JSON file"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'data', 'bns_constitution_mapping.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load BNS-Constitution mapping: {e}")
        return None


@app.route('/api/bns-constitution/all', methods=['GET'])
def get_all_bns_sections():
    """Get all BNS sections with Constitution mapping"""
    try:
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        return jsonify({
            'total_sections': data['metadata']['total_sections'],
            'sections': data['bns_sections'],
            'timestamp': data['metadata']['last_updated']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bns-constitution/section/<int:bns_id>', methods=['GET'])
def get_bns_section(bns_id):
    """Get specific BNS section with Constitution mapping"""
    try:
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        for section in data['bns_sections']:
            if section['bns_id'] == bns_id:
                return jsonify(section)
        
        return jsonify({'error': f'BNS Section {bns_id} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bns-constitution/victim-rights/<int:bns_id>', methods=['GET'])
def get_victim_rights(bns_id):
    """Get victim rights for a specific BNS section"""
    try:
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        for section in data['bns_sections']:
            if section['bns_id'] == bns_id:
                return jsonify({
                    'bns_id': bns_id,
                    'title': section['title'],
                    'victim_rights': section.get('victim_rights', []),
                    'support_organizations': section.get('support_organizations', [])
                })
        
        return jsonify({'error': f'BNS Section {bns_id} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bns-constitution/remedies/<int:bns_id>', methods=['GET'])
def get_remedies(bns_id):
    """Get remedies and paths for victim in a specific BNS section"""
    try:
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        for section in data['bns_sections']:
            if section['bns_id'] == bns_id:
                return jsonify({
                    'bns_id': bns_id,
                    'title': section['title'],
                    'remedies_and_paths': section.get('remedies_and_paths', []),
                    'support_organizations': section.get('support_organizations', [])
                })
        
        return jsonify({'error': f'BNS Section {bns_id} not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bns-constitution/search', methods=['GET'])
def search_bns_constitution():
    """Search BNS sections by keyword"""
    try:
        keyword = request.args.get('keyword', '').lower()
        if not keyword:
            return jsonify({'error': 'Keyword required'}), 400
        
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        results = []
        for section in data['bns_sections']:
            # Search in title, description, victim_rights, and remedies
            search_text = f"{section['title']} {section.get('description', '')}".lower()
            
            if any(right.lower().count(keyword) for right in section.get('victim_rights', [])):
                search_text += ' MATCH'
            
            if keyword in search_text:
                results.append({
                    'bns_id': section['bns_id'],
                    'title': section['title'],
                    'description': section.get('description', ''),
                    'punishment': section.get('punishment', {})
                })
        
        return jsonify({
            'keyword': keyword,
            'results_count': len(results),
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bns-constitution/support-services', methods=['GET'])
def get_support_services():
    """Get all support services and helplines"""
    try:
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        return jsonify({
            'services': data['victim_support_services'],
            'compensation_info': data['victim_compensation_framework'],
            'procedural_safeguards': data['procedural_safeguards_for_victims']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bns-constitution/constitution-articles', methods=['GET'])
def get_constitution_articles():
    """Get relevant Constitution articles for victim protection"""
    try:
        data = load_bns_constitution_data()
        if not data:
            return jsonify({'error': 'Data not available'}), 500
        
        return jsonify(data['constitutional_framework'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize RAG system on startup
    print("=" * 60)
    print("🏛️  NYAY AI - Multilingual Legal Rights Assistant")
    print("   न्याय AI - बहुभाषीय कायदेशीर हक्क सहाय्यक")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get('GEMINI_API_KEY'):
        print("⚠️  WARNING: GEMINI_API_KEY not set!")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        print("   App will run in offline mode without AI responses.")
    else:
        print("✅ Gemini API key found")
    
    # Show LLM mode
    from llm import _GENAI_OK
    if _GENAI_OK and os.environ.get('GEMINI_API_KEY'):
        print("🤖 LLM Mode: Gemini AI (offline fallback ready)")
    else:
        print("📴 LLM Mode: Offline only (Gemini SDK not available or no API key)")

    # Initialize RAG
    try:
        init_rag()
    except Exception as e:
        print(f"⚠️  RAG initialization: {e}")

    # Try to start on port 5000; if busy, find a free port
    import socket
    port = 5000
    for candidate in range(5000, 5005):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', candidate))
        sock.close()
        if result != 0:   # port is free
            port = candidate
            break
    else:
        print("⚠️  Ports 5000-5004 all in use. Trying OS-assigned port.")
        port = 0

    print(f"\n🌐 Starting server at http://127.0.0.1:{port}")
    print("   Press Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=port)
