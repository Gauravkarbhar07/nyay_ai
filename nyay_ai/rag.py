# rag.py - Retrieval-Augmented Generation Pipeline
# Uses sentence-transformers + FAISS when available.
# Falls back to TF-IDF (sklearn) when available.
# Final fallback: pure-Python keyword search (no dependencies).

import os

LAWS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'laws.txt')
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'embeddings_cache.pkl')

# Global variables for the retrieval system
chunks = []
embeddings = None
use_transformer = False
use_tfidf = False
tfidf_vectorizer = None
tfidf_matrix = None

# Try loading heavy dependencies – silently degrade if unavailable
try:
    import numpy as np
    import pickle
    _numpy_ok = True
except Exception as _e:
    print(f"[RAG] numpy unavailable ({_e}), using pure-Python keyword fallback")
    _numpy_ok = False

if _numpy_ok:
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        use_transformer = True
        print("[RAG] Using SentenceTransformer + FAISS for retrieval")
    except Exception:
        print("[RAG] SentenceTransformer/FAISS not available")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            use_tfidf = True
            print("[RAG] Using TF-IDF fallback")
        except Exception:
            print("[RAG] sklearn not available, using keyword fallback")
else:
    print("[RAG] Running in keyword-only mode (no ML dependencies)")


def load_and_chunk_laws(file_path=LAWS_FILE, chunk_size=300):
    """
    Load the laws text file and split it into chunks for retrieval.
    Each chunk is a meaningful section of the law.
    """
    global chunks
    
    if not os.path.exists(file_path):
        print(f"[RAG] Warning: Laws file not found at {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by sections (paragraphs separated by blank lines)
    raw_chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
    
    # Further split large chunks
    final_chunks = []
    for chunk in raw_chunks:
        if len(chunk) > chunk_size * 2:
            # Split by sentences
            sentences = chunk.split('. ')
            current = ''
            for sent in sentences:
                if len(current) + len(sent) < chunk_size * 2:
                    current += sent + '. '
                else:
                    if current:
                        final_chunks.append(current.strip())
                    current = sent + '. '
            if current:
                final_chunks.append(current.strip())
        else:
            final_chunks.append(chunk)
    
    chunks = [c for c in final_chunks if len(c) > 50]
    print(f"[RAG] Loaded {len(chunks)} law chunks")
    return chunks


def build_index():
    """
    Build the vector index from law chunks.
    Uses SentenceTransformer + FAISS if available, TF-IDF if available, else keyword only.
    """
    global use_transformer, use_tfidf

    if not chunks:
        load_and_chunk_laws()

    if use_transformer:
        _build_faiss_index()
    elif use_tfidf:
        _build_tfidf_index()
    else:
        print("[RAG] Index: keyword-only mode")


def _build_faiss_index():
    """Build FAISS index using multilingual sentence transformers"""
    global embeddings

    if not _numpy_ok:
        return

    try:
        import faiss
        import pickle
        from sentence_transformers import SentenceTransformer

        # Check cache
        if os.path.exists(CACHE_FILE):
            print("[RAG] Loading embeddings from cache...")
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                if cache.get('chunks_count') == len(chunks):
                    embeddings = cache['embeddings']
                    print("[RAG] Cache loaded successfully")
                    return

        print("[RAG] Building FAISS index with multilingual embeddings...")
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embeddings = model.encode(chunks, show_progress_bar=True)

        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({'embeddings': embeddings, 'chunks_count': len(chunks)}, f)

        print(f"[RAG] FAISS index built with {len(chunks)} vectors")

    except Exception as e:
        print(f"[RAG] FAISS index build failed: {e}. Falling back to TF-IDF")
        global use_transformer, use_tfidf
        use_transformer = False
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            use_tfidf = True
            _build_tfidf_index()
        except Exception:
            use_tfidf = False


def _build_tfidf_index():
    """Build TF-IDF index as fallback (requires sklearn + numpy)"""
    global tfidf_vectorizer, tfidf_matrix, use_tfidf

    if not _numpy_ok:
        use_tfidf = False
        return

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("[RAG] Building TF-IDF index...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            analyzer='char_wb',
            min_df=1,
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(chunks)
        print(f"[RAG] TF-IDF index built with {len(chunks)} documents")
    except Exception as e:
        print(f"[RAG] TF-IDF build failed: {e}")
        use_tfidf = False


def retrieve_relevant_laws(query, top_k=5):
    """
    Retrieve the most relevant law sections for the given query.
    Returns a list of relevant text chunks.
    """
    global chunks, embeddings, use_transformer, use_tfidf, tfidf_vectorizer, tfidf_matrix

    if not chunks:
        load_and_chunk_laws()
        build_index()

    if not chunks:
        return ['कायदेशीर माहिती उपलब्ध नाही / Legal information not available']

    if use_transformer and embeddings is not None:
        results = _retrieve_with_faiss(query, top_k)
    elif use_tfidf and tfidf_vectorizer is not None:
        results = _retrieve_with_tfidf(query, top_k)
    else:
        # Pure-Python keyword fallback (no external dependencies)
        results = _retrieve_with_keywords(query, top_k)
    
    # Return only chunks for backward compatibility
    return [item[0] for item in results]


def retrieve_relevant_laws_with_scores(query, top_k=5):
    """
    Retrieve the most relevant law sections with cosine similarity scores.
    Returns a list of tuples: (chunk_text, cosine_similarity_score, chunk_index)
    """
    global chunks, embeddings, use_transformer, use_tfidf, tfidf_vectorizer, tfidf_matrix

    if not chunks:
        load_and_chunk_laws()
        build_index()

    if not chunks:
        return [('कायदेशीर माहिती उपलब्ध नाही / Legal information not available', 0.0, 0)]

    if use_transformer and embeddings is not None:
        return _retrieve_with_faiss(query, top_k)
    elif use_tfidf and tfidf_vectorizer is not None:
        return _retrieve_with_tfidf(query, top_k)
    else:
        # Pure-Python keyword fallback (no external dependencies)
        return _retrieve_with_keywords(query, top_k)


def _retrieve_with_faiss(query, top_k):
    """Retrieve using FAISS similarity search. Returns (chunk, score, index) tuples"""
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        query_embedding = model.encode([query])
        
        # Normalize embeddings for cosine similarity
        emb_matrix = np.array(embeddings).astype('float32')
        query_vec = np.array(query_embedding).astype('float32')
        
        # Compute cosine similarity
        emb_norm = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        
        scores = np.dot(emb_norm, query_norm.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.1:
                results.append((chunks[idx], float(scores[idx]), int(idx)))
        
        return results if results else [(chunks[0], 0.0, 0)]
        
    except Exception as e:
        print(f"[RAG] FAISS retrieval failed: {e}")
        return _retrieve_with_keywords(query, top_k)


def _retrieve_with_tfidf(query, top_k):
    """Retrieve using TF-IDF similarity. Returns (chunk, score, index) tuples"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vec = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:
                results.append((chunks[idx], float(scores[idx]), int(idx)))
        
        return results if results else _retrieve_with_keywords(query, top_k)
        
    except Exception as e:
        print(f"[RAG] TF-IDF retrieval failed: {e}")
        return _retrieve_with_keywords(query, top_k)


def _retrieve_with_keywords(query, top_k):
    """Simple keyword-based retrieval as last resort. Returns (chunk, score, index) tuples"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        # Count matching words
        score = sum(1 for word in query_words if word in chunk_lower)
        # Bonus for section number matches
        if any(char.isdigit() for char in query):
            import re
            nums = re.findall(r'\d+', query)
            for num in nums:
                if num in chunk:
                    score += 5
        scored_chunks.append((score, i))
    
    scored_chunks.sort(reverse=True)
    
    results = []
    for score, i in scored_chunks[:top_k]:
        if score > 0:
            # Normalize score to 0-1 range for consistency
            normalized_score = min(score / 10.0, 1.0)
            results.append((chunks[i], normalized_score, i))
    
    if not results:
        # Return general rights information
        if len(chunks) >= 2:
            results = [(chunks[0], 0.0, 0), (chunks[1], 0.0, 1)]
        elif len(chunks) >= 1:
            results = [(chunks[0], 0.0, 0)]
    
    return results


def initialize():
    """Initialize the RAG system"""
    print("[RAG] Initializing RAG system...")
    load_and_chunk_laws()
    build_index()
    print("[RAG] RAG system ready!")


def get_bns_constitution_mapping(search_term=None):
    """
    Load BNS-Constitution mapping from JSON file.
    If search_term provided, filters by BNS section or topic.
    Returns the full mapping data or filtered sections.
    """
    import json
    
    mapping_file = os.path.join(os.path.dirname(__file__), 'data', 'bns_constitution_mapping.json')
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        if not search_term:
            return mapping_data
        
        # Filter sections by search term
        search_lower = search_term.lower()
        filtered_sections = []
        
        for section in mapping_data.get('bns_sections', []):
            section_id = str(section.get('bns_id', '')).lower()
            title = section.get('title', '').lower()
            description = section.get('description', '').lower()
            
            if (search_lower in section_id or 
                search_lower in title or 
                search_lower in description):
                filtered_sections.append(section)
        
        # Return filtered or full mapping
        if filtered_sections:
            result = mapping_data.copy()
            result['bns_sections'] = filtered_sections
            return result
        
        return mapping_data
    
    except Exception as e:
        print(f"[RAG] Error loading mapping: {e}")
        return {'metadata': {}, 'bns_sections': []}


# Initialize when module is loaded
if __name__ != '__main__':
    try:
        initialize()
    except Exception as e:
        print(f"[RAG] Initialization warning: {e}")
