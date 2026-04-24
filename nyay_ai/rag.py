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
        return _retrieve_with_faiss(query, top_k)
    elif use_tfidf and tfidf_vectorizer is not None:
        return _retrieve_with_tfidf(query, top_k)
    else:
        # Pure-Python keyword fallback (no external dependencies)
        return _retrieve_with_keywords(query, top_k)


def _retrieve_with_faiss(query, top_k):
    """Retrieve using FAISS similarity search"""
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
        
        return [chunks[i] for i in top_indices if scores[i] > 0.1]
        
    except Exception as e:
        print(f"[RAG] FAISS retrieval failed: {e}")
        return _retrieve_with_keywords(query, top_k)


def _retrieve_with_tfidf(query, top_k):
    """Retrieve using TF-IDF similarity"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_vec = tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [chunks[i] for i in top_indices if scores[i] > 0.01]
        return results if results else _retrieve_with_keywords(query, top_k)
        
    except Exception as e:
        print(f"[RAG] TF-IDF retrieval failed: {e}")
        return _retrieve_with_keywords(query, top_k)


def _retrieve_with_keywords(query, top_k):
    """Simple keyword-based retrieval as last resort"""
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
    top_indices = [i for score, i in scored_chunks[:top_k] if score > 0]
    
    if not top_indices:
        # Return general rights information
        return [chunks[0], chunks[1]] if len(chunks) >= 2 else chunks
    
    return [chunks[i] for i in top_indices]


def initialize():
    """Initialize the RAG system"""
    print("[RAG] Initializing RAG system...")
    load_and_chunk_laws()
    build_index()
    print("[RAG] RAG system ready!")


# Initialize when module is loaded
if __name__ != '__main__':
    try:
        initialize()
    except Exception as e:
        print(f"[RAG] Initialization warning: {e}")
