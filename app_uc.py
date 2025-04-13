import os
import json
import string
import numpy as np
from collections import Counter, defaultdict
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer  # For sparse keyword search
import scipy.sparse

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - update these paths to match your file locations
JSON_FILE_PATH = 'merged_data.json'  # Path to your JSON file
CLIPS_DIRECTORY = 'clips/chunks'  # Directory containing all video clips
FULL_VIDEO_PATH = 'hkairport.mp4'  # Path to the full original video
FULL_VIDEO_FILENAME = os.path.basename(FULL_VIDEO_PATH)  # Extract just the filename
TFIDF_MODEL_PATH = 'tfidf_model.pkl'  # Path to store TF-IDF model
EMBEDDINGS_FILE_PATH = 'embeddings.npz'  # Path to embeddings

# Search configuration
RRF_CONSTANT = 1   # Constant for Reciprocal Rank Fusion
SEARCH_RESULTS_COUNT = 20  # Number of results to retrieve
FINAL_RESULTS_COUNT = 5   # Final number of results to return

# Global variables for data storage
CLIP_DESCRIPTIONS = []
CLIP_TOKEN_COUNTS = []
CLIP_IDS = []
CLIP_TEXTS = []
CLIP_METADATA = []
CLIPS_DATA = None  # This will store the full JSON data

# TF-IDF related variables
TFIDF_VECTORIZER = None
TFIDF_MATRIX = None

# BERT embeddings (only loaded, not generated)
BERT_AVAILABLE = False
CLIP_EMBEDDINGS = None

# FAISS index related variables
FAISS_INDEX = None
FAISS_DIM = None  # Embedding dimension

# Try to import FAISS with careful error handling
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS library loaded successfully")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS library not available - vector search will be disabled")

def preprocess_text(text):
    """Basic text preprocessing: lowercase, remove punctuation, and tokenize"""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into tokens (words)
    tokens = text.split()
    return tokens

def calculate_jaccard_similarity(query_tokens, clip_tokens):
    """Calculate Jaccard similarity between query and clip tokens"""
    # Create Counter for query tokens
    query_count = Counter(query_tokens)
    
    # Calculate intersection (common tokens)
    intersection = sum((query_count & clip_tokens).values())
    
    # Calculate union (all tokens)
    union = sum((query_count | clip_tokens).values())
    
    # Calculate Jaccard similarity (intersection over union)
    if union == 0:
        return 0
    return intersection / union

def prepare_tfidf_index():
    """Prepare TF-IDF sparse index for keyword search"""
    global TFIDF_VECTORIZER, TFIDF_MATRIX, CLIP_TEXTS
    
    try:
        # Check if we already have a saved model
        if os.path.exists(TFIDF_MODEL_PATH):
            logger.info(f"Loading TF-IDF model from {TFIDF_MODEL_PATH}")
            with open(TFIDF_MODEL_PATH, 'rb') as f:
                saved_data = pickle.load(f)
                TFIDF_VECTORIZER = saved_data['vectorizer']
                TFIDF_MATRIX = saved_data['matrix']
            logger.info(f"Loaded TF-IDF matrix with shape {TFIDF_MATRIX.shape}")
            return True
        
        # Create and fit TF-IDF vectorizer - with explicit single process to avoid semaphore leaks
        logger.info("Creating TF-IDF sparse index")
        TFIDF_VECTORIZER = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,            # Minimum document frequency
            max_df=0.9,          # Maximum document frequency
            n_jobs=1             # Force single process to avoid semaphore leaks
        )
        
        # Transform texts to TF-IDF matrix
        TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(CLIP_TEXTS)
        
        # Save for future use
        os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
        with open(TFIDF_MODEL_PATH, 'wb') as f:
            pickle.dump({
                'vectorizer': TFIDF_VECTORIZER,
                'matrix': TFIDF_MATRIX
            }, f)
        
        logger.info(f"Created TF-IDF matrix with shape {TFIDF_MATRIX.shape}")
        return True
    except Exception as e:
        logger.error(f"Error preparing TF-IDF index: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def prepare_faiss_index():
    """Prepare FAISS index for fast similarity search"""
    global FAISS_INDEX, CLIP_EMBEDDINGS, FAISS_DIM
    
    if not FAISS_AVAILABLE:
        logger.warning("FAISS not available - skipping index preparation")
        return False
    
    try:
        # Check if we have embeddings
        if CLIP_EMBEDDINGS is None:
            logger.warning("No embeddings available for FAISS indexing")
            return False
        
        logger.info("Building FAISS index for vector search")
        
        # Set embedding dimension
        FAISS_DIM = CLIP_EMBEDDINGS.shape[1]
        
        # Convert embeddings to correct format (float32)
        embeddings_float32 = CLIP_EMBEDDINGS.astype(np.float32)
        
        # Choose index type based on number of vectors
        if len(CLIP_EMBEDDINGS) < 10000:
            # For smaller datasets, use exact inner product search
            FAISS_INDEX = faiss.IndexFlatIP(FAISS_DIM)
        else:
            # For larger datasets, use IVF index for faster approximate search
            nlist = int(4 * np.sqrt(len(CLIP_EMBEDDINGS)))  # Number of centroids
            quantizer = faiss.IndexFlatIP(FAISS_DIM)
            FAISS_INDEX = faiss.IndexIVFFlat(quantizer, FAISS_DIM, nlist, faiss.METRIC_INNER_PRODUCT)
            # Need to train this index
            FAISS_INDEX.train(embeddings_float32)
            # Set search parameters
            FAISS_INDEX.nprobe = min(nlist, 16)  # Number of centroids to visit during search
        
        # Add vectors to index
        FAISS_INDEX.add(embeddings_float32)
        logger.info(f"Built FAISS index for {len(CLIP_EMBEDDINGS)} clip embeddings")
        return True
    except Exception as e:
        logger.error(f"Error preparing FAISS index: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_embeddings():
    """Load pre-computed embeddings from file"""
    global CLIP_EMBEDDINGS, BERT_AVAILABLE
    
    if not os.path.exists(EMBEDDINGS_FILE_PATH):
        logger.warning(f"Embeddings file not found at {EMBEDDINGS_FILE_PATH}")
        return False
    
    try:
        logger.info(f"Loading embeddings from {EMBEDDINGS_FILE_PATH}")
        data = np.load(EMBEDDINGS_FILE_PATH, allow_pickle=True)
        
        # Handle different possible formats
        if isinstance(data, np.ndarray):
            # If it's just a numpy array
            CLIP_EMBEDDINGS = data
            logger.info(f"Loaded embeddings array with shape {CLIP_EMBEDDINGS.shape}")
        elif isinstance(data, np.lib.npyio.NpzFile):
            # If it's an npz file with multiple arrays
            if 'embeddings' in data:
                CLIP_EMBEDDINGS = data['embeddings']
                logger.info(f"Loaded 'embeddings' array with shape {CLIP_EMBEDDINGS.shape}")
            else:
                # Try to find the largest array that looks like embeddings
                largest_key = None
                largest_size = 0
                for key in data.keys():
                    if hasattr(data[key], 'ndim') and data[key].ndim == 2 and data[key].shape[1] > 50:  # Typical embedding dimension check
                        if largest_key is None or data[key].shape[0] > largest_size:
                            largest_key = key
                            largest_size = data[key].shape[0]
                
                if largest_key:
                    CLIP_EMBEDDINGS = data[largest_key]
                    logger.info(f"Loaded '{largest_key}' array with shape {CLIP_EMBEDDINGS.shape}")
                else:
                    logger.error("Could not find embeddings in the NPZ file")
                    return False
        
        # Verify embeddings shape matches number of clips
        if len(CLIP_IDS) > 0 and CLIP_EMBEDDINGS.shape[0] != len(CLIP_IDS):
            logger.warning(f"Number of embeddings ({CLIP_EMBEDDINGS.shape[0]}) doesn't match number of clips ({len(CLIP_IDS)})")
            # If fewer embeddings than clips, we'll only use embeddings for the first n clips
            if CLIP_EMBEDDINGS.shape[0] < len(CLIP_IDS):
                logger.warning(f"Using embeddings only for the first {CLIP_EMBEDDINGS.shape[0]} clips")
        
        logger.info(f"Successfully loaded embeddings, shape: {CLIP_EMBEDDINGS.shape}")
        BERT_AVAILABLE = True
        return True
    except Exception as e:
        logger.error(f"Error loading embeddings: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        CLIP_EMBEDDINGS = None
        BERT_AVAILABLE = False
        return False

def faiss_search(query_embedding, top_k=SEARCH_RESULTS_COUNT):
    """Search for similar clips using FAISS"""
    global FAISS_INDEX
    
    if FAISS_INDEX is None or query_embedding is None:
        return []
    
    try:
        # Convert query to float32 and reshape for FAISS
        query_float32 = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = FAISS_INDEX.search(query_float32, top_k)
        
        # Convert to list of (clip_idx, score) pairs
        results = [(int(idx), float(distances[0][i])) for i, idx in enumerate(indices[0]) if idx >= 0]
        
        return results
    except Exception as e:
        logger.error(f"Error in FAISS search: {e}")
        return []

def keyword_search(query, top_k=SEARCH_RESULTS_COUNT):
    """Perform keyword search using TF-IDF similarity"""
    global TFIDF_VECTORIZER, TFIDF_MATRIX
    
    if TFIDF_VECTORIZER is None or TFIDF_MATRIX is None:
        logger.warning("TF-IDF index not available for keyword search")
        return []
    
    try:
        # Transform query to TF-IDF space
        query_vector = TFIDF_VECTORIZER.transform([query])
        
        # Calculate cosine similarity between query and all documents
        # This uses the fact that for normalized vectors, dot product = cosine similarity
        similarities = (TFIDF_MATRIX * query_vector.T).toarray().flatten()
        
        # Get top k indices
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Convert to list of (clip_idx, score) pairs
        results = [(idx, float(similarities[idx])) for idx in top_indices if similarities[idx] > 0]
        
        return results
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []

def reciprocal_rank_fusion(ranked_lists, k=RRF_CONSTANT):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    Args:
        ranked_lists: List of lists, each containing (item_id, score) tuples
        k: Constant in RRF formula (default: 60)
        
    Returns:
        List of (item_id, combined_score) tuples sorted by combined_score
    """
    # Dictionary to store final scores
    fusion_scores = defaultdict(float)
    
    # Process each ranked list
    for ranked_list in ranked_lists:
        # Calculate score for each item based on its rank
        for rank, (item_id, _) in enumerate(ranked_list):
            fusion_scores[item_id] += 1.0 / (k + rank + 1)  # Adding 1 to rank because ranks start at 0
    
    # Sort items by fusion score (descending)
    result = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    return result

def load_json_data():
    """Load clip data from JSON file and prepare for search"""
    global CLIPS_DATA, CLIP_DESCRIPTIONS, CLIP_TOKEN_COUNTS, CLIP_IDS, CLIP_TEXTS, CLIP_METADATA
    
    try:
        logger.info(f"Loading JSON data from {JSON_FILE_PATH}")
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            CLIPS_DATA = json.load(f)
        
        # Clear existing data
        CLIP_DESCRIPTIONS = []
        CLIP_TOKEN_COUNTS = []
        CLIP_IDS = []
        CLIP_TEXTS = []
        CLIP_METADATA = []
        
        # Determine the structure of the JSON data
        if isinstance(CLIPS_DATA, dict) and "clips" in CLIPS_DATA:
            clips = CLIPS_DATA["clips"]
            logger.info(f"Loaded {len(clips)} clips from JSON data (dictionary format)")
        elif isinstance(CLIPS_DATA, list):
            clips = CLIPS_DATA
            logger.info(f"Loaded {len(clips)} clips from JSON data (list format)")
        else:
            logger.warning("JSON data has an unexpected format")
            return False
        
        # Process each clip
        for clip in clips:
            # Extract clip ID
            clip_id = clip.get("filename", "")
            if not clip_id:
                clip_id = clip.get("id", "")
            
            # Extract metadata
            metadata = {
                "id": clip.get("id", ""),
                "duration": clip.get("duration", 0)
            }
            
            # Extract description
            description = clip.get("description", "")
            
            # Extract objects
            objects_text = ""
            if "objects" in clip:
                objects = clip["objects"]
                if isinstance(objects, list):
                    objects_text = " ".join(objects)
                elif isinstance(objects, str):
                    objects_text = objects
            
            # Combine description and objects for rich text representation
            combined_text = f"{description} {objects_text}".strip()
            
            # Preprocess and tokenize text
            tokens = preprocess_text(combined_text)
            
            # Add to our data structures
            CLIP_IDS.append(clip_id)
            CLIP_TEXTS.append(combined_text)
            CLIP_DESCRIPTIONS.append(description)
            CLIP_TOKEN_COUNTS.append(Counter(tokens))
            CLIP_METADATA.append(metadata)
        
        logger.info(f"Processed {len(CLIP_IDS)} clips for search")
        return True
    except Exception as e:
        logger.error(f"Error loading JSON data: {str(e)}")
        return False

def get_clip_details(clip_id):
    """Get detailed information about a clip from the JSON data"""
    if not CLIPS_DATA:
        return None
    
    # Handle dict with "clips" key
    if isinstance(CLIPS_DATA, dict) and "clips" in CLIPS_DATA:
        for clip in CLIPS_DATA["clips"]:
            if clip.get("filename") == clip_id or clip.get("id") == clip_id:
                return clip
    
    # Handle list of clips
    elif isinstance(CLIPS_DATA, list):
        for clip in CLIPS_DATA:
            if clip.get("filename") == clip_id or clip.get("id") == clip_id:
                return clip
    
    return None

def get_query_embedding(query):
    """Get a pre-computed embedding for a query using matching words"""
    global CLIP_TEXTS, CLIP_EMBEDDINGS
    
    # If no embeddings available, return None
    if CLIP_EMBEDDINGS is None:
        return None
    
    # Simple method: find the clip text that has the most words in common with the query
    query_tokens = set(preprocess_text(query))
    best_match_idx = None
    max_common_words = -1
    
    for i, text in enumerate(CLIP_TEXTS):
        # Skip if we don't have embeddings for this clip
        if i >= len(CLIP_EMBEDDINGS):
            continue
            
        text_tokens = set(preprocess_text(text))
        common_words = len(query_tokens.intersection(text_tokens))
        
        if common_words > max_common_words:
            max_common_words = common_words
            best_match_idx = i
    
    # If we found a good match, use its embedding
    if best_match_idx is not None and max_common_words > 0:
        logger.info(f"Using embedding from clip {best_match_idx} for query (matched {max_common_words} words)")
        return CLIP_EMBEDDINGS[best_match_idx]
    
    # If no good match, use average of all embeddings
    logger.info("Using average of all embeddings for query")
    return np.mean(CLIP_EMBEDDINGS, axis=0)

def enhanced_search(query, top_k=FINAL_RESULTS_COUNT, methods=None):
    """Enhanced search using Jaccard, TF-IDF, FAISS and reciprocal rank fusion"""
    global CLIP_TOKEN_COUNTS, CLIP_IDS, CLIP_TEXTS, CLIP_METADATA
    global TFIDF_MATRIX, FAISS_INDEX, CLIP_EMBEDDINGS
    
    if not CLIP_TOKEN_COUNTS or len(CLIP_TOKEN_COUNTS) == 0:
        return {"error": "No clip data loaded."}
    
    # Default: use all available methods
    if methods is None:
        methods = {
            "jaccard": True, 
            "tfidf": TFIDF_VECTORIZER is not None and TFIDF_MATRIX is not None,
            "faiss": FAISS_INDEX is not None and CLIP_EMBEDDINGS is not None
        }
    
    try:
        # Preprocess query for different search methods
        query_tokens = preprocess_text(query)
        
        # Store results from different search methods
        search_results = {}
        methods_used = {k: False for k in ["jaccard", "tfidf", "faiss"]}
        
        # 1. Jaccard similarity search (lexical matching)
        if methods.get("jaccard", False):
            jaccard_results = []
            for i, clip_tokens in enumerate(CLIP_TOKEN_COUNTS):
                similarity = calculate_jaccard_similarity(query_tokens, clip_tokens)
                jaccard_results.append((i, similarity))
            
            # Sort and keep top results
            jaccard_results.sort(key=lambda x: x[1], reverse=True)
            jaccard_results = jaccard_results[:SEARCH_RESULTS_COUNT]
            search_results["jaccard"] = jaccard_results
            methods_used["jaccard"] = True
        
        # 2. TF-IDF keyword search (if available and selected)
        if methods.get("tfidf", False) and TFIDF_VECTORIZER is not None and TFIDF_MATRIX is not None:
            tfidf_results = keyword_search(query, SEARCH_RESULTS_COUNT)
            search_results["tfidf"] = tfidf_results
            methods_used["tfidf"] = True
        
        # 3. FAISS vector search (if available and selected)
        if methods.get("faiss", False) and FAISS_INDEX is not None and CLIP_EMBEDDINGS is not None:
            # Get query embedding using lexical matching since we don't generate embeddings
            query_embedding = get_query_embedding(query)
            if query_embedding is not None:
                faiss_results = faiss_search(query_embedding, SEARCH_RESULTS_COUNT)
                search_results["faiss"] = faiss_results
                methods_used["faiss"] = True
        
        # If no methods were used or all failed, return error
        if not search_results:
            return {"error": "No search methods available or selected"}
            
        # Combine results using Reciprocal Rank Fusion
        ranked_lists = [results for method, results in search_results.items() if results]
        fused_results = reciprocal_rank_fusion(ranked_lists, k=RRF_CONSTANT)
        
        # Extract top-k results
        top_results = fused_results[:top_k]
        
        # Format final results
        final_results = []
        for clip_idx, rrf_score in top_results:
            # Skip invalid indices
            if clip_idx >= len(CLIP_IDS):
                continue
            
            clip_id = CLIP_IDS[clip_idx]
            
            # Get detailed clip info from JSON if available
            clip_details = get_clip_details(clip_id)
            
            # Get scores from individual methods for this clip
            method_scores = {
                method: next((score for idx, score in results if idx == clip_idx), 0.0)
                for method, results in search_results.items()
            }
            
            # Create result entry
            result = {
                "id": CLIP_METADATA[clip_idx].get("id", clip_idx),
                "filename": clip_id,
                "similarity": float(rrf_score),  # RRF score
                "jaccard_similarity": method_scores.get("jaccard", 0.0),
                "tfidf_similarity": method_scores.get("tfidf", 0.0),
                "faiss_similarity": method_scores.get("faiss", 0.0),
                "text": CLIP_TEXTS[clip_idx][:150] + "..." if len(CLIP_TEXTS[clip_idx]) > 150 else CLIP_TEXTS[clip_idx],
                "duration": CLIP_METADATA[clip_idx].get("duration", 0)
            }
            
            # Add additional details from JSON if available
            if clip_details:
                if "description" in clip_details:
                    result["description"] = clip_details["description"]
                if "objects" in clip_details:
                    result["objects"] = clip_details["objects"]
            
            final_results.append(result)
        
        return {
            "results": final_results,
            "methods_used": methods_used,
            "total_results": len(fused_results)
        }
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

@app.route('/')
def index():
    tfidf_status = "enabled" if TFIDF_VECTORIZER is not None and TFIDF_MATRIX is not None else "disabled"
    faiss_status = "enabled" if FAISS_INDEX is not None else "disabled"
    
    return render_template('index.html', 
                          full_video_path=FULL_VIDEO_PATH, 
                          full_video_filename=FULL_VIDEO_FILENAME,
                          tfidf_status=tfidf_status,
                          faiss_status=faiss_status)

@app.route('/search', methods=['POST'])
def handle_search():
    """Handle search request"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = int(data.get('top_k', FINAL_RESULTS_COUNT))
        
        # Get selected methods from request
        selected_methods = {
            "jaccard": data.get('use_text', True),
            "tfidf": data.get('use_keyword', True),
            "faiss": data.get('use_vector', True)
        }
        
        if not query:
            return jsonify({"success": False, "error": "Query is empty"})
        
        # Use the enhanced search with fused results and selected methods
        results = enhanced_search(query, top_k, selected_methods)
        
        if "error" in results:
            return jsonify({"success": False, "error": results["error"]})
        else:
            return jsonify({
                "success": True, 
                "results": results["results"],
                "methods_used": results.get("methods_used", {}),
                "total_results": results.get("total_results", 0)
            })
    except Exception as e:
        logger.error(f"Error handling search: {str(e)}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/clips/<path:filename>')
def serve_clip(filename):
    """Serve a video clip file"""
    return send_from_directory(CLIPS_DIRECTORY, filename)

@app.route('/full_video')
def serve_full_video():
    """Serve the full video file"""
    return send_from_directory(os.path.dirname(FULL_VIDEO_PATH), 
                              FULL_VIDEO_FILENAME)

def create_template():
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create an improved index.html with search methods and collapsible content
    index_path = os.path.join('templates', 'index.html')
    with open(index_path, 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Video Search</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }
        .search-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .video-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-video {
            width: 100%;
            max-height: 400px;
            border-radius: 5px;
        }
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .clip-card {
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.3s;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .clip-card:hover {
            transform: translateY(-5px);
        }
        .clip-video {
            width: 100%;
            height: 180px;
        }
        .loading-spinner {
            display: none;
            text-align: center;
            padding: 20px;
        }
        .card-simple-title {
            padding: 8px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
        }
        .add-more-card {
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #e9ecef;
            border: 2px dashed #adb5bd;
            border-radius: 10px;
            height: 224px; /* Same height as clip card (180px video + ~44px title bar) */
            cursor: pointer;
            transition: all 0.2s;
        }
        .add-more-card:hover {
            background-color: #dee2e6;
        }
        .add-more-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background-color: #007bff;
            color: white;
            border: none;
            font-size: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            transition: all 0.2s;
        }
        .add-more-card:hover .add-more-btn {
            background-color: #0069d9;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.3);
            transform: scale(1.1);
        }
        .method-toggle {
            display: none; /* Hidden as requested */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4"><i class="fas fa-video me-2"></i>Intelligent Video Search</h1>
        
        <!-- Search Section -->
        <div class="search-container">
            <h3 class="mb-3"><i class="fas fa-search me-2"></i>Search Videos</h3>
            <div class="input-group mb-3">
                <input type="text" id="search-input" class="form-control" 
                    placeholder="Describe what you're looking for..." aria-label="Search query">
                <button class="btn btn-primary" type="button" id="search-button">
                    <i class="fas fa-search me-2"></i>Search
                </button>
            </div>
            
            <!-- Method Toggle Options (Hidden) -->
            <div class="method-toggle">
                <!-- Hidden as requested -->
            </div>
        </div>
        
        <!-- Main Video Player -->
        <div class="video-container">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3><i class="fas fa-play-circle me-2"></i>Video Player</h3>
                <button id="reset-video-button" class="btn btn-secondary">
                    <i class="fas fa-undo me-2"></i>Return to Full Video
                </button>
            </div>
            <video id="main-video" class="main-video" controls>
                <source src="/full_video" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <!-- Loading Spinner -->
        <div id="loading-spinner" class="loading-spinner">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-2">Searching for relevant clips...</p>
        </div>
        
        <!-- Results Section -->
        <div id="results" style="display: none;">
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3><i class="fas fa-list me-2"></i>Search Results</h3>
                <span id="results-count" class="badge bg-primary">0 clips found</span>
            </div>
            <div class="results-grid" id="results-container">
                <!-- Results will be populated here -->
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Store all search results
            let allResults = [];
            let currentlyShownResults = 0;
            
            // Function to search for clips
            document.getElementById('search-button').addEventListener('click', function() {
                performSearch();
            });
            
            // Enable pressing Enter in the search box
            document.getElementById('search-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
            
            // Return to full video button
            document.getElementById('reset-video-button').addEventListener('click', function() {
                playInMainPlayer('/full_video');
            });
            
            function performSearch() {
                const query = document.getElementById('search-input').value.trim();
                if (!query) {
                    alert('Please enter a search query');
                    return;
                }
                
                // Always use all search methods (mixed search)
                const useText = true;
                const useKeyword = true; 
                const useVector = true;
                
                // Show loading spinner, hide results
                document.getElementById('loading-spinner').style.display = 'block';
                document.getElementById('results').style.display = 'none';
                
                fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        query: query,
                        use_text: useText,
                        use_keyword: useKeyword,
                        use_vector: useVector,
                        top_k: 20 // Request up to 20 results
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading spinner
                    document.getElementById('loading-spinner').style.display = 'none';
                    
                    if (data.success && data.results && data.results.length > 0) {
                        // Store all results
                        allResults = data.results;
                        currentlyShownResults = 0;
                        
                        // Clear previous results
                        document.getElementById('results-container').innerHTML = '';
                        
                        // Show just the first result
                        addNextResult();
                        
                        // Show results section
                        document.getElementById('results').style.display = 'block';
                        document.getElementById('results-count').textContent = 
                            `Showing 1 of ${data.results.length} clips found`;
                    } else {
                        alert(data.error || 'No results found or an error occurred');
                        if (data.error) {
                            console.error('Error:', data.error);
                        }
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('loading-spinner').style.display = 'none';
                    alert('An error occurred while searching');
                });
            }
            
            function addNextResult() {
                if (currentlyShownResults >= allResults.length) {
                    removeAddMoreButton();
                    return; // No more results to show
                }
                
                // Remove existing "add more" button if it exists
                removeAddMoreButton();
                
                // Get the next result
                const clip = allResults[currentlyShownResults];
                
                // Create result element
                const resultElement = document.createElement('div');
                resultElement.className = 'clip-card';
                
                // Create simple HTML with just video and clip number
                resultElement.innerHTML = `
                    <video class="clip-video" controls muted>
                        <source src="/clips/${clip.filename}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                    <div class="card-simple-title">
                        <strong>Clip ${clip.id}</strong>
                        <button class="btn btn-sm btn-outline-primary play-btn">
                            <i class="fas fa-play me-1"></i>Play
                        </button>
                    </div>
                `;
                
                // Add to container
                document.getElementById('results-container').appendChild(resultElement);
                
                // Add event listener for play button
                resultElement.querySelector('.play-btn').addEventListener('click', function(e) {
                    e.stopPropagation();
                    playInMainPlayer(`/clips/${clip.filename}`);
                });
                
                // Make card clickable to play video
                resultElement.addEventListener('click', function(e) {
                    // Don't trigger if clicking on buttons
                    if (e.target.tagName === 'BUTTON' || e.target.closest('button')) {
                        return;
                    }
                    playInMainPlayer(`/clips/${clip.filename}`);
                });
                
                // Increment counter
                currentlyShownResults++;
                
                // Update counter display
                document.getElementById('results-count').textContent = 
                    `Showing ${currentlyShownResults} of ${allResults.length} clips found`;
                
                // Add the "Show More" button if there are more results
                if (currentlyShownResults < allResults.length) {
                    addShowMoreButton();
                }
            }
            
            function addShowMoreButton() {
                const addMoreElement = document.createElement('div');
                addMoreElement.className = 'add-more-card';
                addMoreElement.id = 'add-more-button';
                
                addMoreElement.innerHTML = `
                    <div class="add-more-btn">
                        <i class="fas fa-plus"></i>
                    </div>
                `;
                
                // Add to container
                document.getElementById('results-container').appendChild(addMoreElement);
                
                // Add event listener
                addMoreElement.addEventListener('click', function() {
                    addNextResult();
                });
            }
            
            function removeAddMoreButton() {
                const addMoreButton = document.getElementById('add-more-button');
                if (addMoreButton) {
                    addMoreButton.remove();
                }
            }
            
            function playInMainPlayer(src) {
                const mainVideo = document.getElementById('main-video');
                mainVideo.src = src;
                mainVideo.play();
                
                // Scroll to the main video player
                mainVideo.scrollIntoView({ behavior: 'smooth' });
            }
        });
    </script>
</body>
</html>''')

if __name__ == '__main__':
    # Create template
    create_template()
    
    # Load data at startup
    print("Loading data from JSON file...")
    success = load_json_data()
    
    if success:
        print(f"Successfully loaded and processed {len(CLIP_IDS)} clips for searching")
        
        # Build TF-IDF index
        print("Building TF-IDF index for keyword search...")
        tfidf_success = prepare_tfidf_index()
        if tfidf_success:
            print("Successfully built TF-IDF index")
        else:
            print("Failed to build TF-IDF index")
            
        # Load pre-computed embeddings
        print("Loading pre-computed embeddings...")
        embeddings_success = load_embeddings()
        if embeddings_success:
            print("Successfully loaded pre-computed embeddings")
            
            # Build FAISS index
            if FAISS_AVAILABLE:
                print("Building FAISS index for vector search...")
                faiss_success = prepare_faiss_index()
                if faiss_success:
                    print("Successfully built FAISS index")
                else:
                    print("Failed to build FAISS index")
            else:
                print("FAISS library not available - vector search will be disabled")
        else:
            print("Failed to load embeddings - vector search will be disabled")
    else:
        print("Error loading data")
    
    # Run the app with threading disabled to avoid resource leaks
    app.run(debug=False, host='0.0.0.0', port=4000, threaded=True)