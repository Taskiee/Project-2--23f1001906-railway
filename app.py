import os
import json
import subprocess
import atexit
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis import Redis
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Constants
REPO_URL = "https://github.com/Taskiee/Project-2---23f1001906"
LOCAL_PATH = "./repo"
FOLDERS = ["GA1", "GA2"]
EMBEDDINGS_FILE = "embeddings.json"

# Initialize Redis for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = Redis.from_url(redis_url, decode_responses=True)

# Rate limiter configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=redis_url,
    default_limits=["5 per minute"]
)

# Global variables for embeddings
embeddings_data = {}
embedding_model = None

def initialize_embedding_model():
    """Initialize the embedding model with error handling"""
    global embedding_model
    try:
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    except Exception as e:
        print(f"Failed to initialize embedding model: {str(e)}")
        raise

def clone_repository():
    """Clone repository if it doesn't exist"""
    if not os.path.exists(LOCAL_PATH):
        print("Cloning repository...")
        try:
            subprocess.run(["git", "clone", REPO_URL, LOCAL_PATH], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {str(e)}")
            raise

def extract_data():
    """Extract question-answer pairs from repository"""
    data = {}
    for folder in FOLDERS:
        folder_path = os.path.join(LOCAL_PATH, folder)
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".txt"):
                        q_path = os.path.join(root, file)
                        try:
                            with open(q_path, "r", encoding="utf-8") as f:
                                question = f.read().strip()
                            
                            base_name = os.path.splitext(file)[0]
                            for ext in [".py", ".sh"]:
                                solution_file = os.path.join(root, f"{base_name}{ext}")
                                if os.path.exists(solution_file):
                                    data[question] = {"solution_file": solution_file}
                                    break
                        except Exception as e:
                            print(f"Error processing file {q_path}: {str(e)}")
                            continue
    return data

def generate_embeddings(data):
    """Generate embeddings for all questions"""
    embeddings = {}
    for question, meta in data.items():
        try:
            vector = embedding_model.encode(question).tolist()
            embeddings[question] = {
                "embedding": vector,
                "solution_file": meta["solution_file"]
            }
        except Exception as e:
            print(f"Error generating embedding for question: {question[:50]}...: {str(e)}")
            continue
    return embeddings

def save_embeddings(embeddings):
    """Save embeddings to file"""
    try:
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f, indent=2)
        print(f"Saved {len(embeddings)} embeddings to {EMBEDDINGS_FILE}")
    except Exception as e:
        print(f"Failed to save embeddings: {str(e)}")
        raise

def load_embeddings():
    """Load embeddings from file"""
    global embeddings_data
    try:
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, "r") as f:
                embeddings_data = json.load(f)
            print(f"Loaded {len(embeddings_data)} embeddings from {EMBEDDINGS_FILE}")
        else:
            print("No embeddings file found, will initialize new one")
            initialize_app()
    except Exception as e:
        print(f"Failed to load embeddings: {str(e)}")
        raise

def find_best_match(question):
    """Find the best matching question using cosine similarity"""
    input_embedding = embedding_model.encode(question).reshape(1, -1)
    best_match = None
    best_similarity = -1
    
    for stored_question, data in embeddings_data.items():
        stored_embedding = np.array(data["embedding"]).reshape(1, -1)
        similarity = cosine_similarity(input_embedding, stored_embedding)[0][0]
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = stored_question

    return best_match, best_similarity

def execute_script(file_path):
    """Execute a script file and return output"""
    try:
        if file_path.endswith(".py"):
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                check=True
            )
        elif file_path.endswith(".sh"):
            result = subprocess.run(
                ["bash", file_path],
                capture_output=True,
                text=True,
                check=True
            )
        else:
            return "Unsupported file type"
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

@app.route("/", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def handle_request():
    """Main API endpoint"""
    if request.method == "GET":
        return jsonify({
            "status": "active",
            "endpoints": {
                "POST /": "Submit questions",
                "GET /health": "Service health check"
            }
        })
    
    question = request.form.get("question") or request.json.get("question")
    if not question:
        return jsonify({"error": "Question parameter is required"}), 400
    
    if not embeddings_data:
        return jsonify({"error": "Embeddings not initialized"}), 503

    try:
        best_match, similarity_score = find_best_match(question)
        
        if best_match and embeddings_data[best_match]["solution_file"]:
            solution_file = embeddings_data[best_match]["solution_file"]
            answer = execute_script(solution_file)
            return jsonify({
                "question": best_match,
                "answer": answer,
                "similarity_score": float(similarity_score)  # Convert numpy float to Python float
            })
        return jsonify({"error": "No matching solution found"}), 404
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "redis_connected": redis_client.ping(),
        "embeddings_loaded": len(embeddings_data) > 0,
        "model_initialized": embedding_model is not None
    })

@app.route("/reinitialize", methods=["POST"])
def reinitialize():
    """Endpoint to manually reinitialize embeddings"""
    try:
        initialize_app()
        return jsonify({"status": "success", "message": "Embeddings reinitialized"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def initialize_app():
    """Initialize application data"""
    global embeddings_data
    
    print("Initializing application...")
    clone_repository()
    data = extract_data()
    embeddings = generate_embeddings(data)
    save_embeddings(embeddings)
    embeddings_data = embeddings
    print(f"Initialized with {len(embeddings)} question-answer pairs")

def cleanup():
    """Cleanup function to run on exit"""
    print("Cleaning up...")

# Register cleanup function
atexit.register(cleanup)

# Initialize when starting
if __name__ == "__main__":
    try:
        initialize_embedding_model()
        load_embeddings()
        port = int(os.environ.get("PORT", 8000))
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        raise
else:
    # For production with gunicorn or similar
    initialize_embedding_model()
    load_embeddings()
