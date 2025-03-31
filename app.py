import os
import json
import subprocess
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from redis import Redis
from sentence_transformers import SentenceTransformer

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

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

def clone_repository():
    """Clone repository if it doesn't exist"""
    if not os.path.exists(LOCAL_PATH):
        print("Cloning repository...")
        subprocess.run(["git", "clone", REPO_URL, LOCAL_PATH], check=True)

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
                        with open(q_path, "r", encoding="utf-8") as f:
                            question = f.read().strip()
                        
                        base_name = os.path.splitext(file)[0]
                        for ext in [".py", ".sh"]:
                            solution_file = os.path.join(root, f"{base_name}{ext}")
                            if os.path.exists(solution_file):
                                data[question] = {"solution_file": solution_file}
                                break
    return data

def generate_embeddings(data):
    """Generate embeddings for all questions"""
    embeddings = {}
    for question, meta in data.items():
        vector = embedding_model.encode(question).tolist()
        embeddings[question] = {
            "embedding": vector,
            "solution_file": meta["solution_file"]
        }
    return embeddings

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
    
    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        return jsonify({"error": "Embeddings not initialized"}), 503

    # Find closest matching question
    input_embedding = embedding_model.encode(question).tolist()
    best_match = None
    best_similarity = -1
    
    for stored_question, data in embeddings.items():
        similarity = sum(a*b for a,b in zip(input_embedding, data["embedding"]))
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = stored_question

    if best_match and embeddings[best_match]["solution_file"]:
        solution_file = embeddings[best_match]["solution_file"]
        answer = execute_script(solution_file)
        return jsonify({
            "question": best_match,
            "answer": answer,
            "similarity_score": best_similarity
        })
    return jsonify({"error": "No matching solution found"}), 404

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "redis_connected": redis_client.ping()
    })

def initialize_app():
    """Initialize application data"""
    clone_repository()
    data = extract_data()
    embeddings = generate_embeddings(data)
    
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f, indent=2)
    
    print(f"Initialized with {len(embeddings)} question-answer pairs")

if __name__ == "__main__":
    initialize_app()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
