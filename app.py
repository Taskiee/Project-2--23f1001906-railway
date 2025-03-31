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
LOCAL_PATH = os.path.abspath("./repo")
FOLDERS = ["GA1", "GA2"]
EMBEDDINGS_FILE = os.path.abspath("embeddings.json")

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
    """Clone repository if it doesn't exist with enhanced error handling"""
    print(f"\n=== CLONING REPOSITORY ===")
    print(f"Target directory: {LOCAL_PATH}")
    
    if os.path.exists(LOCAL_PATH):
        print("Repository already exists")
        return True

    try:
        result = subprocess.run(
            ["git", "clone", REPO_URL, LOCAL_PATH],
            check=True,
            capture_output=True,
            text=True
        )
        print("Clone successful")
        print(f"Clone output: {result.stdout[:200]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Clone failed: {e.stderr}")
        return False

def extract_data():
    """Extract question-answer pairs with detailed validation"""
    print("\n=== EXTRACTING DATA ===")
    data = {}
    valid_pairs = 0
    
    for folder in FOLDERS:
        folder_path = os.path.join(LOCAL_PATH, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            continue

        print(f"\nProcessing {folder}:")
        q_files = [f for f in os.listdir(folder_path) if f.endswith('_Q.txt')]
        
        if not q_files:
            print(f"❌ No question files found in {folder}")
            continue

        for q_file in sorted(q_files):
            base_num = q_file.split('_')[0]
            q_path = os.path.join(folder_path, q_file)
            
            try:
                with open(q_path, "r", encoding="utf-8") as f:
                    question = f.read().strip()
                
                if not question:
                    print(f"⚠️ Empty question file: {q_file}")
                    continue

                ans_files = [f for f in os.listdir(folder_path) 
                           if f.startswith(f"{base_num}_Ans.")]
                
                if not ans_files:
                    print(f"⚠️ No answer file for {q_file}")
                    continue

                solution_file = os.path.join(folder_path, ans_files[0])
                data[question] = {"solution_file": solution_file}
                valid_pairs += 1
                print(f"✅ {q_file} -> {ans_files[0]}")

            except Exception as e:
                print(f"❌ Error processing {q_file}: {str(e)}")
    
    print(f"\nTotal valid question-answer pairs: {valid_pairs}")
    return data

def generate_embeddings(data):
    """Generate embeddings with progress tracking"""
    if not data:
        raise ValueError("No valid question-answer pairs found")
    
    print("\n=== GENERATING EMBEDDINGS ===")
    embeddings = {}
    for i, (question, meta) in enumerate(data.items(), 1):
        try:
            vector = embedding_model.encode(question).tolist()
            embeddings[question] = {
                "embedding": vector,
                "solution_file": meta["solution_file"]
            }
            if i % 5 == 0 or i == len(data):
                print(f"Processed {i}/{len(data)} questions")
        except Exception as e:
            print(f"Error encoding question {i}: {str(e)}")
    
    return embeddings

def execute_script(file_path):
    """Execute scripts with extended format support"""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.py':
            result = subprocess.run(
                ["python", file_path],
                capture_output=True,
                text=True,
                check=True
            )
        elif file_ext == '.sh':
            result = subprocess.run(
                ["bash", file_path],
                capture_output=True,
                text=True,
                check=True
            )
        elif file_ext == '.js':
            result = subprocess.run(
                ["node", file_path],
                capture_output=True,
                text=True,
                check=True
            )
        elif file_ext == '.sql':
            result = subprocess.run(
                ["sqlite3", ":memory:", f".read {file_path}"],
                capture_output=True,
                text=True,
                check=True
            )
        elif file_ext == '.txt':
            with open(file_path, 'r') as f:
                return f.read()
        else:
            return f"Unsupported file type: {file_ext}"
        
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Execution Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

@app.route("/", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def handle_request():
    """Enhanced API endpoint with better error handling"""
    if request.method == "GET":
        return jsonify({
            "status": "active",
            "endpoints": {
                "POST /": "Submit questions",
                "GET /health": "Service health check",
                "GET /debug": "Debug information"
            }
        })
    
    try:
        question = request.form.get("question") or request.json.get("question")
        if not question:
            return jsonify({"error": "Question parameter is required"}), 400
        
        if not os.path.exists(EMBEDDINGS_FILE):
            return jsonify({
                "error": "Embeddings not initialized",
                "solution": "Run initialize_app() manually first"
            }), 503

        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)
            
        if not embeddings:
            return jsonify({"error": "Embeddings file is empty"}), 503

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
                "similarity_score": best_similarity,
                "solution_file": os.path.basename(solution_file),
                "solution_type": os.path.splitext(solution_file)[1]
            })
        return jsonify({"error": "No matching solution found"}), 404

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check"""
    status = {
        "status": "healthy",
        "redis_connected": redis_client.ping(),
        "embeddings_exist": os.path.exists(EMBEDDINGS_FILE),
        "repo_exists": os.path.exists(LOCAL_PATH)
    }
    return jsonify(status)

@app.route("/debug", methods=["GET"])
def debug_info():
    """Debug endpoint"""
    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)
    except:
        embeddings = {}
    
    return jsonify({
        "current_directory": os.getcwd(),
        "repo_contents": subprocess.getoutput(f"find {LOCAL_PATH} -type f | head -n 20"),
        "embeddings_count": len(embeddings),
        "sample_question": next(iter(embeddings.keys())) if embeddings else None,
        "system_info": {
            "python_version": subprocess.getoutput("python --version"),
            "git_status": subprocess.getoutput("git status")
        }
    })

def initialize_app():
    """Robust initialization with full validation"""
    print("\n" + "="*40)
    print("INITIALIZING APPLICATION".center(40))
    print("="*40)
    
    # Verify and clone repository
    if not clone_repository():
        raise RuntimeError("Repository initialization failed")
    
    # Extract data with validation
    data = extract_data()
    if not data:
        raise ValueError("No valid question-answer pairs found")
    
    # Generate embeddings
    embeddings = generate_embeddings(data)
    
    # Save embeddings with verification
    try:
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        print(f"\nSaved embeddings to {EMBEDDINGS_FILE}")
        print(f"File verification - exists: {os.path.exists(EMBEDDINGS_FILE)}")
        print(f"File size: {os.path.getsize(EMBEDDINGS_FILE)} bytes")
        
    except Exception as e:
        raise IOError(f"Failed to save embeddings: {str(e)}")
    
    print("\nInitialization completed successfully!")
    print("="*40 + "\n")

if __name__ == "__main__":
    try:
        initialize_app()
    except Exception as e:
        print(f"\n❌ Initialization failed: {str(e)}")
        print("Attempting to create empty embeddings as fallback...")
        try:
            with open(EMBEDDINGS_FILE, "w") as f:
                json.dump({}, f)
            print("Created empty embeddings.json as fallback")
        except:
            print("Failed to create fallback file")
    
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
