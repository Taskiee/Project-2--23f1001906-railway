import os
import json
import subprocess
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from redis import Redis
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Question Answering API", version="1.0")

# Constants
REPO_URL = "https://github.com/Taskiee/Project-2---23f1001906"
LOCAL_PATH = "./repo"
FOLDERS = ["GA1", "GA2"]
EMBEDDINGS_FILE = "/tmp/embeddings.json"  # Using /tmp for Railway compatibility

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis for rate limiting
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    redis_client = Redis.from_url(redis_url, decode_responses=True)
    redis_client.ping()  # Test connection
    logger.info("Connected to Redis successfully")
except Exception as e:
    logger.error(f"Redis connection failed: {str(e)}")
    redis_client = None

# Rate limiter configuration
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=redis_url if redis_client else "memory://",
    enabled=bool(redis_client)
)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        {"error": "Rate limit exceeded"},
        status_code=429,
        headers={"Retry-After": str(exc.retry_after)}
    )

# Initialize embedding model
try:
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {str(e)}")
    embedding_model = None

def clone_repository():
    """Clone repository if it doesn't exist"""
    if not os.path.exists(LOCAL_PATH):
        logger.info("Cloning repository...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, LOCAL_PATH],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("Repository cloned successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            raise RuntimeError("Repository cloning failed")

def extract_data():
    """Extract question-answer pairs from repository"""
    data = {}
    for folder in FOLDERS:
        folder_path = os.path.join(LOCAL_PATH, folder)
        if os.path.exists(folder_path):
            q_files = [f for f in os.listdir(folder_path) if f.endswith('_Q.txt')]
            
            for q_file in q_files:
                base_num = q_file.split('_')[0]
                q_path = os.path.join(folder_path, q_file)
                
                try:
                    with open(q_path, "r", encoding="utf-8") as f:
                        question = f.read().strip()
                    
                    ans_files = [f for f in os.listdir(folder_path) 
                               if f.startswith(f"{base_num}_Ans.")]
                    
                    if ans_files:
                        solution_file = os.path.join(folder_path, ans_files[0])
                        data[question] = {"solution_file": solution_file}
                    else:
                        logger.warning(f"No answer file found for {q_file}")
                except Exception as e:
                    logger.error(f"Error processing {q_file}: {str(e)}")
    return data

def generate_embeddings(data):
    """Generate embeddings for all questions"""
    if not embedding_model:
        raise RuntimeError("Embedding model not available")
    
    embeddings = {}
    for question, meta in data.items():
        try:
            vector = embedding_model.encode(question).tolist()
            embeddings[question] = {
                "embedding": vector,
                "solution_file": meta["solution_file"]
            }
        except Exception as e:
            logger.error(f"Error generating embedding for question: {str(e)}")
    return embeddings

def execute_script(file_path):
    """Execute a script file and return output"""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"
    
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
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
        else:
            return f"Unsupported file type: {file_ext}"
        
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

@app.get("/")
@app.post("/")
@limiter.limit("5 per minute")
async def handle_request(
    request: Request,
    question: Optional[str] = Form(None)
):
    """Main API endpoint"""
    if request.method == "GET":
        return {
            "status": "active",
            "endpoints": {
                "POST /": "Submit questions",
                "GET /health": "Service health check"
            },
            "redis_connected": bool(redis_client and redis_client.ping()),
            "model_loaded": bool(embedding_model)
        }
    
    # Try to get question from form data or JSON body
    if not question:
        try:
            form_data = await request.form()
            question = form_data.get("question")
        except:
            try:
                json_data = await request.json()
                question = json_data.get("question")
            except:
                pass
    
    if not question:
        raise HTTPException(status_code=400, detail="Question parameter is required")
    
    try:
        with open(EMBEDDINGS_FILE, "r") as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Embeddings not initialized. Please wait for initialization to complete."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error loading embeddings: {str(e)}"
        )

    # Find closest matching question
    try:
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
            return {
                "question": best_match,
                "answer": answer,
                "similarity_score": best_similarity,
                "solution_file": os.path.basename(solution_file)
            }
        raise HTTPException(status_code=404, detail="No matching solution found")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing your question: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if embedding_model else "degraded",
        "redis_connected": bool(redis_client and redis_client.ping()),
        "model_loaded": bool(embedding_model),
        "embeddings_available": os.path.exists(EMBEDDINGS_FILE)
    }

async def initialize_app():
    """Initialize the application components"""
    try:
        logger.info("Initializing application...")
        clone_repository()
        data = extract_data()
        logger.info(f"Found {len(data)} question-answer pairs")
        
        embeddings = generate_embeddings(data)
        logger.info(f"Generated embeddings for {len(embeddings)} questions")
        
        with open(EMBEDDINGS_FILE, "w") as f:
            json.dump(embeddings, f, indent=2)
        logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}")
        
        if embeddings:
            sample_q = next(iter(embeddings.items()))
            logger.info(f"Sample question: {sample_q[0][:50]}...")
        return True
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Run initialization on startup"""
    await initialize_app()

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
