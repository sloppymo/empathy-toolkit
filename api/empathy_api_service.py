#!/usr/bin/env python3
"""
Empathy API Service

A REST API service that makes your empathy models available through HTTP endpoints.
Features:
- Multiple model endpoints (OpenAI, Simple, Local)
- Response analysis and metrics
- Rate limiting and authentication
- Logging and monitoring
"""

import os
import json
import time
import logging
import argparse
import uuid
import threading
import queue
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from waitress import serve
import sys

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import the simple empathy model
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from simple_empathy_model import SimpleEmpathyModel
    SIMPLE_MODEL_AVAILABLE = True
except ImportError:
    SIMPLE_MODEL_AVAILABLE = False

# Try to import the response evaluator
try:
    from response_quality_evaluator import ResponseEvaluator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    # Create a minimal version for standalone use
    class ResponseEvaluator:
        def __init__(self):
            # Define empathy indicators
            self.validation_phrases = [
                "valid", "understand", "makes sense", "natural", "reasonable", 
                "understandable", "normal", "okay to feel", "it's okay", "it is okay",
                "you're not alone", "you are not alone", "many people", "common",
                "i hear you", "that sounds", "that must be", "that's really",
                "that is really", "i can imagine", "it's difficult", "it is difficult"
            ]
            
            self.reflection_phrases = [
                "you feel", "you're feeling", "you are feeling", "you seem", 
                "you sound", "you mentioned", "you've been", "you have been",
                "you're going through", "you are going through", "you've experienced",
                "you have experienced", "it sounds like you", "it seems like you",
                "you're saying", "you are saying"
            ]
            
            self.support_phrases = [
                "here for you", "support", "help", "listen", "there for you",
                "by your side", "alongside you", "with you", "together", "through this",
                "i'm here", "i am here", "we can", "let's", "let us", "take care",
                "be gentle", "be kind", "self-care", "self care"
            ]
        
        def count_empathy_indicators(self, text):
            """Count empathy indicators in a response"""
            text = text.lower()
            
            validation_count = sum(1 for phrase in self.validation_phrases if phrase in text)
            reflection_count = sum(1 for phrase in self.reflection_phrases if phrase in text)
            support_count = sum(1 for phrase in self.support_phrases if phrase in text)
            
            return {
                "validation": validation_count,
                "reflection": reflection_count,
                "support": support_count,
                "total": validation_count + reflection_count + support_count
            }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("empathy_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("empathy_api")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API keys (for a real application, use a more secure storage)
API_KEYS = {}
if os.getenv("API_KEY"):
    API_KEYS[os.getenv("API_KEY")] = "default"

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_times = {}
    
    def is_allowed(self, api_key):
        current_time = time.time()
        
        # Initialize if first request
        if api_key not in self.request_times:
            self.request_times[api_key] = []
        
        # Remove requests older than 1 minute
        self.request_times[api_key] = [t for t in self.request_times[api_key] if current_time - t < 60]
        
        # Check if under limit
        if len(self.request_times[api_key]) < self.requests_per_minute:
            self.request_times[api_key].append(current_time)
            return True
        
        return False

rate_limiter = RateLimiter()

# Request queue for handling concurrent requests
request_queue = queue.Queue()

# Initialize models
openai_client = None
simple_model = None
evaluator = ResponseEvaluator()

def init_openai():
    """Initialize OpenAI client"""
    global openai_client
    
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not available")
        return
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized")
    else:
        logger.warning("OpenAI API key not found")

def init_simple_model():
    """Initialize simple empathy model"""
    global simple_model
    
    if not SIMPLE_MODEL_AVAILABLE:
        logger.warning("Simple empathy model not available")
        return
    
    try:
        simple_model = SimpleEmpathyModel()
        logger.info("Simple empathy model initialized")
    except Exception as e:
        logger.error(f"Error initializing simple model: {str(e)}")

def get_response_from_openai(prompt, model_id, system_message=None):
    """Get a response from an OpenAI model"""
    if not openai_client:
        return "OpenAI API not configured"
    
    if not system_message:
        system_message = "You are an empathetic assistant that provides supportive and compassionate responses."
    
    try:
        response = openai_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting response from OpenAI: {str(e)}")
        return f"Error: {str(e)}"

def get_response_from_simple_model(prompt):
    """Get a response from the simple empathy model"""
    if not simple_model:
        return "Simple model not available"
    
    try:
        return simple_model.generate_response(prompt)
    except Exception as e:
        logger.error(f"Error getting response from simple model: {str(e)}")
        return f"Error: {str(e)}"

def get_response_from_local_model(prompt, model_path):
    """Get a response from a local model"""
    try:
        # Try to import the local_empathy_trainer module
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from local_empathy_trainer import LocalEmpathyTrainer
        
        trainer = LocalEmpathyTrainer()
        return trainer.generate_response(prompt, model_path=model_path)
    except Exception as e:
        logger.error(f"Error getting response from local model: {str(e)}")
        return f"Error loading local model: {str(e)}"

def evaluate_response(response):
    """Evaluate the quality of a response"""
    try:
        # Count empathy indicators
        indicators = evaluator.count_empathy_indicators(response)
        
        # Calculate simple empathy score (0-10)
        empathy_score = min(10, indicators["total"] * 2)
        
        return {
            "validation": indicators["validation"],
            "reflection": indicators["reflection"],
            "support": indicators["support"],
            "total_indicators": indicators["total"],
            "empathy_score": empathy_score
        }
    except Exception as e:
        logger.error(f"Error evaluating response: {str(e)}")
        return {
            "validation": 0,
            "reflection": 0,
            "support": 0,
            "total_indicators": 0,
            "empathy_score": 0
        }

def log_request(request_data, response_data):
    """Log request and response data"""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "response": response_data
        }
        
        # Append to log file
        with open("logs/api_requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Error logging request: {str(e)}")

def process_request_queue():
    """Process requests from the queue"""
    while True:
        try:
            # Get request from queue
            request_data = request_queue.get()
            
            # Process request
            model_type = request_data.get("model_type", "simple")
            prompt = request_data.get("prompt", "")
            model_id = request_data.get("model_id", "ft:gpt-3.5-turbo-0125:valis::BYfKr10K")
            system_message = request_data.get("system_message", "You are an empathetic assistant that provides supportive and compassionate responses.")
            model_path = request_data.get("model_path", "./local_empathy_model")
            
            # Get response based on model type
            if model_type == "openai":
                response_text = get_response_from_openai(prompt, model_id, system_message)
            elif model_type == "simple":
                response_text = get_response_from_simple_model(prompt)
            elif model_type == "local":
                response_text = get_response_from_local_model(prompt, model_path)
            else:
                response_text = "Unsupported model type"
            
            # Evaluate response
            evaluation = evaluate_response(response_text)
            
            # Create response data
            response_data = {
                "id": str(uuid.uuid4()),
                "response": response_text,
                "metrics": evaluation,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log request and response
            log_request(request_data, response_data)
            
            # Mark task as done
            request_queue.task_done()
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            request_queue.task_done()

# API routes
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "openai": openai_client is not None,
            "simple": simple_model is not None
        }
    })

@app.route("/api/generate", methods=["POST"])
def generate_response():
    """Generate a response"""
    # Check API key
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    
    # Check rate limit
    if not rate_limiter.is_allowed(api_key):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    # Get request data
    try:
        request_data = request.get_json()
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Validate request
    if "prompt" not in request_data:
        return jsonify({"error": "Missing prompt"}), 400
    
    # Add request to queue
    request_queue.put(request_data)
    
    # Get response based on model type
    model_type = request_data.get("model_type", "simple")
    prompt = request_data.get("prompt", "")
    model_id = request_data.get("model_id", "ft:gpt-3.5-turbo-0125:valis::BYfKr10K")
    system_message = request_data.get("system_message", "You are an empathetic assistant that provides supportive and compassionate responses.")
    model_path = request_data.get("model_path", "./local_empathy_model")
    
    # Get response based on model type
    if model_type == "openai":
        response_text = get_response_from_openai(prompt, model_id, system_message)
    elif model_type == "simple":
        response_text = get_response_from_simple_model(prompt)
    elif model_type == "local":
        response_text = get_response_from_local_model(prompt, model_path)
    else:
        return jsonify({"error": "Unsupported model type"}), 400
    
    # Evaluate response
    evaluation = evaluate_response(response_text)
    
    # Create response data
    response_data = {
        "id": str(uuid.uuid4()),
        "response": response_text,
        "metrics": evaluation,
        "timestamp": datetime.now().isoformat()
    }
    
    return jsonify(response_data)

@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models"""
    # Check API key
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    
    models = []
    
    # Add OpenAI models if available
    if openai_client:
        models.append({
            "id": "ft:gpt-3.5-turbo-0125:valis::BYfKr10K",
            "name": "Fine-tuned Empathy Model",
            "type": "openai",
            "description": "OpenAI fine-tuned model for empathetic responses"
        })
        
        models.append({
            "id": "gpt-3.5-turbo",
            "name": "GPT-3.5 Turbo",
            "type": "openai",
            "description": "OpenAI base model for comparison"
        })
    
    # Add simple model if available
    if simple_model:
        models.append({
            "id": "simple",
            "name": "Simple Empathy Model",
            "type": "simple",
            "description": "Template-based empathy model (no API key required)"
        })
    
    return jsonify({"models": models})

@app.route("/api/evaluate", methods=["POST"])
def evaluate_text():
    """Evaluate text for empathy metrics"""
    # Check API key
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    
    # Check rate limit
    if not rate_limiter.is_allowed(api_key):
        return jsonify({"error": "Rate limit exceeded"}), 429
    
    # Get request data
    try:
        request_data = request.get_json()
    except:
        return jsonify({"error": "Invalid JSON"}), 400
    
    # Validate request
    if "text" not in request_data:
        return jsonify({"error": "Missing text"}), 400
    
    # Evaluate text
    text = request_data.get("text", "")
    evaluation = evaluate_response(text)
    
    return jsonify({
        "id": str(uuid.uuid4()),
        "text": text,
        "metrics": evaluation,
        "timestamp": datetime.now().isoformat()
    })

def main():
    parser = argparse.ArgumentParser(description="Empathy API Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--generate-key", action="store_true", help="Generate a new API key")
    
    args = parser.parse_args()
    
    # Generate API key if requested
    if args.generate_key:
        api_key = str(uuid.uuid4())
        API_KEYS[api_key] = "generated"
        print(f"Generated API key: {api_key}")
        print("You can use this key in the X-API-Key header for API requests")
        print("For permanent use, add it to your .env file as API_KEY=<key>")
    
    # Initialize models
    init_openai()
    init_simple_model()
    
    # Start request queue processing thread
    threading.Thread(target=process_request_queue, daemon=True).start()
    
    # Start server
    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        logger.info(f"Starting server on {args.host}:{args.port}")
        serve(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
