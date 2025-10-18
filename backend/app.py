import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from agentic_engine import run_agentic_query

# === Load environment variables from .env ===
load_dotenv()

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)   # allow frontend (React) to access Flask backend

# === Azure API Configuration (still used inside LangChain) ===
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://psacodesprint2025.azure-api.net")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1-nano")
API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

# === Health check endpoint ===
@app.route("/")
def home():
    return "✅ Flask backend running with Agentic AI (LangChain + Azure)"

# === Main route: handle user queries ===
@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Get query from frontend
        data = request.get_json()
        query = data.get("query", "")

        # Send query to LangChain agent (in agentic_engine.py)
        result = run_agentic_query(query)

        # Return AI response to frontend
        return jsonify({"response": result})
    except Exception as e:
        print("❌ Backend Error:", e)
        return jsonify({"error": str(e)}), 500


# === Run Flask app ===
if __name__ == "__main__":
    app.run(port=5000, debug=True)
