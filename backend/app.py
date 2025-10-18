import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS   # 👈 NEW: allows frontend requests
from dotenv import load_dotenv

# === Load environment variables from .env ===
load_dotenv()

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)  # 👈 Enable Cross-Origin Resource Sharing (lets React call Flask)

# === Azure API Configuration ===
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = "https://psacodesprint2025.azure-api.net"
DEPLOYMENT_NAME = "gpt-4.1-nano"  # your deployment name in Azure
API_VERSION = "2025-01-01-preview"

# === Health check endpoint ===
@app.route("/")
def home():
    return "✅ Flask backend running (PSA Code Sprint API Gateway mode)"

# === Main route for LLM queries ===
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")

    # Construct the Azure API URL
    url = f"{AZURE_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    payload = {
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 200,
        "temperature": 0.7
    }

    try:
        # Send request to Azure OpenAI (PSA API Gateway)
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()  # raise an error for bad status codes
        result = r.json()

        # Return the AI response to frontend
        return jsonify({"response": result["choices"][0]["message"]["content"]})

    except requests.exceptions.RequestException as e:
        # Network or HTTP error
        print("❌ Network/API Error:", e)
        return jsonify({"error": str(e)}), 500

    except Exception as e:
        # Any other unexpected error
        print("❌ Unexpected Error:", e)
        return jsonify({"error": str(e)}), 500


# === Run the Flask app ===
if __name__ == "__main__":
    app.run(port=5000, debug=True)
