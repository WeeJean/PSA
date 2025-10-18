import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from agentic_engine import run_agentic_query   # ✅ import your LangGraph agent

# === Load environment variables from .env ===
load_dotenv()

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)  # allow frontend (React) to access Flask backend

# === PowerBI Dashboard ===
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")
REPORT_ID = os.getenv("REPORT_ID")

# === Azure API Configuration (used inside LangChain) ===
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://psacodesprint2025.azure-api.net")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1-nano")
API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

# === Health check endpoint ===
@app.route("/")
def home():
    return "✅ Flask backend running with Agentic AI (LangChain + Power BI)"

# === Main route for conversational AI queries ===
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        query = data.get("query", "")
        # Call the LangGraph + pandas agent
        result = run_agentic_query(query)
        return jsonify({"response": result})
    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": str(e)}), 500

# === Power BI embed token endpoint (unchanged) ===
@app.route("/get-embed-token")
def get_embed_token():
    try:
        # Step 1: Get Azure AD access token
        url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": "https://analysis.windows.net/powerbi/api/.default",
        }
        token_response = requests.post(url, data=data)
        access_token = token_response.json().get("access_token")

        # Step 2: Get embed details from Power BI REST API
        headers = {"Authorization": f"Bearer {access_token}"}
        report_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}"
        report_info = requests.get(report_url, headers=headers).json()
        embed_url = report_info["embedUrl"]

        # Step 3: Generate embed token
        embed_token_url = (
            f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}/GenerateToken"
        )
        embed_token_body = {"accessLevel": "View"}
        embed_token_response = requests.post(embed_token_url, headers=headers, json=embed_token_body)
        embed_token = embed_token_response.json()["token"]

        return jsonify({
            "embedUrl": embed_url,
            "reportId": REPORT_ID,
            "accessToken": embed_token
        })
    except Exception as e:
        print("❌ Power BI Error:", e)
        return jsonify({"error": str(e)}), 500

# === Run the Flask app ===
if __name__ == "__main__":
    app.run(port=5000, debug=True)
