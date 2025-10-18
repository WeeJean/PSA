import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS 
from dotenv import load_dotenv

# === Load environment variables from .env ===
load_dotenv()

# === Flask App Setup ===
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (lets React call Flask)

# === PowerBI Dashboard === 
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")
REPORT_ID = os.getenv("REPORT_ID")

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
    
@app.route("/get-embed-token")
def get_embed_token():
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
    embed_token_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}/GenerateToken"
    embed_token_body = {"accessLevel": "View"}
    embed_token_response = requests.post(embed_token_url, headers=headers, json=embed_token_body)
    embed_token = embed_token_response.json()["token"]

    return jsonify({
        "embedUrl": embed_url,
        "reportId": REPORT_ID,
        "accessToken": embed_token
    })

# === Run the Flask app ===
if __name__ == "__main__":
    app.run(port=5000, debug=True)
