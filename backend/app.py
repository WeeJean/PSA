# app.py
import os
import json
import requests
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv

from agent_engine import run_agentic_query, suggest_next_queries


# Local modules
from insight_engine import (
    get_basic_info,
    DATA_PATH,
    _df,
)

# Load env from backend/.env
load_dotenv()

# === Power BI Dashboard secrets ===
CLIENT_ID = os.getenv("CLIENT_ID")
WORKSPACE_ID = os.getenv("WORKSPACE_ID")
REPORT_ID = os.getenv("REPORT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TENANT_ID = os.getenv("TENANT_ID")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:5174", "http://localhost:5174"]}}, supports_credentials=True)

# === Azure API Configuration (used inside LangChain) ===
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://psacodesprint2025.azure-api.net")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1-nano")
API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

# ---------- Health / Misc ----------
@app.get("/")
def root():
    return "✅ Flask backend running (LangChain insight mode)."

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/routes")
def routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

# ---------- Power BI embed token ----------
@app.get("/get-embed-token")
def get_embed_token():
    """
    Returns { embedUrl, reportId, accessToken } for embedding.
    Includes robust error handling so the frontend sees useful details.
    """
    print(TENANT_ID, CLIENT_ID)
    try:
        # 1) AAD access token
        if not all([TENANT_ID, CLIENT_ID, CLIENT_SECRET]):
            return jsonify({"error": "Missing TENANT/CLIENT env vars"}), 500

        oauth_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "scope": "https://analysis.windows.net/powerbi/api/.default",
        }
        tr = requests.post(oauth_url, data=data, timeout=(10, 30))
        tr.raise_for_status()
        access_token = tr.json().get("access_token")
        if not access_token:
            return jsonify({"error": "No access_token from AAD", "details": tr.text}), 502

        headers = {"Authorization": f"Bearer {access_token}"}

        # 2) Report info
        if not all([WORKSPACE_ID, REPORT_ID]):
            return jsonify({"error": "Missing WORKSPACE_ID or REPORT_ID"}), 500

        report_url = f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}"
        rinfo = requests.get(report_url, headers=headers, timeout=(10, 30))
        rinfo.raise_for_status()
        embed_url = rinfo.json().get("embedUrl")
        if not embed_url:
            return jsonify({"error": "Missing embedUrl", "details": rinfo.text}), 502

        # 3) Embed token
        embed_token_url = (
            f"https://api.powerbi.com/v1.0/myorg/groups/{WORKSPACE_ID}/reports/{REPORT_ID}/GenerateToken"
        )
        et = requests.post(embed_token_url, headers=headers, json={"accessLevel": "View"}, timeout=(10, 30))
        et.raise_for_status()
        embed_token = et.json().get("token")
        if not embed_token:
            return jsonify({"error": "Missing embed token", "details": et.text}), 502

        return jsonify({"embedUrl": embed_url, "reportId": REPORT_ID, "accessToken": embed_token}), 200
    except requests.RequestException as e:
        return jsonify({"error": "Power BI request failed", "details": str(e)}), 502
    except Exception as e:
        return jsonify({"error": "Unexpected error", "details": str(e)}), 500

# ---------- Debug / Data helpers ----------
@app.get("/debug/coverage")
def debug_coverage():
    print(TENANT_ID, CLIENT_ID)
    try:
        import pandas as pd

        # Params
        metric = request.args.get("metric", "ArrivalAccuracy(FinalBTR)")
        group = request.args.get("group", "BU")

        if "Week" not in _df.columns:
            return jsonify({"error": "Week not present in dataframe"}), 400
        if metric not in _df.columns:
            return jsonify({"error": f"Metric '{metric}' not in columns"}), 400
        if group not in _df.columns:
            return jsonify({"error": f"Group '{group}' not in columns"}), 400

        d = _df.copy()
        out = {
            "overall": {
                "rows": int(len(d)),
                "weeks_distinct": int(d["Week"].nunique()),
                "metric_non_null": int(d[metric].notna().sum()),
            },
            "by_group": [],
        }

        g = d.groupby(group, dropna=False)
        for k, sub in g:
            out["by_group"].append(
                {
                    group: None if pd.isna(k) else str(k),
                    "rows": int(len(sub)),
                    "weeks_distinct": int(sub["Week"].nunique()),
                    "metric_non_null": int(sub[metric].notna().sum()),
                }
            )

        out["sample_weeks_overall"] = d["Week"].dropna().astype(str).head(8).tolist()
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": "debug/coverage failed", "details": str(e)}), 500


@app.get("/debug/preview-weeks")
def debug_preview_weeks():
    try:
        group = request.args.get("group", "BU")
        value = request.args.get("value")  # e.g., APAC
        if "Week" not in _df.columns:
            return jsonify({"error": "Week not present in dataframe"}), 400
        if group not in _df.columns:
            return jsonify({"error": f"Group '{group}' not in columns"}), 400

        d = _df.copy()
        if value is not None:
            d = d[d[group] == value]

        weeks = d["Week"].dropna().astype(str).tolist()
        return jsonify(
            {"group": group, "value": value, "count": len(weeks), "distinct": len(set(weeks)), "first_20": weeks[:20]}
        ), 200
    except Exception as e:
        return jsonify({"error": "debug/preview-weeks failed", "details": str(e)}), 500

@app.get("/joke-test")
def joke_test():
    base = os.getenv("AZURE_OPENAI_ENDPOINT")
    ver = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if not all([base, ver, deploy, key]):
        return jsonify({"error": "Missing Azure env vars"}), 500
    url = f"{base}/deployments/{deploy}/chat/completions?api-version={ver}"
    headers = {"api-key": key, "Content-Type": "application/json"}
    payload = {"messages": [{"role": "user", "content": "can you tell me an IT joke in 20 words?"}]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(10, 60))
        r.raise_for_status()
        data = r.json()
        joke = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return jsonify({"joke": joke})
    except requests.RequestException as e:
        return jsonify({"error": "Upstream error", "details": str(e)}), 502


@app.get("/distinct")
def distinct():
    col = request.args.get("col", "BU")
    if col not in _df.columns:
        return jsonify({"error": f"Column '{col}' not found", "columns": list(_df.columns)}), 400
    vals = sorted({str(v) for v in _df[col].dropna().unique().tolist()})
    return jsonify({"column": col, "count": len(vals), "values": vals}), 200


@app.get("/peek")
def peek():
    try:
        col = request.args.get("col", "ArrivalAccuracy(FinalBTR)")
        n = int(request.args.get("n", 8))
        if col not in _df.columns:
            return jsonify({"error": f"Column '{col}' not found", "columns": list(_df.columns)}), 400
        return (
            jsonify(
                {"path": str(DATA_PATH), "dtype": str(_df[col].dtype), "samples": _df[col].astype(str).head(n).tolist()}
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": "peek failed", "details": str(e)}), 500


@app.get("/data-info")
def data_info():
    try:
        return jsonify(get_basic_info()), 200
    except Exception as e:
        return jsonify({"error": "data-info failed", "details": str(e)}), 500

@app.post("/ask")
def ask_unified():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    # Optional context sent by the client
    last_question = (data.get("last_question") or "").strip()
    recent_suggestions = data.get("recent_suggestions") or []

    try:
        answer = run_agentic_query(question)

        # optional context to specialize suggestions
        context = {
            "question": question,
            "answer_excerpt": str(answer)[:800],
        }
        chips = suggest_next_queries(
            context=context,
            limit=5,
            last_question=question or last_question,
            ban_list=recent_suggestions if isinstance(recent_suggestions, list) else []
        )

        return jsonify({
            "text": answer,
            "details": {"suggestions": chips}
        }), 200
    except Exception as e:
        return jsonify({"error": "ask failed", "details": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask from:", __file__)
    app.run(port=8000, debug=True)
