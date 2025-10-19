# app.py
import os, re
import json
import requests
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv

from agent_engine import run_agentic_query
from insight_engine import get_llm

# Local modules
from insight_engine import (
    explain,
    get_basic_info,
    DATA_PATH,
    pd,           # if you use it in debug routes
    _df,
    force_recoerce,
    ALIASES,
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
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5173", "http://localhost:5173"]}}, supports_credentials=True)

# === Azure API Configuration (used inside LangChain) ===
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://psacodesprint2025.azure-api.net")
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4.1-nano")
API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

def _extract_suggestions(text: str, limit: int = 5) -> list[str]:
    """Turn an 'Actions' paragraph into 3–5 short chips."""
    if not text:
        return []
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]

    cands = []
    for l in lines:
        if l.startswith(("-", "*")) or re.match(r"^\d+\.\s+", l):
            cands.append(l)
        elif re.match(r"^(increase|reduce|review|investigate|optimi[sz]e|follow up|alert|escalate|deploy|pilot|monitor|coordinate|train|audit|fix|patch|tune|rebalance|re-route|standardi[sz]e|communicate|benchmark|validate)\b", l, re.I):
            cands.append(l)

    cleaned = [re.sub(r"^[-*\d.]+\s+", "", x).strip() for x in cands]
    out, seen = [], set()
    for s in cleaned:
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s[:120])
            if len(out) >= limit:
                break
    return out

# optional: use the model to synthesize chips if extraction fails
def _llm_suggestions(context: dict, limit: int = 5) -> list[str]:
    try:
        llm = get_llm()
        prompt = (
            "From the context below, produce "
            f"{limit} short, imperative next-step suggestions (4–10 words each). "
            "Return ONLY a JSON array of strings.\n\n"
            f"CONTEXT:\n{json.dumps(context, default=str)}"
        )
        raw = llm.invoke(prompt).content
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x)[:120] for x in arr][:limit]
    except Exception:
        pass
    return []

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

# @app.get("/llm-config")
# def llm_config():
#     try:
#         from agent_factory import _make_llm
#         llm = _make_llm()
#         return jsonify({
#             "llm_class": llm.__class__.__name__,
#             "azure_endpoint": getattr(llm, "azure_endpoint", None),
#             "azure_deployment": getattr(llm, "azure_deployment", None),
#             "model_or_name": getattr(llm, "model_name", None) or getattr(llm, "model", None),
#         }), 200
#     except Exception as e:
#         import traceback
#         return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

# @app.get("/llm-selftest")
# def llm_selftest():
#     try:
#         from agent_factory import _make_llm
#         llm = _make_llm()
#         _ = llm.invoke([{"role": "user", "content": "ping"}])
#         return jsonify({"ok": True}), 200
#     except Exception as e:
#         import traceback
#         return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

@app.post("/recoerce")
def recoerce():
    try:
        return jsonify(force_recoerce()), 200
    except Exception as e:
        return jsonify({"error": "recoerce failed", "details": str(e)}), 500


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

# @app.get("/agent-selftest")
# def agent_selftest():
#     try:
#         agent = make_agent()
#         return jsonify({"executor_class": agent.__class__.__name__, "ok": True})
#     except Exception as e:
#         import traceback
#         return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500
    
# ---------- Single agent endpoint ----------
# @app.post("/agent-ask")
# def agent_ask():
#     try:
#         body = request.get_json(silent=True) or {}
#         q = (body.get("question") or "").strip()
#         history = body.get("history", [])
#         if not q:
#             return jsonify({"answer_type":"error","message":"Missing 'question'","payload":{}}), 400

#         agent = make_agent()

#         # Convert simple history into LC messages
#         history_msgs = []
#         for t in history:
#             role = (t.get("role") or "").lower()
#             content = t.get("content") or ""
#             if role == "human":
#                 history_msgs.append(HumanMessage(content=content))
#             elif role in ("ai","assistant"):
#                 history_msgs.append(AIMessage(content=content))

#         # Call the simple executor
#         result = agent.invoke({"input": q, "chat_history": history_msgs})
#         content = result if isinstance(result, str) else result.get("output", "")

#         try:
#             envelope = json.loads(content)
#         except Exception:
#             envelope = {"answer_type":"text","message":str(content),"payload":{},"tool_calls":[]}

#         for key in ("answer_type","message","payload"):
#             envelope.setdefault(key, "" if key!="payload" else {})
#         return jsonify(envelope), 200

#     except Exception as e:
#         import traceback
#         return jsonify({
#         "answer_type": "error",
#         "message": "agent failed",
#         "payload": {"details": str(e), "trace": traceback.format_exc()}
#     }), 500

@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    q = (data.get("question") or data.get("query") or "").strip()
    if not q:
        return jsonify({"error": "Missing 'question'"}), 400
    try:
        result = run_agentic_query(q)  # {text, suggestions[]}
        return jsonify({
            "text": result["text"],
            "details": {"suggestions": result.get("suggestions", [])}
        }), 200
    except Exception as e:
        return jsonify({"error": "agent failed", "details": str(e)}), 500

    
if __name__ == "__main__":
    print("Starting Flask from:", __file__)
    app.run(port=8000, debug=True)
