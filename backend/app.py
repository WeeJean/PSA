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

from agent_engine import run_agentic_query


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
load_dotenv(Path(__file__).with_name(".env"))

# === Power BI Dashboard secrets ===
CLIENT_ID = "d4513e50-29a7-4f57-a41f-68fae5006b67"
WORKSPACE_ID = "41675240-7b6e-4163-a0ed-52b5c3b13e01"
REPORT_ID = "06bdda3d-459c-4632-8784-d43e6b208aab"
CLIENT_SECRET = "uF08Q~1sS-bSDi4bZe8JuOyPrIZglZ4zRqgKLbMp"
TENANT_ID = "27fa816c-95b5-4431-90d9-4d0ac1986f71"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}})

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

@app.get("/llm-config")
def llm_config():
    try:
        from agent_factory import _make_llm
        llm = _make_llm()
        return jsonify({
            "llm_class": llm.__class__.__name__,
            "azure_endpoint": getattr(llm, "azure_endpoint", None),
            "azure_deployment": getattr(llm, "azure_deployment", None),
            "model_or_name": getattr(llm, "model_name", None) or getattr(llm, "model", None),
        }), 200
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.get("/llm-selftest")
def llm_selftest():
    try:
        from agent_factory import _make_llm
        llm = _make_llm()
        _ = llm.invoke([{"role": "user", "content": "ping"}])
        return jsonify({"ok": True}), 200
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

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
def ask_unified():
    """
    Unified ask endpoint.

    Body:
      {
        "question": "Explain APAC performance" | "Show WoW trend for ArrivalAccuracy(FinalBTR) in APAC",
        "filters": { "Region": ["APAC"] },   # optional; used only in 'pipeline' mode
        "mode": "agent" | "pipeline"         # optional; default = "agent"
      }

    Returns (normalized):
      - mode = "pipeline": { "mode":"pipeline", "text": <summary>, "details": {...} }
      - mode = "agent":    { "mode":"agent",    "text": <answer>,  "raw": {} }
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("query") or "").strip()
    mode = (data.get("mode") or "agent").lower()
    filters = data.get("filters")

    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        if mode == "pipeline":
            details, summary = None, None
            # insight_engine.explain returns (merged_dict, summary_text)
            merged, summary = explain(question, filters)
            return jsonify({
                "mode": "pipeline",
                "text": summary,
                "details": merged
            }), 200

        # default = agent
        answer = run_agentic_query(question)
        return jsonify({
            "mode": "agent",
            "text": answer,
            "raw": {}
        }), 200

    except Exception as e:
        return jsonify({"error": "ask failed", "details": str(e)}), 500

    
if __name__ == "__main__":
    print("Starting Flask from:", __file__)
    app.run(port=8000, debug=True)
