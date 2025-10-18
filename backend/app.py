# app.py
import os, json, requests
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# load env from backend/.env
load_dotenv(Path(__file__).with_name(".env"))

# app.py (imports)
from insight_engine import explain, get_basic_info, DATA_PATH, pd, _df, force_recoerce, ALIASES, get_agent

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173","http://127.0.0.1:5173"]}})

@app.get("/")
def root():
    return "✅ Flask backend running (LangChain insight mode)."

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/routes")
def routes():
    return jsonify(sorted([str(r) for r in app.url_map.iter_rules()]))

# --- DEBUG: data coverage by Week / BU ---
from insight_engine import _df

@app.get("/debug/coverage")
def debug_coverage():
    try:
        import pandas as pd

        # Params
        metric = request.args.get("metric", "ArrivalAccuracy(FinalBTR)")
        group  = request.args.get("group", "BU")

        if "Week" not in _df.columns:
            return jsonify({"error": "Week not present in dataframe"}), 400
        if metric not in _df.columns:
            return jsonify({"error": f"Metric '{metric}' not in columns"}), 400
        if group not in _df.columns:
            return jsonify({"error": f"Group '{group}' not in columns"}), 400

        d = _df.copy()
        # simple stats
        out = {
            "overall": {
                "rows": int(len(d)),
                "weeks_distinct": int(d["Week"].nunique()),
                "metric_non_null": int(d[metric].notna().sum()),
            },
            "by_group": []
        }

        # by-group coverage: weeks distinct and metric non-null count
        g = d.groupby(group, dropna=False)
        for k, sub in g:
            out["by_group"].append({
                group: None if pd.isna(k) else str(k),
                "rows": int(len(sub)),
                "weeks_distinct": int(sub["Week"].nunique()),
                "metric_non_null": int(sub[metric].notna().sum()),
            })

        # top 8 weeks overall
        out["sample_weeks_overall"] = d["Week"].dropna().astype(str).head(8).tolist()
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": "debug/coverage failed", "details": str(e)}), 500


@app.get("/debug/preview-weeks")
def debug_preview_weeks():
    try:
        group = request.args.get("group", "BU")
        value = request.args.get("value")   # e.g., APAC
        if "Week" not in _df.columns:
            return jsonify({"error": "Week not present in dataframe"}), 400
        if group not in _df.columns:
            return jsonify({"error": f"Group '{group}' not in columns"}), 400

        d = _df.copy()
        if value is not None:
            d = d[d[group] == value]

        weeks = d["Week"].dropna().astype(str).tolist()
        return jsonify({
            "group": group,
            "value": value,
            "count": len(weeks),
            "distinct": len(set(weeks)),
            "first_20": weeks[:20],
        }), 200
    except Exception as e:
        return jsonify({"error": "debug/preview-weeks failed", "details": str(e)}), 500

@app.post("/recoerce")
def recoerce():
    try:
        return jsonify(force_recoerce()), 200
    except Exception as e:
        return jsonify({"error": "recoerce failed", "details": str(e)}), 500
    
# Optional: keep your old raw-Azure sanity route if you like
@app.get("/joke-test")
def joke_test():
    base   = os.getenv("AZURE_OPENAI_ENDPOINT")
    ver    = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION")
    deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    key    = os.getenv("AZURE_OPENAI_API_KEY")
    if not all([base, ver, deploy, key]):
        return jsonify({"error":"Missing Azure env vars"}), 500
    url = f"{base}/deployments/{deploy}/chat/completions?api-version={ver}"
    headers = {"api-key": key, "Content-Type": "application/json"}
    payload = {"messages":[{"role":"user","content":"can you tell me an IT joke in 20 words?"}]}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=(10,60)); r.raise_for_status()
        data = r.json(); joke = data.get("choices",[{}])[0].get("message",{}).get("content","")
        return jsonify({"joke": joke})
    except requests.RequestException as e:
        return jsonify({"error":"Upstream error","details":str(e)}), 502

@app.get("/distinct")
def distinct():
    from insight_engine import _df
    col = request.args.get("col", "BU")
    if col not in _df.columns:
        return jsonify({"error": f"Column '{col}' not found", "columns": list(_df.columns)}), 400
    vals = sorted({str(v) for v in _df[col].dropna().unique().tolist()})
    return jsonify({"column": col, "count": len(vals), "values": vals}), 200

# 🚀 LangChain-powered insights route
@app.post("/ask")
def ask():
    """
    Body:
      { "question": "Explain this page for APAC", "filters": { "BU": ["APAC"] } }
    """
    data = request.get_json(silent=True) or {}
    question = data.get("question") or data.get("query") or "Explain this page."
    filters  = data.get("filters")

    try:
        merged, summary = explain(question, filters)  # <-- LangChain path
        return jsonify({"summary": summary, "details": merged})
    except Exception as e:
        # send full error back so we can see the upstream 404 body
        return jsonify({"error": "Insight engine failed", "details": str(e)}), 500

@app.get("/peek")
def peek():
    try:
        col = request.args.get("col", "ArrivalAccuracy(FinalBTR)")
        n = int(request.args.get("n", 8))
        if col not in _df.columns:
            return jsonify({"error": f"Column '{col}' not found",
                            "columns": list(_df.columns)}), 400
        return jsonify({
            "path": str(DATA_PATH),
            "dtype": str(_df[col].dtype),
            "samples": _df[col].astype(str).head(n).tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": "peek failed", "details": str(e)}), 500
    
@app.get("/data-info")
def data_info():
    try:
        return jsonify(get_basic_info()), 200
    except Exception as e:
        # return JSON instead of an HTML debugger page
        return jsonify({"error": "data-info failed", "details": str(e)}), 500

@app.post("/agent-ask")
def agent_ask():
    try:
        body = request.get_json(silent=True) or {}
        q = (body.get("question") or "").strip()
        if not q:
            return jsonify({"error": "Missing 'question'"}), 400

        agent = get_agent()

        # Different LC versions accept different shapes; try both
        try:
            result = agent.invoke({"input": q})
        except Exception:
            result = agent.invoke(q)

        if isinstance(result, dict) and "output" in result:
            answer = result["output"]
            raw = {k: v for k, v in result.items() if k != "output"}
        else:
            answer = result
            raw = {}

        return jsonify({"answer": answer, "raw": raw}), 200
    except Exception as e:
        return jsonify({"error": "agent failed", "details": str(e)}), 500

if __name__ == "__main__":
    print("Starting Flask from:", __file__)
    app.run(host="127.0.0.1", port=8000, debug=True)
