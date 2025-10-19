# agent_engine.py
import os, re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool

# bring your data tools in
from insight_engine import summarize_metric, get_basic_info, explain, _df, kpi_snapshot, _norm_filters, _json, anomalies_by_group

load_dotenv()

# --- Tools ---
@tool("data_info", return_direct=False)
def data_info_tool() -> dict:
    """Basic info about current dataset."""
    return get_basic_info()

@tool("debug_coverage", return_direct=False)
def debug_coverage_tool(metric: str = "ArrivalAccuracy(FinalBTR)", group: str = "BU") -> dict:
    """Quick coverage stats by group/week."""
    import pandas as pd
    if "Week" not in _df.columns:
        return {"error": "Week not present in dataframe"}
    if metric not in _df.columns:
        return {"error": f"Metric '{metric}' not in columns"}
    if group not in _df.columns:
        return {"error": f"Group '{group}' not in columns"}

    d = _df.copy()
    out = {
        "overall": {
            "rows": int(len(d)),
            "weeks_distinct": int(d["Week"].nunique()),
            "metric_non_null": int(d[metric].notna().sum()),
        },
        "by_group": []
    }
    g = d.groupby(group, dropna=False)
    for k, sub in g:
        out["by_group"].append({
            group: None if pd.isna(k) else str(k),
            "rows": int(len(sub)),
            "weeks_distinct": int(sub["Week"].nunique()),
            "metric_non_null": int(sub[metric].notna().sum()),
        })
    return out

@tool("kpi_snapshot", return_direct=False)
def kpi_tool(filters_json: str = "") -> str:
    """
    Compute KPI snapshot for the current dataset and optional filters.
    filters_json: JSON string of an include-filter dict, e.g. {"Region":["APAC"],"BU":["SINGAPORE"]}
    Returns JSON string with KPI fields.
    """
    f = _norm_filters(filters_json)
    return _json.dumps(kpi_snapshot(f), default=str)

@tool("trend_wow", return_direct=False)
def trend_tool(metric: str, filters_json: str = "") -> str:
    """
    Compute week-over-week trend for a metric within optional filters.
    metric: exact column name (e.g., "ArrivalAccuracy(FinalBTR)" or "Berth Time(hours):ATU-ATB")
    filters_json: JSON string filters.
    Returns JSON string with latest_week, previous_week, current_mean, previous_mean, delta...
    """
    f = _norm_filters(filters_json)
    return _json.dumps(summarize_metric(metric, f), default=str)

@tool("anomalies_by_group", return_direct=False)
def anomalies_tool(metric: str, group_col: str = "BU", filters_json: str = "", top_n: int = 3) -> str:
    """
    Find highest/lowest groups by z-score on a metric.
    metric: exact column name
    group_col: grouping column (default "BU")
    filters_json: JSON string filters
    top_n: how many groups to return for each side
    Returns JSON string with highest/lowest arrays.
    """
    f = _norm_filters(filters_json)
    out = anomalies_by_group(metric, group_col=group_col, top_n=int(top_n), filters=f)
    return _json.dumps(out, default=str)

@tool("distinct_values", return_direct=False)
def distinct_tool(column: str) -> str:
    """
    List distinct values for a column (e.g., 'Region' or 'BU').
    Returns JSON string with values.
    """
    if column not in _df.columns:
        return _json.dumps({"error": f"Column '{column}' not found", "columns": list(_df.columns)})
    vals = sorted({str(v) for v in _df[column].dropna().unique().tolist()})
    return _json.dumps({"column": column, "count": len(vals), "values": vals})

@tool("peek_column", return_direct=False)
def peek_tool(column: str, n: int = 8) -> str:
    """
    Peek the first n values of a column to understand its content.
    Returns JSON string with dtype and samples.
    """
    if column not in _df.columns:
        return _json.dumps({"error": f"Column '{column}' not found", "columns": list(_df.columns)})
    s = _df[column].head(int(n))
    return _json.dumps({
        "dtype": str(_df[column].dtype),
        "samples": s.astype(str).tolist()
    })

tools = [data_info_tool, debug_coverage_tool, kpi_tool, trend_tool, anomalies_tool, distinct_tool, peek_tool]

# --- LLM ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0.0
)

# --- Prompt ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are PSA’s data analytics assistant. Decide which tools to use and answer concisely.\n"
     "If the user mentions a Region (APAC/EMEA/AMERICAS/ME) or a BU (e.g., SINGAPORE), pass it via filters_json.\n"
     "Always finish with 3–5 short, imperative next-step suggestions (bulleted or numbered)."),
    MessagesPlaceholder("messages"),
    # REQUIRED for tools agent:
    ("system", "Use the tools above. Keep responses business-friendly."),
    ("placeholder", "{agent_scratchpad}"),
])

# --- Agent (LangChain tools agent) ---
agent_runnable = create_openai_tools_agent(llm, tools, prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=False)

SUGG_VERB_RX = r"^(increase|reduce|review|investigate|optimi[sz]e|monitor|coordinate|escalate|deploy|pilot|train|audit|fix|patch|tune|rebalance|re[- ]?route|communicate|benchmark|validate|improve|lower|boost|align)\b"

def _extract_suggestions(text: str, limit: int = 5) -> list[str]:
    if not text:
        return []
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    cands = []
    for l in lines:
        if l.startswith(("-", "*")) or re.match(r"^\d+\.\s+", l):
            cands.append(l)
        elif re.match(SUGG_VERB_RX, l, re.I):
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

def run_agentic_query(query: str) -> dict:
    """Returns {text, suggestions[]}"""
    result = agent.invoke({"messages": [("user", query)]})
    # Try to pull the final assistant text robustly
    text = ""
    if isinstance(result, dict):
        if "output" in result and result["output"]:
            text = result["output"]
        elif "messages" in result and isinstance(result["messages"], list) and result["messages"]:
            last = result["messages"][-1]
            text = getattr(last, "content", "") if not isinstance(last, tuple) else last[1]
    else:
        text = str(result)

    suggestions = _extract_suggestions(text, 5)
    return {"text": text or "(no text)", "suggestions": suggestions}
