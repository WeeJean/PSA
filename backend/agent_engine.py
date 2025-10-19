# agent_engine.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool

# bring your data tools in
from insight_engine import PSA_STRATEGY_GUARDRAIL, summarize_metric, get_basic_info, explain, _df, kpi_snapshot, _norm_filters, _json, anomalies_by_group, get_llm, BU_TO_REGION

load_dotenv()

# --- Tools ---
@tool("run_pipeline", return_direct=False)
def run_pipeline_tool(question: str, filters: dict | None = None) -> dict:
    """Run the structured pipeline to generate a business explanation."""
    merged, summary = explain(question, filters)
    return {"summary": summary, "details": merged}

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

tools = [run_pipeline_tool, data_info_tool, debug_coverage_tool, kpi_tool, trend_tool, anomalies_tool, distinct_tool, peek_tool]

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
     "You are PSA’s data analytics assistant. Use tools when helpful:\n"
     "- run_pipeline(question, filters?) for structured KPI/Trend/Anomalies + actions\n"
     "- kpi_snapshot/trend_wow/anomalies_by_group for specific analytics\n"
     "- data_info/distinct_values/peek_column for schema exploration.\n"
     "Be concise and explain in business terms."
     "If the user names a site like 'Antwerp', 'Singapore', 'Busan', interpret it as BU (column 'BU'), not Region."
     "When the user says 'in APAC', 'in EMEA', or 'in ME', interpret that as a Region filter (column 'Region'), \
 not part of the metric name. Metrics always match existing column headers exactly, such as \
 'Carbon Abatement (Tonnes)' or 'Bunker Saved(USD)'."
    ),
    # Optional chat history support
    MessagesPlaceholder(variable_name="chat_history", optional=True),

    # The current user input
    ("human", "{input}"),

    # REQUIRED by create_openai_tools_agent:
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Agent (LangChain tools agent) ---
agent_runnable = create_openai_tools_agent(llm, tools, prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=False)

# --- Suggestion generator: produce ONLY prompts the agent can run with tools ---
# --- helpers for dedupe / similarity ---
def _normalize_prompt_text(s: str) -> list[str]:
    import re
    stop = {"the","a","an","for","of","to","in","on","by","with","this","that","these","those","and","or"}
    t = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    toks = [w for w in t.split() if w and w not in stop]
    return toks

def _too_similar(a: str, b: str, thresh: float = 0.6) -> bool:
    A, B = set(_normalize_prompt_text(a)), set(_normalize_prompt_text(b))
    if not A or not B: 
        return False
    inter = len(A & B); union = len(A | B)
    return (inter / union) >= thresh

def suggest_next_queries(
    context: dict | None = None,
    limit: int = 6,
    last_question: str | None = None,
    ban_list: list[str] | None = None,
) -> list[str]:
    """
    Produce diversified next-step queries. Buckets:
      - KPI snapshot
      - Trend WoW (or MoM)
      - Rank best performers
      - Rank worst performers
      - Anomalies by group
      - Compare 2 scopes
      - Drill into drivers
      - Distinct values (scoping help)
      - Peek column
      - Prescriptive actions (steps to take)

    Returns short imperatives (4–12 words), no trailing period.
    """
    allowed_metrics = [
        "ArrivalAccuracy(FinalBTR)",
        "Berth Time(hours):ATU-ATB",
        "AssuredPortTimeAchieved(%)",
        "Carbon Abatement (Tonnes)",
        "Bunker Saved(USD)",
    ]
    allowed_bu_list = (sorted(_df["BU"].astype(str).str.strip().str.upper().unique().tolist()) if ("BU" in _df.columns and not _df.empty) else [])
    allowed_bu_list = [b for b in allowed_bu_list if b in BU_TO_REGION]
    allowed_regions_list = sorted({BU_TO_REGION[b] for b in allowed_bu_list})
    ban_list = ban_list or []
    ctx = context or {}
    import json as _json
    ctx_text = _json.dumps(ctx, default=str)[:3000]

    prompt = f"""You produce NEXT-STEP QUERIES the PSA analytics copilot can run or answer.
Return ONLY a JSON array of {limit+3} short strings. No prose, no keys, no markdown.

Rules:
- Start with an imperative verb (Show/Summarize/Rank/Investigate/Compare/List/Peek/Recommend).
- 4–12 words, no trailing period.
- Use concrete scopes when helpful.

IMPORTANT SCOPE RULES (STRICT):
- BU must be chosen ONLY from: {allowed_bu_list}
- Region must be chosen ONLY from: {allowed_regions_list}
- Do NOT invent new BUs or Regions. If a scope isn't in the lists, OMIT it.

- <metric> must be one of:
  {allowed_metrics}

Cover a VARIETY of intents; include at least one from each where sensible:
  • KPI snapshot         → “Summarize KPI snapshot for <scope>”
  • Trend WoW / MoM      → “Show WoW trend for <metric> in <scope>”
  • Rank best            → “Rank best 3 BUs by <metric> in <scope>”
  • Rank worst           → “Rank worst 3 BUs by <metric> in <scope>”
  • Anomalies            → “Find anomalies by BU for <metric> in <scope>”
  • Compare scopes       → “Compare APAC vs EMEA on <metric>”
  • Drivers / drilldown  → “Investigate drivers of <metric> in <scope>”
  • Distinct values      → “List distinct values of BU”
  • Peek column          → “Peek column <name> (first 8 values)”
  • Prescriptive steps   → “Recommend next 3 actions to improve <metric> in <scope>”

DO NOT repeat or paraphrase the user's current question or any items in DO_NOT_REPEAT.

NETWORK STRATEGY:
{PSA_STRATEGY_GUARDRAIL}
Bias queries toward cross-BU/region comparisons, transshipment connectivity, schedule integrity,
and actions that improve end-to-end network performance.

CONTEXT:
{ctx_text}

CURRENT_QUESTION:
{(last_question or "")[:300]}

DO_NOT_REPEAT:
{_json.dumps(ban_list[:10], ensure_ascii=False)}
"""

    llm = get_llm()
    raw = llm.invoke(prompt).content.strip()

    # Parse & validate
    import json, re
    try:
        items = json.loads(raw)
        if not isinstance(items, list): raise ValueError("not list")
    except Exception:
        items = []

    verb = re.compile(r"^(Show|Summarize|Compare|Rank|Investigate|List|Peek|Recommend)\b", re.I)
    # broader intent set
    intent = re.compile(
        r"(KPI|snapshot|WoW|trend|week|month|MoM|rank|best|worst|top|bottom|"
        r"anomal|outlier|compare|vs|drivers?|drill|distinct|unique|values?|list|"
        r"peek|sample|column|actions?|steps?|improve)",
        re.I,
    )

    curated, seen = [], set()
    for s in items:
        if not isinstance(s, str): continue
        t = s.strip().rstrip(".")
        if not t or not verb.search(t) or not intent.search(t): continue
        if last_question and _too_similar(t, last_question): continue
        if any(_too_similar(t, b) for b in ban_list): continue
        k = t.lower()
        if k in seen: continue
        seen.add(k)
        curated.append(t)

    # Diversify by bucket (one each, in order of usefulness)
    def bucket_key(t: str) -> str:
        tl = t.lower()
        if "kpi" in tl or "snapshot" in tl: return "kpi"
        if "wow" in tl or "trend" in tl or "month" in tl or "mom" in tl: return "trend"
        if "rank" in tl and ("best" in tl or "top" in tl): return "best"
        if "rank" in tl and ("worst" in tl or "bottom" in tl): return "worst"
        if "anomal" in tl or "outlier" in tl: return "anom"
        if "compare" in tl or " vs " in tl: return "compare"
        if "driver" in tl or "drill" in tl: return "drivers"
        if "distinct" in tl or "unique" in tl or "values" in tl or "list " in tl: return "distinct"
        if "peek" in tl or "sample" in tl or "column" in tl: return "peek"
        if "action" in tl or "steps" in tl or "improve" in tl or "recommend" in tl: return "actions"
        return "other"

    buckets = {k: [] for k in ["kpi","trend","best","worst","anom","compare","drivers","distinct","peek","actions"]}
    for t in curated:
        k = bucket_key(t)
        if k in buckets and not buckets[k]:
            buckets[k] = [t]

    diversified = []
    order = ["kpi","trend","best","worst","anom","compare","drivers","distinct","peek","actions"]
    for k in order:
        diversified += buckets[k]
        if len(diversified) >= limit: break

    # Fallback if LLM gave nothing useful
    if not diversified:
        diversified = [
            "Summarize KPI snapshot for APAC",
            "Show WoW trend for ArrivalAccuracy(FinalBTR) in APAC",
            "Rank worst 3 BUs by ArrivalAccuracy(FinalBTR) in APAC",
            "Recommend next 3 actions to improve ArrivalAccuracy(FinalBTR) in APAC",
        ]
    return diversified[:limit]



def run_agentic_query(query: str) -> str:
    """Return final text answer (string)."""
    try:
        result = agent.invoke({"input": query})
        if isinstance(result, dict) and "output" in result:
            return result["output"]
        return str(result)
    except Exception as e:
        return f"Error: {e}"
