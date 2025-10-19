# agent_engine_v2.py
import os, re
import json as _json
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool

# ==== bring your data tools in (v2, month-based) ====
from insight_engine import (
    get_basic_info,
    kpi_snapshot,
    summarize_metric,
    apply_filters,   # used in anomalies tool
    _df,             # raw df for some tools
    BU_TO_REGION,
    get_llm,
    _col
)

load_dotenv()

# ---- small helper to parse filters_json safely (kept from v1 behaviour) ----
def _norm_filters(filters_json: str | None):
    if not filters_json:
        return {}
    try:
        f = _json.loads(filters_json)
        if isinstance(f, dict):
            return f
    except Exception:
        pass
    return {}

# --- Tools ---

@tool("data_info", return_direct=False)
def data_info_tool() -> dict:
    """Basic info about current dataset."""
    return get_basic_info()

@tool("kpi_snapshot", return_direct=False)
def kpi_tool(filters_json: str = "") -> str:
    """
    Compute KPI snapshot (month-agnostic aggregates) for the current dataset and optional filters.
    filters_json: JSON string of include filters, e.g. {"Region":["APAC"],"BU":["SINGAPORE"]}
    Returns JSON string with KPI fields.
    """
    f = _norm_filters(filters_json)
    return _json.dumps(kpi_snapshot(f), default=str)

@tool("trend_mom", return_direct=False)
def trend_mom_tool(metric: str, filters_json: str = "") -> str:
    """
    Compute month-over-month trend for a metric within optional filters.
    metric: use canonical metric names (e.g., "ArrivalAccuracy(FinalBTR)", "BunkerSaved(USD)")
    filters_json: JSON string filters, e.g. {"BU":["SINGAPORE"]}.
    Returns JSON with latest_month, previous_month, current_mean, previous_mean, delta_percent/pp.
    """
    f = _norm_filters(filters_json)
    # summarize_metric in v2 defaults to month; pass level="month" to be explicit
    return _json.dumps(summarize_metric(metric, f, level="month"), default=str)

@tool("anomalies_by_group", return_direct=False)
def anomalies_tool(metric: str, group_col: str = "BU", filters_json: str = "", top_n: int = 3) -> str:
    """
    Find highest/lowest groups by z-score on a metric (mean per group), using current dataset.
    Resolves canonical names via engine header map.
    """
    import pandas as pd
    f = _norm_filters(filters_json)

    d = apply_filters(f)

    # Resolve canonical → actual column names
    mcol = _col(metric) or metric
    gcol = _col(group_col) or group_col

    if gcol not in d.columns:
        return _json.dumps({"error": f"Group column '{group_col}' not found (resolved='{gcol}').",
                            "columns": list(d.columns)})

    if mcol not in d.columns:
        return _json.dumps({"error": f"Metric '{metric}' not found (resolved='{mcol}').",
                            "columns": list(d.columns)})

    vals = pd.to_numeric(d[mcol], errors="coerce")
    d = d.assign(_metric=vals).dropna(subset=["_metric"])
    if d.empty:
        return _json.dumps({"error": f"No numeric values for metric '{metric}' after coercion."})

    by_grp = d.groupby(gcol)["_metric"].mean().reset_index()
    mu, sd = by_grp["_metric"].mean(), by_grp["_metric"].std(ddof=0)
    by_grp["z"] = 0.0 if (pd.isna(sd) or sd == 0) else (by_grp["_metric"] - mu) / sd

    highest = by_grp.nlargest(int(top_n), "z").to_dict(orient="records")
    lowest  = by_grp.nsmallest(int(top_n), "z").to_dict(orient="records")

    return _json.dumps({
        "metric": metric,
        "metric_resolved": mcol,
        "group_col": group_col,
        "group_col_resolved": gcol,
        "highest": highest,
        "lowest": lowest,
        "filters_applied": f or {},
    }, default=str)

@tool("metric_value", return_direct=False)
def metric_value_tool(
    metric: str,
    filters_json: str = "",
    month: str | None = None,     # e.g. "2025-09", "September", "Sep 2025", "Sep"
    agg: str = "auto",            # "auto" | "sum" | "mean"
) -> str:
    """
    Return a single metric value for an optional month and filters.
    - month accepts "YYYY-MM", "Sep 2025", "September", or "Sep" (chooses most recent if year omitted)
    - agg: "sum" or "mean"; "auto" = sum for Bunker/Carbon, mean for others (ArrivalAccuracy% handled)
    """
    import calendar
    import pandas as pd
    f = _norm_filters(filters_json)

    d = apply_filters(f)
    tkey = "MonthKey"
    if tkey not in d.columns:
        return _json.dumps({"error": "No 'MonthKey' available in dataset."})

    # Resolve metric
    mcol = _col(metric) or metric
    if mcol not in d.columns:
        return _json.dumps({"error": f"Metric '{metric}' not found (resolved='{mcol}').",
                            "columns": list(d.columns)})

    # Determine aggregation
    metric_l = (metric or "").lower()
    if agg == "auto":
        if "bunker" in metric_l or "carbon" in metric_l:
            agg = "sum"
        else:
            agg = "mean"

    # Optional month selection
    selected_month = None
    dd = d.copy()
    if month:
        m = month.strip()
        # direct YYYY-MM
        if re.match(r"^\d{4}-\d{2}$", m):
            dd = dd[dd[tkey] == m]
            selected_month = m
        else:
            # parse month names, optionally with year
            # examples: "September", "Sep", "Sep 2025"
            parts = m.split()
            month_num = None
            year = None
            # try month name
            for i in range(1,13):
                if calendar.month_name[i].lower().startswith(parts[0].lower()) or \
                   calendar.month_abbr[i].lower() == parts[0].lower():
                    month_num = i
                    break
            if len(parts) >= 2 and re.match(r"^\d{4}$", parts[1]):
                year = int(parts[1])

            if month_num is None:
                return _json.dumps({"error": f"Could not parse month '{month}'."})

            mm = f"{month_num:02d}"
            if year:
                key = f"{year}-{mm}"
                dd = dd[dd[tkey] == key]
                selected_month = key
            else:
                # no year given -> pick the most recent MonthKey that endswith -MM
                pool = dd[dd[tkey].str.endswith(f"-{mm}", na=False)][tkey].dropna().unique().tolist()
                pool = sorted(pool)
                if not pool:
                    return _json.dumps({"error": f"No data for month '{month}'."})
                selected_month = pool[-1]
                dd = dd[dd[tkey] == selected_month]

    vals = pd.to_numeric(dd[mcol], errors="coerce").dropna()
    if vals.empty:
        return _json.dumps({"error": f"No numeric values for '{metric}' with given filters/month.",
                            "resolved_metric": mcol, "selected_month": selected_month, "filters": f})

    # ArrivalAccuracy reported as percent (mean * 100)
    if (_col("ArrivalAccuracy(FinalBTR)") == mcol) or (metric == "ArrivalAccuracy(FinalBTR)"):
        value = float(vals.mean()) * 100.0
        return _json.dumps({
            "metric": metric,
            "metric_resolved": mcol,
            "aggregation": "mean_as_percent",
            "value": round(value, 2),
            "month": selected_month,
            "filters_applied": f or {},
        })

    if agg == "sum":
        value = float(vals.sum())
    else:
        value = float(vals.mean())

    return _json.dumps({
        "metric": metric,
        "metric_resolved": mcol,
        "aggregation": agg,
        "value": round(value, 3),
        "month": selected_month,
        "filters_applied": f or {},
    })

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

tools = [data_info_tool, kpi_tool, trend_mom_tool, anomalies_tool, distinct_tool, peek_tool, metric_value_tool]

# --- LLM (same as before) ---
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0.0
)

# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
    "You are PSA’s data analytics assistant. Use tools when helpful:\n"
    "- kpi_snapshot for KPI aggregates within optional filters\n"
    "- trend_mom for month-over-month trends on a metric\n"
    "- anomalies_by_group to surface outliers by BU/Region\n"
    "- metric_value to return a single metric value for a BU/Region and month (YYYY-MM or month name)\n"  # ← ADD THIS LINE
    "- data_info/distinct_values/peek_column for schema exploration.\n"
    "Be concise and explain in business terms.\n"
    "If the user names a site (Antwerp, Singapore, Busan), interpret it as BU (column 'BU').\n"
    "If the user says 'in APAC', 'in EMEA', or 'in ME', interpret that as a Region filter (column 'Region').\n"
    "Metrics must match canonical names: 'ArrivalAccuracy(FinalBTR)', 'BerthTime(hours):ATU-ATB', "
    "'AssuredPortTimeAchieved(%)', 'CarbonAbatement(Tonnes)', 'BunkerSaved(USD)'.\n"
    ),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Agent (LangChain tools agent) ---
agent_runnable = create_openai_tools_agent(llm, tools, prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=False)

# --- Suggestion generator (kept, but scoped to available BUs/Regions & metrics) ---
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
    ban_metric_list: list[str] | None = None,   # ← NEW: allow banning specific metrics (e.g., ArrivalAccuracy)
) -> list[str]:
    import json as _json
    import json, re, random
    from insight_engine import _df, _col  # ensure we see latest data/schema

    # 1) Build allowed BUs/Regions from DATA
    from insight_engine import BU_TO_REGION
    allowed_bu_list = (
        sorted(_df["BU"].astype(str).str.strip().str.upper().unique().tolist())
        if ("BU" in _df.columns and not _df.empty)
        else []
    )
    allowed_bu_list = [b for b in allowed_bu_list if b in BU_TO_REGION]
    allowed_regions_list = sorted({BU_TO_REGION[b] for b in allowed_bu_list})

    # 2) Build allowed metrics from DATA (canonical names resolved to actual; filter to numeric cols)
    # If you prefer a strict static list, replace this block with your explicit metric names.
    numeric_like = []
    for c in _df.columns:
        if c.startswith("Unnamed"):
            continue
        if str(_df[c].dtype) in ("int64", "float64"):
            numeric_like.append(c)

    # Map actual -> canonical label shown to the LLM. You can customize pretty labels here.
    # Try to keep these 1:1 with your engine’s canonical keys.
    candidates = {
        "BunkerSaved(USD)":          _col("BunkerSaved(USD)")          or "BunkerSaved(USD)",
        "CarbonAbatement(Tonnes)":   _col("CarbonAbatement(Tonnes)")   or "CarbonAbatement(Tonnes)",
        "BerthTime(hours):ATU-ATB":  _col("BerthTime(hours):ATU-ATB")  or "BerthTime(hours):ATU-ATB",
        "AssuredPortTimeAchieved(%)":_col("AssuredPortTimeAchieved(%)")or "AssuredPortTimeAchieved(%)",
        "ArrivalAccuracy(FinalBTR)": _col("ArrivalAccuracy(FinalBTR)") or "ArrivalAccuracy(FinalBTR)",
    }
    allowed_metrics = [k for k, actual in candidates.items() if actual in numeric_like]

    # Optionally: temporarily demote or ban ArrivalAccuracy for demo
    ban_metric_list = set(ban_metric_list or [])
    # Example: ban_metric_list.add("ArrivalAccuracy(FinalBTR)")  # uncomment if you want to fully suppress it

    # If everything got filtered out (shouldn’t happen), fall back to the non-arrival ones:
    if not allowed_metrics:
        allowed_metrics = [
            "BunkerSaved(USD)",
            "CarbonAbatement(Tonnes)",
            "BerthTime(hours):ATU-ATB",
            "AssuredPortTimeAchieved(%)",
            "ArrivalAccuracy(FinalBTR)",
        ]

    # 3) Craft prompt that FORCES metric diversity
    ban_list = ban_list or []
    ctx = context or {}
    ctx_text = _json.dumps(ctx, default=str)[:3000]

    # Make a small rule-string like "Do not use: ArrivalAccuracy(FinalBTR)"
    disallow_clause = ""
    if ban_metric_list:
        disallow_clause = "Do NOT include these metrics: " + ", ".join(sorted(ban_metric_list)) + "\n"

    prompt = f"""
    You are generating NEXT-STEP QUERIES that the PSA analytics copilot can actually answer.

    Return ONLY a JSON array of {limit+3} short strings (no prose, no markdown, no keys).

    ### Rules
    - Start with an action verb (Show, Summarize, Rank, Compare, Investigate, List, Recommend).
    - Keep them short (4–12 words, no trailing periods).
    - Use ONLY valid scopes and metrics listed below.
    - Use a variety of metrics and regions — do not repeat the same metric or region too often.
    - If unsure which region or BU to use, vary across several (e.g. Singapore, Antwerp, Busan, Dammam, etc.)
    - Avoid generic or unanswerable prompts (e.g. “delay causes” or “reasons for low performance”).
    - Avoid always focusing on APAC or Arrival Accuracy.
    - Prefer relevant and diverse KPIs.

    ### Valid metrics
    {allowed_metrics}

    ### Valid BUs
    {allowed_bu_list}

    ### Valid Regions
    {allowed_regions_list}

    ### Example types
    - Summarize KPI snapshot for a BU or Region
    - Show month-over-month trend for a metric in a BU
    - Rank best or worst 3 BUs by a metric
    - Compare two regions or ports on a KPI
    - Investigate anomalies or performance drivers for a metric
    - Recommend next 3 actions to improve a metric

    ### Context
    {_json.dumps(context, default=str)[:1000]}

    ### Current question
    {(last_question or "")[:300]}

    ### Do not repeat
    {_json.dumps(ban_list[:10], ensure_ascii=False)}
    """


    llm = get_llm()
    raw = llm.invoke(prompt).content.strip()

    # 4) Parse & validate
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("not list")
    except Exception:
        items = []

    verb = re.compile(r"^(Show|Summarize|Compare|Rank|Investigate|List|Peek|Recommend)\b", re.I)
    intent = re.compile(
        r"(KPI|snapshot|trend|month|MoM|rank|best|worst|top|bottom|"
        r"anomal|outlier|compare|vs|drivers?|drill|distinct|unique|values?|list|"
        r"peek|sample|column|actions?|steps?|improve)",
        re.I,
    )

    # 5) Post-filter to enforce metric diversity and ban list
    # Detect which canonical metric appears inside a suggestion by substring match.
    def detect_metric(s: str) -> str | None:
        s_low = s.lower()
        for m in allowed_metrics:
            if m.lower() in s_low:
                return m
        return None

    curated, seen_text, seen_metric = [], set(), set()
    for s in items:
        if not isinstance(s, str):
            continue
        t = s.strip().rstrip(".")
        if not t or not verb.search(t) or not intent.search(t):
            continue
        if last_question and _too_similar(t, last_question):
            continue
        if any(_too_similar(t, b) for b in ban_list):
            continue

        m = detect_metric(t)
        # Enforce "one metric max once"
        if (m in ban_metric_list) or (m in seen_metric):
            continue
        # If it mentions NO metric at all, let a small number through (e.g., distinct values / peek column)
        if (m is None) and (sum(1 for x in curated if detect_metric(x) is None) >= 2):
            continue

        k = t.lower()
        if k in seen_text:
            continue
        seen_text.add(k)
        if m:
            seen_metric.add(m)
        curated.append(t)

        if len(curated) >= (limit * 2):  # gather a bit more; we’ll trim
            break

    # 6) Ensure we have at least one for several buckets (light touch)
    def bucket_key(t: str) -> str:
        tl = t.lower()
        if "kpi" in tl or "snapshot" in tl: return "kpi"
        if "trend" in tl or "month" in tl or "mom" in tl: return "trend"
        if "rank" in tl and ("best" in tl or "top" in tl): return "best"
        if "rank" in tl and ("worst" in tl or "bottom" in tl): return "worst"
        if "anomal" in tl or "outlier" in tl: return "anom"
        if "compare" in tl or " vs " in tl: return "compare"
        if "driver" in tl or "drill" in tl: return "drivers"
        if "distinct" in tl or "unique" in tl or "values" in tl or "list " in tl: return "distinct"
        if "peek" in tl or "sample" in tl or "column" in tl: return "peek"
        if "action" in tl or "steps" in tl or "improve" in tl or "recommend" in tl: return "actions"
        return "other"

    buckets = {k: [] for k in ["kpi","trend","best","worst","anom","compare","drivers","distinct","peek","actions","other"]}
    for t in curated:
        buckets[bucket_key(t)].append(t)

    # Interleave buckets to avoid clustering on one intent
    order = ["kpi","trend","compare","best","worst","anom","drivers","distinct","peek","actions","other"]
    out = []
    i = 0
    while len(out) < limit and any(buckets.values()):
        b = order[i % len(order)]
        if buckets[b]:
            out.append(buckets[b].pop(0))
        i += 1

    # Fallback if LLM didn’t cooperate
    if not out:
        # Build deterministic diversified fallbacks that **do not** overuse ArrivalAccuracy
        non_arrival = [m for m in allowed_metrics if m != "ArrivalAccuracy(FinalBTR)"]
        pick = (non_arrival or allowed_metrics)[:3]
        out = [
            f"Summarize KPI snapshot for APAC",
            f"Show MoM trend for {pick[0]} in APAC",
            f"Compare SINGAPORE vs BUSAN on {pick[1] if len(pick)>1 else pick[0]}",
            f"Rank worst 3 BUs by {pick[2] if len(pick)>2 else pick[0]} in APAC",
            f"Investigate drivers of {pick[0]} in ANTWERP",
            f"List distinct values of BU",
        ][:limit]

    return out[:limit]



def run_agentic_query(query: str) -> str:
    """Return final text answer (string)."""
    try:
        result = agent.invoke({"input": query})
        if isinstance(result, dict) and "output" in result:
            return result["output"]
        return str(result)
    except Exception as e:
        return f"Error: {e}"
