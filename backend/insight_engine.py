# insight_engine.py
from __future__ import annotations

import os, json, re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import json as _json

# Map your BU → Region (edit as you like)
BU_TO_REGION = {
    "ANTWERP": "EMEA",
    "BUSAN": "APAC",
    "DAMMAM": "ME",           # Middle East
    "JAKARTA": "APAC",
    "LAEMCHABANG": "APAC",
    "MUMBAI": "APAC",
    "PANAMACITY": "AMERICAS",
    "SINGAPORE": "APAC",
    "TIANJIN": "APAC",
}

def _ensure_region(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" in df.columns:
        return df
    if "BU" in df.columns:
        df["Region"] = (
            df["BU"].astype(str).str.strip().str.upper()
              .map(BU_TO_REGION).fillna("OTHER")
        )
    return df

def _is_arrival_col(name: str) -> bool:
    # strict: uses your exact alias
    return name == ALIASES["arrival_accuracy"][0]

def _coerce_numeric(df: pd.DataFrame) -> list[str]:
    """
    Convert number-like text to real numbers, but only mark a column as numeric if:
      - it has *enough* valid numeric values after coercion (>= max(3, 10% of rows)), and
      - it's not actually a datetime-like column.
    Special-case: ArrivalAccuracy(FinalBTR) maps 'Y'/'N' -> 1/0 and is always numeric.
    """
    numeric_cols: list[str] = []

    # How many valid numeric cells must we see before we accept a column as numeric?
    n_rows = len(df)
    min_valid = max(3, int(0.10 * n_rows))  # at least 3 cells or 10% of rows

    ARRIVAL_COL = ALIASES["arrival_accuracy"][0]  # "ArrivalAccuracy(FinalBTR)"

    for col in df.columns:
        s = df[col]

        # 0) Skip obvious junk columns like "Unnamed: 24"
        if isinstance(col, str) and col.startswith("Unnamed:"):
            continue

        # 1) Special-case arrival accuracy: map Y/N -> 1/0 strictly
        if col == ARRIVAL_COL:
            if s.dtype == "O":
                mapped = s.astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
                coerced = pd.to_numeric(mapped, errors="coerce")
            else:
                coerced = pd.to_numeric(s, errors="coerce")
            df[col] = coerced
            # Accept if we see enough valid 0/1 numbers
            if coerced.notna().sum() >= min_valid:
                numeric_cols.append(col)
            continue

        # 2) If it's already numeric dtype, keep it
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue

        # 3) Detect datetime-like columns & leave them as datetime (not numeric)
        if s.dtype == "O":
            # Try parse a sample; if at least half parse to datetimes, treat as datetime
            dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
            parsed_ratio = dt.notna().mean()
            if parsed_ratio >= 0.50:
                # Keep datetime; write back parsed values (useful for later)
                df[col] = dt
                continue

        # 4) Generic numeric coercion for text columns
        if s.dtype == "O":
            cleaned = (
                s.astype(str)
                 .str.strip()
                 .replace({"": None, "NA": None, "NaN": None}, regex=False)
                 .str.replace(",", "", regex=False)
                 .str.replace("%", "", regex=False)
            )
            coerced = pd.to_numeric(cleaned, errors="coerce")
            valid = coerced.notna().sum()

            # Only accept as numeric if enough valid values exist
            if valid >= min_valid:
                df[col] = coerced
                numeric_cols.append(col)
            # else: leave the original strings as-is (non-numeric)

    return numeric_cols

def force_recoerce() -> dict:
    """
    Re-run numeric coercion (including mapping 'Y'/'N'→1/0 for ArrivalAccuracy(FinalBTR))
    on the in-memory dataframe and report before/after.
    """
    global _num_columns

    # Which column are we coercing specially?
    arrival_col = ALIASES.get("arrival_accuracy", [None])[0]

    before = str(_df[arrival_col].dtype) if (arrival_col and arrival_col in _df.columns) else "N/A"

    # re-run coercion in-place
    _num_columns = _coerce_numeric(_df)

    after = str(_df[arrival_col].dtype) if (arrival_col and arrival_col in _df.columns) else "N/A"

    report = {
        "path": str(DATA_PATH),
        "arrival_col": arrival_col,
        "dtype_before": before,
        "dtype_after": after,
        "numeric_columns": list(_num_columns),
    }
    if arrival_col and arrival_col in _df.columns:
        report["value_counts_after"] = _df[arrival_col].value_counts(dropna=False).to_dict()
        report["sample_after"] = _df[arrival_col].head(12).tolist()
    return report

# ===== Week derivation (robust for your formats) =====
DATE_CANDIDATES = [
    "ATB(LocalTime)",         # prefer this
    "FinalBTR(LocalTime)",    # fallback if ATB is missing/empty
    "ATU (LocalTime)",
    "ABT (LocalTime)",
]

def _normalize_dt_strings(s: pd.Series) -> pd.Series:
    """
    Normalize your date strings:
    - ensure a space between the date and time even if the hour is 1 or 2 digits
      e.g. '23-04-2511:30' -> '23-04-25 11:30', '29-03-258:30' -> '29-03-25 8:30'
    - unify separators, trim spaces, drop empty
    """
    if s.dtype != "O":
        return s
    out = (
        s.astype(str)
         .str.strip()
         .str.replace("\u00a0", " ", regex=False)    # NBSP → space
         .str.replace("/", "-", regex=False)         # just in case
         .str.replace(r"\s+", " ", regex=True)       # collapse internal spaces
         # insert exactly one space between date (dd-mm-yy) and time (H:MM or HH:MM)
         .str.replace(r"^(\d{1,2}-\d{1,2}-\d{2})\s*(\d{1,2}:\d{2})$",
                      r"\1 \2", regex=True)
    )
    out = out.replace({"": None, "NA": None, "NaN": None})
    return out

def _parse_dt_series(s: pd.Series) -> pd.Series:
    """Parse with day-first after normalization; retry a couple of explicit formats if needed."""
    s_norm = _normalize_dt_strings(s)
    dt = pd.to_datetime(s_norm, errors="coerce", dayfirst=True)
    if dt.notna().mean() >= 0.5:
        return dt
    # try explicit patterns if parse rate is too low
    for fmt in ("%d-%m-%y %H:%M", "%d-%m-%Y %H:%M"):
        try_dt = pd.to_datetime(s_norm, format=fmt, errors="coerce")
        if try_dt.notna().mean() > dt.notna().mean():
            dt = try_dt
    return dt

def _best_datetime_column(df: pd.DataFrame) -> tuple[Optional[str], Optional[pd.Series]]:
    """Pick the candidate date column with the highest parse rate."""
    best_col, best_dt, best_rate = None, None, -1.0
    for col in DATE_CANDIDATES:
        if col not in df.columns:
            continue
        dt = _parse_dt_series(df[col])
        rate = dt.notna().mean()
        if rate > best_rate and rate > 0:
            best_col, best_dt, best_rate = col, dt, rate
    return best_col, best_dt

def _ensure_week(df: pd.DataFrame) -> pd.DataFrame:
    """Create 'Week' (YYYY-Www) and 'WeekStart' (Monday date) from the best datetime column."""
    if "Week" in df.columns:
        return df
    col, dt = _best_datetime_column(df)
    if not col or dt is None:
        return df
    iso = dt.dt.isocalendar()  # year, week, day
    df["Week"] = (iso["year"].astype(str) + "-W" + iso["week"].astype(int).astype(str).str.zfill(2))
    df["WeekStart"] = (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()
    # keep the parsed datetimes in the chosen column (prevents accidental numeric coercion)
    df[col] = dt
    return df





# --- Load env from backend/.env ---
load_dotenv(Path(__file__).with_name(".env"))

# ========= LangChain / Azure OpenAI =========
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Prefer OPENAI_API_VERSION if set, otherwise AZURE_OPENAI_API_VERSION
API_VERSION = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
RAW_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://psacodesprint2025.azure-api.net/openai

if not (API_VERSION and DEPLOYMENT and AZURE_KEY and RAW_ENDPOINT):
    raise RuntimeError("Missing Azure env vars for LangChain (endpoint/version/deployment/key). Check backend/.env")

_LLM: Optional[AzureChatOpenAI] = None
_LLM_ENDPOINT_USED: Optional[str] = None  # which endpoint variant succeeded

def _normalize_endpoints(base: str) -> List[str]:
    """Return both endpoint variants (with and without /openai) to handle SDK differences."""
    base = base.rstrip("/")
    if base.endswith("/openai"):
        variants = [base, base.rsplit("/openai", 1)[0]]
    else:
        variants = [base, f"{base}/openai"]
    # unique, preserve order
    out, seen = [], set()
    for v in variants:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def _try_make_llm(endpoint: str) -> AzureChatOpenAI:
    # IMPORTANT: use api_version= (not openai_api_version=)
    return AzureChatOpenAI(
        azure_deployment=DEPLOYMENT,
        azure_endpoint=endpoint,
        api_key=AZURE_KEY,
        api_version=API_VERSION,
        temperature=0.2,
    )

def _warmup_ping(llm: AzureChatOpenAI) -> None:
    # cheap probe to catch 404/401 quickly with readable error
    llm.invoke("ping")

def get_llm() -> AzureChatOpenAI:
    """Lazy-initialize the LLM with endpoint fallback; cache the working one."""
    global _LLM, _LLM_ENDPOINT_USED
    if _LLM is not None:
        return _LLM
    errors = []
    for ep in _normalize_endpoints(RAW_ENDPOINT):
        try:
            cand = _try_make_llm(ep)
            _warmup_ping(cand)
            _LLM = cand
            _LLM_ENDPOINT_USED = ep
            print(f"✅ LangChain LLM ready (endpoint: {ep}, deploy: {DEPLOYMENT}, version: {API_VERSION})")
            return _LLM
        except Exception as e:
            errors.append((ep, str(e)))
    raise RuntimeError(f"LLM init failed. Tried endpoints: {errors}")

# ========= Data loading / helpers =========

# You can set DATA_PATH in backend/.env to point to a specific file.
# If unset, default to data/reference_sample_data.csv
DATA_PATH = Path("data/reference_sample_data.csv")

# Optional: if DATA_PATH is a directory, auto-pick the newest CSV in it.
if DATA_PATH.is_dir():
    csvs = sorted(DATA_PATH.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if csvs:
        DATA_PATH = csvs[0]

_df: pd.DataFrame = pd.DataFrame()
_num_columns: List[str] = []
_loaded_path: Optional[Path] = None

# Aliases locked to your exact headers
ALIASES = {
    "arrival_accuracy":           ["ArrivalAccuracy(FinalBTR)"],
    "berth_time_hrs":             ["Berth Time(hours):ATU-ATB"],
    "assured_port_time_pct":      ["AssuredPortTimeAchieved(%)"],
    "carbon_tonnes":              ["Carbon Abatement (Tonnes)"],
    "bunker_saved_usd":           ["Bunker Saved(USD)"],
    "week":                       ["Week"],
    "date":                       ["ATB(LocalTime)"],
    "group_bu":                   ["BU"],
}

ALIASES["week"] = ["Week"]

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # exact match first
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lowermap = {c.lower(): c for c in df.columns if isinstance(c, str)}
    for c in candidates:
        if isinstance(c, str) and c.lower() in lowermap:
            return lowermap[c.lower()]
    return None

def _try_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        # If your CSV uses a different delimiter, set sep=";" etc.
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def load_data(path: Union[str, Path] = DATA_PATH):
    global _df, _num_columns, _loaded_path
    p = Path(path)
    _df = _try_read(p)
    _df.columns = _df.columns.map(lambda c: c.strip() if isinstance(c, str) else c)
    _df = _ensure_week(_df)            # <-- derive Week
    _df = _ensure_region(_df)   # 👈 add this
    _num_columns = _coerce_numeric(_df) # <-- then coerce numerics
    _loaded_path = p
    return {"ok": True, "rows": len(_df), "cols": list(_df.columns), "path": str(p)}


# eager load (okay to fail softly at boot)
try:
    load_data()
except Exception as e:
    print("⚠️ Data not loaded yet:", e)

def get_basic_info():
    if _df.empty:
        return {"error": "Dataset not loaded properly."}
    return {
        "columns": list(_df.columns),
        "row_count": int(len(_df)),
        "numeric_columns": _num_columns,
        "source_path": str(_loaded_path) if _loaded_path else None,
    }

def apply_filters(filters: Optional[Dict[str, List[Union[str, int, float]]]] = None) -> pd.DataFrame:
    if _df.empty or not filters:
        return _df.copy()
    d = _df.copy()
    for col, allowed in (filters or {}).items():
        if col not in d.columns or not isinstance(allowed, (list, tuple, set)):
            continue
        vals = allowed

        # Case-insensitive, whitespace-tolerant matching for string-like columns
        if pd.api.types.is_object_dtype(d[col]):
            left  = d[col].astype(str).str.strip().str.upper()
            right = pd.Series(list(vals), dtype="object").astype(str).str.strip().str.upper()
            d = d[left.isin(set(right))]
        else:
            d = d[d[col].isin(list(vals))]
    return d


# ========= Analytics helpers =========

def _arrival_col() -> Optional[str]:
    if _df.empty:
        return None
    return _pick_col(_df, ALIASES["arrival_accuracy"])  # no keyword fallback

def kpi_snapshot(filters: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    """
    Compute a compact KPI set.
    - ArrivalAccuracy(FinalBTR): reported as a PERCENT (0–100), assuming the column has been
      coerced to 0/1 (Y/N mapped earlier in _coerce_numeric).
    - Other numeric KPIs are reported as float means or sums.
    """
    if _df.empty:
        return {"error": "Dataset not loaded properly."}

    d = apply_filters(filters)

    def avg(alias_key: str) -> Optional[float]:
        col = _pick_col(d, ALIASES[alias_key])
        if not col:
            return None
        vals = pd.to_numeric(d[col], errors="coerce")
        vals = vals.dropna()
        if vals.empty:
            return None
        # Special case: arrival accuracy is binary 0/1 -> report percent
        if alias_key == "arrival_accuracy":
            return round(float(vals.mean() * 100.0), 2)
        return round(float(vals.mean()), 3)

    def total(alias_key: str) -> Optional[float]:
        col = _pick_col(d, ALIASES[alias_key])
        if not col:
            return None
        vals = pd.to_numeric(d[col], errors="coerce").dropna()
        if vals.empty:
            return None
        return round(float(vals.sum()), 3)

    out = {
        "arrival_accuracy_avg":        avg("arrival_accuracy"),      # percent
        "berth_time_avg_hrs":          avg("berth_time_hrs"),
        "assured_port_time_pct":       avg("assured_port_time_pct"),
        "carbon_total_t":              total("carbon_tonnes"),
        "bunker_saved_usd":            total("bunker_saved_usd"),
        "filters_applied":             filters or {},
    }
    return out


def summarize_metric(metric: str, filters: Optional[Dict] = None) -> Dict[str, Optional[Union[str, float]]]:
    """
    WoW summary for a given metric.
    - Requires a 'Week' column (derived earlier).
    - If metric is ArrivalAccuracy(FinalBTR) interpreted as 0/1, reports current/previous as PERCENT
      and delta as PERCENTAGE POINTS; else delta is % change vs previous mean.
    """
    if _df.empty:
        return {"error": "Dataset not loaded properly."}
    if metric not in _df.columns:
        return {"error": f"Metric '{metric}' not found in dataset."}

    d = apply_filters(filters)

    # resolve week column
    week_col = _pick_col(d, ALIASES["week"]) or _pick_col(d, ["Week"])
    if not week_col:
        return {"error": "No 'week' column found in dataset."}

    # ensure numeric
    if not pd.api.types.is_numeric_dtype(d[metric]):
        d[metric] = pd.to_numeric(d[metric], errors="coerce")

    # collect weeks (sortable)
    weeks = [w for w in pd.unique(d[week_col]) if pd.notna(w)]
    try:
        weeks_sorted = sorted(weeks)
    except Exception:
        weeks_sorted = sorted(map(str, weeks))
    if len(weeks_sorted) < 2:
        return {"error": "Not enough weeks to compute trend."}

    latest, prev = weeks_sorted[-1], weeks_sorted[-2]
    cur = d[d[week_col] == latest]
    old = d[d[week_col] == prev]

    cur_mean = cur[metric].astype(float).mean()
    old_mean = old[metric].astype(float).mean()

    # Is this our arrival-accuracy metric?
    is_arrival = (metric == (ALIASES["arrival_accuracy"][0] if ALIASES.get("arrival_accuracy") else metric))

    if is_arrival:
        # Treat as 0..1; present as percentages; delta in percentage points
        cur_pct = None if pd.isna(cur_mean) else cur_mean * 100.0
        old_pct = None if pd.isna(old_mean) else old_mean * 100.0
        delta_pp = None if (cur_pct is None or old_pct is None) else (cur_pct - old_pct)
        return {
            "metric": metric,
            "latest_week": latest,
            "previous_week": prev,
            "current_mean": None if cur_pct is None else round(float(cur_pct), 2),
            "previous_mean": None if old_pct is None else round(float(old_pct), 2),
            "delta_percent": None if delta_pp is None else round(float(delta_pp), 2),  # percentage points
            "unit": "percentage_points",
            "filters_applied": filters or {},
        }

    # Default: percent change vs previous mean
    if pd.isna(old_mean) or old_mean == 0:
        delta_pct = np.nan
    else:
        delta_pct = (cur_mean - old_mean) / old_mean * 100.0

    return {
        "metric": metric,
        "latest_week": latest,
        "previous_week": prev,
        "current_mean": None if pd.isna(cur_mean) else round(float(cur_mean), 3),
        "previous_mean": None if pd.isna(old_mean) else round(float(old_mean), 3),
        "delta_percent": None if pd.isna(delta_pct) else round(float(delta_pct), 2),
        "filters_applied": filters or {},
    }


def anomalies_by_group(metric: str, group_col: Optional[str] = None, top_n: int = 3, filters: Optional[Dict] = None):
    if _df.empty:
        return {"error": "Dataset not loaded properly."}
    d = apply_filters(filters)
    if group_col is None:
        group_col = _pick_col(d, ALIASES["group_bu"]) or "BU"
    if group_col not in d.columns:
        return {"error": f"Group column '{group_col}' not found."}
    if metric not in d.columns:
        return {"error": f"Metric '{metric}' not found."}

    vals = pd.to_numeric(d[metric], errors="coerce")
    d = d.assign(_metric=vals).dropna(subset=["_metric"])
    if d.empty:
        return {"error": f"No numeric values for metric '{metric}' after coercion."}

    by_grp = d.groupby(group_col)["_metric"].mean().reset_index()
    mu, sd = by_grp["_metric"].mean(), by_grp["_metric"].std(ddof=0)
    by_grp["z"] = 0.0 if (pd.isna(sd) or sd == 0) else (by_grp["_metric"] - mu) / sd

    highest = by_grp.nlargest(top_n, "z").to_dict(orient="records")
    lowest = by_grp.nsmallest(top_n, "z").to_dict(orient="records")
    return {
        "metric": metric,
        "group_col": group_col,
        "highest": highest,
        "lowest": lowest,
        "filters_applied": filters or {},
    }

# ========= LangChain parallel workers =========

def _trend_worker(ctx):
    metric = ctx.get("metric_override") or _arrival_col()
    if not metric:
        return {"Trend": {"error": "Arrival Accuracy column not found."}}
    return {"Trend": summarize_metric(metric, ctx.get("filters"))}

def _kpi_worker(ctx):
    return {"KPI": kpi_snapshot(ctx.get("filters"))}

def _anomaly_worker(ctx):
    metric = ctx.get("metric_override") or _arrival_col()
    if not metric:
        return {"Anomalies": {"error": "Arrival Accuracy column not found."}}
    return {"Anomalies": anomalies_by_group(metric, filters=ctx.get("filters"))}

KPI_CHAIN = RunnableLambda(_kpi_worker)
ANOM_CHAIN = RunnableLambda(_anomaly_worker)
TREND_CHAIN = RunnableLambda(_trend_worker)

PARALLEL = RunnableParallel(KPI=KPI_CHAIN, Anomalies=ANOM_CHAIN, Trend=TREND_CHAIN)

# ========= Prompts & final composition =========

ACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are PSA's network strategist. Based on KPIs, anomalies and trend, "
     "output 3–5 short, imperative, measurable actions with clear owners and timeframes. "
     "Use relative timeframes (e.g., 'within 2 weeks')."),
    ("human", "KPI JSON:\n{kpi}\n\nAnomalies JSON:\n{anom}\n\nTrend JSON:\n{trend}\n\nUser question:\n{q}")
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Senior operations analyst. Write 6–10 concise bullets: "
     "(1) KPIs vs typical, (2) unusually good/bad, (3) likely drivers, "
     "(4) specific next actions with owners + timeframes. Plain English."),
    ("human", "Inputs JSON:\n{inputs}\n\nUser question:\n{q}")
])

def _suggest_actions(par_out: dict, question: str) -> str:
    llm = get_llm()
    msg = ACTION_PROMPT.format_messages(
        kpi=json.dumps(par_out.get("KPI"), default=str),
        anom=json.dumps(par_out.get("Anomalies"), default=str),
        trend=json.dumps(par_out.get("Trend"), default=str),
        q=question
    )
    return llm.invoke(msg).content

def explain(question: str, ui_filters: Optional[Dict] = None, metric_override: Optional[str] = None):
    """
    Run parallel analytics → generate actions → final exec summary.
    Returns: (merged_dict, summary_text)
    """
    ctx = {"filters": ui_filters or {}, "metric_override": metric_override}
    par = PARALLEL.invoke(ctx)  # KPI / Anomalies / Trend in parallel
    actions = _suggest_actions(par, question)
    merged = {**par, "Actions": actions}

    llm = get_llm()
    summary = llm.invoke(SUMMARY_PROMPT.format_messages(
        inputs=json.dumps(merged, default=str), q=question
    )).content
    return merged, summary

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

def get_agent():
    """
    Return an agent-like object with .invoke({"input": question}) -> dict|str.
    Order of attempts:
    1) Newer LC: create_openai_tools_agent + AgentExecutor
    2) Mid LC:  initialize_agent(AgentType.OPENAI_FUNCTIONS)
    3) Fallback: simple_rules_agent (manual routing, no LC agent)
    """
    llm = get_llm()
    tools = [kpi_tool, trend_tool, anomalies_tool, distinct_tool, peek_tool]

    # ---- Attempt 1: Newer LC (0.2.x) ----
    try:
        from langchain.agents import create_openai_tools_agent, AgentExecutor
        try:
            from langchain import hub
            prompt = hub.pull("hwchase17/openai-tools-agent")
        except Exception:
            # Build a minimal prompt if hub unavailable
            from langchain_core.prompts import ChatPromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a precise PSA ops analyst. Use the available tools to answer. "
                 "Prefer KPI -> Trend -> Anomalies. Parse simple filters like 'APAC' -> {\"Region\":[\"APAC\"]} "
                 "and 'SINGAPORE' -> {\"BU\":[\"SINGAPORE\"]}. Be concise."),
                ("human", "{input}"),
            ])
        agent_runnable = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent_runnable, tools=tools, verbose=False)
    except Exception:
        pass

    # ---- Attempt 2: Older LC (0.0.x / 0.1.x): initialize_agent ----
    try:
        from langchain.agents import initialize_agent, AgentType
        return initialize_agent(
            tools=tools, llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,  # or TOOL_CALLING on some builds
            verbose=False,
            handle_parsing_errors=True,
        )
    except Exception:
        pass

    # ---- Attempt 3: Fallback manual "agent" (no LC agent) ----
    class SimpleRulesAgent:
        """Very small router so you're never blocked."""
        def invoke(self, inp: Dict[str, Any]):
            q = inp["input"] if isinstance(inp, dict) else str(inp)
            text = q.strip().upper()

            # naive filter inference
            filters = {}
            # detect 'APAC' / 'EMEA' / 'AMERICAS' / 'ME' tokens
            for region in ["APAC", "EMEA", "AMERICAS", "ME", "MIDDLE EAST"]:
                if region in text:
                    filters = {"Region": [region if region != "MIDDLE EAST" else "ME"]}
                    break
            # detect a BU by peeking common city names (adjust as needed)
            for bu in ["SINGAPORE", "BUSAN", "TIANJIN", "LAEMCHABANG", "JAKARTA", "MUMBAI", "ANTWERP", "PANAMACITY", "DAMMAM"]:
                if bu in text:
                    filters = {"BU": [bu]}
                    break

            # choose metric
            metric = None
            if "ASSURED" in text:
                metric = "AssuredPortTimeAchieved(%)"
            elif "BERTH" in text:
                metric = "Berth Time(hours):ATU-ATB"
            elif "ARRIVAL" in text or "ACCURACY" in text:
                metric = "ArrivalAccuracy(FinalBTR)"

            # call tools directly
            import json as _json
            out = {"filters": filters, "metric": metric}

            out["kpi"] = kpi_snapshot(filters)
            if metric:
                out["trend"] = summarize_metric(metric, filters)
                out["anom"] = anomalies_by_group(metric, filters=filters)
            else:
                # default to arrival accuracy if not specified
                dm = "ArrivalAccuracy(FinalBTR)"
                out["trend"] = summarize_metric(dm, filters)
                out["anom"] = anomalies_by_group(dm, filters=filters)

            # turn into a short answer with LLM for readability
            llm_msg = (
                "Summarize these analytics for a stakeholder in 6 bullets. "
                "Focus on KPI levels, week-over-week change, and top/bottom BUs. "
                f"\nDATA:\n{_json.dumps(out, default=str)}"
            )
            summary = get_llm().invoke(llm_msg).content
            return {"output": summary, "data": out}

    return SimpleRulesAgent()

# ========= Optional: quick CLI self-test =========
if __name__ == "__main__":
    print("→ Data info:", get_basic_info())
    try:
        out, summ = explain("Explain APAC this week.", {"BU": ["APAC"]})
        print("→ Details keys:", list(out.keys()))
        print("→ Summary:\n", summ)
    except Exception as e:
        print("❌ explain() failed:", e)
