# insight_engine_v2.py — month-by-month, flexible column mapper

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ======= Flexible header normalization =======
def _norm_header(h: str) -> str:
    """
    Normalize column header strings so we can match across variants like:
      'data[Arrival Accuracy (Final BTR)]'  → 'ArrivalAccuracy(FinalBTR)'
      'data[Bunker Saved (USD)]'            → 'BunkerSaved(USD)'
      'data[Berth Time (hours): ATU-ATB]'   → 'BerthTime(hours):ATU-ATB'
    Strategy:
      - drop leading 'data[' and trailing ']'
      - collapse whitespace
      - remove spaces around punctuation, then special-case a few
    """
    if not isinstance(h, str):
        return str(h)
    s = h.strip()
    # strip data[...] wrapper
    s = re.sub(r'^\s*data\[(.+?)\]\s*$', r'\1', s, flags=re.I)
    # collapse spaces
    s = re.sub(r'\s+', ' ', s)
    # normalize spaces around punctuation
    s = s.replace(' (', '(').replace('( ', '(').replace(' )', ')')
    s = s.replace(' :', ':').replace(': ', ':')
    s = s.replace(' - ', '-').replace('–', '-').replace('—', '-')
    # remove spaces
    s = s.replace(' ', '')
    return s

# Canonical keys we use internally
CANONICAL = {
    "BU": ["BU"],
    "Region": ["Region"],
    "ArrivalAccuracy(FinalBTR)": ["ArrivalAccuracy(FinalBTR)"],
    "AssuredPortTimeAchieved(%)": ["AssuredPortTimeAchieved(%)"],
    "BunkerSaved(USD)": ["BunkerSaved(USD)", "BunkerSavedUSD", "BunkerSaved"],
    "CarbonAbatement(Tonnes)": ["CarbonAbatement(Tonnes)"],
    "BerthTime(hours):ATU-ATB": ["BerthTime(hours):ATU-ATB"],
    # Date/time variants
    "ATB(LocalTime)": ["ATB(LocalTime)", "ATB(Local)"],
    "ABT(LocalTime)": ["ABT(LocalTime)"],
    "ATU(LocalTime)": ["ATU(LocalTime)"],
    "FinalBTR(LocalTime)": ["FinalBTR(LocalTime)"],
    # Explicit year/month if provided
    "Year": ["Year"],
    "Month": ["Month"],
}

# Regex-based fallbacks for friendly names
REGEX_FALLBACKS = {
    "ArrivalAccuracy(FinalBTR)": re.compile(r"^ArrivalAccuracy\(FinalBTR\)|^ArrivalAccuracy|^OnTime$", re.I),
    "AssuredPortTimeAchieved(%)": re.compile(r"AssuredPortTimeAchieved|\(%\)$", re.I),
    "BunkerSaved(USD)": re.compile(r"^BunkerSaved|Bunker\(USD\)|BunkerSaved\(USD\)", re.I),
    "CarbonAbatement(Tonnes)": re.compile(r"^CarbonAbatement|Tonnes$", re.I),
    "BerthTime(hours):ATU-ATB": re.compile(r"^BerthTime\(hours\):ATU-ATB|BerthTime", re.I),
    "ATB(LocalTime)": re.compile(r"^ATB\(LocalTime\)|ATB\(Local\)|ATB$", re.I),
}

# Business Unit → Region mapping (extend as needed)
BU_TO_REGION = {
    "ANTWERP": "EMEA",
    "BUSAN": "APAC",
    "DAMMAM": "ME",
    "JAKARTA": "APAC",
    "LAEMCHABANG": "APAC",
    "MUMBAI": "APAC",
    "PANAMACITY": "AMERICAS",
    "SINGAPORE": "APAC",
    "TIANJIN": "APAC",
}

# ======= Environment / LLM (optional scaffolding) =======
load_dotenv(Path(__file__).with_name(".env"))

try:
    from langchain_openai import AzureChatOpenAI
    from langchain_core.runnables import RunnableParallel, RunnableLambda
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    AzureChatOpenAI = None
    RunnableParallel = None
    RunnableLambda = None
    ChatPromptTemplate = None

API_VERSION = os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
RAW_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

_LLM = None
def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM
    if AzureChatOpenAI is None:
        raise RuntimeError("LangChain/AzureChatOpenAI not available")
    _LLM = AzureChatOpenAI(
        azure_deployment=DEPLOYMENT,
        azure_endpoint=RAW_ENDPOINT,
        api_key=AZURE_KEY,
        api_version=API_VERSION,
        temperature=0.2,
    )
    _LLM.invoke("ping")
    return _LLM

# ======= Data loading with flexible mapping =======
DATA_PATH = Path("data_data.csv")

_df: pd.DataFrame = pd.DataFrame()
_num_columns: List[str] = []
_loaded_path: Optional[Path] = None
_header_map: Dict[str, str] = {}  # canonical -> actual column name

def _build_header_map(df: pd.DataFrame) -> Dict[str, str]:
    normed = {c: _norm_header(str(c)) for c in df.columns}
    rev = {v: c for c, v in normed.items()}
    mapping = {}

    # exact canonical list first
    def try_set(key: str, candidates: List[str]):
        for cand in candidates:
            n = _norm_header(cand)
            if n in rev:
                mapping[key] = rev[n]
                return True
        return False

    for k, cand_list in CANONICAL.items():
        try_set(k, cand_list)

    # regex fallbacks
    for k, rx in REGEX_FALLBACKS.items():
        if k in mapping:
            continue
        for original_norm, original_col in normed.items():
            # original_norm is the original column name, original_col is normalized; fix variable names:
            pass
    # Correct the above minor mistake:
    mapping2 = dict(mapping)
    for k, rx in REGEX_FALLBACKS.items():
        if k in mapping2:
            continue
        for col_name in df.columns:
            if rx.search(_norm_header(col_name)):
                mapping2[k] = col_name
                break
    return mapping2

def _ensure_region(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" in df.columns:
        return df
    bu_actual = _header_map.get("BU")
    if bu_actual and bu_actual in df.columns:
        df["Region"] = (
            df[bu_actual].astype(str).str.strip().str.upper().map(BU_TO_REGION).fillna("OTHER")
        )
    return df

def _coerce_numeric(df: pd.DataFrame) -> list[str]:
    numeric_cols: list[str] = []
    n_rows = len(df)
    min_valid = max(3, int(0.10 * n_rows))

    # Arrival accuracy coercion (Y/N → 1/0)
    aa_key = "ArrivalAccuracy(FinalBTR)"
    if aa_key in _header_map:
        col = _header_map[aa_key]
        if col in df.columns:
            s = df[col]
            mapped = s.astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
            coerced = pd.to_numeric(mapped, errors="coerce")
            df[col] = coerced
            if coerced.notna().sum() >= min_valid:
                numeric_cols.append(col)

    for col in df.columns:
        if col == _header_map.get(aa_key, ""):
            continue
        if isinstance(col, str) and col.startswith("Unnamed:"):
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(col)
            continue
        if s.dtype == "O":
            # prevent parsing obvious datetime columns
            if any(k in _header_map and _header_map[k] == col for k in ["ATB(LocalTime)", "ABT(LocalTime)", "ATU(LocalTime)", "FinalBTR(LocalTime)"]):
                # parse datetime, but don't add to numeric
                df[col] = pd.to_datetime(s, errors="coerce", dayfirst=True)
                continue
            cleaned = (
                s.astype(str).str.strip()
                 .replace({"": None, "NA": None, "NaN": None}, regex=False)
                 .str.replace(",", "", regex=False)
                 .str.replace("%", "", regex=False)
            )
            coerced = pd.to_numeric(cleaned, errors="coerce")
            if coerced.notna().sum() >= min_valid:
                df[col] = coerced
                numeric_cols.append(col)
    return numeric_cols

def _ensure_month(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a Month key 'YYYY-MM' exists using explicit Year/Month or derived from ATB(LocalTime)."""
    if "MonthKey" in df.columns:
        return df

    # Prefer explicit Year/Month columns if present
    year_col = _header_map.get("Year")
    month_col = _header_map.get("Month")
    if year_col in df.columns if year_col else False:
        if month_col in df.columns if month_col else False:
            y = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
            m = pd.to_numeric(df[month_col], errors="coerce").astype("Int64")
            mask = y.notna() & m.notna()
            mk = pd.Series([None]*len(df), dtype="object")
            mk.loc[mask] = (
                y.astype(int).astype(str).str.zfill(4) + "-" + m.astype(int).astype(str).str.zfill(2)
            ).loc[mask]
            df["MonthKey"] = mk
            return df

    # Else derive from best datetime column
    for key in ["ATB(LocalTime)", "FinalBTR(LocalTime)", "ABT(LocalTime)", "ATU(LocalTime)"]:
        col = _header_map.get(key)
        if col and col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            mk = dt.dt.to_period("M").astype(str)
            df["MonthKey"] = mk
            return df

    # Fallback: nothing to do
    df["MonthKey"] = None
    return df

def _try_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")

def load_data(path: Union[str, Path] = DATA_PATH):
    global _df, _num_columns, _loaded_path, _header_map
    p = Path(path)
    _df = _try_read(p)
    # build header map
    _header_map = _build_header_map(_df)
    # strip whitespace on columns (preserve original names)
    _df.columns = [c for c in _df.columns]
    _df = _ensure_month(_df)
    _df = _ensure_region(_df)
    _num_columns = _coerce_numeric(_df)
    _loaded_path = p
    return {
        "ok": True,
        "rows": len(_df),
        "cols": list(_df.columns),
        "path": str(p),
        "resolved_headers": _header_map,
    }

# eager load (soft)
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
        "resolved_headers": _header_map,
    }

# =========== Filters & helpers ===========
def apply_filters(filters: Optional[Dict[str, List[Union[str, int, float]]]] = None) -> pd.DataFrame:
    if _df.empty or not filters:
        return _df.copy()
    d = _df.copy()
    for col, allowed in (filters or {}).items():
        # map canonical names to actual names if provided
        actual = _header_map.get(col, col)
        if actual not in d.columns:
            continue
        vals = allowed
        if pd.api.types.is_object_dtype(d[actual]):
            left = d[actual].astype(str).str.strip().str.upper()
            right = pd.Series(list(vals), dtype="object").astype(str).str.strip().str.upper()
            d = d[left.isin(set(right))]
        else:
            d = d[d[actual].isin(list(vals))]
    return d

def _col(name: str) -> Optional[str]:
    """Return actual column name for a canonical name."""
    return _header_map.get(name)

# =========== KPI snapshot (means/sums over filters) ===========
def kpi_snapshot(filters: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    if _df.empty:
        return {"error": "Dataset not loaded properly."}
    d = apply_filters(filters)

    def _mean(canon: str, pct: bool=False) -> Optional[float]:
        col = _col(canon)
        if not col or col not in d.columns:
            return None
        vals = pd.to_numeric(d[col], errors="coerce").dropna()
        if vals.empty:
            return None
        m = float(vals.mean())
        return round(m * (100.0 if pct else 1.0), 3)

    def _sum(canon: str) -> Optional[float]:
        col = _col(canon)
        if not col or col not in d.columns:
            return None
        vals = pd.to_numeric(d[col], errors="coerce").dropna()
        if vals.empty:
            return None
        return round(float(vals.sum()), 3)
    
    def _aggregate_top_n(
        group_col_canon: str, 
        metric_col_canon: str, 
        top_n: int = 3
    ) -> List[Dict[str, float]]:
        
        gcol = _col(group_col_canon) # e.g., 'BU' or 'Vessel'
        mcol = _col(metric_col_canon) # e.g., 'Carbon Abatement (Tonnes)'
        
        if not gcol or not mcol or gcol not in d.columns or mcol not in d.columns:
            return []
        
        # 1. Prepare metric column (using existing cleaning method)
        d_temp = d[[gcol, mcol]].copy()
        d_temp['metric_val'] = pd.to_numeric(d_temp[mcol], errors='coerce').fillna(0)
        
        # 2. Group by group_col and sum the metric
        agg = d_temp.groupby(gcol)['metric_val'].sum().reset_index()
        
        # 3. Sort (descending) and select top N
        top = agg.nlargest(top_n, 'metric_val')
        
        # 4. Rename columns for clear output and return
        return top.rename(
            columns={gcol: group_col_canon, 'metric_val': metric_col_canon}
        ).to_dict(orient='records')

    # 1. Base Metrics
    out = {
        "total_rotations": len(d), 
        "arrival_accuracy_avg_pct": _mean("ArrivalAccuracy(FinalBTR)", pct=True),
        "berth_time_avg_hours": _mean("BerthTime(hours):ATU-ATB"),
        "assured_port_time_pct": _mean("AssuredPortTimeAchieved(%)"),
        "carbon_total_tonnes": _sum("CarbonAbatement(Tonnes)"),
        "bunker_saved_usd": _sum("BunkerSaved(USD)"),
        "filters_applied": filters or {},
    }

    # 2. Corrected Top Performer Metrics
    
    # Top 3 Port Units (BU) by Bunker Savings
    out["top_bu_by_bunker_savings"] = _aggregate_top_n(
        group_col_canon="BU", 
        metric_col_canon="BunkerSaved(USD)", # Changed to canonical name (no space)
        top_n=3
    )
    
    # Top 3 Vessels by Bunker Savings
    out["top_vessel_by_bunker_savings"] = _aggregate_top_n( # Key name changed
        group_col_canon="Vessel", 
        metric_col_canon="BunkerSaved(USD)", # Metric changed from CarbonAbatement to BunkerSaved(USD)
        top_n=3
    )
    return out

# =========== Month-over-Month summarizer ===========
def summarize_metric(metric: str, filters: Optional[Dict] = None, level: str = "month") -> Dict[str, Optional[Union[str, float]]]:
    """
    Summarize a metric over time (default: month). Computes current vs previous period and percent/pp delta.
    For ArrivalAccuracy(FinalBTR) we treat underlying data as 0/1 and report percentage.
    """
    if _df.empty:
        return {"error": "Dataset not loaded properly."}
    d = apply_filters(filters)

    # resolve actual metric column
    mcol = _col(metric) or metric
    if mcol not in d.columns:
        return {"error": f"Metric '{metric}' not found in dataset."}

    # choose time key
    if level != "month":
        level = "month"
    tkey = "MonthKey"
    if tkey not in d.columns:
        return {"error": "No 'MonthKey' column available."}

    # coerce numeric
    vals = pd.to_numeric(d[mcol], errors="coerce")
    s = d.assign(_v=vals).dropna(subset=["_v"])
    if s.empty:
        return {"error": f"No numeric values for metric '{metric}' after coercion."}

    # aggregate mean per month
    g = s.groupby(tkey)["_v"].mean().reset_index()
    g = g[g[tkey].notna()].sort_values(tkey)
    if len(g) < 2:
        return {"error": "Not enough months to compute trend."}

    latest, prev = g.iloc[-1], g.iloc[-2]
    cur_mean = float(latest["_v"])
    old_mean = float(prev["_v"])

    # arrival accuracy treated as percent
    is_arrival = metric in ("ArrivalAccuracy(FinalBTR)", _col("ArrivalAccuracy(FinalBTR)"))
    if is_arrival:
        cur_pct = cur_mean * 100.0
        old_pct = old_mean * 100.0
        delta_pp = cur_pct - old_pct
        unit = "percentage_points"
        return {
            "metric": metric,
            "latest_month": latest[tkey],
            "previous_month": prev[tkey],
            "current_mean": round(cur_pct, 2),
            "previous_mean": round(old_pct, 2),
            "delta_percent": round(delta_pp, 2),
            "unit": unit,
            "filters_applied": filters or {},
        }
    else:
        # % change vs previous
        if old_mean == 0 or np.isnan(old_mean):
            delta_pct = np.nan
        else:
            delta_pct = (cur_mean - old_mean) / old_mean * 100.0
        return {
            "metric": metric,
            "latest_month": latest[tkey],
            "previous_month": prev[tkey],
            "current_mean": round(cur_mean, 3),
            "previous_mean": round(old_mean, 3),
            "delta_percent": None if np.isnan(delta_pct) else round(delta_pct, 2),
            "filters_applied": filters or {},
        }



# ========= Optional: quick CLI self-test =========
if __name__ == "__main__":
    print("→ Data info:", get_basic_info())
