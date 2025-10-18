import pandas as pd

# === Load dataset ===
DATA_PATH = "data/Reference sample data.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    print("❌ Error loading Excel file:", e)
    df = pd.DataFrame()

# === Detect time column & preprocess ===
def _detect_time_column():
    """Find a likely date/time column in the dataset."""
    for col in df.columns:
        if any(k in col.lower() for k in ["time", "date", "timestamp"]):
            return col
    return None

time_col = _detect_time_column()

if time_col:
    # Try to parse date/time flexibly
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=[time_col])  # drop invalid dates
    df["day"] = df[time_col].dt.date
    df["week"] = df[time_col].dt.isocalendar().week
    df["month"] = df[time_col].dt.to_period("M").astype(str)
    df["year"] = df[time_col].dt.year
    print(f"✅ Parsed {time_col}: {len(df)} valid rows, {df[time_col].min()} → {df[time_col].max()}")
else:
    print("⚠️ No time column detected — time-based grouping disabled.")


# === Helper functions ===
def get_basic_info():
    """Return basic dataset info."""
    if df.empty:
        return {"error": "Dataset not loaded properly."}
    return {
        "columns": list(df.columns),
        "row_count": len(df)
    }


def _group_by_time(level: str):
    """Return grouped DataFrame by a time level."""
    if level not in ["day", "week", "month", "year"]:
        raise ValueError("Invalid time level. Choose from day, week, month, year.")
    if level not in df.columns:
        raise ValueError(f"'{level}' column not found; dataset may lack time information.")
    return df.groupby(level)


def summarize_metric(metric, level="week"):
    """
    Summarize average and trend for a given metric.
    Handles numeric, percentage, and Y/N-type categorical data.
    Automatically groups by time (day/week/month/year).
    """
    if df.empty:
        return {"error": "Dataset not loaded properly."}

    # --- fuzzy match metric name ---
    columns_lower = {c.lower(): c for c in df.columns}
    metric_lower = metric.lower()
    match = next((columns_lower[c] for c in columns_lower if metric_lower in c), None)
    if not match:
        return {"error": f"Metric '{metric}' not found in dataset."}
    metric = match

    try:
        grouped = _group_by_time(level)

        # --- Clean and normalize metric values ---
        col = df[metric].astype(str).str.strip()

        # Case 1: Y/N categorical → map to 1/0
        if set(col.dropna().unique()) <= {"Y", "N", "y", "n"}:
            clean_metric = col.str.upper().map({"Y": 1, "N": 0})

        # Case 2: Percent values like "95%" → numeric
        elif col.str.contains("%").any():
            clean_metric = pd.to_numeric(col.str.replace("%", ""), errors="coerce")

        # Case 3: Pure numeric
        else:
            clean_metric = pd.to_numeric(col, errors="coerce")

        df["clean_metric"] = clean_metric

        # --- Compute mean per time group ---
        trend = grouped["clean_metric"].mean().dropna()
        if len(trend) < 2:
            return {"error": f"Not enough {level}s to compute trend for '{metric}'."}

        latest, prev = trend.index[-1], trend.index[-2]
        cur_mean, old_mean = trend.iloc[-1], trend.iloc[-2]
        delta = (cur_mean - old_mean) / old_mean * 100 if old_mean != 0 else float("nan")

        return {
            "metric": metric,
            "time_level": level,
            "latest_period": str(latest),
            "previous_period": str(prev),
            "current_mean": float(round(cur_mean, 3)),
            "previous_mean": float(round(old_mean, 3)),
            "delta_percent": float(round(delta, 2))
        }

    except Exception as e:
        return {"error": f"Error analyzing '{metric}': {e}"}
