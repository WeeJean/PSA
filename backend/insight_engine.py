import pandas as pd

# === Load dataset ===
# If you copied the Excel file into backend/, use this:
DATA_PATH = "data/Reference sample data.xlsx"

# Otherwise, use the full path like:
# DATA_PATH = r"C:\Users\kohweejean\Downloads\Reference sample data.xlsx"

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    print("❌ Error loading Excel file:", e)
    df = pd.DataFrame()

# === Functions ===
def get_basic_info():
    """Return basic dataset info."""
    if df.empty:
        return {"error": "Dataset not loaded properly."}
    return {
        "columns": list(df.columns),
        "row_count": len(df)
    }

def summarize_metric(metric):
    """Summarize average and week-to-week trend for a given metric."""
    if df.empty:
        return {"error": "Dataset not loaded properly."}
    if metric not in df.columns:
        return {"error": f"Metric '{metric}' not found in dataset."}

    # Try to detect 'week' column for time-based comparison
    week_col = None
    for col in df.columns:
        if "week" in col.lower():
            week_col = col
            break

    if not week_col:
        return {"error": "No 'week' column found in dataset."}

    weeks = sorted(df[week_col].unique())
    if len(weeks) < 2:
        return {"error": "Not enough weeks to compute trend."}

    latest, prev = weeks[-1], weeks[-2]
    cur = df[df[week_col] == latest]
    old = df[df[week_col] == prev]

    cur_mean = cur[metric].mean()
    old_mean = old[metric].mean()
    delta = (cur_mean - old_mean) / old_mean * 100

    return {
        "metric": metric,
        "latest_week": latest,
        "previous_week": prev,
        "current_mean": round(cur_mean, 3),
        "previous_mean": round(old_mean, 3),
        "delta_percent": round(delta, 2)
    }
