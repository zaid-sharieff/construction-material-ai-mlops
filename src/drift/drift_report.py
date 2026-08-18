# src/drift/drift_report.py

import pandas as pd
from pathlib import Path
from evidently import Report
from evidently.metrics import ValueDrift
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.database.db import SessionLocal
from src.database.models import Detection, Quantification

# ---------------- CONFIG ----------------
MIN_SAMPLES = 30
OUTPUT_DIR = Path("drift_report")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "drift_report.html"

# Thresholds
MAX_NAN_RATIO = 0.4   # >40% NaN → skip
MIN_UNIQUE = 2        # constant columns → skip


# ---------------- HELPERS ----------------
def compute_bbox_area(d):
    return abs(d.x2 - d.x1) * abs(d.y2 - d.y1)


def column_is_valid(ref, cur, col):
    """Strict validation to avoid Evidently runtime warnings"""

    if col not in ref.columns or col not in cur.columns:
        return False

    ref_col = ref[col]
    cur_col = cur[col]

    # Drop NaNs
    ref_clean = ref_col.dropna()
    cur_clean = cur_col.dropna()

    # Empty after cleaning
    if ref_clean.empty or cur_clean.empty:
        return False

    # Too many NaNs
    if ref_col.isna().mean() > MAX_NAN_RATIO:
        return False
    if cur_col.isna().mean() > MAX_NAN_RATIO:
        return False

    # Constant / no variance
    if ref_clean.nunique() < MIN_UNIQUE:
        return False
    if cur_clean.nunique() < MIN_UNIQUE:
        return False

    return True


# ---------------- DATA FETCH ----------------
def fetch_drift_data():
    db = SessionLocal()

    detections = db.query(Detection).order_by(Detection.id.asc()).all()
    quantifications = db.query(Quantification).order_by(Quantification.id.asc()).all()

    db.close()

    if len(detections) < MIN_SAMPLES:
        return None, None, "Not enough detections for drift detection"

    det_df = pd.DataFrame([
        {
            "material": d.material,
            "confidence": float(d.confidence) if d.confidence is not None else np.nan,
            "bbox_area": compute_bbox_area(d)
        }
        for d in detections
    ])

    quant_df = pd.DataFrame([
        {
            "count": q.count,
            "volume_m3": q.volume_m3,
            "mass_kg": q.mass_kg
        }
        for q in quantifications
    ])

    df = pd.concat(
        [det_df.reset_index(drop=True), quant_df.reset_index(drop=True)],
        axis=1
    )

    mid = len(df) // 2
    reference = df.iloc[:mid].copy()
    current = df.iloc[mid:].copy()

    return reference, current, None


# ---------------- DRIFT REPORT ----------------
def generate_drift_report():
    reference_df, current_df, error = fetch_drift_data()

    if error:
        return False, error

    candidate_columns = [
        "confidence",
        "bbox_area",
        "material",
        "count",
        "volume_m3",
        "mass_kg",
    ]

    metrics = []
    skipped = []

    for col in candidate_columns:
        if column_is_valid(reference_df, current_df, col):
            metrics.append(ValueDrift(column=col))
        else:
            skipped.append(col)

    if not metrics:
        return False, "No valid columns available for drift detection"

    report = Report(metrics=metrics)
    snapshot = report.run(
        reference_data=reference_df,
        current_data=current_df
    )
    snapshot.save_html(str(OUTPUT_PATH))

    msg = f"Drift report generated at {OUTPUT_PATH}"
    if skipped:
        msg += f" | Skipped columns: {skipped}"

    return True, msg


if __name__ == "__main__":
    success, message = generate_drift_report()
    print(message)
