"""
Inference: NDVI series -> crop prediction, using YOUR notebook logic.

At training time the crop is known, so the notebook picks CROP_CONFIG[crop].
At inference it isn't, so each crop's config is tried (detect -> validate with
that crop's RULES -> 13 features -> Random Forest). A candidate is accepted only
when the Random Forest agrees with the crop hypothesis that produced it; the
accepted candidate closest to the query date (then most confident) wins.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.lifecycle_logic import FEATURE_COLS, candidate_lifecycles

MODEL_PATH = Path(__file__).resolve().parents[1] / "model_weights" / "rf_model.joblib"
MIN_CONFIDENCE = 0.50          # below this -> "No confident crop match"
MAX_GAP_DAYS = 60              # lifecycle must contain / be within this of the query date
MIN_OBSERVATIONS = 8

_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} missing — run `python train_model.py` first.")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _d(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _gap_days(rec, q):
    if rec["Sowing_Date"] <= q <= rec["Harvest_Date"]:
        return 0
    return min(abs((q - rec["Sowing_Date"]).days), abs((q - rec["Harvest_Date"]).days))


def _series_payload(dates, ndvi, smoothed):
    sm = [None if np.isnan(x) else round(float(x), 4) for x in smoothed]
    return {"ndvi_dates": [_d(x) for x in dates],
            "ndvi_values": [round(float(x), 4) for x in ndvi],
            "smoothed": sm}


def predict_from_series(df: pd.DataFrame, ref_date, ground_truth: str | None = None) -> dict:
    """df: columns Date, NDVI. Returns the JSON payload the frontend renders."""
    b = load_model()
    clf, classes = b["model"], list(b["classes"])
    q = pd.Timestamp(ref_date)

    df = df.dropna().sort_values("Date")
    if len(df) < MIN_OBSERVATIONS:
        return {"error": f"Not enough NDVI observations ({len(df)}) around this point/date."}
    dates = pd.to_datetime(df["Date"].values.astype("datetime64[ns]"))
    ndvi = df["NDVI"].values.astype(float)

    cands = []
    for c in candidate_lifecycles(dates, ndvi):
        if not c["record"]["Is_Valid"] or c["features"] is None:
            continue
        gap = _gap_days(c["record"], q)
        if gap > MAX_GAP_DAYS:
            continue
        x = np.array([[c["features"][k] for k in FEATURE_COLS]])
        proba = clf.predict_proba(x)[0]
        top = classes[int(np.argmax(proba))]
        if top != c["hypothesis"]:            # RF must agree with the rule-set that produced it
            continue
        c.update(gap=gap, proba=proba, conf=float(proba.max()))
        cands.append(c)

    # for the chart when nothing matches, show the smoothing of the closest-looking config
    fallback = candidate_lifecycles(dates, ndvi, crops=["Paddy"])
    fb_smooth = fallback[0]["smoothed"] if fallback else pd.Series(ndvi).rolling(3, center=True, min_periods=1).mean().values
    if not fallback:
        fb_smooth = pd.Series(ndvi).rolling(3, center=True, min_periods=1).mean().values

    gt = (ground_truth or "").strip() or None
    cands = [c for c in cands if c["conf"] >= MIN_CONFIDENCE]
    if not cands:
        return {"predicted_crop": "No confident crop match", "confidence_pct": 0,
                "sowing_date": None, "peak_date": None, "harvest_date": None, "duration_days": None,
                "top3": [], "ground_truth": gt, "correct": False,
                **_series_payload(dates, ndvi, fb_smooth)}

    best = sorted(cands, key=lambda c: (c["gap"], -c["conf"]))[0]
    rec, cyc = best["record"], best["cycle"]
    order = np.argsort(best["proba"])[::-1][:3]
    pred = best["hypothesis"]
    return {
        "predicted_crop": pred,
        "confidence_pct": round(best["conf"] * 100, 1),
        "sowing_date": _d(rec["Sowing_Date"]), "peak_date": _d(rec["Peak_Date"]),
        "harvest_date": _d(rec["Harvest_Date"]), "duration_days": int(rec["Duration_Days"]),
        "sowing_idx": cyc["sowing"], "peak_idx": cyc["peak"], "harvest_idx": cyc["harvest"],
        "top3": [{"crop": classes[i], "confidence": round(float(best["proba"][i]) * 100, 1)} for i in order],
        "ground_truth": gt,
        "correct": bool(gt and gt.lower() == pred.lower()),
        **_series_payload(dates, ndvi, best["smoothed"]),
    }
