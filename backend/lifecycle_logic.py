"""
Crop lifecycle logic — ported 1:1 from your notebook
(crop_data_lifecycle_classification_updates.ipynb).

  * CROP_CONFIG      -> per-crop smoothing / peak / search-window settings
  * detect_cycles    -> rolling-mean smoothing + peak + sowing/harvest search
  * RULES / validate -> per-crop calendar, duration and NDVI rise/drop rules
  * build_features   -> the 13 FEATURE_COLS fed to the Random Forest

Nothing here touches Earth Engine or the web layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_trapz = getattr(np, "trapezoid", None) or np.trapz

CROP_CONFIG = {
    "Sugarcane": {"smooth": 6, "max_cycles": 1, "search_months": 8, "use_prominence": True,  "prominence": 0.08},
    "Onion":     {"smooth": 2, "max_cycles": 1, "search_months": 3, "use_prominence": False, "prominence": None},
    "Paddy":     {"smooth": 3, "max_cycles": 2, "search_months": 3, "use_prominence": True,  "prominence": 0.15},
    "Cotton":    {"smooth": 2, "max_cycles": 2, "search_months": 4, "use_prominence": True,  "prominence": 0.07},
}
DEFAULT_CFG = {"smooth": 3, "max_cycles": 1, "search_months": 8, "use_prominence": False, "prominence": None}

RULES = {
    "Onion":     {"sow_months": [5, 6, 7, 8, 9, 10, 11, 12, 1], "harv_months": [1, 2, 3, 4, 5, 10, 11, 12], "min_dur": 85,  "max_dur": 180, "min_drop": 0.2,  "min_rise": 0.15},
    "Paddy":     {"sow_months": [5, 6, 7, 8],                   "harv_months": [9, 10, 11, 12, 1],          "min_dur": 90,  "max_dur": 180, "min_drop": 0.2,  "min_rise": 0.15},
    "Sugarcane": {"sow_months": [10,11 ,12,1, 2, 3, 4],         "harv_months": [10, 11, 12, 1, 2, 3],       "min_dur": 180, "max_dur": 365, "min_drop": 0.09, "min_rise": 0.12},
    "Cotton":    {"sow_months": [5, 6, 7, 8, 9],                "harv_months": [11, 12, 1, 4],              "min_dur": 65,  "max_dur": 230, "min_drop": 0.15, "min_rise": 0.10},
}

FEATURE_COLS = [
    "Duration_Days", "Peak_NDVI", "Peak_Month",
    "Sowing_NDVI", "Sowing_Month",
    "Harvest_NDVI", "Harvest_Month",
    "AUC", "Mean_NDVI", "Std_NDVI", "NDVI_range",
    "Rise_rate", "Fall_rate",
]


# ── detection (notebook cell 4 / 6) ─────────────────────────────────────────
def split_into_years(dates):
    start, end = dates.min(), dates.max()
    segments, seg_start = [], start
    while seg_start <= end:
        seg_end = seg_start + pd.DateOffset(years=1)
        segments.append((seg_start, seg_end))
        seg_start = seg_end
    return segments


def detect_cycles(dates, ndvi, cfg):
    s = pd.Series(ndvi).rolling(cfg["smooth"], center=True, min_periods=1).mean().values
    max_cycles = cfg["max_cycles"]
    min_dist = max(3, len(s) // (max_cycles * 6))

    if cfg["use_prominence"]:
        peaks, _ = find_peaks(s, prominence=cfg["prominence"], distance=min_dist)
    else:
        peaks, _ = find_peaks(s, height=0.3, distance=min_dist)
    if len(peaks) == 0:
        return [], s

    if len(peaks) > max_cycles:
        top_idx = np.argsort(s[peaks])[-max_cycles:]
        peaks = np.sort(peaks[top_idx])

    window = pd.DateOffset(months=cfg["search_months"])
    cycles = []
    for p in peaks:
        peak_date = dates[p]
        before = np.where((dates >= peak_date - window) & (dates <= peak_date))[0]
        sowing = before[np.argmin(s[before])] if len(before) else 0
        after = np.where((dates >= peak_date) & (dates <= peak_date + window))[0]
        harvest = after[np.argmin(s[after])] if len(after) else len(s) - 1
        cycles.append({"sowing": int(sowing), "peak": int(p), "harvest": int(harvest)})
    return cycles, s


def detect_all(dates, ndvi, cfg):
    """Year-segment loop from notebook cell 6 for one series."""
    smoothed_full = np.full(len(dates), np.nan)
    all_cycles = []
    for seg_idx, (seg_start, seg_end) in enumerate(split_into_years(dates)):
        idx = np.where((dates >= seg_start) & (dates < seg_end))[0]
        if len(idx) < 5:
            continue
        cycles, s = detect_cycles(dates[idx], ndvi[idx], cfg)
        smoothed_full[idx] = s
        for c in cycles:
            all_cycles.append({
                "sowing": int(idx[c["sowing"]]), "peak": int(idx[c["peak"]]),
                "harvest": int(idx[c["harvest"]]), "seg_idx": seg_idx,
            })
    return all_cycles, smoothed_full


# ── lifecycle table row (notebook cell 10) ──────────────────────────────────
def cycle_record(crop, dates, ndvi, smoothed, c):
    sow, peak, harv = (pd.Timestamp(dates[c[k]]) for k in ("sowing", "peak", "harvest"))
    mask = (dates >= sow) & (dates <= harv)
    seg_days = pd.to_timedelta(dates[mask] - sow).days.astype(float)
    seg_ndvi = ndvi[mask]
    auc = float(_trapz(seg_ndvi, seg_days)) if len(seg_days) > 1 else np.nan
    return {
        "Crop": crop,
        "Sowing_Date": sow, "Sowing_Month": sow.month, "Sowing_NDVI": round(float(smoothed[c["sowing"]]), 4),
        "Peak_Date": peak, "Peak_Month": peak.month, "Peak_NDVI": round(float(smoothed[c["peak"]]), 4),
        "Harvest_Date": harv, "Harvest_Month": harv.month, "Harvest_NDVI": round(float(smoothed[c["harvest"]]), 4),
        "Duration_Days": (harv - sow).days,
        "AUC": round(auc, 2) if not np.isnan(auc) else np.nan,
        "Mean_NDVI": round(float(np.nanmean(seg_ndvi)), 4),
        "_seg_ndvi": seg_ndvi,
    }


# ── validation (notebook cell 11) ───────────────────────────────────────────
def validate(row):
    rule = RULES.get(row["Crop"])
    if rule is None:
        return "No rule defined"
    reasons = []
    if row["Sowing_Month"] not in rule["sow_months"]:
        reasons.append(f"Sow month {row['Sowing_Month']} invalid")
    if row["Harvest_Month"] not in rule["harv_months"]:
        reasons.append(f"Harvest month {row['Harvest_Month']} invalid")
    if not (rule["min_dur"] <= row["Duration_Days"] <= rule["max_dur"]):
        reasons.append(f"Duration {row['Duration_Days']}d not in [{rule['min_dur']}-{rule['max_dur']}]")
    drop = row["Peak_NDVI"] - row["Harvest_NDVI"]
    if drop <= rule.get("min_drop", 0.2):
        reasons.append(f"Peak-Harvest NDVI drop {drop:.3f} <= {rule.get('min_drop', 0.2)}")
    rise = row["Peak_NDVI"] - row["Sowing_NDVI"]
    if rise <= rule.get("min_rise", 0.15):
        reasons.append(f"Peak-Sowing NDVI rise {rise:.3f} <= {rule.get('min_rise', 0.15)}")
    return "Valid" if not reasons else "; ".join(reasons)


# ── features (notebook cell 19) ─────────────────────────────────────────────
def build_features(row):
    v = np.asarray(row["_seg_ndvi"], dtype=float)
    if len(v) < 3:
        return None
    return {
        "Duration_Days": row["Duration_Days"],
        "Peak_NDVI": row["Peak_NDVI"], "Peak_Month": row["Peak_Month"],
        "Sowing_NDVI": row["Sowing_NDVI"], "Sowing_Month": row["Sowing_Month"],
        "Harvest_NDVI": row["Harvest_NDVI"], "Harvest_Month": row["Harvest_Month"],
        "AUC": row["AUC"], "Mean_NDVI": row["Mean_NDVI"],
        "Std_NDVI": round(float(np.std(v)), 4),
        "NDVI_range": round(float(np.ptp(v)), 4),
        "Rise_rate": round((row["Peak_NDVI"] - row["Sowing_NDVI"]) / max((row["Peak_Date"] - row["Sowing_Date"]).days, 1), 5),
        "Fall_rate": round((row["Peak_NDVI"] - row["Harvest_NDVI"]) / max((row["Harvest_Date"] - row["Peak_Date"]).days, 1), 5),
    }


def candidate_lifecycles(dates, ndvi, crops=None):
    """
    Inference-time helper. At training time the crop label is known so the
    notebook uses that crop's config. At inference it isn't, so every crop's
    config is tried and each candidate is validated with THAT crop's RULES.

    Returns a list of dicts (one per valid cycle) with 'hypothesis' = crop
    whose config/rules produced it, plus the feature dict.
    """
    out = []
    for crop in (crops or CROP_CONFIG):
        cycles, smoothed = detect_all(dates, ndvi, CROP_CONFIG[crop])
        for c in cycles:
            rec = cycle_record(crop, dates, ndvi, smoothed, c)
            rec["Validation"] = validate(rec)
            rec["Is_Valid"] = rec["Validation"] == "Valid"
            feats = build_features(rec)
            out.append({"hypothesis": crop, "record": rec, "features": feats,
                        "smoothed": smoothed, "cycle": c})
    return out
