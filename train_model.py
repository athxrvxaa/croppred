"""
Train the Random Forest exactly as in your notebook (cells 1-20) and save it.

    python train_model.py                       # uses combined_ndvi.csv
    python train_model.py path/to/your.csv

CSV columns: Farm_ID | Date (or Sentinel_Date, dd-mm-YYYY) | NDVI | crop label
Output:      model_weights/rf_model.joblib
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from backend.lifecycle_logic import (CROP_CONFIG, DEFAULT_CFG, FEATURE_COLS,
                                     build_features, cycle_record, detect_all, validate)

path = sys.argv[1] if len(sys.argv) > 1 else "combined_ndvi.csv"
raw = pd.read_csv(path).rename(columns={"Farm_ID": "Farm", "Sentinel_Date": "Date", "crop label": "Crop"})
raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%Y")
raw = raw[["Farm", "Crop", "Date", "NDVI"]].groupby(["Farm", "Crop", "Date"], as_index=False)["NDVI"].mean()
raw = raw[raw["NDVI"] >= 0]
# your notebook trains on these 4 crops
df = raw[raw["Crop"].isin(CROP_CONFIG)].sort_values(["Crop", "Farm", "Date"]).reset_index(drop=True)
print("Loaded", df.shape, "| crops:", sorted(df.Crop.unique()), "| farms:", df.Farm.nunique())

rows = []
for (crop, farm), g in df.groupby(["Crop", "Farm"]):
    dates = pd.to_datetime(g["Date"].values.astype("datetime64[ns]"))
    ndvi = g["NDVI"].values
    cycles, sm = detect_all(dates, ndvi, CROP_CONFIG.get(crop, DEFAULT_CFG))
    for c in sorted(cycles, key=lambda x: x["peak"]):
        rec = cycle_record(crop, dates, ndvi, sm, c)
        if validate(rec) != "Valid":
            continue
        f = build_features(rec)
        if f:
            rows.append({"Crop": crop, "Farm": farm, **f})
ml = pd.DataFrame(rows).dropna(subset=FEATURE_COLS + ["Crop"]).reset_index(drop=True)
print("Valid samples:", len(ml)); print(ml.groupby("Crop").size())

fc = ml[["Farm", "Crop"]].drop_duplicates().reset_index(drop=True)
a, b = fc["Farm"].values, fc["Crop"].values
i1, t_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(a, b))
i2, v_idx = next(StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42).split(a[i1], b[i1]))
tr, va, te = set(a[i1][i2]), set(a[i1][v_idx]), set(a[t_idx])
ml["split"] = ml["Farm"].map(lambda x: "train" if x in tr else "val" if x in va else "test")
T = {s: ml[ml.split == s] for s in ("train", "val", "test")}

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
clf.fit(T["train"][FEATURE_COLS].values, T["train"]["Crop"].values)
for s in ("val", "test"):
    print(f"\n── {s} ──")
    print(classification_report(T[s]["Crop"], clf.predict(T[s][FEATURE_COLS].values), zero_division=0))

joblib.dump({"model": clf, "feature_cols": FEATURE_COLS, "classes": list(clf.classes_)}, "model_weights/rf_model.joblib")
print("Saved model_weights/rf_model.joblib | classes:", list(clf.classes_))
