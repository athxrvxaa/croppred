"""Offline test: replays real farm series from combined_ndvi.csv (no Earth Engine)."""
import pandas as pd
from backend.predictor import predict_from_series

def _farm(fid):
    d = pd.read_csv("combined_ndvi.csv")
    g = d[d["Farm_ID"] == fid].copy()
    g["Date"] = pd.to_datetime(g["Date"], format="%d-%m-%Y")
    ref = pd.to_datetime(g["ground truth month"].iloc[0], format="%d-%m-%Y")
    return g[["Date", "NDVI"]].groupby("Date", as_index=False).mean(), ref

def test_known_farms():
    for fid, crop in [("ON001", "Onion"), ("P001", "Paddy")]:
        df, ref = _farm(fid)
        r = predict_from_series(df, ref, crop)
        assert r["predicted_crop"] == crop and r["correct"]
        assert r["sowing_date"] < r["peak_date"] < r["harvest_date"]

def test_too_few_points():
    df, ref = _farm("ON001")
    assert "error" in predict_from_series(df.head(3), ref)
