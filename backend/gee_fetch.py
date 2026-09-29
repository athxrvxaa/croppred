"""
Earth Engine NDVI fetch — Sentinel-2 point time series.

No API key / service-account file is used. Earth Engine is initialised against
a Google Cloud *project* using your normal `earthengine authenticate` login.

GEE Project
-----------
Default project: `gee-project-497010`
To change it, edit this line:

    def init_ee(project: str = 'gee-project-497010'):

(or set the GEE_PROJECT environment variable).
"""
from __future__ import annotations

import datetime as dt
import os

import pandas as pd

COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_LIMIT = 20          # scene-level CLOUDY_PIXEL_PERCENTAGE
PIXEL_SCALE = 10          # Sentinel-2 native resolution (m)
MONTHS_BEFORE = 18
MONTHS_AFTER = 18

_ee = None


def init_ee(project: str = 'gee-project-497010'):
    """Initialise Earth Engine once and return the `ee` module."""
    global _ee
    if _ee is not None:
        return _ee
    import ee

    project = os.getenv("GEE_PROJECT", project)
    try:
        ee.Initialize(project=project)
    except Exception:
        # First run on this machine: opens the browser login (no key file).
        try:
            ee.Authenticate()
            ee.Initialize(project=project)
        except Exception as exc:
            raise RuntimeError(
                f"Earth Engine could not initialise for project '{project}'. "
                "Run `earthengine authenticate` once in a terminal, and make sure "
                "the Earth Engine API is enabled for that project."
            ) from exc
    _ee = ee
    return ee


def fetch_ndvi(lat: float, lon: float, ref_date: dt.date,
               months_before: int = MONTHS_BEFORE, months_after: int = MONTHS_AFTER) -> pd.DataFrame:
    """Return a DataFrame[Date, NDVI] (one row per acquisition date, sorted)."""
    ee = init_ee()
    point = ee.Geometry.Point([lon, lat])
    ref = ee.Date(ref_date.isoformat())
    start, end = ref.advance(-months_before, "month"), ref.advance(months_after, "month")

    def tag(img):
        img = ee.Image(img)
        return img.set("Date", img.date().format("YYYY-MM-dd"))

    def add_ndvi(img):
        img = ee.Image(img)
        return img.addBands(img.normalizedDifference(["B8", "B4"]).rename("NDVI"))

    col = (ee.ImageCollection(COLLECTION).filterBounds(point).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_LIMIT)).map(tag)
           .sort("CLOUDY_PIXEL_PERCENTAGE").distinct("Date")   # least-cloudy image per date
           .sort("system:time_start").map(add_ndvi))

    def sample(img):
        img = ee.Image(img)
        v = img.select("NDVI").reduceRegion(ee.Reducer.first(), point, PIXEL_SCALE,
                                            bestEffort=True, maxPixels=16)
        return ee.Feature(None, {"Date": img.date().format("YYYY-MM-dd"), "NDVI": v.get("NDVI")})

    feats = ee.FeatureCollection(col.map(sample)).filter(ee.Filter.notNull(["NDVI"])).getInfo()["features"]
    df = pd.DataFrame([{"Date": pd.Timestamp(f["properties"]["Date"]), "NDVI": float(f["properties"]["NDVI"])}
                       for f in feats])
    if df.empty:
        return pd.DataFrame(columns=["Date", "NDVI"])
    df = df.groupby("Date", as_index=False)["NDVI"].mean()
    return df[df["NDVI"] >= 0].sort_values("Date").reset_index(drop=True)
