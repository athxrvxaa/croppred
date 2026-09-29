# Crop Predictor

CropDynamics' project shell (FastAPI + Sentinel-2 via Earth Engine) running **your notebook's logic**
and **your `index.html` frontend**.

## Run

```bash
pip install -r requirements.txt
earthengine authenticate          # one time, browser login — no API key / JSON key file
uvicorn backend.main:app --reload --port 8000
```
Open http://127.0.0.1:8000  (Windows: double-click `run.bat`, Mac/Linux: `./run.sh`)

## GEE Project

Default project: `gee-project-497010`
To change it, edit this line in `backend/gee_fetch.py`:

```
def init_ee(project: str = 'gee-project-497010'):
```
(or set the `GEE_PROJECT` environment variable). The Earth Engine API must be enabled on that project
and the account you ran `earthengine authenticate` with must have access to it.

## Layout

```
backend/
  gee_fetch.py        Earth Engine NDVI fetch (init_ee + fetch_ndvi)
  lifecycle_logic.py  YOUR notebook: CROP_CONFIG, detect_cycles, RULES/validate, 13 features
  predictor.py        inference: tries each crop's config+rules -> Random Forest -> best cycle
  main.py             FastAPI: GET /  (index.html),  POST /predict
frontend/index.html   your UI (lifecycle markers now come from the backend)
model_weights/rf_model.joblib   Random Forest trained with your notebook code
train_model.py        retrain:  python train_model.py [your.csv]
combined_ndvi.csv     training data (Farm_ID, Date, NDVI, crop label)
notebooks/            your original notebook
tests/                offline tests (no Earth Engine needed)
```

## How prediction works

Your notebook knows the crop when it detects a cycle (it uses that crop's config). A new point has no
label, so `predictor.py` runs detection with **each** crop's `CROP_CONFIG`, validates each cycle with that
crop's `RULES`, builds the 13 features, and asks the Random Forest. A candidate counts only if the model
agrees with the crop hypothesis that produced it, its lifecycle contains (or is within 60 days of) the
query date, and confidence >= 50%. Otherwise the UI shows "No confident crop match".

Tunables at the top of `backend/predictor.py`: `MIN_CONFIDENCE`, `MAX_GAP_DAYS`, `MIN_OBSERVATIONS`.
Crops: Cotton, Onion, Paddy, Sugarcane (same as your notebook). To add one, add it to `CROP_CONFIG` and
`RULES` in `lifecycle_logic.py` and retrain.

Retraining reproduces your notebook exactly (6022 rows, 135 valid cycles, 100% val/test on the farm-level split).
"# croppred" 
