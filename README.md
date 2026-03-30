# Spatial-Temporal Delivery Delay & ETA Engine

A production-grade, end-to-end Machine Learning pipeline and serving engine that predicts delivery arrival times (ETA in minutes) and delay probabilities. The system uses spatial hexagons (H3 indexing), cyclical temporal encoding, and tabular operational metrics.

---

## Key Features

* **H3 Spatial Hexagon Indexing:** Resolution 8 mapping (`h3` package) of pickup/dropoff coordinates to learn historical averages for delays and orders volume per hexagon.
* **Cyclical Temporal Encoding:** Decomposes order timestamps using sine/cosine transforms to encode continuous hour of day and day of week patterns.
* **Operational Feature Matrix:** Handles active driver density ratios, pending queue volume, merchant preparation times, and rain levels (mm/hr).
* **Dual LightGBM Models:** Combines a regressor for continuous ETA prediction and a binary classifier for delay probability (> 35 min threshold).
* **Local Perturbation Explainability:** Computes exact marginal impacts of distance, weather, driver density, prep times, spatial bottlenecks, and cyclic time on a single delivery prediction.
* **FastAPI Microservice:** Serve real-time predictions via POST `/predict_eta` and health audits via GET `/health`.
* **Streamlit Dashboard:** Interactive map allowing dual-click coordinates selection (pickup and dropoff), parameters sliders, H3 boundary overlays, and Plotly attribution chart.

---

## Repository Structure

```text
spatial_eta_engine/
│
├── data/                      # Synthetic spatial delivery records
├── models/                    # Saved pipeline artifacts (.joblib)
├── src/
│   ├── __init__.py
│   ├── dataset.py             # Synthetic delivery logs generator
│   ├── spatial_features.py    # H3 spatial indexing & geohash aggregation logic
│   ├── temporal_features.py   # Sine/Cosine cyclic time transformation & encoders
│   ├── train.py               # Model training script with regression benchmarks
│   └── explainability.py      # Feature attribution logic (global/local)
│
├── api/
│   └── main.py                # FastAPI endpoints (/health, /predict_eta)
│
├── app/
│   └── dashboard.py           # Streamlit app (Interactive Map, ETA predictor, feature impact)
│
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Create and Activate Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Mac/Linux)
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1: Train Models & Generate Synthetic Data
Run the training script to generate the synthetic delivery logs and train the models:
```bash
python src/train.py
```
This generates `data/delivery_records.csv`, fits the preprocessing pipeline and LightGBM models, evaluates performance, and outputs serialization artifacts into `models/eta_pipeline.joblib`.

### Step 2: Launch FastAPI Server
Run the backend web service:
```bash
uvicorn api.main:app --reload
```
The documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
* **GET `/health`**: Returns engine status and metrics.
* **POST `/predict_eta`**: Receives delivery parameters and returns ETA predictions, delay probabilities, and explainability attributions.

### Step 3: Run the Streamlit Dashboard
Launch the frontend interactive simulator:
```bash
streamlit run app/dashboard.py
```
The dashboard runs at [http://localhost:8501](http://localhost:8501) and connects to the FastAPI backend, falling back to local model loading if the backend is offline.


