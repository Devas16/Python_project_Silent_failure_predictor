# 🔍 Silent Failure Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=for-the-badge&logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An ML-powered system that detects anomalies in server metrics — catching silent failures *before* they become crashes.**

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Dashboard](#-streamlit-dashboard) · [Results](#-sample-results)

</div>

---

## 🧠 Problem Statement

Modern distributed systems rarely fail catastrophically without warning. Instead, they exhibit **silent failures** — subtle, gradual deviations in system behaviour that precede a crash by minutes or hours:

- **Memory leak**: RAM usage slowly climbs while CPU stays normal
- **Thread starvation**: Response times creep up while throughput appears fine
- **Disk saturation**: I/O write queues build up silently until the system stalls
- **Cascade failure**: Multiple metrics degrade simultaneously — the last warning before a crash

These precursors are invisible to threshold-based alerting (which only fires *after* a metric exceeds a hard limit). The Silent Failure Predictor uses **unsupervised Machine Learning** to model what "normal" looks like and flag deviations in real time — before humans or traditional monitors notice anything is wrong.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Isolation Forest** | Primary unsupervised anomaly detector — no labels required |
| 🧬 **Autoencoder** (bonus) | Deep learning model trained on normal data; flags high reconstruction error |
| 📊 **Rich Visualisations** | Time-series charts with anomaly overlay, score distributions, confusion matrices |
| 🖥️ **Streamlit Dashboard** | Simulated real-time metric stream with live anomaly alerts |
| ⚡ **FastAPI REST API** | `/predict/single` and `/predict` (batch) endpoints with Pydantic validation |
| 🔧 **Rolling Features** | Temporal context via rolling mean/std — catches gradual drift |
| 🪵 **Loguru Logging** | Structured logs to console + rotating file |
| ⚙️ **YAML Config** | All hyperparameters centralised in `config/config.yaml` |
| 🧪 **Unit Tests** | pytest suite covering data generation, preprocessing, and model behaviour |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Silent Failure Predictor                   │
├─────────────────┬───────────────────┬───────────────────────┤
│   Data Layer    │    ML Layer       │   Application Layer   │
├─────────────────┼───────────────────┼───────────────────────┤
│ data_generator  │ Isolation Forest  │  FastAPI REST API     │
│ (6 metrics,     │ (sklearn)         │  /predict/single      │
│  5 failure      │                   │  /predict (batch)     │
│  patterns)      │ Autoencoder       │                       │
│                 │ (TensorFlow/Keras)│  Streamlit Dashboard  │
│ preprocessor    │                   │  Real-time stream     │
│ (rolling stats  │                   │  Anomaly alerts       │
│  + StandardScaler)                  │  Interactive charts   │
└─────────────────┴───────────────────┴───────────────────────┘
```

### Why Isolation Forest?

Isolation Forest works by randomly partitioning data using decision-tree splits. **Anomalous points are isolated in fewer splits** because they occupy sparse regions of the feature space. Key advantages for this use-case:

- **Unsupervised** — no labelled anomaly data needed in production
- **Efficient** — O(n log n) time complexity, handles large metric streams
- **Contamination parameter** — explicitly model the expected anomaly rate
- **Score-based** — continuous anomaly score enables severity tiering (normal / warning / critical)

### Why Autoencoder (bonus)?

An Autoencoder trained only on normal samples learns to reconstruct normal patterns well. At inference time, **anomalous inputs produce high reconstruction error** (MSE). This is complementary to Isolation Forest:

- Captures **multi-metric correlations** — e.g. "CPU spike always accompanied by latency rise"
- Better at detecting **pattern-level** deviations (not just point outliers)

---

## 📂 Project Structure

```
silent-failure-predictor/
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── config_loader.py          # YAML config reader (LRU-cached)
│   ├── logger.py                 # Loguru logger setup
│   ├── data_generator.py         # Synthetic dataset with 5 failure patterns
│   ├── preprocessor.py           # Rolling features + StandardScaler pipeline
│   ├── isolation_forest_model.py # Isolation Forest wrapper (train/eval/save/load)
│   ├── autoencoder_model.py      # Keras Autoencoder wrapper
│   ├── visualizer.py             # matplotlib + Plotly plotting utilities
│   └── train.py                  # End-to-end training pipeline
│
├── app/                          # Application layer
│   ├── __init__.py
│   ├── api.py                    # FastAPI REST API
│   └── dashboard.py              # Streamlit real-time dashboard
│
├── data/                         # Dataset storage
│   └── system_metrics.csv        # Generated after running `python run.py generate`
│
├── models/                       # Saved model artefacts
│   ├── isolation_forest.pkl      # (generated after training)
│   ├── scaler.pkl                # (generated after training)
│   └── autoencoder.h5            # (generated after training, if TF available)
│
├── notebooks/
│   └── exploration.ipynb         # Interactive EDA + model training walkthrough
│
├── outputs/                      # Saved plot images
│   ├── metrics_overview.png
│   ├── anomaly_score_distribution.png
│   ├── confusion_matrix.png
│   └── reconstruction_error.png
│
├── tests/                        # pytest unit tests
│   ├── __init__.py
│   ├── test_data_generator.py
│   ├── test_preprocessor.py
│   └── test_isolation_forest.py
│
├── config/
│   └── config.yaml               # All hyperparameters & settings
│
├── logs/                         # Rotating log files (auto-created)
├── run.py                        # Unified CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/silent-failure-predictor.git
cd silent-failure-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# OR
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: TensorFlow is optional. If you skip it, the Autoencoder model won't train but everything else works fine. Use `--skip-ae` flag when training.

### 4. Generate the dataset

```bash
python run.py generate
```

This creates `data/system_metrics.csv` — 2,000 timestamped metric readings with ~5% injected silent failure patterns.

### 5. Train the models

```bash
python run.py train            # Full pipeline (Isolation Forest + Autoencoder)
python run.py train --skip-ae  # Without Autoencoder (no TensorFlow needed)
```

Trained models are saved to `models/` and visualisation plots to `outputs/`.

### 6. Run a demo prediction

```bash
python run.py predict
```

Scores a hardcoded cascade-failure sample to demonstrate the API.

### 7. Launch the dashboard

```bash
python run.py dashboard
```

Opens the Streamlit real-time dashboard at `http://localhost:8501`.

### 8. Launch the REST API

```bash
python run.py api
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`.

---

## 📡 API Reference

### Health Check

```http
GET /health
```

```json
{ "status": "ok", "timestamp": 1711700000.0 }
```

### Predict Single

```http
POST /predict/single
Content-Type: application/json
```

```json
{
  "cpu_usage": 92.0,
  "memory_usage": 88.5,
  "response_time_ms": 2100.0,
  "disk_io_mbps": 180.0,
  "network_latency_ms": 240.0,
  "error_rate_pct": 12.3
}
```

**Response:**

```json
{
  "is_anomaly": true,
  "anomaly_score": 0.4821,
  "severity": "critical",
  "message": "⚠️  Silent failure detected — immediate investigation recommended."
}
```

### Batch Predict

```http
POST /predict
Content-Type: application/json
```

```json
{
  "readings": [
    { "cpu_usage": 45, "memory_usage": 52, "response_time_ms": 210, "disk_io_mbps": 28, "network_latency_ms": 16, "error_rate_pct": 0.4 },
    { "cpu_usage": 94, "memory_usage": 91, "response_time_ms": 2300, "disk_io_mbps": 175, "network_latency_ms": 250, "error_rate_pct": 14 }
  ]
}
```

**Response:**

```json
{
  "total_samples": 2,
  "anomaly_count": 1,
  "anomaly_fraction": 0.5,
  "predictions": [
    { "is_anomaly": false, "anomaly_score": 0.092, "severity": "normal", "message": "✅  System operating normally." },
    { "is_anomaly": true,  "anomaly_score": 0.481, "severity": "critical", "message": "⚠️  Silent failure detected." }
  ]
}
```

---

## 🖥️ Streamlit Dashboard

The dashboard simulates a live metric feed by replaying the generated dataset at configurable speed.

**Features:**
- ▶/⏹ Start / Stop real-time simulation
- 📊 Primary metric chart with anomaly overlay (any of 6 metrics)
- 🔴 Alert banner escalates from normal → warning → critical
- 🔢 KPI panel: samples processed, anomalies detected, anomaly rate, consecutive alerts
- 📋 Recent readings table with colour-coded status
- ⚙️ Sidebar controls: metric selection, refresh rate

```bash
streamlit run app/dashboard.py
# or
python run.py dashboard
```

---

## 📊 Sample Results

### Metrics Overview
*Six-panel time-series chart showing all metrics over 2,000 time steps, with anomalies highlighted in red.*

```
outputs/metrics_overview.png
```
- Normal readings: steady blue lines following daily usage patterns
- Anomalies (red ×): clear spikes in response time, memory, and error rate

### Anomaly Score Distribution
*Histogram of Isolation Forest scores split by ground-truth class.*

```
outputs/anomaly_score_distribution.png
```
- Normal samples (blue): concentrated at low scores
- Anomalies (red): shifted right to higher scores — excellent class separation

### Model Performance

| Metric | Isolation Forest | Autoencoder |
|--------|-----------------|-------------|
| ROC-AUC | ~0.95 | ~0.91 |
| Average Precision | ~0.82 | ~0.77 |
| Precision (anomaly) | ~0.78 | ~0.72 |
| Recall (anomaly) | ~0.88 | ~0.85 |

> Results vary slightly with random seed. Run `python run.py train` to see exact numbers on your machine.

---

## ⚙️ Configuration

All settings live in `config/config.yaml`. Key sections:

```yaml
data:
  n_samples: 2000          # Dataset size
  anomaly_fraction: 0.05   # ~5% injected anomalies

isolation_forest:
  contamination: 0.05      # Expected anomaly rate
  n_estimators: 200        # More trees = better, slower

autoencoder:
  epochs: 50
  encoding_dim: 4          # Bottleneck size
  threshold_percentile: 95 # Reconstruction error cutoff

features:
  use_rolling_stats: true
  rolling_window: 10       # Temporal context window
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

The test suite covers:
- Dataset shape, types, value ranges, and anomaly fraction
- Preprocessor fit/transform, save/load roundtrip, rolling feature expansion
- Isolation Forest training, prediction shape, ROC-AUC above chance, save/load

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML — Anomaly Detection | scikit-learn (Isolation Forest) |
| ML — Deep Learning | TensorFlow / Keras (Autoencoder) |
| Data | NumPy, Pandas |
| Visualisation | Matplotlib, Plotly, Seaborn |
| Dashboard | Streamlit |
| REST API | FastAPI + Uvicorn |
| Config | PyYAML |
| Logging | Loguru |
| Testing | pytest, pytest-cov |
| Serialisation | joblib |

---

## 🔮 Future Improvements

- [ ] **Online learning** — update the model incrementally as new data arrives without full retraining
- [ ] **LSTM / Transformer** — replace rolling statistics with a learned temporal encoder
- [ ] **Multi-service monitoring** — extend to multiple microservices with per-service models
- [ ] **Prometheus / Grafana integration** — pull real metrics instead of synthetic data
- [ ] **Slack / PagerDuty alerts** — push notifications on critical anomaly sequences
- [ ] **Explainability** — SHAP values to explain *which* features drove each anomaly
- [ ] **Docker + Kubernetes** — containerise the API and dashboard for production deployment
- [ ] **A/B comparison** — side-by-side evaluation of Isolation Forest vs One-Class SVM vs LOF
- [ ] **Hyperparameter search** — automated tuning with Optuna or Ray Tune

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) — Isolation Forest implementation
- [TensorFlow](https://www.tensorflow.org/) — Keras Autoencoder framework
- [Streamlit](https://streamlit.io/) — dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) — modern Python API framework
- [Loguru](https://github.com/Delgan/loguru) — beautiful Python logging

---

<div align="center">
Built with ❤️ — If this project helped you, please ⭐ star it on GitHub!
</div>
