"""
app/dashboard.py
----------------
Streamlit real-time dashboard for the Silent Failure Predictor.

Features:
  • Simulated live metric stream (new data point every second)
  • Isolation Forest anomaly detection on-the-fly
  • Interactive Plotly charts for all six metrics
  • Alert panel with consecutive-anomaly counter
  • Model performance summary panel

Run:
    streamlit run app/dashboard.py
"""

import time
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
import os

# Make sure project root is on the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_loader import load_config
from src.data_generator import load_dataset
from src.preprocessor import Preprocessor
from src.isolation_forest_model import IsolationForestDetector

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "Silent Failure Predictor",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .alert-critical {
        background: linear-gradient(135deg, #3d1515 0%, #2d0f0f 100%);
        border: 1px solid #ff4b4b;
        border-radius: 10px;
        padding: 14px;
        color: #ff4b4b;
        font-weight: bold;
        text-align: center;
    }
    .alert-normal {
        background: linear-gradient(135deg, #0f3d1a 0%, #0a2912 100%);
        border: 1px solid #22c55e;
        border-radius: 10px;
        padding: 14px;
        color: #22c55e;
        font-weight: bold;
        text-align: center;
    }
    .stMetric > div { background: #1a1f2e; border-radius: 8px; padding: 8px; }
    h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
cfg       = load_config()
dash_cfg  = cfg["dashboard"]
MAX_PTS   = dash_cfg.get("max_display_points", 200)
ALERT_THR = dash_cfg.get("alert_threshold", 3)

METRICS = [
    ("cpu_usage",          "CPU Usage (%)",          "#00C8FF"),
    ("memory_usage",       "Memory Usage (%)",       "#A78BFA"),
    ("response_time_ms",   "Response Time (ms)",     "#34D399"),
    ("disk_io_mbps",       "Disk I/O (MB/s)",        "#FCD34D"),
    ("network_latency_ms", "Network Latency (ms)",   "#F472B6"),
    ("error_rate_pct",     "Error Rate (%)",         "#FB923C"),
]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "buffer"       not in st.session_state: st.session_state.buffer       = pd.DataFrame()
if "alert_count"  not in st.session_state: st.session_state.alert_count  = 0
if "total_alerts" not in st.session_state: st.session_state.total_alerts = 0
if "running"      not in st.session_state: st.session_state.running      = False
if "row_idx"      not in st.session_state: st.session_state.row_idx      = 0

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ML models …")
def load_models():
    model_path  = cfg["isolation_forest"]["model_path"]
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Models not found. Run `python -m src.train` first.")
        st.stop()

    det  = IsolationForestDetector(); det.load(model_path)
    prep = Preprocessor();            prep.load(scaler_path)
    return det, prep


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    return load_dataset()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.shields.io/badge/Silent%20Failure%20Predictor-ML-blue?style=for-the-badge")
    st.markdown("---")
    st.header("⚙️ Controls")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False

    if st.button("🔄 Reset Stream", use_container_width=True):
        st.session_state.buffer       = pd.DataFrame()
        st.session_state.alert_count  = 0
        st.session_state.total_alerts = 0
        st.session_state.row_idx      = 0
        st.session_state.running      = False

    st.markdown("---")
    selected_metric = st.selectbox(
        "📊 Primary Chart Metric",
        options=[m[0] for m in METRICS],
        format_func=lambda x: next(m[1] for m in METRICS if m[0] == x),
    )

    refresh_ms = st.slider("Refresh Interval (ms)", 500, 3000, 1000, step=250)

    st.markdown("---")
    st.header("🤖 Model Info")
    st.markdown(f"""
    - **Algorithm**: Isolation Forest  
    - **Estimators**: {cfg['isolation_forest']['n_estimators']}  
    - **Contamination**: {cfg['isolation_forest']['contamination']}  
    - **Features**: {len(cfg['features']['features_list'])} base + rolling  
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using scikit-learn + Streamlit")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("🔍 Silent Failure Predictor")
st.markdown("*Real-time anomaly detection in system metrics — catch crashes before they happen.*")
st.markdown("---")

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
detector, preprocessor = load_models()
full_df                = load_data()

# ---------------------------------------------------------------------------
# Top KPI row
# ---------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

buf = st.session_state.buffer
n_buf = len(buf)
n_anom = int(buf["is_anomaly"].sum()) if n_buf > 0 and "is_anomaly" in buf.columns else 0

kpi1.metric("📡 Samples Processed",   n_buf)
kpi2.metric("🚨 Anomalies Detected",  n_anom)
kpi3.metric("📈 Anomaly Rate",
            f"{(n_anom/n_buf*100):.1f}%" if n_buf > 0 else "—")
kpi4.metric("⚡ Consecutive Alerts", st.session_state.alert_count)

st.markdown("---")

# ---------------------------------------------------------------------------
# Alert banner
# ---------------------------------------------------------------------------
alert_placeholder = st.empty()

if st.session_state.alert_count >= ALERT_THR:
    alert_placeholder.markdown(
        f'<div class="alert-critical">🔴  CRITICAL: {st.session_state.alert_count} consecutive anomalies detected!'
        f'  System may be approaching failure.  Investigate immediately.</div>',
        unsafe_allow_html=True
    )
elif n_anom > 0:
    alert_placeholder.markdown(
        '<div class="alert-normal">🟡  WARNING: Anomalies detected but within normal operating bounds.</div>',
        unsafe_allow_html=True
    )
else:
    alert_placeholder.markdown(
        '<div class="alert-normal">🟢  System operating normally. No anomalies detected.</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
chart_placeholder  = st.empty()
detail_placeholder = st.empty()

def render_charts(df: pd.DataFrame):
    """Render primary metric chart + mini-sparklines for all metrics."""
    if len(df) == 0:
        chart_placeholder.info("▶ Press Start to begin the real-time simulation.")
        return

    # Primary large chart
    primary_color = next(m[2] for m in METRICS if m[0] == selected_metric)
    fig = go.Figure()

    norm = df[df["is_anomaly"] == 0]
    anom = df[df["is_anomaly"] == 1]

    fig.add_trace(go.Scatter(
        x=norm["timestamp"], y=norm[selected_metric],
        mode="lines", name="Normal",
        line=dict(color=primary_color, width=2),
    ))
    if len(anom) > 0:
        fig.add_trace(go.Scatter(
            x=anom["timestamp"], y=anom[selected_metric],
            mode="markers", name="🚨 Anomaly",
            marker=dict(color="#FF4B4B", size=10, symbol="x-open", line=dict(width=2)),
        ))

    fig.update_layout(
        title=f"Live Stream — {next(m[1] for m in METRICS if m[0]==selected_metric)}",
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_title="Time", yaxis_title="Value",
        legend=dict(orientation="h", y=1.1),
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # Sparklines (2 rows × 3 cols)
    cols = detail_placeholder.columns(3)
    for i, (col_name, label, color) in enumerate(METRICS):
        if col_name not in df.columns:
            continue
        mini = go.Figure()
        mini.add_trace(go.Scatter(
            x=df["timestamp"], y=df[col_name],
            mode="lines", line=dict(color=color, width=1.5), showlegend=False
        ))
        if len(anom) > 0:
            mini.add_trace(go.Scatter(
                x=anom["timestamp"], y=anom[col_name],
                mode="markers", marker=dict(color="#FF4B4B", size=6, symbol="x"),
                showlegend=False
            ))
        mini.update_layout(
            title=label, template="plotly_dark",
            height=170, margin=dict(l=20, r=10, t=30, b=20),
        )
        cols[i % 3].plotly_chart(mini, use_container_width=True)


render_charts(buf)

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
if st.session_state.running:
    idx = st.session_state.row_idx

    if idx >= len(full_df):
        st.session_state.running = False
        st.info("✅ Dataset exhausted. Press Reset to start again.")
    else:
        row = full_df.iloc[[idx]].copy()
        row["timestamp"] = datetime.now()

        # Live inference
        X = preprocessor.transform(row[cfg["features"]["features_list"]])
        labels, scores = detector.predict(X)
        row["is_anomaly"]    = int(labels[0])
        row["anomaly_score"] = float(scores[0])

        # Update buffer
        buf = pd.concat([st.session_state.buffer, row], ignore_index=True).tail(MAX_PTS)
        st.session_state.buffer   = buf
        st.session_state.row_idx += 1

        # Update alert counter
        if labels[0] == 1:
            st.session_state.alert_count  += 1
            st.session_state.total_alerts += 1
        else:
            st.session_state.alert_count = 0

        time.sleep(refresh_ms / 1000)
        st.rerun()

# ---------------------------------------------------------------------------
# Historical data table
# ---------------------------------------------------------------------------
if len(buf) > 0:
    st.markdown("---")
    st.subheader("📋 Recent Readings")
    display_df = buf.tail(20).copy()
    display_df["status"] = display_df["is_anomaly"].map({0: "✅ Normal", 1: "🚨 Anomaly"})
    st.dataframe(
        display_df[["timestamp", "cpu_usage", "memory_usage", "response_time_ms",
                    "network_latency_ms", "error_rate_pct", "status"]].style.applymap(
            lambda v: "color: #ff4b4b" if v == "🚨 Anomaly" else "color: #22c55e",
            subset=["status"]
        ),
        use_container_width=True,
    )
