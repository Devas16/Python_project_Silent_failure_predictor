"""
visualizer.py
-------------
Plotting utilities for system metrics and anomaly detection results.

All plots are saved to outputs/ and returned as matplotlib Figure objects
so they can also be embedded in notebooks or the Streamlit dashboard.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers & CI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config_loader import load_config
from src.logger import log


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_cfg      = load_config()["visualization"]
_OUT_DIR  = _cfg.get("output_dir", "outputs/")
_DPI      = _cfg.get("figure_dpi", 150)
_ANOM_CLR = _cfg.get("anomaly_color", "#FF4B4B")
_NORM_CLR = _cfg.get("normal_color",  "#4B9CFF")
_FIG_SIZE = tuple(_cfg.get("figure_size", [18, 12]))

os.makedirs(_OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Metrics Overview (matplotlib)
# ---------------------------------------------------------------------------

def plot_metrics_overview(df: pd.DataFrame, filename: str = "metrics_overview.png") -> str:
    """
    Six-panel time-series plot of all metrics with anomaly highlights.

    Args:
        df:       DataFrame with timestamp, metric columns, and is_anomaly.
        filename: Output filename (saved inside outputs/).

    Returns:
        Absolute path to the saved PNG.
    """
    plt.style.use("dark_background")
    metrics = [
        ("cpu_usage",          "CPU Usage (%)",          "#00C8FF"),
        ("memory_usage",       "Memory Usage (%)",       "#A78BFA"),
        ("response_time_ms",   "Response Time (ms)",     "#34D399"),
        ("disk_io_mbps",       "Disk I/O (MB/s)",        "#FCD34D"),
        ("network_latency_ms", "Network Latency (ms)",   "#F472B6"),
        ("error_rate_pct",     "Error Rate (%)",         "#FB923C"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=_FIG_SIZE)
    fig.suptitle("System Metrics — Silent Failure Predictor", fontsize=16, y=1.01,
                 color="white", fontweight="bold")

    anomaly_mask = df["is_anomaly"].astype(bool)

    for ax, (col, label, color) in zip(axes.flat, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue

        x     = df["timestamp"] if "timestamp" in df.columns else df.index
        y     = df[col]

        # Background normal line
        ax.plot(x, y, color=color, linewidth=0.8, alpha=0.8, label="Normal")

        # Overlay anomaly scatter
        if anomaly_mask.sum() > 0:
            ax.scatter(
                x[anomaly_mask], y[anomaly_mask],
                color=_ANOM_CLR, s=25, zorder=5, label="Anomaly"
            )

        ax.set_title(label, fontsize=11, color="white")
        ax.set_xlabel("Time", fontsize=8, color="#aaa")
        ax.set_ylabel(label.split("(")[-1].rstrip(")").strip() if "(" in label else label,
                      fontsize=8, color="#aaa")
        ax.tick_params(colors="#aaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.grid(alpha=0.15)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color=_NORM_CLR, linewidth=2, label="Normal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ANOM_CLR,
               markersize=8, label="Anomaly", linestyle="None"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               framealpha=0.3, fontsize=10)

    plt.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    plt.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    log.info(f"Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 2. Anomaly Score Distribution (matplotlib)
# ---------------------------------------------------------------------------

def plot_anomaly_score_distribution(
        scores:   np.ndarray,
        labels:   np.ndarray,
        filename: str = "anomaly_score_distribution.png",
) -> str:
    """
    Histogram of anomaly scores split by true class.

    Args:
        scores:   Continuous anomaly score per sample (higher = more anomalous).
        labels:   Ground-truth binary labels (0/1).
        filename: Output PNG filename.

    Returns:
        Absolute path to saved PNG.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))

    normal_scores = scores[labels == 0]
    anom_scores   = scores[labels == 1]

    bins = 60
    ax.hist(normal_scores, bins=bins, color=_NORM_CLR, alpha=0.7, label="Normal", density=True)
    ax.hist(anom_scores,   bins=bins, color=_ANOM_CLR, alpha=0.7, label="Anomaly", density=True)

    ax.set_title("Anomaly Score Distribution", fontsize=14, color="white")
    ax.set_xlabel("Anomaly Score (higher → more suspicious)", fontsize=11, color="#aaa")
    ax.set_ylabel("Density", fontsize=11, color="#aaa")
    ax.legend(fontsize=11)
    ax.tick_params(colors="#aaa")
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    plt.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    log.info(f"Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 3. Confusion Matrix (matplotlib)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: list, filename: str = "confusion_matrix.png") -> str:
    """
    Render a 2×2 confusion matrix heatmap.

    Args:
        cm:       2×2 list as returned by sklearn.metrics.confusion_matrix.
        filename: Output PNG filename.

    Returns:
        Absolute path to saved PNG.
    """
    import matplotlib.colors as mcolors

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(6, 5))

    cm_arr = np.array(cm)
    im     = ax.imshow(cm_arr, cmap="Blues")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"], color="white", fontsize=12)
    ax.set_yticklabels(["Normal", "Anomaly"], color="white", fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12, color="#aaa")
    ax.set_ylabel("True",      fontsize=12, color="#aaa")
    ax.set_title("Confusion Matrix", fontsize=14, color="white")

    total = cm_arr.sum()
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            ax.text(j, i, f"{val}\n({val/total*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if val < cm_arr.max() / 2 else "black",
                    fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    plt.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    log.info(f"Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Interactive Plotly Chart (for Streamlit)
# ---------------------------------------------------------------------------

def build_plotly_figure(df: pd.DataFrame, metric: str = "cpu_usage") -> go.Figure:
    """
    Build an interactive Plotly time-series chart for the Streamlit dashboard.

    Args:
        df:     DataFrame with timestamp, metric column, and is_anomaly.
        metric: Column name to plot.

    Returns:
        Plotly Figure object (caller renders it with st.plotly_chart).
    """
    anom = df[df["is_anomaly"] == 1]
    norm = df[df["is_anomaly"] == 0]

    x_col = "timestamp" if "timestamp" in df.columns else df.index

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=norm[x_col], y=norm[metric],
        mode="lines",
        name="Normal",
        line=dict(color=_NORM_CLR, width=1.5),
    ))

    fig.add_trace(go.Scatter(
        x=anom[x_col], y=anom[metric],
        mode="markers",
        name="Anomaly",
        marker=dict(color=_ANOM_CLR, size=8, symbol="x"),
    ))

    pretty = metric.replace("_", " ").title()
    fig.update_layout(
        title=f"{pretty} — Anomaly Detection",
        xaxis_title="Time",
        yaxis_title=pretty,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Autoencoder Reconstruction Error Plot
# ---------------------------------------------------------------------------

def plot_reconstruction_error(
        errors:    np.ndarray,
        threshold: float,
        labels:    np.ndarray,
        filename:  str = "reconstruction_error.png",
) -> str:
    """
    Plot per-sample reconstruction error with threshold line.

    Args:
        errors:    Array of reconstruction MSE values.
        threshold: Decision threshold (samples above = anomaly).
        labels:    Ground-truth binary labels.
        filename:  Output PNG filename.

    Returns:
        Absolute path to saved PNG.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(14, 5))

    idx     = np.arange(len(errors))
    colors  = [_ANOM_CLR if l else _NORM_CLR for l in labels]

    ax.scatter(idx, errors, c=colors, s=10, alpha=0.7)
    ax.axhline(threshold, color="#FBBF24", linewidth=1.5,
               linestyle="--", label=f"Threshold = {threshold:.4f}")

    ax.set_title("Autoencoder Reconstruction Error per Sample", fontsize=14, color="white")
    ax.set_xlabel("Sample Index", fontsize=11, color="#aaa")
    ax.set_ylabel("Reconstruction MSE", fontsize=11, color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.grid(alpha=0.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_NORM_CLR,
               markersize=8, label="Normal", linestyle="None"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_ANOM_CLR,
               markersize=8, label="Anomaly", linestyle="None"),
        Line2D([0], [0], color="#FBBF24", linewidth=1.5, linestyle="--",
               label=f"Threshold ({threshold:.4f})"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    plt.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    log.info(f"Saved → {path}")
    return path
