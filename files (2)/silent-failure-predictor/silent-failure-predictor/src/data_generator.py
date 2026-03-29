"""
data_generator.py
-----------------
Generates a realistic synthetic dataset of system metrics that mimics
production telemetry from a web/microservice application.

The dataset contains six metrics sampled every minute:
    - cpu_usage          (%)
    - memory_usage       (%)
    - response_time_ms   (milliseconds)
    - disk_io_mbps       (MB/s)
    - network_latency_ms (milliseconds)
    - error_rate_pct     (%)

A configurable fraction of rows (~5%) are "injected" silent failures —
subtle or dramatic deviations that represent pre-crash system behaviour.
The `is_anomaly` column is the ground-truth label (1 = anomaly, 0 = normal).

Usage:
    from src.data_generator import generate_dataset
    df = generate_dataset()
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.logger import log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(save: bool = True) -> pd.DataFrame:
    """
    Generate synthetic system-metrics dataset and (optionally) persist it.

    Args:
        save: When True the CSV is written to the path specified in config.

    Returns:
        pd.DataFrame with columns:
            timestamp, cpu_usage, memory_usage, response_time_ms,
            disk_io_mbps, network_latency_ms, error_rate_pct, is_anomaly
    """
    cfg       = load_config()
    data_cfg  = cfg["data"]
    n         = data_cfg["n_samples"]
    seed      = data_cfg["random_seed"]
    anom_frac = data_cfg["anomaly_fraction"]
    out_path  = data_cfg["output_path"]

    rng = np.random.default_rng(seed)

    log.info(f"Generating dataset | n_samples={n} | anomaly_fraction={anom_frac}")

    # -----------------------------------------------------------------------
    # 1. Normal baseline signals
    # -----------------------------------------------------------------------
    m = cfg["data"]["metrics"]

    timestamps = _make_timestamps(n)

    # Add smooth daily-cycle trend to CPU and memory to look realistic
    t = np.linspace(0, 4 * np.pi, n)  # Two full sine cycles across the series
    daily_cpu_wave    = 8 * np.sin(t)
    daily_memory_wave = 5 * np.sin(t + np.pi / 4)

    cpu    = _clamp(rng.normal(m["cpu_usage"]["mean"],    m["cpu_usage"]["std"],    n) + daily_cpu_wave,    0, 100)
    mem    = _clamp(rng.normal(m["memory_usage"]["mean"], m["memory_usage"]["std"], n) + daily_memory_wave, 0, 100)
    rt     = _clamp(rng.normal(m["response_time_ms"]["mean"], m["response_time_ms"]["std"], n), 50, 1000)
    diskio = _clamp(rng.normal(m["disk_io_mbps"]["mean"],     m["disk_io_mbps"]["std"],     n),  0, 200)
    netlat = _clamp(rng.normal(m["network_latency_ms"]["mean"], m["network_latency_ms"]["std"], n), 1, 100)
    errrt  = _clamp(rng.normal(m["error_rate_pct"]["mean"],   m["error_rate_pct"]["std"],   n),  0,  5)

    is_anomaly = np.zeros(n, dtype=int)

    # -----------------------------------------------------------------------
    # 2. Inject silent failure patterns
    # -----------------------------------------------------------------------
    n_anomalies = int(n * anom_frac)
    anomaly_indices = rng.choice(n, size=n_anomalies, replace=False)

    # Five distinct failure "signatures"
    patterns = [
        _pattern_memory_leak,
        _pattern_cpu_spike,
        _pattern_slow_response,
        _pattern_cascade_failure,
        _pattern_disk_saturation,
    ]

    for idx in anomaly_indices:
        pattern_fn = patterns[rng.integers(len(patterns))]
        cpu[idx], mem[idx], rt[idx], diskio[idx], netlat[idx], errrt[idx] = pattern_fn(
            cpu[idx], mem[idx], rt[idx], diskio[idx], netlat[idx], errrt[idx], rng
        )
        is_anomaly[idx] = 1

    # -----------------------------------------------------------------------
    # 3. Assemble DataFrame
    # -----------------------------------------------------------------------
    df = pd.DataFrame({
        "timestamp":          timestamps,
        "cpu_usage":          np.round(cpu,    2),
        "memory_usage":       np.round(mem,    2),
        "response_time_ms":   np.round(rt,     2),
        "disk_io_mbps":       np.round(diskio, 2),
        "network_latency_ms": np.round(netlat, 2),
        "error_rate_pct":     np.round(errrt,  4),
        "is_anomaly":         is_anomaly,
    })

    log.info(f"Dataset created | rows={len(df)} | anomalies={is_anomaly.sum()} "
             f"({is_anomaly.mean() * 100:.1f}%)")

    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Dataset saved → {out_path}")

    return df


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the dataset from CSV.  Generates it first if the file is absent.

    Args:
        path: Override the default path from config.yaml.

    Returns:
        pd.DataFrame as returned by generate_dataset().
    """
    if path is None:
        path = load_config()["data"]["output_path"]

    if not os.path.exists(path):
        log.warning(f"Dataset not found at '{path}'. Generating now …")
        return generate_dataset(save=True)

    df = pd.read_csv(path, parse_dates=["timestamp"])
    log.info(f"Dataset loaded from '{path}' | rows={len(df)}")
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_timestamps(n: int, start: str = "2024-01-01 00:00:00") -> pd.DatetimeIndex:
    """Create a DatetimeIndex with 1-minute intervals."""
    base = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    return pd.to_datetime([base + timedelta(minutes=i) for i in range(n)])


def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip array values to [lo, hi]."""
    return np.clip(arr, lo, hi)


# ── Failure Pattern Functions ────────────────────────────────────────────────

def _pattern_memory_leak(cpu, mem, rt, diskio, netlat, errrt, rng):
    """Gradually exhausting RAM — memory climbs while CPU stays normal."""
    mem    = _clamp(mem + rng.uniform(30, 45), 0, 100)
    rt     = _clamp(rt  + rng.uniform(100, 400), 50, 5000)
    errrt  = _clamp(errrt + rng.uniform(1, 5), 0, 100)
    return cpu, mem, rt, diskio, netlat, errrt


def _pattern_cpu_spike(cpu, mem, rt, diskio, netlat, errrt, rng):
    """Runaway process — CPU pegged near 100% causing latency."""
    cpu   = _clamp(cpu + rng.uniform(40, 55), 0, 100)
    rt    = _clamp(rt  + rng.uniform(200, 600), 50, 5000)
    errrt = _clamp(errrt + rng.uniform(0.5, 3), 0, 100)
    return cpu, mem, rt, diskio, netlat, errrt


def _pattern_slow_response(cpu, mem, rt, diskio, netlat, errrt, rng):
    """DB/network bottleneck — response time explodes while CPU looks fine."""
    rt     = _clamp(rt     + rng.uniform(800, 2500), 50, 5000)
    netlat = _clamp(netlat + rng.uniform(100, 300),   1,  500)
    errrt  = _clamp(errrt  + rng.uniform(2,   8),     0,  100)
    return cpu, mem, rt, diskio, netlat, errrt


def _pattern_cascade_failure(cpu, mem, rt, diskio, netlat, errrt, rng):
    """Everything degrades simultaneously — precursor to full crash."""
    cpu    = _clamp(cpu    + rng.uniform(30, 50), 0,   100)
    mem    = _clamp(mem    + rng.uniform(25, 40), 0,   100)
    rt     = _clamp(rt     + rng.uniform(500, 1500), 50, 5000)
    diskio = _clamp(diskio + rng.uniform(50, 150),    0,  500)
    netlat = _clamp(netlat + rng.uniform(50, 150),    1,  500)
    errrt  = _clamp(errrt  + rng.uniform(5,  20),     0,  100)
    return cpu, mem, rt, diskio, netlat, errrt


def _pattern_disk_saturation(cpu, mem, rt, diskio, netlat, errrt, rng):
    """Disk I/O saturation — writes queue up, application stalls."""
    diskio = _clamp(diskio + rng.uniform(150, 300),  0, 500)
    rt     = _clamp(rt     + rng.uniform(300, 800), 50, 5000)
    cpu    = _clamp(cpu    + rng.uniform(10, 25),    0, 100)
    return cpu, mem, rt, diskio, netlat, errrt


# ---------------------------------------------------------------------------
# CLI entry point (python -m src.data_generator)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset(save=True)
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean() * 100:.2f}%)")
