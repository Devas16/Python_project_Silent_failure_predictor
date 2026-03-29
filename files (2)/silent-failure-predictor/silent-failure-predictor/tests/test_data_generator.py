"""
tests/test_data_generator.py
-----------------------------
Unit tests for the data generation module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_dataset, load_dataset


class TestGenerateDataset:
    def test_returns_dataframe(self):
        df = generate_dataset(save=False)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self):
        df = generate_dataset(save=False)
        expected = [
            "timestamp", "cpu_usage", "memory_usage",
            "response_time_ms", "disk_io_mbps",
            "network_latency_ms", "error_rate_pct", "is_anomaly",
        ]
        assert list(df.columns) == expected

    def test_row_count(self):
        df = generate_dataset(save=False)
        assert len(df) == 2000   # default n_samples from config

    def test_anomaly_fraction(self):
        df = generate_dataset(save=False)
        frac = df["is_anomaly"].mean()
        # Allow ±2% around configured 5%
        assert 0.03 <= frac <= 0.07

    def test_cpu_range(self):
        df = generate_dataset(save=False)
        assert df["cpu_usage"].between(0, 100).all()

    def test_memory_range(self):
        df = generate_dataset(save=False)
        assert df["memory_usage"].between(0, 100).all()

    def test_response_time_positive(self):
        df = generate_dataset(save=False)
        assert (df["response_time_ms"] > 0).all()

    def test_no_nulls(self):
        df = generate_dataset(save=False)
        assert df.isnull().sum().sum() == 0

    def test_is_anomaly_binary(self):
        df = generate_dataset(save=False)
        assert set(df["is_anomaly"].unique()).issubset({0, 1})

    def test_timestamps_monotonic(self):
        df = generate_dataset(save=False)
        assert df["timestamp"].is_monotonic_increasing
