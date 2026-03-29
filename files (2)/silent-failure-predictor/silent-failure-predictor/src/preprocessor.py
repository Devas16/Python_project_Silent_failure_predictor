"""
preprocessor.py
---------------
Feature engineering and preprocessing pipeline for the system-metrics dataset.

Steps performed:
  1. Drop non-numeric columns (timestamp, is_anomaly) before scaling.
  2. Optionally compute rolling statistics (mean + std) over a sliding window
     to capture temporal context — critical for detecting gradual drift.
  3. StandardScaler normalisation so that Isolation Forest and the Autoencoder
     are not biased by metrics with large absolute ranges.

All fitted transformers are preserved so the same pipeline can be applied to
unseen / streaming data at inference time.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from src.config_loader import load_config
from src.logger import log


class Preprocessor:
    """
    Stateful preprocessing pipeline.

    Attributes:
        scaler         : Fitted StandardScaler instance.
        feature_names  : List of feature column names after rolling expansion.
        _fitted        : Whether fit() has been called.
    """

    # Default save path for the fitted scaler
    _SCALER_PATH = os.path.join("models", "scaler.pkl")

    def __init__(self):
        cfg = load_config()
        feat_cfg = cfg["features"]

        self._use_rolling = feat_cfg.get("use_rolling_stats", True)
        self._window      = feat_cfg.get("rolling_window", 10)
        self._base_feats  = feat_cfg.get("features_list", [
            "cpu_usage", "memory_usage", "response_time_ms",
            "disk_io_mbps", "network_latency_ms", "error_rate_pct",
        ])

        self.scaler        = StandardScaler()
        self.feature_names: list = []
        self._fitted       = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the scaler on *df* and return the scaled feature matrix.

        Args:
            df: Raw dataframe containing at minimum the base feature columns.

        Returns:
            2-D numpy array (n_samples, n_features) — scaled.
        """
        features_df = self._build_features(df)
        self.feature_names = list(features_df.columns)

        X_scaled = self.scaler.fit_transform(features_df.values)
        self._fitted = True

        log.info(
            f"Preprocessor fitted | features={self.feature_names} "
            f"| shape={X_scaled.shape}"
        )
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply the already-fitted scaler to new data.

        Args:
            df: Raw dataframe with the same base columns used during fit.

        Returns:
            2-D scaled numpy array.

        Raises:
            RuntimeError: If fit_transform() has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "Preprocessor has not been fitted. "
                "Call fit_transform() first or load a saved scaler."
            )
        features_df = self._build_features(df)
        return self.scaler.transform(features_df.values)

    def save(self, path: Optional[str] = None) -> str:
        """Persist the fitted scaler to disk."""
        path = path or self._SCALER_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"scaler": self.scaler, "feature_names": self.feature_names}, path)
        log.info(f"Scaler saved → {path}")
        return path

    def load(self, path: Optional[str] = None) -> "Preprocessor":
        """Restore a previously saved scaler from disk."""
        path = path or self._SCALER_PATH
        payload = joblib.load(path)
        self.scaler        = payload["scaler"]
        self.feature_names = payload["feature_names"]
        self._fitted       = True
        log.info(f"Scaler loaded ← {path} | features={self.feature_names}")
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and optionally expand base features with rolling statistics.

        Rolling mean  captures smoothed trend — gradual drift stands out.
        Rolling std   captures volatility   — sudden instability stands out.
        NaN rows produced at the window boundary are forward-filled to avoid
        dropping data at the start of the series.
        """
        # Select only the base metric columns
        base = df[self._base_feats].copy()

        if not self._use_rolling:
            return base

        rolling_frames = [base]  # always keep the raw features

        for col in self._base_feats:
            roll = base[col].rolling(window=self._window, min_periods=1)
            rolling_frames.append(
                roll.mean().rename(f"{col}_roll_mean")
            )
            rolling_frames.append(
                roll.std().fillna(0).rename(f"{col}_roll_std")
            )

        combined = pd.concat(rolling_frames, axis=1)
        combined = combined.ffill().fillna(0)  # forward-fill then zero-fill edges
        return combined


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Preprocessor]:
    """
    One-shot convenience wrapper used by training scripts.

    Returns:
        X_scaled     : Scaled feature matrix.
        y            : Ground-truth anomaly labels (0/1) as numpy array.
        preprocessor : Fitted Preprocessor instance.
    """
    prep = Preprocessor()
    X    = prep.fit_transform(df)
    y    = df["is_anomaly"].values if "is_anomaly" in df.columns else np.zeros(len(df), dtype=int)
    return X, y, prep
