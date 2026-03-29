"""
isolation_forest_model.py
--------------------------
Isolation Forest anomaly detector.

WHY ISOLATION FOREST?
---------------------
Isolation Forest is an unsupervised algorithm that works by randomly
partitioning data using decision-tree splits.  Anomalous points require
*fewer* splits to isolate because they occupy sparse regions of the feature
space.  This makes it:
  • Parameter-light  — only *contamination* and *n_estimators* matter.
  • Scalable         — O(n log n) time, works on large datasets.
  • Unsupervised     — no labels required; ideal for real-world monitoring
                       where "normal" is plentiful but anomalies are rare/unlabelled.

SILENT FAILURES in this context are metric readings that deviate subtly
from normal operating patterns — not necessarily catastrophic spikes, but
the quiet precursors: memory creep, latency drift, rising error rates.
Isolation Forest scores each sample with an *anomaly_score* (the lower the
more anomalous) and thresholds it to a binary prediction.

Usage:
    from src.isolation_forest_model import IsolationForestDetector
    model = IsolationForestDetector()
    model.train(X_train)
    preds, scores = model.predict(X_test)
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from typing import Tuple, Optional

from src.config_loader import load_config
from src.logger import log


class IsolationForestDetector:
    """
    Wrapper around scikit-learn's IsolationForest with save/load,
    evaluation, and structured logging.

    Attributes:
        model          : Trained IsolationForest instance.
        contamination  : Expected anomaly fraction passed to the algorithm.
        _fitted        : Whether the model has been trained.
    """

    def __init__(self):
        cfg = load_config()["isolation_forest"]

        self.contamination = cfg.get("contamination", 0.05)
        self._model_path   = cfg.get("model_path", "models/isolation_forest.pkl")

        self.model = IsolationForest(
            n_estimators  = cfg.get("n_estimators", 200),
            contamination = self.contamination,
            max_samples   = cfg.get("max_samples", "auto"),
            random_state  = cfg.get("random_state", 42),
            n_jobs        = -1,   # use all CPU cores
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Core Methods
    # ------------------------------------------------------------------

    def train(self, X: np.ndarray) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest on the training feature matrix.

        Args:
            X: 2-D numpy array (n_samples, n_features) — scaled.

        Returns:
            self (for method chaining).
        """
        log.info(f"Training Isolation Forest | n_estimators={self.model.n_estimators} "
                 f"| contamination={self.contamination} | X.shape={X.shape}")
        self.model.fit(X)
        self._fitted = True
        log.info("Isolation Forest training complete.")
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly labels and raw anomaly scores.

        sklearn's IsolationForest returns:
          • decision_function → higher = more normal  (we negate for intuition)
          • predict           → +1 normal, -1 anomaly

        We convert to binary labels: 0 = normal, 1 = anomaly.

        Args:
            X: 2-D numpy array (n_samples, n_features) — scaled.

        Returns:
            (labels, scores)
              labels  : int array, 0 = normal | 1 = anomaly
              scores  : float array, higher = more anomalous
        """
        self._check_fitted()
        raw_preds    = self.model.predict(X)               # +1 or -1
        anomaly_score = -self.model.decision_function(X)  # negate: high = anomalous

        # Convert sklearn convention (+1/-1) → (0/1)
        labels = np.where(raw_preds == -1, 1, 0)

        log.info(f"Prediction done | anomalies detected={labels.sum()} / {len(labels)}")
        return labels, anomaly_score

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return continuous anomaly scores (higher = more suspicious)."""
        self._check_fitted()
        return -self.model.decision_function(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Evaluate model performance against ground-truth labels.

        Args:
            X:      Scaled feature matrix.
            y_true: Binary ground-truth labels (0/1).

        Returns:
            Dictionary with accuracy metrics and sklearn's full report.
        """
        self._check_fitted()
        y_pred, scores = self.predict(X)

        report  = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"],
                                        output_dict=True)
        cm      = confusion_matrix(y_true, y_pred).tolist()
        roc_auc = roc_auc_score(y_true, scores)
        avg_pr  = average_precision_score(y_true, scores)

        metrics = {
            "classification_report": report,
            "confusion_matrix":      cm,
            "roc_auc":               round(roc_auc, 4),
            "average_precision":     round(avg_pr, 4),
        }

        log.info(f"Evaluation | ROC-AUC={roc_auc:.4f} | Avg-Precision={avg_pr:.4f}")
        log.info("\n" + classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Persist the fitted model to disk using joblib."""
        self._check_fitted()
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"Isolation Forest saved → {path}")
        return path

    def load(self, path: Optional[str] = None) -> "IsolationForestDetector":
        """Restore a previously saved model from disk."""
        path = path or self._model_path
        self.model  = joblib.load(path)
        self._fitted = True
        log.info(f"Isolation Forest loaded ← {path}")
        return self

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first."
            )
