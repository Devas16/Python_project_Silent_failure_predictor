"""
train.py
--------
End-to-end training pipeline for the Silent Failure Predictor.

Steps:
  1. Load (or generate) the system-metrics dataset.
  2. Preprocess: feature engineering + StandardScaler.
  3. Train Isolation Forest (primary model).
  4. Optionally train Autoencoder (bonus model).
  5. Evaluate both models and print metrics.
  6. Save models, scaler, and all visualisation plots.

Run:
    python -m src.train
    python -m src.train --skip-autoencoder
"""

import argparse
import sys
import numpy as np

from src.config_loader import load_config
from src.data_generator import load_dataset
from src.preprocessor import prepare_data
from src.isolation_forest_model import IsolationForestDetector
from src.visualizer import (
    plot_metrics_overview,
    plot_anomaly_score_distribution,
    plot_confusion_matrix,
    plot_reconstruction_error,
)
from src.logger import log


def parse_args():
    parser = argparse.ArgumentParser(description="Train Silent Failure Predictor")
    parser.add_argument(
        "--skip-autoencoder", action="store_true",
        help="Skip Autoencoder training (faster, no TensorFlow required)"
    )
    return parser.parse_args()


def run_isolation_forest(X: np.ndarray, y: np.ndarray):
    """Train, evaluate and save the Isolation Forest model."""
    log.info("=" * 60)
    log.info("  ISOLATION FOREST PIPELINE")
    log.info("=" * 60)

    detector = IsolationForestDetector()
    detector.train(X)

    labels, scores = detector.predict(X)
    metrics        = detector.evaluate(X, y)

    log.info(f"ROC-AUC          : {metrics['roc_auc']}")
    log.info(f"Average Precision: {metrics['average_precision']}")

    detector.save()

    # Plots
    plot_anomaly_score_distribution(scores, y)
    plot_confusion_matrix(metrics["confusion_matrix"])

    return detector, labels, scores, metrics


def run_autoencoder(X: np.ndarray, y: np.ndarray):
    """Train, evaluate and save the Autoencoder (requires TensorFlow)."""
    log.info("=" * 60)
    log.info("  AUTOENCODER PIPELINE")
    log.info("=" * 60)

    try:
        from src.autoencoder_model import AutoencoderDetector
        from sklearn.metrics import classification_report, roc_auc_score
    except ImportError as e:
        log.error(f"Cannot train Autoencoder: {e}")
        return None

    # Train only on normal samples
    normal_mask = y == 0
    X_normal    = X[normal_mask]

    ae = AutoencoderDetector(input_dim=X.shape[1])
    ae.train(X_normal)

    ae_labels, ae_errors = ae.predict(X)
    ae_roc               = roc_auc_score(y, ae_errors)

    log.info(f"\n{classification_report(y, ae_labels, target_names=['Normal','Anomaly'])}")
    log.info(f"Autoencoder ROC-AUC: {ae_roc:.4f}")

    ae.save()
    plot_reconstruction_error(ae_errors, ae.threshold, y)

    return ae


def main():
    args = parse_args()
    cfg  = load_config()

    log.info("╔══════════════════════════════════════════════╗")
    log.info("║   Silent Failure Predictor — Training        ║")
    log.info("╚══════════════════════════════════════════════╝")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    df = load_dataset()
    log.info(f"Dataset shape: {df.shape}")
    log.info(f"Class balance: {df['is_anomaly'].value_counts().to_dict()}")

    # Metrics overview plot
    plot_metrics_overview(df)

    # ------------------------------------------------------------------
    # 2. Preprocessing
    # ------------------------------------------------------------------
    X, y, preprocessor = prepare_data(df)
    preprocessor.save()
    log.info(f"Feature matrix shape: {X.shape}")

    # ------------------------------------------------------------------
    # 3. Isolation Forest
    # ------------------------------------------------------------------
    detector, if_labels, if_scores, if_metrics = run_isolation_forest(X, y)

    # ------------------------------------------------------------------
    # 4. Autoencoder (optional)
    # ------------------------------------------------------------------
    if not args.skip_autoencoder:
        ae = run_autoencoder(X, y)
        if ae is None:
            log.warning("Autoencoder skipped due to import error.")
    else:
        log.info("Autoencoder training skipped (--skip-autoencoder flag).")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("  TRAINING COMPLETE — Summary")
    log.info("=" * 60)
    log.info(f"  Total samples   : {len(df)}")
    log.info(f"  True anomalies  : {y.sum()}")
    log.info(f"  IF detected     : {if_labels.sum()}")
    log.info(f"  IF ROC-AUC      : {if_metrics['roc_auc']}")
    log.info(f"  Models saved    : models/")
    log.info(f"  Plots saved     : outputs/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
