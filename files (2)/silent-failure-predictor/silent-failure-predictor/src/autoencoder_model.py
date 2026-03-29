"""
autoencoder_model.py
---------------------
Autoencoder-based anomaly detector — bonus model.

HOW IT WORKS:
-------------
An Autoencoder is a neural network trained to *reconstruct* its input.
When trained exclusively on normal data it learns a compact internal
representation of "what normal looks like".  At inference time, samples
that differ from the training distribution (i.e. anomalies) will be
reconstructed poorly — their Mean Squared Error (MSE) will be high.

We set a threshold at the 95th percentile of reconstruction errors on the
training set.  Any test sample whose error exceeds this threshold is
flagged as an anomaly.

This approach is complementary to Isolation Forest:
  • Autoencoder is better at detecting *pattern* deviations in correlated metrics.
  • Isolation Forest is better at detecting *point* outliers.

Architecture:
    Input (n_features)
      → Dense(32, relu) → Dense(encoding_dim, relu)   # encoder
      → Dense(32, relu) → Dense(n_features, sigmoid)  # decoder

Usage:
    from src.autoencoder_model import AutoencoderDetector
    ae = AutoencoderDetector(input_dim=18)
    ae.train(X_normal)        # train ONLY on normal samples
    labels, errors = ae.predict(X_test)
"""

import os
import numpy as np
from typing import Tuple, Optional

from src.config_loader import load_config
from src.logger import log

# Lazy import TensorFlow so the project doesn't crash if TF is not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    log.warning("TensorFlow not installed. AutoencoderDetector will not be available.")


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detector.

    The model is trained on *normal* samples only so that it learns to
    reconstruct normal behaviour.  Anomalies produce high reconstruction error.

    Args:
        input_dim: Number of input features (must match training data width).
    """

    def __init__(self, input_dim: int):
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for AutoencoderDetector. "
                "Install it with: pip install tensorflow"
            )

        cfg            = load_config()["autoencoder"]
        self._cfg      = cfg
        self.input_dim = input_dim
        self.threshold: Optional[float] = None

        self._model_path = cfg.get("model_path", "models/autoencoder.h5")
        self._model      = self._build_model(
            encoding_dim=cfg.get("encoding_dim", 4),
            lr=cfg.get("learning_rate", 0.001),
        )
        self._fitted     = False

    # ------------------------------------------------------------------
    # Model Construction
    # ------------------------------------------------------------------

    def _build_model(self, encoding_dim: int, lr: float) -> "keras.Model":
        """Construct the Autoencoder graph."""
        inp = keras.Input(shape=(self.input_dim,), name="input")

        # Encoder
        encoded = keras.layers.Dense(32, activation="relu", name="enc_1")(inp)
        encoded = keras.layers.Dense(encoding_dim, activation="relu", name="bottleneck")(encoded)

        # Decoder
        decoded = keras.layers.Dense(32, activation="relu", name="dec_1")(encoded)
        decoded = keras.layers.Dense(self.input_dim, activation="sigmoid", name="output")(decoded)

        model = keras.Model(inputs=inp, outputs=decoded, name="autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )
        log.info(f"Autoencoder built | input_dim={self.input_dim} | bottleneck={encoding_dim}")
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X_normal: np.ndarray) -> "AutoencoderDetector":
        """
        Train on normal-only data.  Labels are not required.

        Args:
            X_normal: 2-D array of *normal* scaled samples.

        Returns:
            self
        """
        cfg = self._cfg
        log.info(
            f"Training Autoencoder | epochs={cfg['epochs']} "
            f"| batch_size={cfg['batch_size']} | X.shape={X_normal.shape}"
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0
            ),
        ]

        self._model.fit(
            X_normal, X_normal,             # input == target for autoencoders
            epochs          = cfg["epochs"],
            batch_size      = cfg["batch_size"],
            validation_split= cfg["validation_split"],
            callbacks       = callbacks,
            verbose         = 0,
        )

        # Compute reconstruction errors on training data to set threshold
        recon_errors    = self._reconstruction_errors(X_normal)
        pct             = cfg.get("threshold_percentile", 95)
        self.threshold  = float(np.percentile(recon_errors, pct))
        self._fitted    = True

        log.info(
            f"Autoencoder trained | reconstruction error "
            f"p{pct}={self.threshold:.6f}"
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify samples using reconstruction error.

        Args:
            X: 2-D scaled feature matrix.

        Returns:
            (labels, errors)
              labels : int array 0/1 — 1 = anomaly
              errors : float array of per-sample reconstruction MSE
        """
        self._check_fitted()
        errors = self._reconstruction_errors(X)
        labels = (errors > self.threshold).astype(int)

        log.info(
            f"Autoencoder predict | anomalies={labels.sum()} / {len(labels)} "
            f"| threshold={self.threshold:.6f}"
        )
        return labels, errors

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample reconstruction MSE."""
        return self._reconstruction_errors(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save Keras model weights and threshold."""
        self._check_fitted()
        path = path or self._model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._model.save(path)
        # Save threshold alongside the model
        np.save(path.replace(".h5", "_threshold.npy"), np.array([self.threshold]))
        log.info(f"Autoencoder saved → {path}")
        return path

    def load(self, path: Optional[str] = None) -> "AutoencoderDetector":
        """Restore saved Keras model and threshold."""
        path = path or self._model_path
        self._model    = keras.models.load_model(path)
        thr_path       = path.replace(".h5", "_threshold.npy")
        self.threshold = float(np.load(thr_path)[0])
        self._fitted   = True
        log.info(f"Autoencoder loaded ← {path} | threshold={self.threshold:.6f}")
        return self

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE between input and reconstruction."""
        X_pred = self._model.predict(X, verbose=0)
        return np.mean(np.power(X - X_pred, 2), axis=1)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "Model is not trained. Call train() or load() first."
            )
