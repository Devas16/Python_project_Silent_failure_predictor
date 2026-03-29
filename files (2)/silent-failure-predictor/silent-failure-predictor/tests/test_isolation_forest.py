"""
tests/test_isolation_forest.py
-------------------------------
Unit tests for the Isolation Forest anomaly detector.
"""

import pytest
import numpy as np
from src.data_generator import generate_dataset
from src.preprocessor import prepare_data
from src.isolation_forest_model import IsolationForestDetector


@pytest.fixture(scope="module")
def prepared_data():
    """Shared fixture: generate data and preprocess once per module."""
    df = generate_dataset(save=False)
    X, y, prep = prepare_data(df)
    return X, y, prep


class TestIsolationForestDetector:
    def test_train_does_not_raise(self, prepared_data):
        X, y, _ = prepared_data
        det = IsolationForestDetector()
        det.train(X)   # should not raise

    def test_predict_shape(self, prepared_data):
        X, y, _ = prepared_data
        det = IsolationForestDetector()
        det.train(X)
        labels, scores = det.predict(X)
        assert labels.shape == (len(X),)
        assert scores.shape == (len(X),)

    def test_labels_binary(self, prepared_data):
        X, y, _ = prepared_data
        det = IsolationForestDetector()
        det.train(X)
        labels, _ = det.predict(X)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_roc_auc_above_chance(self, prepared_data):
        X, y, _ = prepared_data
        det = IsolationForestDetector()
        det.train(X)
        metrics = det.evaluate(X, y)
        assert metrics["roc_auc"] > 0.5, "ROC-AUC should be above random chance"

    def test_predict_without_fit_raises(self):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            det.predict(np.zeros((5, 10)))

    def test_save_and_load(self, tmp_path, prepared_data):
        X, y, _ = prepared_data
        save_path = str(tmp_path / "test_if.pkl")

        det = IsolationForestDetector()
        det.train(X)
        det.save(save_path)

        det2 = IsolationForestDetector()
        det2.load(save_path)
        labels2, _ = det2.predict(X)
        assert labels2.shape == (len(X),)
