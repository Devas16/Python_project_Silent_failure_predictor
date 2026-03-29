"""
tests/test_preprocessor.py
---------------------------
Unit tests for the Preprocessor pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import generate_dataset
from src.preprocessor import Preprocessor, prepare_data


@pytest.fixture(scope="module")
def raw_df():
    return generate_dataset(save=False)


class TestPreprocessor:
    def test_fit_transform_returns_array(self, raw_df):
        prep = Preprocessor()
        X    = prep.fit_transform(raw_df)
        assert isinstance(X, np.ndarray)

    def test_no_nans_after_fit_transform(self, raw_df):
        prep = Preprocessor()
        X    = prep.fit_transform(raw_df)
        assert not np.isnan(X).any()

    def test_feature_names_set(self, raw_df):
        prep = Preprocessor()
        prep.fit_transform(raw_df)
        assert len(prep.feature_names) > 0

    def test_transform_without_fit_raises(self, raw_df):
        prep = Preprocessor()
        with pytest.raises(RuntimeError):
            prep.transform(raw_df)

    def test_rolling_expands_features(self, raw_df):
        prep_roll   = Preprocessor()
        prep_roll._use_rolling = True
        X_roll      = prep_roll.fit_transform(raw_df)

        prep_no_roll = Preprocessor()
        prep_no_roll._use_rolling = False
        X_base       = prep_no_roll.fit_transform(raw_df)

        assert X_roll.shape[1] > X_base.shape[1]

    def test_prepare_data_returns_three_items(self, raw_df):
        X, y, prep = prepare_data(raw_df)
        assert X.shape[0] == len(raw_df)
        assert y.shape[0] == len(raw_df)
        assert prep._fitted

    def test_scaled_mean_approx_zero(self, raw_df):
        prep = Preprocessor()
        X    = prep.fit_transform(raw_df)
        # After StandardScaler the column means should be near 0
        assert np.abs(X.mean(axis=0)).max() < 0.1

    def test_save_load_roundtrip(self, raw_df, tmp_path):
        save_path = str(tmp_path / "scaler.pkl")
        prep = Preprocessor()
        X1   = prep.fit_transform(raw_df)
        prep.save(save_path)

        prep2 = Preprocessor()
        prep2.load(save_path)
        X2 = prep2.transform(raw_df)

        np.testing.assert_array_almost_equal(X1, X2)
