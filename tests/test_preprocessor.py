"""
Tests for the Data Preprocessor Module
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import (
    _identify_feature_types,
    build_preprocessing_pipeline,
)


@pytest.fixture
def sample_dataframe():
    """Create a small sample DataFrame for testing."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "annual_income": [300000, 500000, 750000, 1000000, 1500000],
        "loan_amount": [100000, 200000, 300000, 400000, 500000],
        "credit_score": [650, 700, 750, 800, 850],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "education": ["Graduate", "Not Graduate", "Graduate", "Graduate", "Not Graduate"],
    })


class TestIdentifyFeatureTypes:
    """Tests for _identify_feature_types."""

    def test_separates_numerical_and_categorical(self, sample_dataframe):
        numerical, categorical = _identify_feature_types(sample_dataframe)

        assert "age" in numerical
        assert "annual_income" in numerical
        assert "gender" in categorical
        assert "education" in categorical

    def test_no_overlap(self, sample_dataframe):
        numerical, categorical = _identify_feature_types(sample_dataframe)

        overlap = set(numerical) & set(categorical)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_columns_classified(self, sample_dataframe):
        numerical, categorical = _identify_feature_types(sample_dataframe)

        all_classified = set(numerical) | set(categorical)
        assert all_classified == set(sample_dataframe.columns)


class TestBuildPreprocessingPipeline:
    """Tests for build_preprocessing_pipeline."""

    def test_pipeline_creates_successfully(self):
        pipeline = build_preprocessing_pipeline(
            numerical_cols=["age", "income"],
            categorical_cols=["gender"],
        )
        assert pipeline is not None

    def test_pipeline_transforms_data(self, sample_dataframe):
        numerical, categorical = _identify_feature_types(sample_dataframe)
        pipeline = build_preprocessing_pipeline(numerical, categorical)

        X_transformed = pipeline.fit_transform(sample_dataframe)

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(sample_dataframe)
        # Should have more columns than numerical due to one-hot encoding
        assert X_transformed.shape[1] >= len(numerical)

    def test_numerical_scaling(self, sample_dataframe):
        numerical, categorical = _identify_feature_types(sample_dataframe)
        pipeline = build_preprocessing_pipeline(numerical, categorical)

        X_transformed = pipeline.fit_transform(sample_dataframe)

        # After StandardScaler, numerical features should have ~0 mean
        for i in range(len(numerical)):
            col_mean = X_transformed[:, i].mean()
            assert abs(col_mean) < 1e-10, (
                f"Column {i} ({numerical[i]}) mean = {col_mean}, expected ~0"
            )
