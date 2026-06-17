"""
Tests for the Model Trainer Module
"""

import numpy as np
import pytest

from src.models.trainer import (
    get_models,
    train_single_model,
    cross_validate_model,
)


@pytest.fixture
def sample_training_data():
    """Create a small synthetic dataset for testing model training."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear boundary

    return X, y


class TestGetModels:
    """Tests for get_models."""

    def test_returns_four_models(self):
        models = get_models()
        assert len(models) == 4

    def test_contains_expected_model_names(self):
        models = get_models()
        expected = {
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM",
        }
        assert set(models.keys()) == expected

    def test_models_have_fit_predict(self):
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"
            assert hasattr(model, "predict_proba"), f"{name} missing predict_proba()"


class TestTrainSingleModel:
    """Tests for train_single_model."""

    def test_trains_without_error(self, sample_training_data):
        X, y = sample_training_data
        models = get_models()

        for name, model in models.items():
            fitted = train_single_model(model, X, y, name)
            assert fitted is not None

    def test_fitted_model_can_predict(self, sample_training_data):
        X, y = sample_training_data
        models = get_models()

        for name, model in models.items():
            fitted = train_single_model(model, X, y, name)
            predictions = fitted.predict(X[:5])

            assert len(predictions) == 5
            assert set(predictions).issubset({0, 1})


class TestCrossValidateModel:
    """Tests for cross_validate_model."""

    def test_returns_valid_result(self, sample_training_data):
        X, y = sample_training_data
        models = get_models()
        model = models["Logistic Regression"]

        result = cross_validate_model(model, X, y, "Logistic Regression", cv=3)

        assert "model_name" in result
        assert "mean_score" in result
        assert "std_score" in result
        assert "all_scores" in result
        assert 0 <= result["mean_score"] <= 1
        assert len(result["all_scores"]) == 3
