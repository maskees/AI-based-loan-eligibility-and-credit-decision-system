"""
Tests for the Explainability Modules (SHAP & LIME)
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def trained_model_and_data():
    """Train a simple model for explainability testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    feature_names = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]

    return model, X, y, feature_names


class TestSHAPExplainer:
    """Tests for the SHAP explainer."""

    def test_shap_explainer_initializes(self, trained_model_and_data):
        from src.explainability.shap_explainer import SHAPExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, X[:20], feature_names)

        assert explainer is not None
        assert explainer.feature_names == feature_names

    def test_shap_values_computed(self, trained_model_and_data):
        from src.explainability.shap_explainer import SHAPExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, X[:20], feature_names)
        values = explainer.compute_shap_values()

        assert values is not None
        assert values.shape[0] == 20
        assert values.shape[1] == 5

    def test_top_features_returns_correct_count(self, trained_model_and_data):
        from src.explainability.shap_explainer import SHAPExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, X[:20], feature_names)
        explainer.compute_shap_values()

        top = explainer.get_top_features(0, top_n=3)

        assert len(top) == 3
        assert all("feature" in f for f in top)
        assert all("shap_value" in f for f in top)
        assert all("impact" in f for f in top)

    def test_global_importance_sorted(self, trained_model_and_data):
        from src.explainability.shap_explainer import SHAPExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, X[:20], feature_names)
        explainer.compute_shap_values()

        importances = explainer.get_global_feature_importance()

        assert len(importances) == 5
        # Should be sorted descending by importance
        for i in range(len(importances) - 1):
            assert importances[i]["importance"] >= importances[i + 1]["importance"]


class TestLIMEExplainer:
    """Tests for the LIME explainer."""

    def test_lime_explainer_initializes(self, trained_model_and_data):
        from src.explainability.lime_explainer import LIMEExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = LIMEExplainer(model, X, feature_names)

        assert explainer is not None

    def test_lime_explanation_as_list(self, trained_model_and_data):
        from src.explainability.lime_explainer import LIMEExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = LIMEExplainer(model, X, feature_names)

        explanation = explainer.get_explanation_as_list(X[0], num_features=3)

        assert len(explanation) == 3
        assert all("feature_rule" in e for e in explanation)
        assert all("contribution" in e for e in explanation)

    def test_what_if_analysis(self, trained_model_and_data):
        from src.explainability.lime_explainer import LIMEExplainer

        model, X, _, feature_names = trained_model_and_data
        explainer = LIMEExplainer(model, X, feature_names)

        result = explainer.what_if_analysis(X[0], feature_index=0, new_value=5.0)

        assert "original_prediction" in result
        assert "modified_prediction" in result
        assert "probability_change" in result
        assert "feature_changed" in result
        assert result["feature_changed"] == "feature_a"
