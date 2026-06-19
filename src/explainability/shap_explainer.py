"""
SHAP Explainer Module
=====================
Provides global and local model explanations using SHAP (SHapley Additive
exPlanations). Generates summary plots, waterfall plots, force plots, and
feature importance rankings.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wrapper around SHAP for generating model explanations.

    Attributes
    ----------
    model : estimator
        A fitted scikit-learn compatible model.
    explainer : shap.Explainer
        The SHAP explainer instance.
    shap_values : np.ndarray
        Computed SHAP values for the provided data.
    feature_names : list[str]
        Names of the input features.
    """

    def __init__(
        self,
        model: Any,
        X_data: np.ndarray,
        feature_names: list[str] | None = None,
    ):
        """
        Initialize the SHAP explainer.

        Parameters
        ----------
        model : estimator
            Fitted model (XGBoost, LightGBM, RandomForest, etc.).
        X_data : np.ndarray
            Background data for computing SHAP values (typically X_test
            or a sample of X_train).
        feature_names : list[str], optional
            Names of the features.
        """
        self.model = model
        self.feature_names = feature_names
        self.X_data = X_data

        # Use TreeExplainer for tree-based models, KernelExplainer as fallback
        try:
            self.explainer = shap.TreeExplainer(model)
            logger.info("Using SHAP TreeExplainer ✓")
        except Exception:
            logger.info("Falling back to SHAP KernelExplainer (slower)...")
            self.explainer = shap.KernelExplainer(
                model.predict_proba, shap.sample(X_data, 100)
            )

        self.shap_values = None

    def compute_shap_values(self, X: np.ndarray | None = None) -> np.ndarray:
        """
        Compute SHAP values for the given data.

        Parameters
        ----------
        X : np.ndarray, optional
            Data to explain. Defaults to the data provided at initialization.

        Returns
        -------
        np.ndarray
            SHAP values array.
        """
        if X is None:
            X = self.X_data

        logger.info("Computing SHAP values for %d samples...", X.shape[0])
        self.shap_values = self.explainer.shap_values(X)

        # For binary classification, shap_values may be:
        #   - a list [class_0_array, class_1_array]  → take index [1]
        #   - a 3-D ndarray of shape (n_samples, n_features, n_classes) → take [..., 1]
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Positive class
        elif self.shap_values.ndim == 3:
            self.shap_values = self.shap_values[..., 1]  # Positive class

        logger.info("SHAP values computed ✓ Shape: %s", self.shap_values.shape)
        return self.shap_values

    def plot_summary(self, save: bool = True) -> plt.Figure:
        """
        Generate a SHAP summary plot showing global feature importance
        with distribution of impact.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("SHAP Summary Plot — Feature Impact on Loan Decision", fontsize=13)
        plt.tight_layout()

        if save:
            self._save_figure("shap_summary_plot.png")

        return fig

    def plot_feature_importance(self, save: bool = True) -> plt.Figure:
        """
        Generate a SHAP bar plot showing mean absolute SHAP values
        (global feature importance ranking).
        """
        if self.shap_values is None:
            self.compute_shap_values()

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values,
            self.X_data,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
        )
        plt.title("SHAP Feature Importance — Mean |SHAP Value|", fontsize=13)
        plt.tight_layout()

        if save:
            self._save_figure("shap_feature_importance.png")

        return fig

    def plot_waterfall(
        self,
        sample_index: int = 0,
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate a SHAP waterfall plot for a single prediction,
        showing how each feature pushes the prediction from the base value.

        Parameters
        ----------
        sample_index : int
            Index of the sample in X_data to explain.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        fig, ax = plt.subplots(figsize=(10, 6))

        explanation = shap.Explanation(
            values=self.shap_values[sample_index],
            base_values=self.explainer.expected_value
            if not isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value[1],
            data=self.X_data[sample_index],
            feature_names=self.feature_names,
        )

        shap.waterfall_plot(explanation, show=False)
        plt.title(
            f"SHAP Waterfall — Applicant #{sample_index + 1} Decision Breakdown",
            fontsize=13,
        )
        plt.tight_layout()

        if save:
            self._save_figure(f"shap_waterfall_sample_{sample_index}.png")

        return fig

    def get_top_features(
        self,
        sample_index: int,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get the top N contributing features for a single prediction.

        Parameters
        ----------
        sample_index : int
            Index of the sample to explain.
        top_n : int
            Number of top features to return.

        Returns
        -------
        list[dict]
            List of dicts with 'feature', 'shap_value', and 'impact' keys.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        values = self.shap_values[sample_index]
        abs_values = np.abs(values)
        top_indices = np.argsort(abs_values)[-top_n:][::-1]

        features = []
        for idx in top_indices:
            i = int(idx)  # ensure plain Python int for list indexing
            name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            shap_val = float(values[i])
            features.append({
                "feature": name,
                "shap_value": round(shap_val, 4),
                "impact": "positive" if shap_val > 0 else "negative",
                "feature_value": float(self.X_data[sample_index, i]),
            })

        return features

    def get_global_feature_importance(self) -> list[dict[str, float]]:
        """
        Get global feature importance based on mean |SHAP values|.

        Returns
        -------
        list[dict]
            Sorted list of feature importances.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        indices = np.argsort(mean_abs_shap)[::-1]

        importances = []
        for idx in indices:
            i = int(idx)  # ensure plain Python int for list indexing
            name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            importances.append({
                "feature": name,
                "importance": round(float(mean_abs_shap[i]), 4),
            })

        return importances

    def plot_force(
        self,
        sample_index: int = 0,
        matplotlib: bool = True,
        save: bool = True,
    ) -> Any:
        """
        Generate a SHAP force plot for a single prediction.

        Force plots show how features push the prediction from the base
        value toward the final output. Best viewed in Jupyter notebooks
        with ``shap.initjs()`` called first.

        Parameters
        ----------
        sample_index : int
            Index of the sample in X_data to explain.
        matplotlib : bool
            If True, render as a matplotlib figure (static).
            If False, return an interactive HTML object (for notebooks).
        save : bool
            Whether to save the plot (only works with matplotlib=True).

        Returns
        -------
        matplotlib Figure or shap ForcePlot object
        """
        if self.shap_values is None:
            self.compute_shap_values()

        base_value = (
            self.explainer.expected_value
            if not isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value[1]
        )

        if matplotlib:
            fig, ax = plt.subplots(figsize=(14, 3))
            shap.force_plot(
                base_value,
                self.shap_values[sample_index],
                self.X_data[sample_index],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False,
            )
            plt.title(
                f"SHAP Force Plot — Applicant #{sample_index + 1}",
                fontsize=12,
                pad=40,
            )
            plt.tight_layout()

            if save:
                self._save_figure(f"shap_force_sample_{sample_index}.png")

            return fig
        else:
            # Interactive HTML version for notebooks
            return shap.force_plot(
                base_value,
                self.shap_values[sample_index],
                self.X_data[sample_index],
                feature_names=self.feature_names,
            )

    def get_explanation_object(
        self,
        sample_index: int | None = None,
    ) -> shap.Explanation:
        """
        Return a full ``shap.Explanation`` object for native SHAP
        visualisation in Jupyter notebooks.

        Parameters
        ----------
        sample_index : int, optional
            If provided, returns an Explanation for a single sample.
            If None, returns the Explanation for all samples.

        Returns
        -------
        shap.Explanation
            A SHAP Explanation object compatible with all shap plot functions.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        base_value = (
            self.explainer.expected_value
            if not isinstance(self.explainer.expected_value, list)
            else self.explainer.expected_value[1]
        )

        if sample_index is not None:
            return shap.Explanation(
                values=self.shap_values[sample_index],
                base_values=base_value,
                data=self.X_data[sample_index],
                feature_names=self.feature_names,
            )
        else:
            return shap.Explanation(
                values=self.shap_values,
                base_values=np.full(self.shap_values.shape[0], base_value),
                data=self.X_data,
                feature_names=self.feature_names,
            )

    def _save_figure(self, filename: str) -> None:
        """Save the current matplotlib figure to reports directory."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = REPORTS_DIR / filename
        plt.savefig(filepath, bbox_inches="tight", dpi=150)
        logger.info("Saved SHAP plot → %s", filepath)
