"""
LIME Explainer Module
=====================
Provides local, interpretable model explanations using LIME (Local
Interpretable Model-agnostic Explanations). Useful for per-prediction
explanations and "What-If" analysis.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Wrapper around LIME for generating local prediction explanations.

    Attributes
    ----------
    model : estimator
        A fitted model with ``predict_proba`` method.
    explainer : LimeTabularExplainer
        The LIME explainer instance.
    feature_names : list[str]
        Names of the input features.
    """

    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        feature_names: list[str] | None = None,
        class_names: list[str] | None = None,
    ):
        """
        Initialize the LIME explainer.

        Parameters
        ----------
        model : estimator
            Fitted model with ``predict_proba`` method.
        X_train : np.ndarray
            Training data used to build the LIME explainer's
            local neighborhood distributions.
        feature_names : list[str], optional
            Names of the features.
        class_names : list[str], optional
            Names of the target classes. Defaults to ["Rejected", "Approved"].
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ["Rejected", "Approved"]

        self.explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        logger.info("LIME explainer initialized ✓")

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ):
        """
        Generate a LIME explanation for a single prediction.

        Parameters
        ----------
        instance : np.ndarray
            A single data point (1D array) to explain.
        num_features : int
            Number of top features to include in the explanation.

        Returns
        -------
        lime.explanation.Explanation
            The LIME explanation object.
        """
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
        )

        logger.info(
            "LIME explanation generated for instance (predicted class: %s)",
            self.class_names[explanation.predict_proba.argmax()]
            if hasattr(explanation, "predict_proba")
            else "unknown",
        )

        return explanation

    def get_explanation_as_list(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Get the LIME explanation as a structured list of feature contributions.

        Parameters
        ----------
        instance : np.ndarray
            A single data point to explain.
        num_features : int
            Number of features in the explanation.

        Returns
        -------
        list[dict]
            Each dict has keys: 'feature_rule', 'contribution', 'impact'.
        """
        explanation = self.explain_instance(instance, num_features)
        exp_list = explanation.as_list()

        results = []
        for feature_rule, contribution in exp_list:
            results.append({
                "feature_rule": feature_rule,
                "contribution": round(float(contribution), 4),
                "impact": "positive" if contribution > 0 else "negative",
            })

        return results

    def plot_explanation(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        sample_label: str = "Applicant",
        save: bool = True,
    ) -> plt.Figure:
        """
        Generate a LIME explanation plot for a single instance.

        Parameters
        ----------
        instance : np.ndarray
            A single data point to explain.
        num_features : int
            Number of features to show.
        sample_label : str
            Label for the plot title.
        save : bool
            Whether to save the plot to disk.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        """
        explanation = self.explain_instance(instance, num_features)

        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.title(
            f"LIME Explanation — {sample_label}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        if save:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            filepath = REPORTS_DIR / f"lime_explanation_{sample_label.lower().replace(' ', '_')}.png"
            fig.savefig(filepath, bbox_inches="tight", dpi=150)
            logger.info("Saved LIME plot → %s", filepath)

        return fig

    def what_if_analysis(
        self,
        instance: np.ndarray,
        feature_index: int,
        new_value: float,
        num_features: int = 10,
    ) -> dict[str, Any]:
        """
        Perform a "What-If" analysis: change a single feature value
        and compare the prediction before and after.

        Parameters
        ----------
        instance : np.ndarray
            Original data point.
        feature_index : int
            Index of the feature to modify.
        new_value : float
            New value for the feature.
        num_features : int
            Number of features for LIME explanation.

        Returns
        -------
        dict
            Contains original and modified predictions with explanations.
        """
        # Original prediction
        original_proba = self.model.predict_proba(instance.reshape(1, -1))[0]

        # Modified instance
        modified_instance = instance.copy()
        modified_instance[feature_index] = new_value
        modified_proba = self.model.predict_proba(modified_instance.reshape(1, -1))[0]

        feature_name = (
            self.feature_names[feature_index]
            if self.feature_names
            else f"feature_{feature_index}"
        )

        result = {
            "feature_changed": feature_name,
            "original_value": float(instance[feature_index]),
            "new_value": float(new_value),
            "original_prediction": {
                "class": self.class_names[original_proba.argmax()],
                "probability": round(float(original_proba.max()), 4),
                "probabilities": {
                    name: round(float(p), 4)
                    for name, p in zip(self.class_names, original_proba)
                },
            },
            "modified_prediction": {
                "class": self.class_names[modified_proba.argmax()],
                "probability": round(float(modified_proba.max()), 4),
                "probabilities": {
                    name: round(float(p), 4)
                    for name, p in zip(self.class_names, modified_proba)
                },
            },
            "probability_change": round(
                float(modified_proba[1] - original_proba[1]), 4
            ),
        }

        logger.info(
            "What-If: %s changed from %.2f → %.2f | "
            "Approval probability: %.4f → %.4f (Δ=%+.4f)",
            feature_name,
            result["original_value"],
            result["new_value"],
            original_proba[1],
            modified_proba[1],
            result["probability_change"],
        )

        return result

    def explain_and_compare(
        self,
        instance: np.ndarray,
        shap_explainer: Any,
        num_features: int = 10,
    ) -> dict[str, Any]:
        """
        Generate both LIME and SHAP explanations for a single instance
        and return them in a structured dict for side-by-side comparison.

        Parameters
        ----------
        instance : np.ndarray
            A single data point (1D array) to explain.
        shap_explainer : SHAPExplainer
            An initialised ``SHAPExplainer`` instance (from
            ``src.explainability.shap_explainer``).
        num_features : int
            Number of top features to include.

        Returns
        -------
        dict
            Contains ``lime_explanation``, ``shap_explanation``,
            ``prediction``, and ``comparison_table``.
        """
        # LIME explanation
        lime_explanation = self.get_explanation_as_list(instance, num_features)

        # SHAP explanation
        shap_top = shap_explainer.get_top_features(0, num_features)

        # Prediction
        proba = self.model.predict_proba(instance.reshape(1, -1))[0]
        prediction_class = self.class_names[proba.argmax()]

        # Build comparison table
        comparison = []
        lime_dict = {
            e["feature_rule"].split(" ")[0]: e for e in lime_explanation
        }
        shap_dict = {s["feature"]: s for s in shap_top}

        all_features = list(
            dict.fromkeys(
                [s["feature"] for s in shap_top]
                + [e["feature_rule"] for e in lime_explanation]
            )
        )

        for feat in all_features:
            row = {"feature": feat}
            if feat in shap_dict:
                row["shap_value"] = shap_dict[feat]["shap_value"]
                row["shap_impact"] = shap_dict[feat]["impact"]
            if feat in lime_dict:
                row["lime_contribution"] = lime_dict[feat]["contribution"]
                row["lime_impact"] = lime_dict[feat]["impact"]
            comparison.append(row)

        result = {
            "prediction": prediction_class,
            "probability": {
                name: round(float(p), 4)
                for name, p in zip(self.class_names, proba)
            },
            "lime_explanation": lime_explanation,
            "shap_explanation": shap_top,
            "comparison_table": comparison,
        }

        logger.info(
            "SHAP vs LIME comparison generated for instance "
            "(prediction: %s, P=%.4f)",
            prediction_class,
            float(proba.max()),
        )

        return result
