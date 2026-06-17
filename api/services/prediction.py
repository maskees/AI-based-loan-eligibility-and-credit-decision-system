"""
Prediction Service
==================
Business logic layer that loads the trained model and preprocessing pipeline,
processes incoming loan applications, and generates predictions with
risk categorization and SHAP-based explanations.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    BEST_MODEL_FILENAME,
    PREPROCESSING_PIPELINE_FILENAME,
    FEATURE_NAMES_FILENAME,
    RISK_THRESHOLDS,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service class that encapsulates the full prediction pipeline:
    load model → preprocess input → predict → classify risk → explain.
    """

    def __init__(self):
        """Initialize and load all required artifacts."""
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load the trained model, preprocessor, and feature names from disk."""
        try:
            model_path = MODELS_DIR / BEST_MODEL_FILENAME
            pipeline_path = MODELS_DIR / PREPROCESSING_PIPELINE_FILENAME
            names_path = MODELS_DIR / FEATURE_NAMES_FILENAME

            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(pipeline_path)
            self.feature_names = joblib.load(names_path)

            logger.info("✓ Model loaded from %s", model_path)
            logger.info("✓ Preprocessor loaded from %s", pipeline_path)
            logger.info("✓ Feature names loaded (%d features)", len(self.feature_names))

        except FileNotFoundError as e:
            logger.error(
                "Could not load model artifacts: %s. "
                "Please run the training pipeline first.",
                e,
            )
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if all model artifacts are loaded."""
        return all([self.model, self.preprocessor, self.feature_names])

    def predict(self, application_data: dict) -> dict[str, Any]:
        """
        Generate a prediction for a single loan application.

        Parameters
        ----------
        application_data : dict
            Dictionary of applicant features matching the LoanApplication schema.

        Returns
        -------
        dict
            Prediction result with loan_id, prediction, probability,
            risk_category, top_factors, and timestamp.
        """
        # Convert to DataFrame for preprocessing
        input_df = pd.DataFrame([application_data])

        # Preprocess the input using the saved pipeline
        X_processed = self.preprocessor.transform(input_df)

        # Generate prediction
        prediction_proba = self.model.predict_proba(X_processed)[0]
        approval_probability = float(prediction_proba[1])
        prediction_label = "Approved" if approval_probability >= 0.5 else "Rejected"

        # Classify risk
        risk_category = self._classify_risk(approval_probability)

        # Generate a unique loan ID
        loan_id = f"CL-{uuid.uuid4().hex[:8].upper()}"

        result = {
            "loan_id": loan_id,
            "prediction": prediction_label,
            "probability": round(approval_probability, 4),
            "risk_category": risk_category,
            "top_factors": [],  # Will be populated by explain endpoint
            "timestamp": datetime.now(),
            "raw_probabilities": {
                "rejected": round(float(prediction_proba[0]), 4),
                "approved": round(float(prediction_proba[1]), 4),
            },
        }

        logger.info(
            "Prediction: %s [%s] (P=%.4f, Risk=%s)",
            loan_id, prediction_label, approval_probability, risk_category,
        )

        return result

    def predict_with_explanation(
        self,
        application_data: dict,
        top_n: int = 5,
    ) -> dict[str, Any]:
        """
        Generate a prediction along with SHAP-based feature explanations.

        Parameters
        ----------
        application_data : dict
            Applicant features.
        top_n : int
            Number of top features to include.

        Returns
        -------
        dict
            Prediction result with top_factors populated.
        """
        # Get base prediction
        result = self.predict(application_data)

        # Compute SHAP explanations
        try:
            from src.explainability.shap_explainer import SHAPExplainer

            input_df = pd.DataFrame([application_data])
            X_processed = self.preprocessor.transform(input_df)

            explainer = SHAPExplainer(
                model=self.model,
                X_data=X_processed,
                feature_names=self.feature_names,
            )
            top_factors = explainer.get_top_features(0, top_n)
            result["top_factors"] = top_factors

        except Exception as e:
            logger.warning("Could not generate SHAP explanation: %s", e)

        return result

    def _classify_risk(self, probability: float) -> str:
        """
        Classify the applicant's risk level based on approval probability.

        Parameters
        ----------
        probability : float
            Probability of loan approval (0.0 to 1.0).

        Returns
        -------
        str
            Risk category: "Low", "Medium", or "High".
        """
        if probability >= RISK_THRESHOLDS["Low"]:
            return "Low"
        elif probability >= RISK_THRESHOLDS["Medium"]:
            return "Medium"
        else:
            return "High"

    def get_model_info(self) -> dict[str, Any]:
        """Return metadata about the loaded model."""
        return {
            "model_name": type(self.model).__name__,
            "model_type": str(type(self.model)),
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names or [],
            "training_metric": "roc_auc",
        }
