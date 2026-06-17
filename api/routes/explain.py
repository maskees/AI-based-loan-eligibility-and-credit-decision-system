"""
Explain Route
=============
FastAPI router for model explanation endpoints using SHAP and LIME.
"""

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import ExplainRequest, ExplainResponse, TopFactor
from api.services.prediction import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainability"])

_service: PredictionService | None = None


def _get_service() -> PredictionService:
    """Get or initialize the prediction service."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


@router.post(
    "/",
    response_model=ExplainResponse,
    summary="Explain a Loan Decision",
    description=(
        "Get a detailed explanation of why a loan application was "
        "approved or rejected, using SHAP and/or LIME."
    ),
)
async def explain_prediction(request: ExplainRequest) -> ExplainResponse:
    """
    Generate an explanation for a loan application prediction.

    Supports three methods:
    - ``shap``: Shapley-value-based feature attribution
    - ``lime``: Local linear approximation explanation
    - ``both``: Returns both SHAP and LIME explanations
    """
    try:
        service = _get_service()
        app_data = request.application.model_dump()

        # Get base prediction
        prediction_result = service.predict(app_data)

        # Preprocess input for explainers
        input_df = pd.DataFrame([app_data])
        X_processed = service.preprocessor.transform(input_df)

        shap_explanation = None
        lime_explanation = None

        # SHAP explanation
        if request.method in ("shap", "both"):
            try:
                from src.explainability.shap_explainer import SHAPExplainer

                shap_exp = SHAPExplainer(
                    model=service.model,
                    X_data=X_processed,
                    feature_names=service.feature_names,
                )
                shap_factors = shap_exp.get_top_features(0, request.top_n_features)
                shap_explanation = [
                    TopFactor(
                        feature=f["feature"],
                        shap_value=f["shap_value"],
                        impact=f["impact"],
                        feature_value=f.get("feature_value"),
                    )
                    for f in shap_factors
                ]
            except Exception as e:
                logger.warning("SHAP explanation failed: %s", e)

        # LIME explanation
        if request.method in ("lime", "both"):
            try:
                from src.explainability.lime_explainer import LIMEExplainer

                lime_exp = LIMEExplainer(
                    model=service.model,
                    X_train=X_processed,
                    feature_names=service.feature_names,
                )
                lime_factors = lime_exp.get_explanation_as_list(
                    X_processed[0], request.top_n_features
                )
                lime_explanation = lime_factors
            except Exception as e:
                logger.warning("LIME explanation failed: %s", e)

        return ExplainResponse(
            loan_id=prediction_result["loan_id"],
            prediction=prediction_result["prediction"],
            probability=prediction_result["probability"],
            shap_explanation=shap_explanation,
            lime_explanation=lime_explanation,
            timestamp=prediction_result["timestamp"],
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training pipeline first.",
        )
    except Exception as e:
        logger.exception("Explanation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}",
        )
