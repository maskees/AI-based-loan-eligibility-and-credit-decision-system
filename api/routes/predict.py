"""
Predict Route
=============
FastAPI router for loan prediction endpoints.
"""

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import (
    LoanApplication,
    PredictionResponse,
    TopFactor,
)
from api.services.prediction import PredictionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Lazy-loaded prediction service
_service: PredictionService | None = None


def _get_service() -> PredictionService:
    """Get or initialize the prediction service (singleton pattern)."""
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Predict Loan Eligibility",
    description=(
        "Submit a loan application and receive an instant prediction "
        "with approval probability, risk category, and top contributing factors."
    ),
)
async def predict_loan(application: LoanApplication) -> PredictionResponse:
    """
    Process a loan application and return a prediction.

    The endpoint:
    1. Validates the input via Pydantic schema
    2. Preprocesses the features
    3. Runs the ML model prediction
    4. Classifies the risk level
    5. Generates SHAP-based explanation (top factors)
    """
    try:
        service = _get_service()
        result = service.predict_with_explanation(
            application.model_dump(),
            top_n=5,
        )

        # Convert top_factors to schema-compatible format
        top_factors = [
            TopFactor(
                feature=f["feature"],
                shap_value=f.get("shap_value"),
                impact=f["impact"],
                feature_value=f.get("feature_value"),
            )
            for f in result.get("top_factors", [])
        ]

        return PredictionResponse(
            loan_id=result["loan_id"],
            prediction=result["prediction"],
            probability=result["probability"],
            risk_category=result["risk_category"],
            top_factors=top_factors,
            timestamp=result["timestamp"],
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Please run the training pipeline first. "
                "See README.md for instructions."
            ),
        )
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/batch",
    summary="Batch Predict Loan Eligibility",
    description="Submit multiple loan applications for batch prediction.",
)
async def predict_batch(
    applications: list[LoanApplication],
) -> list[PredictionResponse]:
    """Process multiple loan applications in batch."""
    try:
        service = _get_service()
        results = []

        for app in applications:
            result = service.predict_with_explanation(app.model_dump(), top_n=5)
            top_factors = [
                TopFactor(
                    feature=f["feature"],
                    shap_value=f.get("shap_value"),
                    impact=f["impact"],
                    feature_value=f.get("feature_value"),
                )
                for f in result.get("top_factors", [])
            ]
            results.append(
                PredictionResponse(
                    loan_id=result["loan_id"],
                    prediction=result["prediction"],
                    probability=result["probability"],
                    risk_category=result["risk_category"],
                    top_factors=top_factors,
                    timestamp=result["timestamp"],
                )
            )

        return results

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training pipeline first.",
        )
    except Exception as e:
        logger.exception("Batch prediction failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}",
        )
