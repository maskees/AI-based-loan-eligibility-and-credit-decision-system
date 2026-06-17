"""
FastAPI Application — CreditLens AI
====================================
Main entry point for the REST API backend.
Serves loan eligibility predictions and model explanations.

Run with:
    uv run uvicorn api.main:app --reload
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, explain
from api.schemas import HealthResponse, ModelInfoResponse
from src.config import API_VERSION, LOG_FORMAT, LOG_LEVEL

# ---------------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CreditLens AI — Loan Eligibility API",
    description=(
        "AI-based Loan Eligibility & Credit Decision System with Explainable AI.\n\n"
        "This API provides:\n"
        "- **Loan Prediction**: Submit an application, get instant approval/rejection\n"
        "- **Risk Scoring**: Low / Medium / High risk categorization\n"
        "- **Explainability**: SHAP & LIME explanations for every decision\n"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware (allows Streamlit dashboard to connect)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your dashboard's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------
api_prefix = f"/api/{API_VERSION}"

app.include_router(predict.router, prefix=api_prefix)
app.include_router(explain.router, prefix=api_prefix)


# ---------------------------------------------------------------------------
# Root & Health endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CreditLens AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": f"/api/{API_VERSION}/health",
    }


@app.get(
    f"/api/{API_VERSION}/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
)
async def health_check():
    """Check if the API and model are operational."""
    model_loaded = False
    try:
        from api.services.prediction import PredictionService
        service = PredictionService()
        model_loaded = service.is_loaded
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.get(
    f"/api/{API_VERSION}/model/info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Model Information",
)
async def model_info():
    """Get metadata about the loaded model."""
    try:
        from api.services.prediction import PredictionService
        service = PredictionService()
        info = service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        return ModelInfoResponse(
            model_name="Not Loaded",
            model_type="N/A",
            n_features=0,
            feature_names=[],
            training_metric="roc_auc",
        )


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Log startup information."""
    logger.info("=" * 60)
    logger.info("CreditLens AI API — Starting up")
    logger.info("API Version: %s", API_VERSION)
    logger.info("Docs: http://localhost:8000/docs")
    logger.info("=" * 60)
