"""
Pydantic Schemas for the CreditLens AI API
===========================================
Defines request and response models for data validation and serialization.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    """Schema for a loan application submission."""

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Applicant's age (18-100)",
        examples=[35],
    )
    gender: Literal["Male", "Female", "Other"] = Field(
        ...,
        description="Applicant's gender",
        examples=["Male"],
    )
    education: Literal["Graduate", "Not Graduate"] = Field(
        ...,
        description="Highest education level",
        examples=["Graduate"],
    )
    employment_status: Literal["Employed", "Self-Employed", "Unemployed"] = Field(
        ...,
        description="Current employment status",
        examples=["Employed"],
    )
    marital_status: Literal["Married", "Single", "Divorced"] = Field(
        default="Single",
        description="Marital status",
        examples=["Married"],
    )
    annual_income: float = Field(
        ...,
        gt=0,
        description="Annual income in INR/USD",
        examples=[750000.0],
    )
    loan_amount: float = Field(
        ...,
        gt=0,
        description="Requested loan amount",
        examples=[2500000.0],
    )
    loan_term: int = Field(
        ...,
        ge=1,
        le=30,
        description="Loan repayment term in years",
        examples=[15],
    )
    loan_purpose: str = Field(
        default="Personal",
        description="Purpose of the loan",
        examples=["Home"],
    )
    credit_score: int = Field(
        ...,
        ge=300,
        le=900,
        description="CIBIL/Credit score (300-900)",
        examples=[720],
    )
    number_of_dependents: int = Field(
        ...,
        ge=0,
        description="Number of financial dependents",
        examples=[2],
    )
    previous_defaults: int = Field(
        ...,
        ge=0,
        description="Number of previous loan defaults",
        examples=[0],
    )
    assets_value: float = Field(
        ...,
        ge=0,
        description="Total value of assets owned",
        examples=[5000000.0],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "gender": "Male",
                    "education": "Graduate",
                    "employment_status": "Employed",
                    "marital_status": "Married",
                    "annual_income": 750000.0,
                    "loan_amount": 2500000.0,
                    "loan_term": 15,
                    "loan_purpose": "Home",
                    "credit_score": 720,
                    "number_of_dependents": 2,
                    "previous_defaults": 0,
                    "assets_value": 5000000.0,
                }
            ]
        }
    }


class ExplainRequest(BaseModel):
    """Schema for requesting an explanation for a loan application."""

    application: LoanApplication
    method: Literal["shap", "lime", "both"] = Field(
        default="shap",
        description="Explainability method to use",
    )
    top_n_features: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top contributing features to return",
    )


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class TopFactor(BaseModel):
    """A single contributing factor in the prediction explanation."""

    feature: str = Field(..., description="Name of the feature")
    shap_value: float | None = Field(None, description="SHAP value contribution")
    impact: str = Field(..., description="Direction of impact (Positive/Negative/Neutral)")
    feature_value: Any | None = Field(None, description="Actual value of the feature")


class PredictionResponse(BaseModel):
    """Schema for prediction API response."""

    loan_id: str = Field(..., description="Unique loan application ID")
    prediction: Literal["Approved", "Rejected"] = Field(
        ..., description="Loan decision"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Approval probability (0–1)"
    )
    risk_category: Literal["Low", "Medium", "High"] = Field(
        ..., description="Risk classification"
    )
    top_factors: list[TopFactor] = Field(
        default_factory=list, description="Top features influencing the decision"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Prediction timestamp"
    )


class ExplainResponse(BaseModel):
    """Schema for explanation API response."""
    loan_id: str
    prediction: str
    probability: float
    shap_explanation: list[TopFactor] | None = None
    lime_explanation: list[dict] | None = None
    timestamp: datetime


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    model_type: str
    n_features: int
    feature_names: list[str]
    training_metric: str
    training_score: float | None = None


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str
    status_code: int
