"""
Tests for the FastAPI Application
"""

import pytest
from fastapi.testclient import TestClient


class TestRootEndpoint:
    """Tests for the root and health endpoints."""

    def test_root_returns_api_info(self):
        from api.main import app

        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "CreditLens AI API"
        assert "docs" in data

    def test_docs_accessible(self):
        # pyrefly: ignore [missing-import]
        from api.main import app

        client = TestClient(app)
        response = client.get("/docs")

        assert response.status_code == 200


class TestPredictEndpoint:
    """Tests for the /predict endpoint input validation."""

    def _get_client(self):
        from api.main import app
        return TestClient(app)

    def test_predict_rejects_invalid_age(self):
        client = self._get_client()
        response = client.post(
            "/api/v1/predict/",
            json={
                "age": 10,  # Below minimum 18
                "gender": "Male",
                "education": "Graduate",
                "employment_status": "Employed",
                "annual_income": 500000,
                "loan_amount": 200000,
                "loan_term": 10,
                "credit_score": 700,
                "number_of_dependents": 0,
                "previous_defaults": 0,
                "assets_value": 1000000,
            },
        )
        assert response.status_code == 422  # Validation error

    def test_predict_rejects_invalid_credit_score(self):
        client = self._get_client()
        response = client.post(
            "/api/v1/predict/",
            json={
                "age": 30,
                "gender": "Male",
                "education": "Graduate",
                "employment_status": "Employed",
                "annual_income": 500000,
                "loan_amount": 200000,
                "loan_term": 10,
                "credit_score": 100,  # Below minimum 300
                "number_of_dependents": 0,
                "previous_defaults": 0,
                "assets_value": 1000000,
            },
        )
        assert response.status_code == 422

    def test_predict_rejects_missing_fields(self):
        client = self._get_client()
        response = client.post(
            "/api/v1/predict/",
            json={"age": 30},  # Missing required fields
        )
        assert response.status_code == 422
