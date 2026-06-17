"""
CreditLens AI — Project Configuration
======================================

Centralized configuration for all project paths, constants, and settings.
Uses pathlib for cross-platform path handling and python-dotenv for
environment variable management.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env file (if present)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
# Root of the project (parent of the `src/` directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"

# Reports and visualizations
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Notebooks
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------
# Primary dataset filename (in data/raw/)
DATASET_FILENAME = os.getenv("DATASET_FILENAME", "loan_approval_dataset.csv")
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME

# Target variable name
TARGET_COLUMN = "loan_status"

# Train/test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature Configuration
# ---------------------------------------------------------------------------

# Categorical features (will be one-hot encoded)
CATEGORICAL_FEATURES = [
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

# Numerical features (will be scaled)
NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

# Engineered feature names (created during feature engineering)
ENGINEERED_FEATURES = [
    "debt_to_income_ratio",
    "loan_to_income_ratio",
    "credit_utilization",
    "employment_stability_score",
]

# All features used for model training
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

# Models to train and compare
MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]

# Cross-validation folds
CV_FOLDS = 5

# Primary evaluation metric
PRIMARY_METRIC = "roc_auc"

# Optuna hyperparameter tuning
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 600  # seconds (10 minutes max)

# Performance targets
TARGET_AUC_ROC = 0.85
TARGET_F1_SCORE = 0.80

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"

# ---------------------------------------------------------------------------
# Model Serialization
# ---------------------------------------------------------------------------
BEST_MODEL_FILENAME = "best_model.joblib"
BEST_MODEL_PATH = MODELS_DIR / BEST_MODEL_FILENAME

PREPROCESSOR_FILENAME = "preprocessor.joblib"
PREPROCESSING_PIPELINE_FILENAME = PREPROCESSOR_FILENAME
PREPROCESSOR_PATH = MODELS_DIR / PREPROCESSOR_FILENAME

FEATURE_NAMES_FILENAME = "feature_names.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / FEATURE_NAMES_FILENAME

# ---------------------------------------------------------------------------
# Risk Classification
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "Low": 0.70,
    "Medium": 0.40,
    "High": 0.00
}

# ---------------------------------------------------------------------------
# Dashboard Configuration
# ---------------------------------------------------------------------------
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# SMOTE Configuration (Class Balancing)
# ---------------------------------------------------------------------------
SMOTE_RANDOM_STATE = RANDOM_STATE
SMOTE_K_NEIGHBORS = 5

# ---------------------------------------------------------------------------
# Ensure directories exist
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Create all required project directories if they don't exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        NOTEBOOKS_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Auto-create directories on import
ensure_directories()
