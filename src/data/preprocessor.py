"""
Data Preprocessor Module
========================
Handles data cleaning, encoding, scaling, and class balancing using
scikit-learn Pipelines and ColumnTransformers for full reproducibility.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    MODELS_DIR,
    PREPROCESSING_PIPELINE_FILENAME,
    FEATURE_NAMES_FILENAME,
    RANDOM_STATE,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


def _identify_feature_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify which columns in the DataFrame are numerical vs categorical,
    filtered to only include columns that actually exist.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (features only, no target).

    Returns
    -------
    tuple of (numerical_cols, categorical_cols)
    """
    available_cols = set(df.columns)
    numerical_cols = [c for c in NUMERICAL_FEATURES if c in available_cols]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in available_cols]

    # Auto-detect any remaining columns
    for col in df.columns:
        if col in numerical_cols or col in categorical_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    logger.info("Numerical features (%d): %s", len(numerical_cols), numerical_cols)
    logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)

    return numerical_cols, categorical_cols


def build_preprocessing_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that handles both numerical
    and categorical preprocessing in a single, reproducible pipeline.

    Numerical Pipeline:
        1. Impute missing values with the median
        2. Standard scale (zero mean, unit variance)

    Categorical Pipeline:
        1. Impute missing values with the most frequent value
        2. One-hot encode (drop first to avoid multicollinearity)

    Parameters
    ----------
    numerical_cols : list[str]
        Names of numerical feature columns.
    categorical_cols : list[str]
        Names of categorical feature columns.

    Returns
    -------
    ColumnTransformer
        The fitted-ready preprocessing pipeline.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",  # Drop any columns not in our feature lists
    )

    logger.info("Built preprocessing pipeline ✓")
    return preprocessor


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | None = None,
    apply_smote: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    """
    Full preprocessing pipeline: identify features, fit transformer,
    transform data, and optionally apply SMOTE.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Testing feature matrix.
    y_train : pd.Series, optional
        Training target vector (needed for SMOTE).
    apply_smote : bool
        Whether to apply SMOTE oversampling on the training set.

    Returns
    -------
    tuple of (X_train_processed, X_test_processed, y_train_resampled, y_test, feature_names)
        Processed arrays and the list of transformed feature names.
    """
    numerical_cols, categorical_cols = _identify_feature_types(X_train)

    # Build and fit the pipeline on training data
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Extract feature names from the fitted pipeline
    feature_names = _get_feature_names(preprocessor, numerical_cols, categorical_cols)

    logger.info(
        "Preprocessing complete: %d features → %d transformed features",
        X_train.shape[1],
        X_train_processed.shape[1],
    )

    # Apply SMOTE for class balancing
    y_train_resampled = None
    if apply_smote and y_train is not None:
        X_train_processed, y_train_resampled = _apply_smote(
            X_train_processed, y_train
        )

    # Save the fitted pipeline and feature names for inference
    _save_pipeline(preprocessor, feature_names)

    return (
        X_train_processed,
        X_test_processed,
        y_train_resampled if y_train_resampled is not None else (y_train.values if y_train is not None else None),
        None,  # y_test is passed separately
        feature_names,
    )


def _get_feature_names(
    preprocessor: ColumnTransformer,
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> list[str]:
    """
    Extract the final feature names after preprocessing.
    """
    feature_names = list(numerical_cols)

    # Get one-hot encoded feature names
    if categorical_cols:
        try:
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_feature_names)
        except Exception:
            # Fallback: generate generic names
            logger.warning("Could not extract categorical feature names, using generic names")
            for i in range(
                preprocessor.transform(
                    pd.DataFrame(columns=numerical_cols + categorical_cols)
                ).shape[1]
                - len(numerical_cols)
            ):
                feature_names.append(f"cat_feature_{i}")

    return feature_names


def _apply_smote(
    X: np.ndarray,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    tuple of (X_resampled, y_resampled)
    """
    logger.info("Class distribution before SMOTE: %s", dict(y.value_counts()))

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    unique, counts = np.unique(y_resampled, return_counts=True)
    logger.info("Class distribution after SMOTE: %s", dict(zip(unique, counts)))
    logger.info(
        "SMOTE applied: %d → %d samples",
        len(y),
        len(y_resampled),
    )

    return X_resampled, y_resampled


def _save_pipeline(
    preprocessor: ColumnTransformer,
    feature_names: list[str],
) -> None:
    """Save the fitted preprocessing pipeline and feature names to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = MODELS_DIR / PREPROCESSING_PIPELINE_FILENAME
    joblib.dump(preprocessor, pipeline_path)
    logger.info("Saved preprocessing pipeline → %s", pipeline_path)

    names_path = MODELS_DIR / FEATURE_NAMES_FILENAME
    joblib.dump(feature_names, names_path)
    logger.info("Saved feature names → %s", names_path)


def load_preprocessing_pipeline() -> tuple[ColumnTransformer, list[str]]:
    """
    Load the previously saved preprocessing pipeline and feature names.

    Returns
    -------
    tuple of (preprocessor, feature_names)
    """
    pipeline_path = MODELS_DIR / PREPROCESSING_PIPELINE_FILENAME
    names_path = MODELS_DIR / FEATURE_NAMES_FILENAME

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found at {pipeline_path}. "
            "Please run the training pipeline first."
        )

    preprocessor = joblib.load(pipeline_path)
    feature_names = joblib.load(names_path)

    logger.info("Loaded preprocessing pipeline from %s", pipeline_path)
    return preprocessor, feature_names
