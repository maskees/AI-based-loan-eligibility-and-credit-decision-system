"""
Model Trainer Module
====================
Trains multiple ML models, performs cross-validation, and saves the best model.
Supports: Logistic Regression, Random Forest, XGBoost, LightGBM.
"""

import logging
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    CV_FOLDS,
    MODELS_DIR,
    BEST_MODEL_FILENAME,
    RANDOM_STATE,
    PRIMARY_METRIC,
)

logger = logging.getLogger(__name__)


def get_models() -> dict[str, Any]:
    """
    Return a dictionary of all models to be trained and compared.

    Returns
    -------
    dict[str, estimator]
        Model name → scikit-learn compatible estimator.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

    return models


def train_single_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "Model",
) -> Any:
    """
    Train a single model on the training data.

    Parameters
    ----------
    model : estimator
        A scikit-learn compatible model.
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    model_name : str
        Name for logging purposes.

    Returns
    -------
    estimator
        The fitted model.
    """
    logger.info("Training %s...", model_name)
    model.fit(X_train, y_train)
    logger.info("  ✓ %s training complete", model_name)

    return model


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    cv: int | None = None,
    scoring: str | None = None,
) -> dict[str, float]:
    """
    Perform stratified k-fold cross-validation on a model.

    Parameters
    ----------
    model : estimator
        A scikit-learn compatible model.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    model_name : str
        Name for logging.
    cv : int, optional
        Number of folds. Defaults to ``config.CV_FOLDS``.
    scoring : str, optional
        Scoring metric. Defaults to ``config.PRIMARY_METRIC``.

    Returns
    -------
    dict with keys: 'model_name', 'mean_score', 'std_score', 'all_scores'
    """
    if cv is None:
        cv = CV_FOLDS
    if scoring is None:
        scoring = PRIMARY_METRIC

    logger.info(
        "Cross-validating %s (%d-fold, metric=%s)...",
        model_name, cv, scoring,
    )

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    result = {
        "model_name": model_name,
        "mean_score": round(float(np.mean(scores)), 4),
        "std_score": round(float(np.std(scores)), 4),
        "all_scores": [round(float(s), 4) for s in scores],
    }

    logger.info(
        "  ✓ %s: %s = %.4f (±%.4f)",
        model_name, scoring, result["mean_score"], result["std_score"],
    )

    return result


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """
    Train all models, cross-validate each, and return results.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.

    Returns
    -------
    tuple of (trained_models, cv_results)
        - trained_models: dict[str, fitted_estimator]
        - cv_results: list of cross-validation result dicts, sorted by score
    """
    models = get_models()
    trained_models: dict[str, Any] = {}
    cv_results: list[dict[str, float]] = []

    logger.info("=" * 60)
    logger.info("TRAINING %d MODELS", len(models))
    logger.info("=" * 60)

    for name, model in models.items():
        # Cross-validate first (on unfitted model)
        cv_result = cross_validate_model(model, X_train, y_train, name)
        cv_results.append(cv_result)

        # Then train on the full training set
        fitted_model = train_single_model(model, X_train, y_train, name)
        trained_models[name] = fitted_model

    # Sort by mean score (descending)
    cv_results.sort(key=lambda x: x["mean_score"], reverse=True)

    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION RESULTS (sorted by %s):", PRIMARY_METRIC)
    for i, result in enumerate(cv_results, 1):
        logger.info(
            "  %d. %s: %.4f (±%.4f)",
            i, result["model_name"], result["mean_score"], result["std_score"],
        )
    logger.info("=" * 60)

    return trained_models, cv_results


def select_best_model(
    trained_models: dict[str, Any],
    cv_results: list[dict[str, float]],
) -> tuple[str, Any]:
    """
    Select the best model based on cross-validation results.

    Parameters
    ----------
    trained_models : dict[str, estimator]
        Dictionary of trained models.
    cv_results : list[dict]
        Cross-validation results, sorted by score.

    Returns
    -------
    tuple of (best_model_name, best_model)
    """
    best_name = cv_results[0]["model_name"]
    best_model = trained_models[best_name]

    logger.info(
        "🏆 Best model: %s (%.4f %s)",
        best_name, cv_results[0]["mean_score"], PRIMARY_METRIC,
    )

    return best_name, best_model


def save_model(model: Any, filename: str | None = None) -> str:
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : estimator
        The trained model to save.
    filename : str, optional
        Filename for the saved model. Defaults to ``config.BEST_MODEL_FILENAME``.

    Returns
    -------
    str
        Path to the saved model file.
    """
    if filename is None:
        filename = BEST_MODEL_FILENAME

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info("Saved model → %s", filepath)

    return str(filepath)


def load_model(filename: str | None = None) -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    filename : str, optional
        Filename of the saved model. Defaults to ``config.BEST_MODEL_FILENAME``.

    Returns
    -------
    estimator
        The loaded model.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    if filename is None:
        filename = BEST_MODEL_FILENAME

    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model not found at {filepath}. "
            "Please run the training pipeline first."
        )

    model = joblib.load(filepath)
    logger.info("Loaded model from %s", filepath)

    return model
