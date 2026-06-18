"""
Hyperparameter Tuner Module
============================
Uses Optuna for Bayesian hyperparameter optimization of gradient boosting
models (XGBoost and LightGBM).
"""

import logging
from typing import Any

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import (
    CV_FOLDS,
    OPTUNA_N_TRIALS,
    OPTUNA_TIMEOUT,
    RANDOM_STATE,
    PRIMARY_METRIC,
)

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _xgboost_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Optuna objective function for XGBoost hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "verbosity": 0,
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(
        model, X, y,
        cv=CV_FOLDS,
        scoring=PRIMARY_METRIC,
        n_jobs=-1,
    )

    return float(np.mean(scores))


def _lightgbm_objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Optuna objective function for LightGBM hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    scores = cross_val_score(
        model, X, y,
        cv=CV_FOLDS,
        scoring=PRIMARY_METRIC,
        n_jobs=-1,
    )

    return float(np.mean(scores))


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int | None = None,
    timeout: int | None = None,
) -> tuple[Any, dict, float]:
    """
    Run Optuna hyperparameter optimization for the specified model.

    Parameters
    ----------
    model_name : str
        Name of the model to tune. Must be "XGBoost" or "LightGBM".
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training target vector.
    n_trials : int, optional
        Number of Optuna trials. Defaults to ``config.OPTUNA_N_TRIALS``.
    timeout : int, optional
        Maximum time in seconds. Defaults to ``config.OPTUNA_TIMEOUT``.

    Returns
    -------
    tuple of (best_model, best_params, best_score)
        - best_model: Fitted model with optimal hyperparameters.
        - best_params: Dictionary of the best hyperparameters.
        - best_score: Best cross-validation score achieved.

    Raises
    ------
    ValueError
        If an unsupported model name is provided.
    """
    if n_trials is None:
        n_trials = OPTUNA_N_TRIALS
    if timeout is None:
        timeout = OPTUNA_TIMEOUT

    logger.info("=" * 60)
    logger.info("HYPERPARAMETER TUNING: %s", model_name)
    logger.info("Trials: %d | Timeout: %ds | Metric: %s", n_trials, timeout, PRIMARY_METRIC)
    logger.info("=" * 60)

    # Select objective function
    if model_name == "XGBoost":
        objective = lambda trial: _xgboost_objective(trial, X_train, y_train)
        ModelClass = XGBClassifier
    elif model_name == "LightGBM":
        objective = lambda trial: _lightgbm_objective(trial, X_train, y_train)
        ModelClass = LGBMClassifier
    else:
        raise ValueError(
            f"Tuning not supported for '{model_name}'. "
            "Supported models: 'XGBoost', 'LightGBM'."
        )

    # Create and run study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_tuning",
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Extract best results
    best_params = study.best_params
    best_score = study.best_value

    logger.info("Best %s score: %.4f", PRIMARY_METRIC, best_score)
    logger.info("Best parameters: %s", best_params)

    # Train the final model with best params
    if model_name == "XGBoost":
        best_params.update({
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "verbosity": 0,
        })
    elif model_name == "LightGBM":
        best_params.update({
            "random_state": RANDOM_STATE,
            "verbose": -1,
        })

    best_model = ModelClass(**best_params)
    best_model.fit(X_train, y_train)

    logger.info("✓ Best %s model trained with optimal hyperparameters", model_name)

    return best_model, best_params, best_score


def get_tuning_summary(study: optuna.Study) -> dict:
    """
    Extract a summary of the Optuna study for reporting.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.

    Returns
    -------
    dict
        Summary including best trial, number of trials, and duration.
    """
    return {
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
    }
