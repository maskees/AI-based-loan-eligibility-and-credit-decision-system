"""
Model Evaluator Module
======================
Generates comprehensive evaluation metrics, visualizations, and
comparison tables for all trained models.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Use a clean plotting style
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({"figure.figsize": (10, 6), "figure.dpi": 150})


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> dict[str, Any]:
    """
    Compute a full suite of evaluation metrics for a single model.

    Parameters
    ----------
    model : estimator
        Fitted model with ``predict`` and ``predict_proba`` methods.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels.
    model_name : str
        Name for logging and result labelling.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = _get_probabilities(model, X_test)

    metrics = {
        "model_name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "log_loss": round(log_loss(y_test, y_proba), 4) if y_proba is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    logger.info(
        "%s → Acc=%.4f | F1=%.4f | AUC=%.4f | Precision=%.4f | Recall=%.4f",
        model_name,
        metrics["accuracy"],
        metrics["f1_score"],
        metrics["roc_auc"] or 0,
        metrics["precision"],
        metrics["recall"],
    )

    return metrics


def evaluate_all_models(
    trained_models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Evaluate all trained models and return a sorted list of results.

    Parameters
    ----------
    trained_models : dict[str, estimator]
        Dictionary of model name → fitted model.
    X_test : np.ndarray
        Test feature matrix.
    y_test : np.ndarray
        True test labels.

    Returns
    -------
    list[dict]
        List of evaluation results, sorted by AUC-ROC (descending).
    """
    results = []
    for name, model in trained_models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

    # Sort by AUC-ROC
    results.sort(
        key=lambda x: x["roc_auc"] if x["roc_auc"] is not None else 0,
        reverse=True,
    )

    return results


def create_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Create a clean comparison DataFrame of all model metrics.

    Parameters
    ----------
    results : list[dict]
        List of evaluation results from ``evaluate_all_models``.

    Returns
    -------
    pd.DataFrame
        Formatted comparison table.
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1_score"],
            "AUC-ROC": r["roc_auc"],
            "Log Loss": r["log_loss"],
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Model")
    logger.info("\n%s", df.to_string())

    return df


def plot_confusion_matrices(
    results: list[dict[str, Any]],
    y_test: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Plot confusion matrices for all models in a grid layout."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        cm = np.array(result["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
        )
        ax.set_title(result["model_name"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        _save_figure(fig, "confusion_matrices.png")

    return fig


def plot_roc_curves(
    results: list[dict[str, Any]],
    y_test: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Plot ROC curves for all models on a single chart."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for result in results:
        if result["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                label=f"{result['model_name']} (AUC = {roc_auc:.4f})",
                linewidth=2,
            )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save_figure(fig, "roc_curves.png")

    return fig


def plot_precision_recall_curves(
    results: list[dict[str, Any]],
    y_test: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Plot Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for result in results:
        if result["y_proba"] is not None:
            precision, recall, _ = precision_recall_curve(y_test, result["y_proba"])
            ax.plot(
                recall, precision,
                label=f"{result['model_name']}",
                linewidth=2,
            )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(
        "Precision-Recall Curves — Model Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        _save_figure(fig, "precision_recall_curves.png")

    return fig


def plot_metrics_comparison(
    comparison_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Plot a grouped bar chart comparing all metrics across models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    plot_df = comparison_df[metrics_to_plot]

    plot_df.plot(kind="bar", ax=ax, width=0.8, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        _save_figure(fig, "metrics_comparison.png")

    return fig


def _get_probabilities(model: Any, X: np.ndarray) -> np.ndarray | None:
    """Safely extract probability predictions for the positive class."""
    try:
        probas = model.predict_proba(X)
        return probas[:, 1]
    except AttributeError:
        logger.warning("Model does not support predict_proba")
        return None


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a matplotlib figure to the reports directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = REPORTS_DIR / filename
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
    logger.info("Saved plot → %s", filepath)
