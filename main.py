"""
CreditLens AI — Main Training Pipeline
========================================
End-to-end pipeline that:
1. Loads and validates the dataset
2. Engineers features
3. Preprocesses data (scaling, encoding, SMOTE)
4. Trains all 4 models with cross-validation
5. Selects the best model
6. Optionally tunes hyperparameters with Optuna
7. Evaluates on the test set
8. Generates SHAP/LIME explanations
9. Saves all artifacts (model, plots, reports)

Run with:
    uv run python main.py
"""

import json
import logging
import sys

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    PRIMARY_METRIC,
)
from src.data.loader import (
    load_raw_dataset,
    validate_dataset,
    get_dataset_summary,
    split_data,
)
from src.data.preprocessor import preprocess_data
from src.features.engineer import engineer_features
from src.models.trainer import (
    train_all_models,
    select_best_model,
    save_model,
)
from src.models.evaluator import (
    evaluate_all_models,
    create_comparison_table,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_metrics_comparison,
)
from src.utils.helpers import setup_logging, timer

# Configure logging
logger = setup_logging("creditlens.pipeline")


@timer
def run_pipeline(
    tune_hyperparams: bool = False,
    tune_trials: int = 50,
) -> None:
    """
    Execute the complete training pipeline.

    Parameters
    ----------
    tune_hyperparams : bool
        Whether to run Optuna hyperparameter tuning on the best model.
    tune_trials : int
        Number of Optuna trials (only used if tune_hyperparams=True).
    """
    logger.info("=" * 70)
    logger.info("   CreditLens AI — Training Pipeline")
    logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Step 1: Load & Validate Data
    # -----------------------------------------------------------------------
    logger.info("\n📥 STEP 1: Loading Dataset")
    df = load_raw_dataset()
    df = validate_dataset(df)

    summary = get_dataset_summary(df)
    logger.info("Dataset shape: %s", summary["shape"])
    if "class_distribution" in summary:
        logger.info("Class distribution: %s", summary["class_distribution"])

    # -----------------------------------------------------------------------
    # Step 2: Feature Engineering
    # -----------------------------------------------------------------------
    logger.info("\n🔧 STEP 2: Feature Engineering")
    df = engineer_features(df)

    # -----------------------------------------------------------------------
    # Step 3: Train/Test Split
    # -----------------------------------------------------------------------
    logger.info("\n✂️ STEP 3: Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(df)

    # -----------------------------------------------------------------------
    # Step 4: Preprocessing (Scaling, Encoding, SMOTE)
    # -----------------------------------------------------------------------
    logger.info("\n⚙️ STEP 4: Preprocessing")
    X_train_processed, X_test_processed, y_train_resampled, _, feature_names = (
        preprocess_data(X_train, X_test, y_train, apply_smote=True)
    )

    # -----------------------------------------------------------------------
    # Step 5: Model Training & Cross-Validation
    # -----------------------------------------------------------------------
    logger.info("\n🧠 STEP 5: Model Training")
    trained_models, cv_results = train_all_models(X_train_processed, y_train_resampled)

    # -----------------------------------------------------------------------
    # Step 6: Select Best Model
    # -----------------------------------------------------------------------
    logger.info("\n🏆 STEP 6: Model Selection")
    best_name, best_model = select_best_model(trained_models, cv_results)

    # -----------------------------------------------------------------------
    # Step 7: Optional Hyperparameter Tuning
    # -----------------------------------------------------------------------
    if tune_hyperparams and best_name in ("XGBoost", "LightGBM"):
        logger.info("\n🔬 STEP 7: Hyperparameter Tuning")
        from src.models.tuner import tune_model

        tuned_model, best_params, best_score = tune_model(
            best_name, X_train_processed, y_train_resampled, n_trials=tune_trials,
        )
        # Replace the model
        best_model = tuned_model
        trained_models[best_name] = tuned_model
        logger.info("Tuned %s: %s = %.4f", best_name, PRIMARY_METRIC, best_score)
    else:
        logger.info("\n⏩ STEP 7: Skipping hyperparameter tuning (tune_hyperparams=False)")

    # -----------------------------------------------------------------------
    # Step 8: Evaluation on Test Set
    # -----------------------------------------------------------------------
    logger.info("\n📊 STEP 8: Evaluation")
    results = evaluate_all_models(trained_models, X_test_processed, y_test)
    comparison_df = create_comparison_table(results)

    # Save comparison as JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_json = comparison_df.reset_index().to_dict(orient="records")
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_json, f, indent=2)
    logger.info("Saved comparison → %s", REPORTS_DIR / "model_comparison.json")

    # -----------------------------------------------------------------------
    # Step 9: Generate Visualizations
    # -----------------------------------------------------------------------
    logger.info("\n📈 STEP 9: Generating Visualizations")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)
    plot_metrics_comparison(comparison_df)

    # -----------------------------------------------------------------------
    # Step 10: Generate Explanations
    # -----------------------------------------------------------------------
    logger.info("\n🔍 STEP 10: Generating Explanations")
    try:
        from src.explainability.shap_explainer import SHAPExplainer

        shap_exp = SHAPExplainer(
            model=best_model,
            X_data=X_test_processed[:200],  # Use subset for speed
            feature_names=feature_names,
        )
        shap_exp.compute_shap_values()
        shap_exp.plot_summary()
        shap_exp.plot_feature_importance()
        shap_exp.plot_waterfall(sample_index=0)
        logger.info("SHAP explanations generated ✓")
    except Exception as e:
        logger.warning("SHAP explanation generation failed: %s", e)

    # -----------------------------------------------------------------------
    # Step 11: Save Best Model
    # -----------------------------------------------------------------------
    logger.info("\n💾 STEP 11: Saving Best Model")
    save_model(best_model)

    # -----------------------------------------------------------------------
    # Done!
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("   ✅ PIPELINE COMPLETE")
    logger.info("   Best model: %s", best_name)
    logger.info("   Saved to: %s", MODELS_DIR)
    logger.info("   Reports: %s", REPORTS_DIR)
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("  1. Start API:       uv run uvicorn api.main:app --reload")
    logger.info("  2. Start Dashboard: uv run streamlit run dashboard/app.py")


if __name__ == "__main__":
    # Parse simple CLI flags
    tune = "--tune" in sys.argv
    trials = 50
    for i, arg in enumerate(sys.argv):
        if arg == "--trials" and i + 1 < len(sys.argv):
            trials = int(sys.argv[i + 1])

    run_pipeline(tune_hyperparams=tune, tune_trials=trials)
