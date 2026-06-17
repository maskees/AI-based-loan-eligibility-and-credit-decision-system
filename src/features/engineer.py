"""
Feature Engineering Module
===========================
Creates derived financial features that improve model predictive power.
These engineered features capture domain-specific relationships between
raw features (e.g., Debt-to-Income ratio, Loan-to-Income ratio).

Designed for the Kaggle Loan Approval Prediction dataset with columns:
    person_age, person_gender, person_education, person_income,
    person_emp_exp, person_home_ownership, loan_amnt, loan_intent,
    loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
    credit_score, previous_loan_defaults_on_file, loan_status
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the dataset.

    This function creates new columns and should be called BEFORE
    preprocessing (scaling/encoding), as it operates on raw feature values.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset with original features.

    Returns
    -------
    pd.DataFrame
        The dataset with additional engineered features.
    """
    df = df.copy()
    initial_cols = len(df.columns)
    logger.info("Starting feature engineering on %d rows...", len(df))

    df = _create_debt_to_income_ratio(df)
    df = _create_loan_to_income_ratio(df)
    df = _create_employment_stability_score(df)
    df = _create_credit_risk_bands(df)
    df = _create_loan_amount_bins(df)
    df = _create_age_group(df)

    new_features = len(df.columns) - initial_cols
    logger.info(
        "Feature engineering complete: %d new features added",
        new_features,
    )

    return df


def _create_debt_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Debt-to-Income Ratio (DTI) — a key metric in lending decisions.
    Measures what fraction of income goes toward the loan obligation.

    Formula: DTI = loan_amnt / person_income
    A lower DTI signals lower financial stress.
    """
    if "loan_amnt" in df.columns and "person_income" in df.columns:
        df["debt_to_income_ratio"] = np.where(
            df["person_income"] > 0,
            df["loan_amnt"] / df["person_income"],
            0.0,
        )
        logger.info("  ✓ Created: debt_to_income_ratio")
    else:
        logger.warning("  ✗ Skipped debt_to_income_ratio (missing columns)")

    return df


def _create_loan_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loan-to-Income Ratio (LTI) — how large the requested loan is
    relative to the applicant's annual earnings.

    Uses loan_percent_income directly if available, otherwise computes
    loan_amnt / person_income. This is kept separate from DTI for
    clarity and to allow independent feature selection.
    """
    if "loan_percent_income" in df.columns:
        # The dataset already has this as a direct column — use it as-is
        # and create a derived squared version for non-linearity
        df["loan_to_income_ratio"] = df["loan_percent_income"] ** 2
        logger.info("  ✓ Created: loan_to_income_ratio (squared loan_percent_income)")
    elif "loan_amnt" in df.columns and "person_income" in df.columns:
        df["loan_to_income_ratio"] = np.where(
            df["person_income"] > 0,
            df["loan_amnt"] / df["person_income"],
            0.0,
        )
        logger.info("  ✓ Created: loan_to_income_ratio")
    else:
        logger.warning("  ✗ Skipped loan_to_income_ratio (missing columns)")

    return df


def _create_employment_stability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment Stability Score — a binned score based on years of employment.

    Uses 'person_emp_exp' column from the dataset.

    Bins:
        0-1 years   → 1 (Unstable)
        2-4 years   → 2 (Developing)
        5-9 years   → 3 (Stable)
        10+ years   → 4 (Very Stable)
    """
    if "person_emp_exp" in df.columns:
        bins = [-np.inf, 1, 4, 9, np.inf]
        labels = [1, 2, 3, 4]
        df["employment_stability_score"] = pd.cut(
            df["person_emp_exp"],
            bins=bins,
            labels=labels,
        ).astype(float)
        logger.info("  ✓ Created: employment_stability_score")
    else:
        logger.warning(
            "  ✗ Skipped employment_stability_score (missing 'person_emp_exp')"
        )

    return df


def _create_credit_risk_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credit Risk Bands — categorize credit scores into risk tiers.

    Bands (based on CIBIL/FICO-like ranges):
        300-549  → 1 (Very Poor)
        550-649  → 2 (Poor)
        650-749  → 3 (Fair)
        750-849  → 4 (Good)
        850-900  → 5 (Excellent)
    """
    if "credit_score" in df.columns:
        bins = [299, 549, 649, 749, 849, 900]
        labels = [1, 2, 3, 4, 5]
        df["credit_risk_band"] = pd.cut(
            df["credit_score"].clip(300, 900),
            bins=bins,
            labels=labels,
        ).astype(float)
        logger.info("  ✓ Created: credit_risk_band")
    else:
        logger.warning("  ✗ Skipped credit_risk_band (missing 'credit_score')")

    return df


def _create_loan_amount_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loan Amount Bins — categorize loan amounts into buckets using quantiles.
    This helps the model capture non-linear relationships with loan size.
    """
    if "loan_amnt" in df.columns:
        df["loan_amount_bin"] = pd.qcut(
            df["loan_amnt"],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop",
        ).astype(float)
        logger.info("  ✓ Created: loan_amount_bin")
    else:
        logger.warning("  ✗ Skipped loan_amount_bin (missing 'loan_amnt')")

    return df


def _create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Age Group — bin ages into meaningful life stages.

    Bins:
        18-25  → 1 (Young Adult)
        26-35  → 2 (Early Career)
        36-45  → 3 (Mid Career)
        46-55  → 4 (Senior)
        56+    → 5 (Pre-Retirement)
    """
    if "person_age" in df.columns:
        bins = [-np.inf, 25, 35, 45, 55, np.inf]
        labels = [1, 2, 3, 4, 5]
        df["age_group"] = pd.cut(
            df["person_age"],
            bins=bins,
            labels=labels,
        ).astype(float)
        logger.info("  ✓ Created: age_group")
    else:
        logger.warning("  ✗ Skipped age_group (missing 'person_age')")

    return df


def get_engineered_feature_descriptions() -> dict[str, str]:
    """
    Return human-readable descriptions of all engineered features.
    Useful for documentation and dashboard display.
    """
    return {
        "debt_to_income_ratio": (
            "Ratio of loan amount to annual income. "
            "Lower values indicate less financial strain."
        ),
        "loan_to_income_ratio": (
            "Squared loan-to-income percentage, capturing non-linear "
            "effects of high loan burden relative to income."
        ),
        "employment_stability_score": (
            "Binned employment experience: 1=Unstable (<2yr), 2=Developing (2-4yr), "
            "3=Stable (5-9yr), 4=Very Stable (10+yr)."
        ),
        "credit_risk_band": (
            "Credit score categorized into risk tiers: "
            "1=Very Poor, 2=Poor, 3=Fair, 4=Good, 5=Excellent."
        ),
        "loan_amount_bin": (
            "Loan amount bucketed into 5 quantile-based groups "
            "to capture non-linear effects."
        ),
        "age_group": (
            "Age categorized into life stages: 1=Young Adult, "
            "2=Early Career, 3=Mid Career, 4=Senior, 5=Pre-Retirement."
        ),
    }
