"""
Feature Engineering Module
===========================
Creates derived financial features that improve model predictive power.
These engineered features capture domain-specific relationships between
raw features (e.g., Debt-to-Income ratio, Loan-to-Income ratio).
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
    logger.info("Starting feature engineering on %d rows...", len(df))

    df = _create_debt_to_income_ratio(df)
    df = _create_loan_to_income_ratio(df)
    df = _create_income_per_dependent(df)
    df = _create_employment_stability_score(df)
    df = _create_credit_risk_bands(df)
    df = _create_loan_amount_bins(df)

    logger.info(
        "Feature engineering complete: %d new features added",
        df.shape[1] - len(df.columns),
    )

    return df


def _create_debt_to_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Debt-to-Income Ratio (DTI) — a key metric in lending decisions.
    Measures what fraction of income goes toward existing debt obligations.

    Formula: DTI = total_debt / annual_income
    A lower DTI signals lower financial stress.
    """
    if "loan_amount" in df.columns and "annual_income" in df.columns:
        df["debt_to_income_ratio"] = np.where(
            df["annual_income"] > 0,
            df["loan_amount"] / df["annual_income"],
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

    Formula: LTI = loan_amount / annual_income
    Lenders typically cap this at 4–5x.
    """
    if "loan_amount" in df.columns and "annual_income" in df.columns:
        df["loan_to_income_ratio"] = np.where(
            df["annual_income"] > 0,
            df["loan_amount"] / df["annual_income"],
            0.0,
        )
        logger.info("  ✓ Created: loan_to_income_ratio")
    else:
        logger.warning("  ✗ Skipped loan_to_income_ratio (missing columns)")

    return df


def _create_income_per_dependent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Income Per Dependent — discretionary income available per family member.

    Formula: income_per_dependent = annual_income / (number_of_dependents + 1)
    The +1 accounts for the applicant themselves.
    """
    if "annual_income" in df.columns and "number_of_dependents" in df.columns:
        df["income_per_dependent"] = df["annual_income"] / (
            df["number_of_dependents"] + 1
        )
        logger.info("  ✓ Created: income_per_dependent")
    else:
        logger.warning("  ✗ Skipped income_per_dependent (missing columns)")

    return df


def _create_employment_stability_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment Stability Score — a binned score based on years of employment.

    Bins:
        0-1 years   → 1 (Unstable)
        2-4 years   → 2 (Developing)
        5-9 years   → 3 (Stable)
        10+ years   → 4 (Very Stable)
    """
    if "employment_length" in df.columns:
        bins = [-np.inf, 1, 4, 9, np.inf]
        labels = [1, 2, 3, 4]
        df["employment_stability_score"] = pd.cut(
            df["employment_length"],
            bins=bins,
            labels=labels,
        ).astype(float)
        logger.info("  ✓ Created: employment_stability_score")
    else:
        logger.warning(
            "  ✗ Skipped employment_stability_score (missing 'employment_length')"
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
    if "loan_amount" in df.columns:
        df["loan_amount_bin"] = pd.qcut(
            df["loan_amount"],
            q=5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop",
        ).astype(float)
        logger.info("  ✓ Created: loan_amount_bin")
    else:
        logger.warning("  ✗ Skipped loan_amount_bin (missing 'loan_amount')")

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
            "How large the loan is relative to yearly earnings. "
            "Typically capped at 4-5x by lenders."
        ),
        "income_per_dependent": (
            "Annual income divided by number of dependents (+1 for self). "
            "Higher values mean more disposable income."
        ),
        "employment_stability_score": (
            "Binned employment length: 1=Unstable (<2yr), 2=Developing (2-4yr), "
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
    }
