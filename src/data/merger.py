"""
Data Merger Module
==================
Merges multiple loan/credit datasets into a single unified DataFrame.
Handles column name harmonization, value mapping, and schema alignment
across datasets with different naming conventions.

Supported datasets:
    1. Kaggle Loan Approval Prediction Dataset (~45K rows, 14 cols)
    2. Kaggle Credit Risk Dataset by Laotse (~32K rows, 12 cols)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, TARGET_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column Mapping: Credit Risk Dataset → Primary Dataset schema
# ---------------------------------------------------------------------------
CREDIT_RISK_COLUMN_MAP = {
    "person_emp_length": "person_emp_exp",
    "cb_person_default_on_file": "previous_loan_defaults_on_file",
}

# Value mapping for the defaults column (Y/N → Yes/No)
DEFAULTS_VALUE_MAP = {
    "Y": "Yes",
    "N": "No",
}


def load_and_merge_datasets(
    primary_filename: str = "loan_approval_dataset.csv",
    additional_filenames: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load and merge all available datasets from data/raw/ into a single
    unified DataFrame.

    Parameters
    ----------
    primary_filename : str
        The primary dataset filename.
    additional_filenames : list[str], optional
        List of additional dataset filenames to merge. If None,
        auto-detects any CSV files in data/raw/ that aren't the primary.

    Returns
    -------
    pd.DataFrame
        The merged dataset with harmonized column names.
    """
    # Load primary dataset
    primary_path = RAW_DATA_DIR / primary_filename
    if not primary_path.exists():
        raise FileNotFoundError(f"Primary dataset not found: {primary_path}")

    primary_df = pd.read_csv(primary_path)
    logger.info(
        "Loaded primary dataset: %s (%d rows × %d cols)",
        primary_filename, *primary_df.shape,
    )

    # Auto-detect additional datasets if not specified
    if additional_filenames is None:
        additional_filenames = [
            f.name for f in RAW_DATA_DIR.glob("*.csv")
            if f.name != primary_filename
        ]

    if not additional_filenames:
        logger.info("No additional datasets found — using primary only.")
        return primary_df

    # Load and harmonize each additional dataset
    all_dfs = [primary_df]

    for filename in additional_filenames:
        filepath = RAW_DATA_DIR / filename
        if not filepath.exists():
            logger.warning("Dataset not found, skipping: %s", filepath)
            continue

        df = pd.read_csv(filepath)
        logger.info(
            "Loaded additional dataset: %s (%d rows × %d cols)",
            filename, *df.shape,
        )

        # Harmonize column names and values
        df = _harmonize_dataset(df, filename)
        all_dfs.append(df)

    # Merge all datasets using UNION of columns (keeps all features).
    # Missing columns in any dataset will be filled with NaN,
    # which the preprocessor's imputer will handle automatically.
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure target column exists
    if TARGET_COLUMN not in merged_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in merged dataset. "
            f"Available columns: {list(merged_df.columns)}"
        )

    # Log which columns have missing values from the merge
    merge_nulls = merged_df.isnull().sum()
    cols_with_nulls = merge_nulls[merge_nulls > 0]
    if len(cols_with_nulls) > 0:
        logger.info(
            "Columns with NaN after merge (will be imputed): %s",
            {col: int(n) for col, n in cols_with_nulls.items()},
        )

    # Shuffle the merged data to mix records from different sources
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(
        "Merged dataset: %d rows × %d cols (from %d sources)",
        *merged_df.shape, len(all_dfs),
    )

    # Log class distribution
    if TARGET_COLUMN in merged_df.columns:
        dist = merged_df[TARGET_COLUMN].value_counts().to_dict()
        logger.info("Merged class distribution: %s", dist)

    return merged_df


def _harmonize_dataset(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Harmonize a dataset's column names and values to match the
    primary dataset schema.

    Parameters
    ----------
    df : pd.DataFrame
        The raw additional dataset.
    filename : str
        Filename for logging and identification.

    Returns
    -------
    pd.DataFrame
        Harmonized DataFrame.
    """
    df = df.copy()

    # Apply column renaming if this is the credit risk dataset
    if "person_emp_length" in df.columns or "cb_person_default_on_file" in df.columns:
        rename_map = {
            old: new for old, new in CREDIT_RISK_COLUMN_MAP.items()
            if old in df.columns
        }
        if rename_map:
            df = df.rename(columns=rename_map)
            logger.info(
                "  Renamed columns in %s: %s", filename, rename_map
            )

    # Map default values: Y/N → Yes/No
    if "previous_loan_defaults_on_file" in df.columns:
        current_values = set(df["previous_loan_defaults_on_file"].dropna().unique())
        if current_values.issubset({"Y", "N"}):
            df["previous_loan_defaults_on_file"] = (
                df["previous_loan_defaults_on_file"].map(DEFAULTS_VALUE_MAP)
            )
            logger.info("  Mapped default values: Y/N → Yes/No in %s", filename)

    # Drop columns that don't exist in the primary schema
    # (loan_grade is specific to the credit risk dataset)
    cols_to_drop = [c for c in ["loan_grade"] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info("  Dropped extra columns in %s: %s", filename, cols_to_drop)

    return df
