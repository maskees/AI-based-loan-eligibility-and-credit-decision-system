"""
Data Loader Module
==================
Handles loading raw CSV datasets, schema validation, and train/test splitting.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    DATASET_FILENAME,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def load_raw_dataset(filename: str | None = None) -> pd.DataFrame:
    """
    Load the raw loan dataset from a CSV file.

    Parameters
    ----------
    filename : str, optional
        Name of the CSV file inside ``data/raw/``.
        Defaults to the value in ``config.DATASET_FILENAME``.

    Returns
    -------
    pd.DataFrame
        The raw dataset as a Pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist.
    """
    if filename is None:
        filename = DATASET_FILENAME

    filepath = RAW_DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please place your CSV file in '{RAW_DATA_DIR}/'."
        )

    logger.info("Loading dataset from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded dataset: %d rows × %d columns", *df.shape)

    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic validation on the loaded dataset.

    Checks
    ------
    - Target column exists
    - No completely empty columns
    - Minimum row count (at least 100 rows)

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    pd.DataFrame
        The validated dataset (unchanged if all checks pass).

    Raises
    ------
    ValueError
        If validation fails.
    """
    # Check target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop completely empty columns
    empty_cols = df.columns[df.isnull().all()].tolist()
    if empty_cols:
        logger.warning("Dropping completely empty columns: %s", empty_cols)
        df = df.drop(columns=empty_cols)

    # Minimum row check
    if len(df) < 100:
        raise ValueError(
            f"Dataset has only {len(df)} rows. "
            "A minimum of 100 rows is required for meaningful training."
        )

    logger.info("Dataset validation passed ✓")
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Generate a summary of the dataset for quick inspection.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to summarize.

    Returns
    -------
    dict
        Summary containing shape, dtypes, missing values, and class balance.
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().mean() * 100).round(2).to_dict(),
    }

    if TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts()
        summary["class_distribution"] = target_counts.to_dict()
        summary["class_balance_ratio"] = round(
            target_counts.min() / target_counts.max(), 4
        )

    return summary


def split_data(
    df: pd.DataFrame,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into train and test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed dataset with features and target.
    test_size : float, optional
        Fraction of data to use for testing. Defaults to ``config.TEST_SIZE``.
    random_state : int, optional
        Random seed for reproducibility. Defaults to ``config.RANDOM_STATE``.

    Returns
    -------
    tuple of (X_train, X_test, y_train, y_test)
        The split feature matrices and target vectors.
    """
    if test_size is None:
        test_size = TEST_SIZE
    if random_state is None:
        random_state = RANDOM_STATE

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Split data: train=%d rows, test=%d rows (test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )

    return X_train, X_test, y_train, y_test
