"""
Utility Helpers
===============
Common utility functions used across the project — logging setup,
file I/O, timing decorators, etc.
"""

import logging
import time
from functools import wraps
from pathlib import Path

from src.config import LOG_FORMAT, LOG_LEVEL


def setup_logging(name: str = "creditlens") -> logging.Logger:
    """
    Configure and return a project logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    return logger


def timer(func):
    """
    Decorator that logs the execution time of a function.

    Usage::

        @timer
        def my_slow_function():
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if elapsed < 1:
            logger.info("⏱ %s completed in %.0f ms", func.__name__, elapsed * 1000)
        elif elapsed < 60:
            logger.info("⏱ %s completed in %.2f s", func.__name__, elapsed)
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            logger.info("⏱ %s completed in %dm %.1fs", func.__name__, minutes, seconds)

        return result

    return wrapper


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The ensured directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_currency(value: float, symbol: str = "₹") -> str:
    """
    Format a number as currency with Indian/standard notation.

    Parameters
    ----------
    value : float
        The monetary value to format.
    symbol : str
        Currency symbol. Default is Indian Rupee (₹).

    Returns
    -------
    str
        Formatted currency string.
    """
    if value >= 1_00_00_000:  # 1 Crore
        return f"{symbol}{value / 1_00_00_000:.2f} Cr"
    elif value >= 1_00_000:  # 1 Lakh
        return f"{symbol}{value / 1_00_000:.2f} L"
    elif value >= 1_000:
        return f"{symbol}{value / 1_000:.1f} K"
    else:
        return f"{symbol}{value:,.0f}"
