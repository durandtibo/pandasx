r"""Contain statistics utility functions."""

from __future__ import annotations

__all__ = ["compute_statistics_continuous", "compute_statistics_continuous_series", "quantile"]

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kurtosis, skew

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


def compute_statistics_continuous(data: np.ndarray) -> dict[str, float]:
    r"""Return several descriptive statistics for the data with
    continuous values.

    Args:
        data: The data to analyze.

    Returns:
        The descriptive statistics for the input data.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from flamme.utils.stats import compute_statistics_continuous
    >>> compute_statistics_continuous(np.arange(101))
    {'mean': 50.0, 'std': 29.15...,
     'skewness': 0.0, 'kurtosis': -1.20..., 'min': 0.0, 'q001': 0.1, 'q01': 1.0,
     'q05': 5.0, 'q10': 10.0, 'q25': 25.0, 'median': 50.0, 'q75': 75.0, 'q90': 90.0,
     'q95': 95.0, 'q99': 99.0, 'q999': 99.9, 'max': 100.0, '>0': 100, '<0': 0, '=0': 1}

    ```
    """
    array = data.ravel().astype(np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": float("nan"),
            "q001": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "q999": float("nan"),
            "max": float("nan"),
            ">0": 0,
            "<0": 0,
            "=0": 0,
        }
    quantiles = quantile(array, q=[0.001, 0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999])
    return {
        "mean": np.mean(array).item(),
        "std": np.std(array).item(),
        "skewness": skew(array).item(),
        "kurtosis": kurtosis(array).item(),
        "min": np.min(array).item(),
        "q001": quantiles[0.001],
        "q01": quantiles[0.01],
        "q05": quantiles[0.05],
        "q10": quantiles[0.1],
        "q25": quantiles[0.25],
        "median": np.median(array).item(),
        "q75": quantiles[0.75],
        "q90": quantiles[0.9],
        "q95": quantiles[0.95],
        "q99": quantiles[0.99],
        "q999": quantiles[0.999],
        "max": np.max(array).item(),
        ">0": (array > 0).sum().item(),
        "<0": (array < 0).sum().item(),
        "=0": (array == 0).sum().item(),
    }


def compute_statistics_continuous_series(series: pl.Series) -> dict[str, float]:
    r"""Return several descriptive statistics for the data with
    continuous values.

    Args:
        series: The series to analyze.

    Returns:
        The descriptive statistics for the input data.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from flamme.utils.stats import compute_statistics_continuous_series
    >>> compute_statistics_continuous_series(pl.Series(list(range(101))))
    {'mean': 50.0, 'std': 29.15...,
     'skewness': 0.0, 'kurtosis': -1.20..., 'min': 0.0, 'q001': 0.1, 'q01': 1.0,
     'q05': 5.0, 'q10': 10.0, 'q25': 25.0, 'median': 50.0, 'q75': 75.0, 'q90': 90.0,
     'q95': 95.0, 'q99': 99.0, 'q999': 99.9, 'max': 100.0, '>0': 100, '<0': 0, '=0': 1,
     'count': 101, 'num_nulls': 0, 'nunique': 101, 'num_non_nulls': 101}

    ```
    """
    stats = {
        "count": int(series.shape[0]),
        "num_nulls": int(series.null_count()),
        "nunique": series.n_unique(),
    }
    stats["num_non_nulls"] = stats["count"] - stats["num_nulls"]
    return compute_statistics_continuous(series.drop_nulls().to_numpy()) | stats


def quantile(array: np.ndarray, q: Sequence[float]) -> dict[float, float]:
    r"""Compute the q-th quantile of the data.

    Args:
        array: The input data.
        q: The quantiles to compute. Values must be between 0 and 1
            inclusive.

    Returns:
        A dictionary with the quantiles values.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from flamme.utils.stats import quantile
    >>> quantile(np.arange(101), q=[0.001, 0.01, 0.05, 0.1, 0.25, 0.75, 0.9, 0.95, 0.99, 0.999])
    {0.001: 0.1, 0.01: 1.0, 0.05: 5.0, 0.1: 10.0, 0.25: 25.0, 0.75: 75.0,
     0.9: 90.0, 0.95: 95.0, 0.99: 99.0, 0.999: 99.9}

    ```
    """
    array = array.ravel()
    if array.size == 0:
        return {v: float("nan") for v in q}
    return dict(zip(q, np.quantile(array.astype(np.float64), q).tolist()))
