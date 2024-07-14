from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_allclose

from flamme.utils.stats import (
    compute_statistics_continuous,
    compute_statistics_continuous_series,
    quantile,
)

###################################################
#     Tests for compute_statistics_continuous     #
###################################################


def test_compute_statistics_continuous() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous(np.arange(101)),
        {
            "mean": 50.0,
            "std": 29.154759474226502,
            "skewness": 0.0,
            "kurtosis": -1.2,
            "min": 0.0,
            "q001": 0.1,
            "q01": 1.0,
            "q05": 5.0,
            "q10": 10.0,
            "q25": 25.0,
            "median": 50.0,
            "q75": 75.0,
            "q90": 90.0,
            "q95": 95.0,
            "q99": 99.0,
            "q999": 99.9,
            "max": 100.0,
            ">0": 100,
            "<0": 0,
            "=0": 1,
        },
        atol=1e-2,
    )


def test_compute_statistics_continuous_empty() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous(np.array([])),
        {
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
        },
        equal_nan=True,
    )


def test_compute_statistics_continuous_with_nan() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous(np.array([np.nan, *list(range(101)), np.nan])),
        {
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
            ">0": 100,
            "<0": 0,
            "=0": 1,
        },
        equal_nan=True,
    )


def test_compute_statistics_continuous_single_numeric_value() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous(np.array([1, 1, 1, 1, 1])),
        {
            "mean": 1.0,
            "std": 0.0,
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": 1.0,
            "q001": 1.0,
            "q01": 1.0,
            "q05": 1.0,
            "q10": 1.0,
            "q25": 1.0,
            "median": 1.0,
            "q75": 1.0,
            "q90": 1.0,
            "q95": 1.0,
            "q99": 1.0,
            "q999": 1.0,
            "max": 1.0,
            ">0": 5,
            "<0": 0,
            "=0": 0,
        },
        equal_nan=True,
    )


def test_compute_statistics_continuous_only_nans() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous(np.asarray([np.nan, np.nan, np.nan, np.nan])),
        {
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
        },
        equal_nan=True,
    )


##########################################################
#     Tests for compute_statistics_continuous_series     #
##########################################################


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.Float32, pl.Float64])
def test_compute_statistics_continuous_series(dtype: pl.DataType) -> None:
    assert objects_are_allclose(
        compute_statistics_continuous_series(pl.Series(list(range(101)), dtype=dtype)),
        {
            "count": 101,
            "num_nulls": 0,
            "num_non_nulls": 101,
            "nunique": 101,
            "mean": 50.0,
            "std": 29.154759474226502,
            "skewness": 0.0,
            "kurtosis": -1.2,
            "min": 0.0,
            "q001": 0.1,
            "q01": 1.0,
            "q05": 5.0,
            "q10": 10.0,
            "q25": 25.0,
            "median": 50.0,
            "q75": 75.0,
            "q90": 90.0,
            "q95": 95.0,
            "q99": 99.0,
            "q999": 99.9,
            "max": 100.0,
            ">0": 100,
            "<0": 0,
            "=0": 1,
        },
        atol=1e-2,
    )


def test_compute_statistics_continuous_series_empty() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous_series(pl.Series([])),
        {
            "count": 0,
            "num_nulls": 0,
            "num_non_nulls": 0,
            "nunique": 0,
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
        },
        equal_nan=True,
    )


def test_compute_statistics_continuous_series_with_none() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous_series(pl.Series([None, *list(range(101)), None])),
        {
            "count": 103,
            "num_nulls": 2,
            "num_non_nulls": 101,
            "nunique": 102,
            "mean": 50.0,
            "std": 29.154759474226502,
            "skewness": 0.0,
            "kurtosis": -1.2,
            "min": 0.0,
            "q001": 0.1,
            "q01": 1.0,
            "q05": 5.0,
            "q10": 10.0,
            "q25": 25.0,
            "median": 50.0,
            "q75": 75.0,
            "q90": 90.0,
            "q95": 95.0,
            "q99": 99.0,
            "q999": 99.9,
            "max": 100.0,
            ">0": 100,
            "<0": 0,
            "=0": 1,
        },
        atol=1e-2,
    )


def test_compute_statistics_continuous_series_single_numeric_value() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous_series(pl.Series([1, 1, 1, 1, 1])),
        {
            "count": 5,
            "num_nulls": 0,
            "num_non_nulls": 5,
            "nunique": 1,
            "mean": 1.0,
            "std": 0.0,
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": 1.0,
            "q001": 1.0,
            "q01": 1.0,
            "q05": 1.0,
            "q10": 1.0,
            "q25": 1.0,
            "median": 1.0,
            "q75": 1.0,
            "q90": 1.0,
            "q95": 1.0,
            "q99": 1.0,
            "q999": 1.0,
            "max": 1.0,
            ">0": 5,
            "<0": 0,
            "=0": 0,
        },
        equal_nan=True,
    )


def test_compute_statistics_continuous_series_only_nans() -> None:
    assert objects_are_allclose(
        compute_statistics_continuous_series(pl.Series([np.nan, np.nan, np.nan, np.nan])),
        {
            "count": 4,
            "num_nulls": 0,
            "num_non_nulls": 4,
            "nunique": 1,
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
        },
        equal_nan=True,
    )


##############################
#     Tests for quantile     #
##############################


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_quantile(dtype: np.dtype) -> None:
    assert objects_are_allclose(
        quantile(np.arange(101, dtype=dtype), q=[0.1, 0.5, 0.9]),
        {0.1: 10.0, 0.5: 50.0, 0.9: 90.0},
    )


def test_quantile_with_nans() -> None:
    assert objects_are_allclose(
        quantile(np.array([np.nan, *list(range(101)), np.nan]), q=[0.1, 0.5, 0.9]),
        {0.1: float("nan"), 0.5: float("nan"), 0.9: float("nan")},
        equal_nan=True,
    )


def test_quantile_only_nans() -> None:
    assert objects_are_allclose(
        quantile(np.array([np.nan, np.nan, np.nan, np.nan]), q=[0.1, 0.5, 0.9]),
        {0.1: float("nan"), 0.5: float("nan"), 0.9: float("nan")},
        equal_nan=True,
    )


def test_quantile_empty() -> None:
    assert objects_are_allclose(
        quantile(np.array([]), q=[0.1, 0.5, 0.9]),
        {0.1: float("nan"), 0.5: float("nan"), 0.9: float("nan")},
        equal_nan=True,
    )
