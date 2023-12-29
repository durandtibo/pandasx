from __future__ import annotations

import numpy as np
from coola import objects_are_allclose, objects_are_equal
from pandas import DataFrame, Series
from pandas.testing import assert_series_equal
from pytest import mark

from flamme.analyzer import (
    ColumnContinuousAdvancedAnalyzer,
)
from flamme.section import (
    ColumnContinuousAdvancedSection,
    EmptySection,
)

######################################################
#     Tests for ColumnContinuousAdvancedAnalyzer     #
######################################################


def test_column_continuous_analyzer_str() -> None:
    assert str(ColumnContinuousAdvancedAnalyzer(column="col")).startswith(
        "ColumnContinuousAdvancedAnalyzer("
    )


def test_column_continuous_analyzer_series() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert_series_equal(section.series, Series([np.nan] + list(range(101)) + [np.nan], name="col"))


def test_column_continuous_analyzer_column() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.column == "col"


def test_column_continuous_analyzer_nbins_default() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.nbins is None


@mark.parametrize("nbins", (1, 2, 4))
def test_column_continuous_analyzer_nbins(nbins: int) -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col", nbins=nbins).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.nbins == nbins


def test_column_continuous_analyzer_yscale_default() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.yscale == "auto"


@mark.parametrize("yscale", ["linear", "log"])
def test_column_continuous_analyzer_yscale(yscale: str) -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col", yscale=yscale).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.yscale == yscale


def test_column_continuous_analyzer_figsize_default() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.figsize is None


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_column_continuous_analyzer_figsize(figsize: tuple[float, float]) -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col", figsize=figsize).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert section.figsize == figsize


def test_column_continuous_analyzer_get_statistics() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "count": 103,
            "num_nulls": 2,
            "num_non_nulls": 101,
            "nunique": 102,
            "mean": 50.0,
            "std": 29.300170647967224,
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
        },
        atol=1e-2,
    )


def test_column_continuous_analyzer_get_statistics_empty() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col").analyze(DataFrame({"col": []}))
    assert isinstance(section, ColumnContinuousAdvancedSection)
    assert objects_are_allclose(
        section.get_statistics(),
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
        },
        equal_nan=True,
    )


def test_column_continuous_analyzer_get_statistics_missing_column() -> None:
    section = ColumnContinuousAdvancedAnalyzer(column="col2").analyze(DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
