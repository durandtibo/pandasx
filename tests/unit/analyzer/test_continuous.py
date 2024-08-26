from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose, objects_are_equal
from polars.testing import assert_series_equal

from flamme.analyzer import ColumnContinuousAnalyzer
from flamme.section import ColumnContinuousSection, EmptySection


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame({"col": [None, *list(range(101)), None]}, schema={"col": pl.Int64})


##############################################
#     Tests for ColumnContinuousAnalyzer     #
##############################################


def test_column_continuous_analyzer_str() -> None:
    assert str(ColumnContinuousAnalyzer(column="col")).startswith("ColumnContinuousAnalyzer(")


def test_column_continuous_analyzer_series(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert_series_equal(
        section.series,
        pl.Series(values=[None, *list(range(101)), None], name="col", dtype=pl.Int64),
    )


def test_column_continuous_analyzer_column(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.column == "col"


def test_column_continuous_analyzer_nbins_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.nbins is None


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_column_continuous_analyzer_nbins(dataframe: pl.DataFrame, nbins: int) -> None:
    section = ColumnContinuousAnalyzer(column="col", nbins=nbins).analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.nbins == nbins


def test_column_continuous_analyzer_yscale_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_continuous_analyzer_yscale(dataframe: pl.DataFrame, yscale: str) -> None:
    section = ColumnContinuousAnalyzer(column="col", yscale=yscale).analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.yscale == yscale


def test_column_continuous_analyzer_xmin_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmin == "q0"


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None])
def test_column_continuous_analyzer_xmin(dataframe: pl.DataFrame, xmin: float | str | None) -> None:
    section = ColumnContinuousAnalyzer(column="col", xmin=xmin).analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmin == xmin


def test_column_continuous_analyzer_xmax_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmax == "q1"


@pytest.mark.parametrize("xmax", [1.0, "q0.1", None])
def test_column_continuous_analyzer_xmax(dataframe: pl.DataFrame, xmax: float | str | None) -> None:
    section = ColumnContinuousAnalyzer(column="col", xmax=xmax).analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmax == xmax


def test_column_continuous_analyzer_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_continuous_analyzer_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnContinuousAnalyzer(column="col", figsize=figsize).analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert section.figsize == figsize


def test_column_continuous_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(dataframe)
    assert isinstance(section, ColumnContinuousSection)
    assert objects_are_allclose(
        section.get_statistics(),
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


def test_column_continuous_analyzer_analyze_empty() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(pl.DataFrame({"col": []}))
    assert isinstance(section, ColumnContinuousSection)
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
            ">0": 0,
            "<0": 0,
            "=0": 0,
        },
        equal_nan=True,
    )


def test_column_continuous_analyzer_analyze_missing_column() -> None:
    section = ColumnContinuousAnalyzer(column="col2").analyze(pl.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
