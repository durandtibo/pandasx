from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from coola import objects_are_allclose, objects_are_equal
from pandas.testing import assert_series_equal

from flamme.analyzer import ColumnContinuousAnalyzer, ColumnTemporalContinuousAnalyzer
from flamme.section import (
    ColumnContinuousSection,
    ColumnTemporalContinuousSection,
    EmptySection,
)

##############################################
#     Tests for ColumnContinuousAnalyzer     #
##############################################


def test_column_continuous_analyzer_str() -> None:
    assert str(ColumnContinuousAnalyzer(column="col")).startswith("ColumnContinuousAnalyzer(")


def test_column_continuous_analyzer_series() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert_series_equal(section.series, pd.Series([np.nan, *list(range(101)), np.nan], name="col"))


def test_column_continuous_analyzer_column() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.column == "col"


def test_column_continuous_analyzer_nbins_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.nbins is None


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_column_continuous_analyzer_nbins(nbins: int) -> None:
    section = ColumnContinuousAnalyzer(column="col", nbins=nbins).analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.nbins == nbins


def test_column_continuous_analyzer_yscale_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_continuous_analyzer_yscale(yscale: str) -> None:
    section = ColumnContinuousAnalyzer(column="col", yscale=yscale).analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.yscale == yscale


def test_column_continuous_analyzer_xmin_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmin == "q0"


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None])
def test_column_continuous_analyzer_xmin(xmin: float | str | None) -> None:
    section = ColumnContinuousAnalyzer(column="col", xmin=xmin).analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmin == xmin


def test_column_continuous_analyzer_xmax_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmax == "q1"


@pytest.mark.parametrize("xmax", [1.0, "q0.1", None])
def test_column_continuous_analyzer_xmax(xmax: float | str | None) -> None:
    section = ColumnContinuousAnalyzer(column="col", xmax=xmax).analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.xmax == xmax


def test_column_continuous_analyzer_figsize_default() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_continuous_analyzer_figsize(figsize: tuple[float, float]) -> None:
    section = ColumnContinuousAnalyzer(column="col", figsize=figsize).analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
    assert section.figsize == figsize


def test_column_continuous_analyzer_get_statistics() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(
        pd.DataFrame({"col": [np.nan, *list(range(101)), np.nan]})
    )
    assert isinstance(section, ColumnContinuousSection)
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
            ">0": 100,
            "<0": 0,
            "=0": 1,
        },
        atol=1e-2,
    )


def test_column_continuous_analyzer_get_statistics_empty() -> None:
    section = ColumnContinuousAnalyzer(column="col").analyze(pd.DataFrame({"col": []}))
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


def test_column_continuous_analyzer_get_statistics_missing_column() -> None:
    section = ColumnContinuousAnalyzer(column="col2").analyze(pd.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


######################################################
#     Tests for ColumnTemporalContinuousAnalyzer     #
######################################################


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col": np.array([1.2, 4.2, np.nan, 2.2]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


def test_column_temporal_continuous_analyzer_str() -> None:
    assert str(
        ColumnTemporalContinuousAnalyzer(column="col", dt_column="datetime", period="M")
    ).startswith("ColumnTemporalContinuousAnalyzer(")


def test_column_temporal_continuous_analyzer_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.column == "col"


def test_column_temporal_continuous_analyzer_dt_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.dt_column == "datetime"


def test_column_temporal_continuous_analyzer_period(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.period == "M"


def test_column_temporal_continuous_analyzer_yscale_default(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.yscale == "auto"


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_temporal_continuous_analyzer_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M", yscale=yscale
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.yscale == yscale


def test_column_temporal_continuous_analyzer_figsize_default(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col",
        dt_column="datetime",
        period="M",
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize is None


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_continuous_analyzer_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M", figsize=figsize
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert section.figsize == figsize


def test_column_temporal_continuous_analyzer_get_statistics(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(dataframe)
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_get_statistics_empty() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": [], "int": [], "str": [], "datetime": []}))
    assert isinstance(section, ColumnTemporalContinuousSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_get_statistics_missing_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_analyzer_get_statistics_missing_dt_column() -> None:
    section = ColumnTemporalContinuousAnalyzer(
        column="col", dt_column="datetime", period="M"
    ).analyze(pd.DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
