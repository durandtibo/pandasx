from __future__ import annotations

import math

import numpy as np
import pandas as pd
from coola import objects_are_allclose, objects_are_equal
from pandas import DataFrame, Series
from pandas.testing import assert_series_equal
from pytest import mark

from flamme.analyzer import (
    ContinuousDistributionAnalyzer,
    TemporalContinuousDistributionAnalyzer,
)
from flamme.section import (
    ContinuousDistributionSection,
    EmptySection,
    TemporalContinuousDistributionSection,
)
from tests.unit.section.test_continous import STATS_KEYS

####################################################
#     Tests for ContinuousDistributionAnalyzer     #
####################################################


def test_continuous_distribution_analyzer_str() -> None:
    assert str(ContinuousDistributionAnalyzer(column="float")).startswith(
        "ContinuousDistributionAnalyzer("
    )


def test_continuous_distribution_analyzer_series() -> None:
    section = ContinuousDistributionAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ContinuousDistributionSection)
    assert_series_equal(section.series, Series([np.nan] + list(range(101)) + [np.nan], name="col"))


def test_continuous_distribution_analyzer_column() -> None:
    section = ContinuousDistributionAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ContinuousDistributionSection)
    assert section.column == "col"


def test_continuous_distribution_analyzer_nbins_default() -> None:
    section = ContinuousDistributionAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ContinuousDistributionSection)
    assert section.nbins is None


@mark.parametrize("nbins", (1, 2, 4))
def test_continuous_distribution_analyzer_nbins(nbins: int) -> None:
    section = ContinuousDistributionAnalyzer(column="col", nbins=nbins).analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ContinuousDistributionSection)
    assert section.nbins == nbins


def test_continuous_distribution_analyzer_get_statistics() -> None:
    section = ContinuousDistributionAnalyzer(column="col").analyze(
        DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})
    )
    assert isinstance(section, ContinuousDistributionSection)
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "count": 103,
            "num_nulls": 2,
            "num_non_nulls": 101,
            "nunique": 102,
            "mean": 50.0,
            "median": 50.0,
            "min": 0.0,
            "max": 100.0,
            "std": 29.300170647967224,
            "q01": 1.0,
            "q05": 5.0,
            "q10": 10.0,
            "q25": 25.0,
            "q75": 75.0,
            "q90": 90.0,
            "q95": 95.0,
            "q99": 99.0,
        },
    )


def test_continuous_distribution_analyzer_get_statistics_empty() -> None:
    section = ContinuousDistributionAnalyzer(column="col").analyze(DataFrame({"col": []}))
    assert isinstance(section, ContinuousDistributionSection)
    stats = section.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 0
    assert stats["num_nulls"] == 0
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 0
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_continuous_distribution_analyzer_get_statistics_missing_column() -> None:
    section = ContinuousDistributionAnalyzer(column="col2").analyze(DataFrame({"col": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


############################################################
#     Tests for TemporalContinuousDistributionAnalyzer     #
############################################################


def test_temporal_continuous_distribution_analyzer_str() -> None:
    assert str(
        TemporalContinuousDistributionAnalyzer(column="float", dt_column="datetime", period="M")
    ).startswith("TemporalContinuousDistributionAnalyzer(")


def test_temporal_continuous_distribution_analyzer_get_statistics() -> None:
    section = TemporalContinuousDistributionAnalyzer(
        column="float", dt_column="datetime", period="M"
    ).analyze(
        DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        )
    )
    assert isinstance(section, TemporalContinuousDistributionSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_continuous_distribution_analyzer_get_statistics_empty() -> None:
    section = TemporalContinuousDistributionAnalyzer(
        column="float", dt_column="datetime", period="M"
    ).analyze(DataFrame({"float": [], "int": [], "str": [], "datetime": []}))
    assert isinstance(section, TemporalContinuousDistributionSection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_continuous_distribution_analyzer_get_statistics_missing_column() -> None:
    section = TemporalContinuousDistributionAnalyzer(
        column="float", dt_column="datetime", period="M"
    ).analyze(DataFrame({"datetime": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_continuous_distribution_analyzer_get_statistics_missing_dt_column() -> None:
    section = TemporalContinuousDistributionAnalyzer(
        column="float", dt_column="datetime", period="M"
    ).analyze(DataFrame({"float": []}))
    assert isinstance(section, EmptySection)
    assert objects_are_equal(section.get_statistics(), {})
