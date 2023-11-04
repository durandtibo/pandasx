from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_equal
from pandas import DataFrame

from flamme.analyzer import TemporalContinuousDistributionAnalyzer
from flamme.section import EmptySection, TemporalContinuousDistributionSection

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
