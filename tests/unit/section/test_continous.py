from __future__ import annotations

import math

import numpy as np
from coola import objects_are_allclose
from jinja2 import Template
from pandas import DataFrame
from pytest import fixture

from flamme.section import ContinuousDistributionSection

STATS_KEYS = [
    "mean",
    "median",
    "min",
    "max",
    "std",
    "q01",
    "q05",
    "q10",
    "q25",
    "q75",
    "q90",
    "q95",
    "q99",
]


@fixture
def dataframe() -> DataFrame:
    return DataFrame({"col": [np.nan] + list(range(101)) + [np.nan]})


###################################################
#     Tests for ContinuousDistributionSection     #
###################################################


def test_continuous_distribution_section_get_statistics(dataframe: DataFrame) -> None:
    output = ContinuousDistributionSection(df=dataframe, column="col")
    assert objects_are_allclose(
        output.get_statistics(),
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


def test_continuous_distribution_section_get_statistics_empty_row() -> None:
    output = ContinuousDistributionSection(df=DataFrame({"col": []}), column="col")
    stats = output.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 0
    assert stats["num_nulls"] == 0
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 0
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_continuous_distribution_section_get_statistics_empty_column() -> None:
    output = ContinuousDistributionSection(df=DataFrame({}), column="col")
    stats = output.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 0
    assert stats["num_nulls"] == 0
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 0
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_continuous_distribution_section_get_statistics_only_nans() -> None:
    output = ContinuousDistributionSection(
        df=DataFrame({"col": [np.nan, np.nan, np.nan, np.nan]}), column="col"
    )
    stats = output.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 4
    assert stats["num_nulls"] == 4
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 1
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_continuous_distribution_section_render_html_body(dataframe: DataFrame) -> None:
    output = ContinuousDistributionSection(df=dataframe, column="col")
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_body_args(dataframe: DataFrame) -> None:
    output = ContinuousDistributionSection(df=dataframe, column="col")
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_continuous_distribution_section_render_html_body_empty() -> None:
    output = ContinuousDistributionSection(df=DataFrame({"float": []}), column="col")
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_toc(dataframe: DataFrame) -> None:
    output = ContinuousDistributionSection(df=dataframe, column="col")
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_continuous_distribution_section_render_html_toc_args(dataframe: DataFrame) -> None:
    output = ContinuousDistributionSection(df=dataframe, column="col")
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
