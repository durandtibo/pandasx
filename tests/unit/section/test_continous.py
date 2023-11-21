from __future__ import annotations

import math

import numpy as np
from coola import objects_are_allclose
from jinja2 import Template
from pandas import Series
from pytest import fixture, mark

from flamme.section import ContinuousDistributionSection
from flamme.section.continuous import create_histogram_figure, create_stats_table

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
def series() -> Series:
    return Series([np.nan] + list(range(101)) + [np.nan])


###################################################
#     Tests for ContinuousDistributionSection     #
###################################################


def test_continuous_distribution_section_series(series: Series) -> None:
    assert ContinuousDistributionSection(series=series, column="col").series.equals(series)


def test_continuous_distribution_section_column(series: Series) -> None:
    assert ContinuousDistributionSection(series=series, column="col").column == "col"


def test_continuous_distribution_section_nbins_default(series: Series) -> None:
    assert ContinuousDistributionSection(series=series, column="col").nbins is None


@mark.parametrize("nbins", (1, 2, 4))
def test_continuous_distribution_section_nbins(series: Series, nbins: int) -> None:
    assert ContinuousDistributionSection(series=series, column="col", nbins=nbins).nbins == nbins


def test_continuous_distribution_section_get_statistics(series: Series) -> None:
    section = ContinuousDistributionSection(series=series, column="col")
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


def test_continuous_distribution_section_get_statistics_empty_row() -> None:
    section = ContinuousDistributionSection(series=Series([]), column="col")
    stats = section.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 0
    assert stats["num_nulls"] == 0
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 0
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_continuous_distribution_section_get_statistics_only_nans() -> None:
    section = ContinuousDistributionSection(
        series=Series([np.nan, np.nan, np.nan, np.nan]), column="col"
    )
    stats = section.get_statistics()
    assert len(stats) == 17
    assert stats["count"] == 4
    assert stats["num_nulls"] == 4
    assert stats["num_non_nulls"] == 0
    assert stats["nunique"] == 1
    for key in STATS_KEYS:
        assert math.isnan(stats[key])


def test_continuous_distribution_section_render_html_body(series: Series) -> None:
    section = ContinuousDistributionSection(series=series, column="col")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_body_args(series: Series) -> None:
    section = ContinuousDistributionSection(series=series, column="col")
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_continuous_distribution_section_render_html_body_empty() -> None:
    section = ContinuousDistributionSection(series=Series([]), column="col")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_toc(series: Series) -> None:
    section = ContinuousDistributionSection(series=series, column="col")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_continuous_distribution_section_render_html_toc_args(series: Series) -> None:
    section = ContinuousDistributionSection(series=series, column="col")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


############################################
#    Tests for create_histogram_figure     #
############################################


def test_create_histogram_figure(series: Series) -> None:
    assert isinstance(create_histogram_figure(series=series, column="col"), str)


@mark.parametrize("nbins", (1, 2, 4))
def test_create_histogram_figure_nbins(series: Series, nbins: int) -> None:
    assert isinstance(create_histogram_figure(series=series, column="col", nbins=nbins), str)


#######################################
#    Tests for create_stats_table     #
#######################################


def test_create_stats_table() -> None:
    assert isinstance(
        create_stats_table(
            stats={
                "count": 103,
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
            column="col",
        ),
        str,
    )
