from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_equal
from jinja2 import Template
from pandas import DataFrame
from pytest import fixture

from flamme.section import TemporalDiscreteDistributionSection


@fixture
def dataframe() -> DataFrame:
    return DataFrame(
        {
            "col": np.array([1, 42, np.nan, 22]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


#########################################################
#     Tests for TemporalDiscreteDistributionSection     #
#########################################################


def test_temporal_discrete_distribution_section_column(dataframe: DataFrame) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.column == "col"


def test_temporal_discrete_distribution_section_dt_column(dataframe: DataFrame) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.dt_column == "datetime"


def test_temporal_discrete_distribution_section_log_y_default(dataframe: DataFrame) -> None:
    assert not TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    ).log_y


def test_temporal_discrete_distribution_section_log_y(dataframe: DataFrame) -> None:
    assert TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
        log_y=True,
    ).log_y


def test_temporal_discrete_distribution_section_period(dataframe: DataFrame) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.period == "M"


def test_temporal_discrete_distribution_section_get_statistics(dataframe: DataFrame) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_discrete_distribution_section_get_statistics_empty_row() -> None:
    section = TemporalDiscreteDistributionSection(
        df=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_discrete_distribution_section_get_statistics_empty_column() -> None:
    section = TemporalDiscreteDistributionSection(
        df=DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_discrete_distribution_section_render_html_body(dataframe: DataFrame) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_discrete_distribution_section_render_html_body_empty_row() -> None:
    section = TemporalDiscreteDistributionSection(
        df=DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_discrete_distribution_section_render_html_body_empty_column() -> None:
    section = TemporalDiscreteDistributionSection(
        df=DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_discrete_distribution_section_render_html_body_args(
    dataframe: DataFrame,
) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_temporal_discrete_distribution_section_render_html_toc(dataframe: DataFrame) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_temporal_discrete_distribution_section_render_html_toc_args(
    dataframe: DataFrame,
) -> None:
    section = TemporalDiscreteDistributionSection(
        df=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
