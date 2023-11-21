from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_equal
from jinja2 import Template
from pandas import DataFrame

from flamme.section import TemporalContinuousDistributionSection

###########################################################
#     Tests for TemporalContinuousDistributionSection     #
###########################################################


def test_temporal_continuous_distribution_section_get_statistics() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_continuous_distribution_section_get_statistics_empty_row() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame({"float": [], "datetime": []}),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_continuous_distribution_section_get_statistics_empty_column() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame({}),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_temporal_continuous_distribution_section_render_html_body() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_continuous_distribution_section_render_html_body_empty_row() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame({"float": [], "datetime": []}),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_continuous_distribution_section_render_html_body_empty_column() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame({}),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_continuous_distribution_section_render_html_body_args() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_temporal_continuous_distribution_section_render_html_toc() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_temporal_continuous_distribution_section_render_html_toc_args() -> None:
    section = TemporalContinuousDistributionSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        column="float",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
