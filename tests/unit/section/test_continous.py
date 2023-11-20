from __future__ import annotations

import numpy as np
from coola import objects_are_allclose
from jinja2 import Template
from pandas import DataFrame

from flamme.section import ContinuousDistributionSection

###################################################
#     Tests for ContinuousDistributionSection     #
###################################################


def test_continuous_distribution_section_get_statistics() -> None:
    output = ContinuousDistributionSection(
        df=DataFrame({"col": np.array([1.2, 4.2, np.nan, 2.2])}),
        column="col",
    )
    assert objects_are_allclose(output.get_statistics(), {})


def test_continuous_distribution_section_get_statistics_empty_row() -> None:
    output = ContinuousDistributionSection(df=DataFrame({"col": []}), column="col")
    assert objects_are_allclose(output.get_statistics(), {})


def test_continuous_distribution_section_get_statistics_empty_column() -> None:
    output = ContinuousDistributionSection(df=DataFrame({}), column="col")
    assert objects_are_allclose(output.get_statistics(), {})


def test_continuous_distribution_section_render_html_body() -> None:
    output = ContinuousDistributionSection(
        df=DataFrame({"col": np.array([1.2, 4.2, np.nan, 2.2])}), column="col"
    )
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_body_args() -> None:
    output = ContinuousDistributionSection(
        df=DataFrame({"col": np.array([1.2, 4.2, np.nan, 2.2])}), column="col"
    )
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_continuous_distribution_section_render_html_body_empty() -> None:
    output = ContinuousDistributionSection(df=DataFrame({"float": []}), column="col")
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_toc() -> None:
    output = ContinuousDistributionSection(
        df=DataFrame({"col": np.array([1.2, 4.2, np.nan, 2.2])}), column="col"
    )
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_continuous_distribution_section_render_html_toc_args() -> None:
    output = ContinuousDistributionSection(
        df=DataFrame({"col": np.array([1.2, 4.2, np.nan, 2.2])}), column="col"
    )
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
