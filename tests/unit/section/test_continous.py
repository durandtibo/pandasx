from __future__ import annotations

from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import ContinuousDistributionSection

###################################################
#     Tests for ContinuousDistributionSection     #
###################################################


def test_continuous_distribution_section_get_statistics() -> None:
    output = ContinuousDistributionSection(column="col")
    assert objects_are_allclose(output.get_statistics(), {})


def test_continuous_distribution_section_get_statistics_empty_row() -> None:
    output = ContinuousDistributionSection(column="col")
    assert objects_are_allclose(output.get_statistics(), {})


def test_continuous_distribution_section_get_statistics_empty_column() -> None:
    output = ContinuousDistributionSection(column="col")
    assert objects_are_allclose(output.get_statistics(), {})


def test_continuous_distribution_section_render_html_body() -> None:
    output = ContinuousDistributionSection(column="col")
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_body_args() -> None:
    output = ContinuousDistributionSection(column="col")
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_continuous_distribution_section_render_html_body_empty() -> None:
    output = ContinuousDistributionSection(column="col")
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_continuous_distribution_section_render_html_toc() -> None:
    output = ContinuousDistributionSection(column="col")
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_continuous_distribution_section_render_html_toc_args() -> None:
    output = ContinuousDistributionSection(column="col")
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
