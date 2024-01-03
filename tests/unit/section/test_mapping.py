from __future__ import annotations

from unittest.mock import Mock

import pytest
from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import BaseSection, SectionDict


@pytest.fixture()
def sections() -> dict:
    return {
        "metric1": Mock(
            spec=BaseSection,
            render_html_body=Mock(return_value="<li>body metric 1</li>"),
            render_html_toc=Mock(return_value="<li>toc metric 1</li>"),
        ),
        "metric2": Mock(
            spec=BaseSection,
            render_html_body=Mock(return_value="<li>body metric 2</li>"),
            render_html_toc=Mock(return_value="<li>toc metric 2</li>"),
        ),
        "metric": SectionDict(
            {
                "metric1": Mock(
                    spec=BaseSection,
                    render_html_body=Mock(return_value="<li>body metric 1</li>"),
                    render_html_toc=Mock(return_value="<li>toc metric 1</li>"),
                ),
                "metric2": Mock(
                    spec=BaseSection,
                    render_html_body=Mock(return_value="<li>body metric 2</li>"),
                    render_html_toc=Mock(return_value="<li>toc metric 2</li>"),
                ),
                "metric": SectionDict(
                    {
                        "metric1": Mock(
                            spec=BaseSection,
                            render_html_body=Mock(return_value="<li>body metric 1</li>"),
                            render_html_toc=Mock(return_value="<li>toc metric 1</li>"),
                        ),
                        "metric2": Mock(
                            spec=BaseSection,
                            render_html_body=Mock(return_value="<li>body metric 2</li>"),
                            render_html_toc=Mock(return_value="<li>toc metric 2</li>"),
                        ),
                    }
                ),
            }
        ),
    }


#################################
#     Tests for SectionDict     #
#################################


def test_capacity_distribution_section_sections() -> None:
    sections = SectionDict(
        {
            "metric1": Mock(spec=BaseSection, get_statistics=Mock(return_value={"acc": 42})),
            "metric2": Mock(spec=BaseSection, get_statistics=Mock(return_value={"ap": 12})),
        }
    ).sections
    assert isinstance(sections, dict)
    assert len(sections) == 2
    assert isinstance(sections["metric1"], BaseSection)
    assert isinstance(sections["metric2"], BaseSection)


@pytest.mark.parametrize("max_toc_depth", [0, 1, 2])
def test_capacity_distribution_section_max_toc_depth(max_toc_depth: int) -> None:
    assert SectionDict(sections={}, max_toc_depth=max_toc_depth).max_toc_depth == max_toc_depth


def test_capacity_distribution_section_get_statistics() -> None:
    section = SectionDict(
        {
            "metric1": Mock(spec=BaseSection, get_statistics=Mock(return_value={"acc": 42})),
            "metric2": Mock(spec=BaseSection, get_statistics=Mock(return_value={"ap": 12})),
        }
    )
    assert objects_are_allclose(
        section.get_statistics(), {"metric1": {"acc": 42}, "metric2": {"ap": 12}}
    )


def test_capacity_distribution_section_render_html_body() -> None:
    section = SectionDict(
        {
            "metric1": Mock(
                spec=BaseSection, render_html_body=Mock(return_value="<li>metric 1</li>")
            ),
            "metric2": Mock(
                spec=BaseSection, render_html_body=Mock(return_value="<li>metric 2</li>")
            ),
        }
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


@pytest.mark.parametrize("max_toc_depth", [0, 1, 2])
def test_capacity_distribution_section_render_html_body_args(
    max_toc_depth: int, sections: dict
) -> None:
    section = SectionDict(
        sections,
        max_toc_depth=max_toc_depth,
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_capacity_distribution_section_render_html_toc(sections: dict) -> None:
    section = SectionDict(sections)
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_capacity_distribution_section_render_html_toc_args(sections: dict) -> None:
    section = SectionDict(sections)
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_capacity_distribution_section_render_html_toc_too_deep(sections: dict) -> None:
    section = SectionDict(sections)
    assert isinstance(Template(section.render_html_toc(max_depth=2, depth=2)).render(), str)


def test_capacity_distribution_section_render_html_toc_empty() -> None:
    section = SectionDict({})
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_capacity_distribution_section_render_html_toc_empty_toc() -> None:
    section = SectionDict(
        {
            "metric1": Mock(spec=BaseSection, render_html_toc=Mock(return_value="")),
            "metric2": Mock(spec=BaseSection, render_html_toc=Mock(return_value="")),
        }
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)
