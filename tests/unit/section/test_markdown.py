from __future__ import annotations

from jinja2 import Template

from flamme.section import MarkdownSection
from flamme.section.markdown import create_section_template
from flamme.testing import markdown_available

#####################################
#     Tests for MarkdownSection     #
#####################################


@markdown_available
def test_markdown_section_str() -> None:
    assert str(MarkdownSection(desc="meow")).startswith("MarkdownSection(")


@markdown_available
def test_markdown_section_get_statistics() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert section.get_statistics() == {}


@markdown_available
def test_markdown_section_get_statistics_empty() -> None:
    section = MarkdownSection(desc="")
    assert section.get_statistics() == {}


@markdown_available
def test_markdown_section_render_html_body() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(Template(section.render_html_body()).render(), str)


@markdown_available
def test_markdown_section_render_html_body_args() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


@markdown_available
def test_markdown_section_render_html_body_empty() -> None:
    section = MarkdownSection(desc="")
    assert isinstance(Template(section.render_html_body()).render(), str)


@markdown_available
def test_markdown_section_render_html_toc() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(Template(section.render_html_toc()).render(), str)


@markdown_available
def test_markdown_section_render_html_toc_args() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)
