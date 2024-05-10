from __future__ import annotations

from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import ContentSection

####################################
#     Tests for ContentSection     #
####################################


def test_content_section_str() -> None:
    assert str(ContentSection(content="meow")).startswith("ContentSection(")


def test_content_section_get_statistics() -> None:
    section = ContentSection(content="meow")
    assert objects_are_allclose(
        section.get_statistics(),
        {},
    )


def test_content_section_render_html_body() -> None:
    section = ContentSection(content="meow")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_content_section_render_html_body_args() -> None:
    section = ContentSection(content="meow")
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_content_section_render_html_toc() -> None:
    section = ContentSection(content="meow")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_content_section_render_html_toc_args() -> None:
    section = ContentSection(content="meow")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
