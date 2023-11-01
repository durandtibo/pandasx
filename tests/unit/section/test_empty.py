from __future__ import annotations

from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import EmptySection

##################################
#     Tests for EmptySection     #
##################################


def test_column_type_section_get_statistics() -> None:
    output = EmptySection()
    assert objects_are_allclose(output.get_statistics(), {})


def test_column_type_section_render_html_body() -> None:
    output = EmptySection()
    assert isinstance(Template(output.render_html_body()).render(), str)


def test_column_type_section_render_html_body_args() -> None:
    output = EmptySection()
    assert isinstance(
        Template(output.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_type_section_render_html_toc() -> None:
    output = EmptySection()
    assert isinstance(Template(output.render_html_toc()).render(), str)


def test_column_type_section_render_html_toc_args() -> None:
    output = EmptySection()
    assert isinstance(
        Template(output.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
