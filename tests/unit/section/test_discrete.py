from __future__ import annotations

from collections import Counter

from coola import objects_are_allclose
from jinja2 import Template
from pytest import mark

from flamme.section import ColumnDiscreteSection

###########################################
#     Tests for ColumnDiscreteSection     #
###########################################


def test_column_discrete_section_figsize_default() -> None:
    assert (
        ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col").figsize
        is None
    )


@mark.parametrize("figsize", ((7, 3), (1.5, 1.5)))
def test_column_discrete_section_figsize(figsize: tuple[float, float]) -> None:
    assert (
        ColumnDiscreteSection(
            counter=Counter({"a": 4, "b": 2, "c": 6}), column="col", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_discrete_section_get_statistics() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "most_common": [("c", 6), ("a", 4), ("b", 2)],
            "nunique": 3,
            "total": 12,
        },
    )


def test_column_discrete_section_get_statistics_empty_row() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 0, "b": 0, "c": 0}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {"most_common": [], "nunique": 0, "total": 0},
    )


def test_column_discrete_section_get_statistics_empty_column() -> None:
    section = ColumnDiscreteSection(counter=Counter({}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {"most_common": [], "nunique": 0, "total": 0},
    )


def test_column_discrete_section_render_html_body() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_discrete_section_render_html_body_args() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_discrete_section_render_html_body_empty() -> None:
    section = ColumnDiscreteSection(counter=Counter({}), column="col")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_discrete_section_render_html_toc() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_discrete_section_render_html_toc_args() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
