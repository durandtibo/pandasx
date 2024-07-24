from __future__ import annotations

from collections import Counter

import pytest
from coola import objects_are_allclose
from jinja2 import Template
from matplotlib import pyplot as plt

from flamme.section import ColumnDiscreteSection
from flamme.section.discrete import (
    create_histogram,
    create_histogram_section,
    create_section_template,
    create_table,
)

###########################################
#     Tests for ColumnDiscreteSection     #
###########################################


def test_column_discrete_section_str() -> None:
    assert str(
        ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    ).startswith("ColumnDiscreteSection(")


def test_column_discrete_section_figsize_default() -> None:
    assert (
        ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col").figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_discrete_section_figsize(figsize: tuple[float, float]) -> None:
    assert (
        ColumnDiscreteSection(
            counter=Counter({"a": 4, "b": 2, "c": 6}), column="col", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_discrete_section_yscale_default() -> None:
    assert (
        ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col").yscale
        == "auto"
    )


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_discrete_section_yscale(yscale: str) -> None:
    assert (
        ColumnDiscreteSection(
            counter=Counter({"a": 4, "b": 2, "c": 6}), column="col", yscale=yscale
        ).yscale
        == yscale
    )


def test_column_discrete_section_get_statistics() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "most_common": [("c", 6), ("a", 4), ("b", 2)],
            "null_values": 0,
            "nunique": 3,
            "total": 12,
        },
    )


def test_column_discrete_section_get_statistics_empty_row() -> None:
    section = ColumnDiscreteSection(counter=Counter({"a": 0, "b": 0, "c": 0}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {"most_common": [], "null_values": 0, "nunique": 0, "total": 0},
    )


def test_column_discrete_section_get_statistics_empty_column() -> None:
    section = ColumnDiscreteSection(counter=Counter({}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {"most_common": [], "null_values": 0, "nunique": 0, "total": 0},
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


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)


##############################################
#     Tests for create_histogram_section     #
##############################################


def test_create_histogram_section() -> None:
    assert isinstance(
        create_histogram_section(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col"), str
    )


def test_create_histogram_section_empty() -> None:
    assert isinstance(create_histogram_section(counter=Counter({}), column="col"), str)


@pytest.mark.parametrize("yscale", ["auto", "linear", "log"])
def test_create_histogram_section_yscale(yscale: str) -> None:
    assert isinstance(
        create_histogram_section(
            counter=Counter({"a": 4, "b": 2, "c": 6}), column="col", yscale=yscale
        ),
        str,
    )


######################################
#     Tests for create_histogram     #
######################################


def test_create_histogram() -> None:
    assert isinstance(
        create_histogram(column="col", names=["a", "b", "c"], counts=[5, 2, 100]), plt.Figure
    )


@pytest.mark.parametrize("yscale", ["auto", "linear", "log"])
def test_create_histogram_yscale(yscale: str) -> None:
    assert isinstance(
        create_histogram(column="col", names=["a", "b", "c"], counts=[5, 2, 100], yscale=yscale),
        plt.Figure,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_histogram_figsize(figsize: tuple[float, float]) -> None:
    assert isinstance(
        create_histogram(column="col", names=["a", "b", "c"], counts=[5, 2, 100], figsize=figsize),
        plt.Figure,
    )


def test_create_histogram_empty() -> None:
    assert create_histogram(column="col", names=[], counts=[]) is None


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(create_table(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col"), str)


def test_create_table_empty() -> None:
    assert isinstance(create_table(counter=Counter({}), column="col"), str)
