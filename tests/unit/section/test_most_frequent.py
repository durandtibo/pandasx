from __future__ import annotations

from collections import Counter

from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import MostFrequentValuesSection
from flamme.section.most_frequent import (
    create_section_template,
    create_table,
    create_table_row,
)

###############################################
#     Tests for MostFrequentValuesSection     #
###############################################


def test_most_frequent_values_section_str() -> None:
    assert str(
        MostFrequentValuesSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    ).startswith("MostFrequentValuesSection(")


def test_most_frequent_values_section_get_statistics() -> None:
    section = MostFrequentValuesSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert objects_are_allclose(
        section.get_statistics(),
        {"most_common": [("c", 6), ("a", 4), ("b", 2)]},
    )


def test_most_frequent_values_section_get_statistics_top2() -> None:
    section = MostFrequentValuesSection(
        counter=Counter({"a": 4, "b": 2, "c": 6}), column="col", top=2
    )
    assert objects_are_allclose(section.get_statistics(), {"most_common": [("c", 6), ("a", 4)]})


def test_most_frequent_values_section_get_statistics_empty_column() -> None:
    section = MostFrequentValuesSection(counter=Counter({}), column="col")
    assert objects_are_allclose(section.get_statistics(), {"most_common": []})


def test_most_frequent_values_section_render_html_body() -> None:
    section = MostFrequentValuesSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_most_frequent_values_section_render_html_body_args() -> None:
    section = MostFrequentValuesSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_most_frequent_values_section_render_html_body_empty() -> None:
    section = MostFrequentValuesSection(counter=Counter({}), column="col")
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_most_frequent_values_section_render_html_toc() -> None:
    section = MostFrequentValuesSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_most_frequent_values_section_render_html_toc_args() -> None:
    section = MostFrequentValuesSection(counter=Counter({"a": 4, "b": 2, "c": 6}), column="col")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(create_table(Counter({"a": 4, "b": 2, "c": 6})), str)


def test_create_table_empty() -> None:
    assert isinstance(create_table(Counter({})), str)


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(create_table_row(Counter({"a": 4, "b": 2, "c": 6})), str)


def test_create_table_row_empty() -> None:
    assert isinstance(create_table_row(Counter({})), str)
