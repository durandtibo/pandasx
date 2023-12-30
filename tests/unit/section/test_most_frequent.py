from __future__ import annotations

from collections import Counter

from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import MostFrequentValuesSection

###############################################
#     Tests for MostFrequentValuesSection     #
###############################################


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
