from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_allclose
from jinja2 import Template
from pandas import DataFrame
from pytest import raises

from flamme.section import NullValueSection, TemporalNullValueSection

######################################
#     Tests for NullValueSection     #
######################################


def test_null_value_section_incorrect_null_count_size() -> None:
    with raises(RuntimeError, match=r"columns \(3\) and null_count \(2\) do not match"):
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1]),
            total_count=np.array([5, 5, 5]),
        )


def test_null_value_section_incorrect_total_count_size() -> None:
    with raises(RuntimeError, match=r"columns \(3\) and total_count \(2\) do not match"):
        NullValueSection(
            columns=["col1", "col2", "col3"],
            null_count=np.array([0, 1, 2]),
            total_count=np.array([5, 5]),
        )


def test_null_value_section_get_statistics() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 1, 2),
            "total_count": (5, 5, 5),
        },
    )


def test_null_value_section_get_statistics_empty_row() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 0, 0]),
        total_count=np.array([0, 0, 0]),
    )
    assert objects_are_allclose(
        section.get_statistics(),
        {
            "columns": ("col1", "col2", "col3"),
            "null_count": (0, 0, 0),
            "total_count": (0, 0, 0),
        },
    )


def test_null_value_section_get_statistics_empty_column() -> None:
    section = NullValueSection(columns=[], null_count=np.array([]), total_count=np.array([]))
    assert objects_are_allclose(
        section.get_statistics(),
        {"columns": (), "null_count": (), "total_count": ()},
    )


def test_null_value_section_render_html_body() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_null_value_section_render_html_body_args() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_null_value_section_render_html_body_empty() -> None:
    section = NullValueSection(
        columns=[],
        null_count=np.array([]),
        total_count=np.array([]),
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_null_value_section_render_html_toc() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_null_value_section_render_html_toc_args() -> None:
    section = NullValueSection(
        columns=["col1", "col2", "col3"],
        null_count=np.array([0, 1, 2]),
        total_count=np.array([5, 5, 5]),
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


##############################################
#     Tests for TemporalNullValueSection     #
##############################################


def test_temporal_null_value_section_get_statistics() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_get_statistics_empty_row() -> None:
    section = TemporalNullValueSection(
        df=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_get_statistics_only_datetime_column() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_render_html_body() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_null_value_section_render_html_body_args() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_temporal_null_value_section_render_html_body_empty() -> None:
    section = TemporalNullValueSection(
        df=DataFrame({"float": [], "int": [], "str": [], "datetime": []}),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_null_value_section_render_html_toc() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_temporal_null_value_section_render_html_toc_args() -> None:
    section = TemporalNullValueSection(
        df=DataFrame(
            {
                "float": np.array([1.2, 4.2, np.nan, 2.2]),
                "int": np.array([np.nan, 1, 0, 1]),
                "str": np.array(["A", "B", None, np.nan]),
                "datetime": pd.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
            }
        ),
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )
