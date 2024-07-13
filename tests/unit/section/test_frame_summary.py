from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose
from jinja2 import Template

from flamme.section import DataFrameSummarySection
from flamme.section.frame_summary import create_table, create_table_row


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2, 1, 2.2],
            "int": [1, 1, 0, 1, 1, 1],
            "str": ["A", "B", None, None, "C", "B"],
        },
        schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
    )


#############################################
#     Tests for DataFrameSummarySection     #
#############################################


def test_dataframe_summary_section_str(dataframe: pl.DataFrame) -> None:
    assert str(DataFrameSummarySection(dataframe)).startswith("DataFrameSummarySection(")


def test_dataframe_summary_section_frame(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummarySection(dataframe).frame is dataframe


@pytest.mark.parametrize("top", [0, 1, 2])
def test_dataframe_summary_section_top(dataframe: pl.DataFrame, top: int) -> None:
    assert DataFrameSummarySection(dataframe, top=top).top == top


def test_dataframe_summary_section_top_incorrect(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"Incorrect top value \(-1\). top must be positive"):
        DataFrameSummarySection(dataframe, top=-1)


def test_dataframe_summary_section_get_columns(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummarySection(dataframe).get_columns() == ("float", "int", "str")


def test_dataframe_summary_section_get_columns_empty() -> None:
    assert DataFrameSummarySection(pl.DataFrame({})).get_columns() == ()


def test_dataframe_summary_section_get_null_count(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummarySection(dataframe).get_null_count() == (1, 0, 2)


def test_dataframe_summary_section_get_null_count_empty() -> None:
    assert DataFrameSummarySection(pl.DataFrame({})).get_null_count() == ()


def test_dataframe_summary_section_get_nunique(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummarySection(dataframe).get_nunique() == (5, 2, 4)


def test_dataframe_summary_section_get_nunique_empty() -> None:
    assert DataFrameSummarySection(pl.DataFrame({})).get_nunique() == ()


def test_dataframe_summary_section_get_dtypes(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummarySection(dataframe).get_dtypes() == (
        pl.Float64(),
        pl.Int64(),
        pl.String(),
    )


def test_dataframe_summary_section_get_dtypes_empty() -> None:
    assert DataFrameSummarySection(pl.DataFrame({})).get_dtypes() == ()


def test_dataframe_summary_section_get_most_frequent_values(dataframe: pl.DataFrame) -> None:
    assert objects_are_allclose(
        DataFrameSummarySection(dataframe).get_most_frequent_values(),
        (
            ((2.2, 2), (1.2, 1), (4.2, 1), (None, 1), (1.0, 1)),
            ((1, 5), (0, 1)),
            (("B", 2), (None, 2), ("A", 1), ("C", 1)),
        ),
    )


def test_dataframe_summary_section_get_most_frequent_values_empty() -> None:
    assert DataFrameSummarySection(pl.DataFrame({})).get_most_frequent_values() == ()


def test_dataframe_summary_section_get_statistics(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummarySection(dataframe).get_statistics() == {
        "columns": ("float", "int", "str"),
        "dtypes": (pl.Float64(), pl.Int64(), pl.String()),
        "null_count": (1, 0, 2),
        "nunique": (5, 2, 4),
    }


def test_dataframe_summary_section_get_statistics_empty_rows() -> None:
    assert DataFrameSummarySection(
        pl.DataFrame(
            {"float": [], "int": [], "str": []},
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    ).get_statistics() == {
        "columns": ("float", "int", "str"),
        "dtypes": (pl.Float64(), pl.Int64(), pl.String()),
        "null_count": (0, 0, 0),
        "nunique": (0, 0, 0),
    }


def test_dataframe_summary_section_get_statistics_empty() -> None:
    assert DataFrameSummarySection(pl.DataFrame({})).get_statistics() == {
        "columns": (),
        "dtypes": (),
        "null_count": (),
        "nunique": (),
    }


def test_column_temporal_null_value_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = DataFrameSummarySection(dataframe)
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_body_args(dataframe: pl.DataFrame) -> None:
    section = DataFrameSummarySection(dataframe)
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_null_value_section_render_html_body_empty_rows() -> None:
    section = DataFrameSummarySection(
        pl.DataFrame(
            {"float": [], "int": [], "str": []},
            schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        )
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_body_empty() -> None:
    section = DataFrameSummarySection(pl.DataFrame({}))
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = DataFrameSummarySection(dataframe)
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_null_value_section_render_html_toc_args(dataframe: pl.DataFrame) -> None:
    section = DataFrameSummarySection(dataframe)
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(
        create_table(
            columns=["float", "int", "str"],
            null_count=(1, 0, 2),
            nunique=(5, 2, 4),
            dtypes=(pl.Float64(), pl.Int64(), pl.String()),
            most_frequent_values=(
                ((2.2, 2), (1.2, 1), (4.2, 1), (None, 1), (1.0, 1)),
                ((1, 5), (0, 1)),
                (("B", 2), (None, 2), ("A", 1), ("C", 1)),
            ),
            total=42,
        ),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(
        create_table(
            columns=[],
            null_count=[],
            nunique=[],
            dtypes=[],
            most_frequent_values=[],
            total=0,
        ),
        str,
    )


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(
        create_table_row(
            column="col",
            null=5,
            nunique=42,
            dtype=pl.Float64(),
            most_frequent_values=[("C", 12), ("A", 5), ("B", 4)],
            total=100,
        ),
        str,
    )


def test_create_table_row_empty() -> None:
    assert isinstance(
        create_table_row(
            column="col",
            null=0,
            nunique=0,
            dtype=pl.Float64(),
            most_frequent_values=[],
            total=0,
        ),
        str,
    )
