from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from polars.testing import assert_frame_equal

from flamme.section import ColumnTemporalNullValueSection
from flamme.section.null_temp_col import (
    add_column_to_figure,
    create_section_template,
    create_table_section,
    create_temporal_null_figure,
    create_temporal_null_figures,
    create_temporal_null_table,
    create_temporal_null_table_row,
    split_figures_by_column,
)


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2],
            "int": [None, 1, 0, 1],
            "str": ["A", "B", None, None],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "float": pl.Float64,
            "int": pl.Int64,
            "str": pl.String,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


@pytest.fixture()
def dataframe_empty() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [],
            "int": [],
            "str": [],
            "datetime": [],
        },
        schema={
            "float": pl.Float64,
            "int": pl.Int64,
            "str": pl.String,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


####################################################
#     Tests for ColumnTemporalNullValueSection     #
####################################################


def test_column_temporal_null_value_section_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnTemporalNullValueSection(
            frame=dataframe,
            columns=["float", "int", "str"],
            dt_column="datetime",
            period="1mo",
        )
    ).startswith("ColumnTemporalNullValueSection(")


def test_column_temporal_null_value_section_frame(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert_frame_equal(section.frame, dataframe)


def test_column_temporal_null_value_section_columns(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert section.columns == ("float", "int", "str")


@pytest.mark.parametrize("dt_column", ["datetime", "str"])
def test_column_temporal_null_value_section_dt_column(
    dataframe: pl.DataFrame, dt_column: str
) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column=dt_column,
        period="1mo",
    )
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["1mo", "D"])
def test_column_temporal_null_value_section_period(dataframe: pl.DataFrame, period: str) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@pytest.mark.parametrize("ncols", [1, 2])
def test_column_temporal_null_value_section_ncols(dataframe: pl.DataFrame, ncols: int) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
        ncols=ncols,
    )
    assert section.ncols == ncols


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_null_value_section_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
        figsize=figsize,
    )
    assert section.figsize == figsize


def test_column_temporal_null_value_section_figsize_default(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert section.figsize == (7, 5)


def test_column_temporal_null_value_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_get_statistics_empty_row(
    dataframe_empty: pl.DataFrame,
) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe_empty,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_get_statistics_only_datetime_column() -> None:
    section = ColumnTemporalNullValueSection(
        frame=pl.DataFrame(
            {
                "datetime": [
                    datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                    datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                ],
            },
            schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        ),
        columns=[],
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_null_value_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_body_args(
    dataframe: pl.DataFrame,
) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_null_value_section_render_html_body_empty(
    dataframe_empty: pl.DataFrame,
) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe_empty,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_null_value_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_null_value_section_render_html_toc_args(
    dataframe: pl.DataFrame,
) -> None:
    section = ColumnTemporalNullValueSection(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)


#################################################
#     Tests for create_temporal_null_figure     #
#################################################


def test_create_temporal_null_figure(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_null_figure(
            frame=dataframe,
            columns=["float", "int", "str"],
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


@pytest.mark.parametrize("ncols", [1, 2])
def test_create_temporal_null_figure_ncols(dataframe: pl.DataFrame, ncols: int) -> None:
    assert isinstance(
        create_temporal_null_figure(
            frame=dataframe,
            columns=["float", "int", "str"],
            dt_column="datetime",
            period="1mo",
            ncols=ncols,
        ),
        str,
    )


##################################################
#     Tests for create_temporal_null_figures     #
##################################################


def test_create_temporal_null_figures(dataframe: pl.DataFrame) -> None:
    figures = create_temporal_null_figures(
        frame=dataframe,
        columns=["float", "int", "str"],
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(figures, list)
    assert len(figures) == 3


def test_create_temporal_null_figures_subset(dataframe: pl.DataFrame) -> None:
    figures = create_temporal_null_figures(
        frame=dataframe,
        columns=["float", "int"],
        dt_column="datetime",
        period="1w",
    )
    assert isinstance(figures, list)
    assert len(figures) == 2


def test_create_temporal_null_figures_empty() -> None:
    assert (
        create_temporal_null_figures(
            frame=pl.DataFrame({}),
            columns=[],
            dt_column="datetime",
            period="1mo",
        )
        == []
    )


def test_create_temporal_null_figures_empty_rows(dataframe_empty: pl.DataFrame) -> None:
    assert (
        create_temporal_null_figures(
            frame=dataframe_empty,
            columns=["float", "int", "str"],
            dt_column="datetime",
            period="1mo",
        )
        == []
    )


##########################################
#     Tests for add_column_to_figure     #
##########################################


def test_add_column_to_figure() -> None:
    assert objects_are_equal(
        add_column_to_figure(columns=["col1", "col2"], figures=["fig1", "fig2"]),
        [
            '<div style="text-align:center">(0) col1\nfig1</div>',
            '<div style="text-align:center">(1) col2\nfig2</div>',
        ],
    )


def test_add_column_to_figure_empty() -> None:
    assert objects_are_equal(add_column_to_figure(columns=[], figures=[]), [])


def test_add_column_to_figure_incorrect_sizes() -> None:
    with pytest.raises(
        RuntimeError, match="The number of column names is different from the number of figures:"
    ):
        add_column_to_figure(columns=["col1", "col2"], figures=["fig1", "fig2", "fig3"])


############################################
#    Tests for split_figures_by_column     #
############################################


def test_split_figures_by_column_ncols_1() -> None:
    assert objects_are_equal(
        split_figures_by_column(figures=["fig1", "fig2", "fig3"], ncols=1),
        ['<div class="col">\n  fig1\n  <hr>\n  fig2\n  <hr>\n  fig3\n</div>'],
    )


def test_split_figures_by_column_ncols_2() -> None:
    assert objects_are_equal(
        split_figures_by_column(figures=["fig1", "fig2", "fig3"], ncols=2),
        ['<div class="col">\n  fig1\n  <hr>\n  fig3\n</div>', '<div class="col">\n  fig2\n</div>'],
    )


def test_split_figures_by_column_ncols_3() -> None:
    assert objects_are_equal(
        split_figures_by_column(figures=["fig1", "fig2", "fig3"], ncols=3),
        [
            '<div class="col">\n  fig1\n</div>',
            '<div class="col">\n  fig2\n</div>',
            '<div class="col">\n  fig3\n</div>',
        ],
    )


def test_split_figures_by_column_empty_ncols_1() -> None:
    assert objects_are_equal(
        split_figures_by_column(figures=[], ncols=1), ['<div class="col">\n  \n</div>']
    )


def test_split_figures_by_column_empty_ncols_2() -> None:
    assert objects_are_equal(
        split_figures_by_column(figures=[], ncols=2),
        [
            '<div class="col">\n  \n</div>',
            '<div class="col">\n  \n</div>',
        ],
    )


#########################################
#    Tests for create_table_section     #
#########################################


def test_create_table_section(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_table_section(
            frame=dataframe,
            columns=["float"],
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


def test_create_table_section_empty(dataframe_empty: pl.DataFrame) -> None:
    assert (
        create_table_section(
            frame=dataframe_empty,
            columns=["float"],
            dt_column="datetime",
            period="1mo",
        )
        == ""
    )


###############################################
#    Tests for create_temporal_null_table     #
###############################################


def test_create_temporal_null_table(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_null_table(
            frame=dataframe,
            column="float",
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


def test_create_temporal_null_table_empty(dataframe_empty: pl.DataFrame) -> None:
    assert (
        create_temporal_null_table(
            frame=dataframe_empty,
            column="float",
            dt_column="datetime",
            period="1mo",
        )
        == ""
    )


###################################################
#    Tests for create_temporal_null_table_row     #
###################################################


def test_create_temporal_null_table_row() -> None:
    assert isinstance(create_temporal_null_table_row(label="meow", null=5, total=42), str)
