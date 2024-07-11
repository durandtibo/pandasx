from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from polars.testing import assert_frame_equal

from flamme.section import TemporalRowCountSection
from flamme.section.count_rows import (
    create_temporal_count_figure,
    create_temporal_count_table,
    prepare_data,
)


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ]
        },
        schema={"datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


#############################################
#     Tests for TemporalRowCountSection     #
#############################################


def test_column_temporal_row_count_section_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="1mo")
    ).startswith("TemporalRowCountSection(")


def test_column_temporal_row_count_section_frame(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="1mo")
    assert_frame_equal(section.frame, dataframe)


@pytest.mark.parametrize("dt_column", ["datetime", "date"])
def test_column_temporal_row_count_section_dt_column(
    dt_column: str, dataframe: pl.DataFrame
) -> None:
    section = TemporalRowCountSection(
        frame=dataframe.with_columns(pl.col("datetime").alias("date")),
        dt_column=dt_column,
        period="1mo",
    )
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["1mo", "1d"])
def test_column_temporal_row_count_section_period(dataframe: pl.DataFrame, period: str) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period=period,
    )
    assert section.period == period


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_row_count_section_figsize(
    dataframe: pl.DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="1mo",
        figsize=figsize,
    )
    assert section.figsize == figsize


def test_column_temporal_row_count_section_figsize_default(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="1mo",
    )
    assert section.figsize is None


def test_column_temporal_row_count_section_missing_dt_column(dataframe: pl.DataFrame) -> None:
    with pytest.raises(
        ValueError, match=r"Datetime column my_datetime is not in the DataFrame \(columns:"
    ):
        TemporalRowCountSection(
            frame=dataframe,
            dt_column="my_datetime",
            period="1mo",
        )


def test_column_temporal_row_count_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_row_count_section_get_statistics_empty_row() -> None:
    section = TemporalRowCountSection(
        frame=pl.DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_temporal_row_count_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_row_count_section_render_html_body_args(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(
        frame=dataframe,
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_row_count_section_render_html_body_empty_rows() -> None:
    section = TemporalRowCountSection(
        frame=pl.DataFrame({"col1": [], "col2": [], "datetime": []}),
        dt_column="datetime",
        period="1mo",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_row_count_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="1mo")
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_row_count_section_render_html_toc_args(dataframe: pl.DataFrame) -> None:
    section = TemporalRowCountSection(frame=dataframe, dt_column="datetime", period="1mo")
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


################################################
#    Tests for create_temporal_count_table     #
################################################


def test_create_temporal_count_table(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_count_table(
            frame=dataframe,
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


def test_create_temporal_count_table_empty() -> None:
    assert isinstance(
        create_temporal_count_table(
            frame=pl.DataFrame({"col1": [], "col2": [], "datetime": pl.to_datetime([])}),
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


#################################################
#    Tests for create_temporal_count_figure     #
#################################################


def test_create_temporal_count_figure(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_count_figure(
            frame=dataframe,
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


def test_create_temporal_count_figure_empty() -> None:
    assert isinstance(
        create_temporal_count_figure(
            frame=pl.DataFrame({"datetime": []}), dt_column="datetime", period="1mo"
        ),
        str,
    )


#################################
#    Tests for prepare_data     #
#################################


def test_prepare_data(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        prepare_data(
            frame=dataframe,
            dt_column="datetime",
            period="1mo",
        ),
        (
            [3, 1, 1, 1],
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_empty() -> None:
    assert objects_are_equal(
        prepare_data(frame=pl.DataFrame({"datetime": []}), dt_column="datetime", period="1mo"),
        ([], []),
    )
