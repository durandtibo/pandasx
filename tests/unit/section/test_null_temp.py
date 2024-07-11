from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from coola import objects_are_allclose, objects_are_equal
from jinja2 import Template
from polars.testing import assert_frame_equal

from flamme.section import TemporalNullValueSection
from flamme.section.null_temp import (
    create_temporal_null_figure,
    create_temporal_null_table,
    prepare_data,
)


@pytest.fixture()
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [None, 1.0, 0.0, 1.0],
            "col2": [None, 1, 0, None],
            "datetime": [
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col1": pl.Float64,
            "col2": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


##############################################
#     Tests for TemporalNullValueSection     #
##############################################


def test_temporal_null_value_section_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalNullValueSection(
            frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
        )
    ).startswith("TemporalNullValueSection(")


def test_temporal_null_value_section_frame(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert_frame_equal(section.frame, dataframe)


def test_temporal_null_value_section_columns(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert section.columns == ("col1", "col2")


@pytest.mark.parametrize("dt_column", ["datetime", "date"])
def test_temporal_null_value_section_dt_column(dt_column: str) -> None:
    section = TemporalNullValueSection(
        frame=pl.DataFrame(
            {
                "col1": np.array([1.2, 4.2, np.nan, 2.2]),
                "col2": np.array([np.nan, 1, np.nan, 1]),
                "datetime": pl.to_datetime(
                    ["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]
                ),
                "date": pl.to_datetime(["2021-01-03", "2021-02-03", "2021-03-03", "2021-04-03"]),
            }
        ),
        columns=["col1", "col2"],
        dt_column=dt_column,
        period="M",
    )
    assert section.dt_column == dt_column


@pytest.mark.parametrize("period", ["M", "D"])
def test_temporal_null_value_section_period(dataframe: pl.DataFrame, period: str) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period=period
    )
    assert section.period == period


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_temporal_null_value_section_figsize(
    dataframe: pl.DataFrame, figsize: tuple[int, int]
) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M", figsize=figsize
    )
    assert section.figsize == figsize


def test_temporal_null_value_section_figsize_default(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert section.figsize is None


def test_temporal_null_value_section_missing_dt_column(dataframe: pl.DataFrame) -> None:
    with pytest.raises(
        ValueError, match=r"Datetime column my_datetime is not in the DataFrame \(columns:"
    ):
        TemporalNullValueSection(
            frame=dataframe, columns=["col1", "col2"], dt_column="my_datetime", period="M"
        )


def test_temporal_null_value_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_get_statistics_empty_row() -> None:
    section = TemporalNullValueSection(
        frame=pl.DataFrame({"col1": [], "col2": [], "datetime": []}),
        columns=["col1", "col2"],
        dt_column="datetime",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_temporal_null_value_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_null_value_section_render_html_body_args(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_temporal_null_value_section_render_html_body_empty_rows() -> None:
    section = TemporalNullValueSection(
        frame=pl.DataFrame({"col1": [], "col2": [], "datetime": []}),
        columns=["col1", "col2"],
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_temporal_null_value_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_temporal_null_value_section_render_html_toc_args(dataframe: pl.DataFrame) -> None:
    section = TemporalNullValueSection(
        frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


################################################
#    Tests for create_temporal_null_figure     #
################################################


def test_create_temporal_null_figure(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_null_figure(
            frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
        ),
        str,
    )


def test_create_temporal_null_figure_empty() -> None:
    assert isinstance(
        create_temporal_null_figure(
            frame=pl.DataFrame({"col1": [], "col2": [], "datetime": pl.to_datetime([])}),
            columns=["col1", "col2"],
            dt_column="datetime",
            period="M",
        ),
        str,
    )


###############################################
#    Tests for create_temporal_null_table     #
###############################################


def test_create_temporal_null_table(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_null_table(
            frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"
        ),
        str,
    )


def test_create_temporal_null_table_empty() -> None:
    assert isinstance(
        create_temporal_null_table(
            frame=pl.DataFrame({"col1": [], "col2": [], "datetime": pl.to_datetime([])}),
            columns=["col1", "col2"],
            dt_column="datetime",
            period="M",
        ),
        str,
    )


#################################
#    Tests for prepare_data     #
#################################


def test_prepare_data(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        prepare_data(frame=dataframe, columns=["col1", "col2"], dt_column="datetime", period="M"),
        (
            np.array([1, 0, 2, 0]),
            np.array([2, 2, 2, 2]),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_subset(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        prepare_data(frame=dataframe, columns=["col1"], dt_column="datetime", period="M"),
        (
            np.array([0, 0, 1, 0]),
            np.array([1, 1, 1, 1]),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_empty_columns(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        prepare_data(frame=dataframe, columns=[], dt_column="datetime", period="M"),
        (
            np.array([0, 0, 0, 0]),
            np.array([0, 0, 0, 0]),
            ["2020-01", "2020-02", "2020-03", "2020-04"],
        ),
    )


def test_prepare_data_empty() -> None:
    assert objects_are_equal(
        prepare_data(
            frame=pl.DataFrame(
                {"col1": [], "col2": [], "datetime": []},
                schema={
                    "col1": pl.Float64,
                    "col2": pl.Int64,
                    "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                },
            ),
            columns=["col1", "col2"],
            dt_column="datetime",
            period="M",
        ),
        (np.array([], dtype=int), np.array([], dtype=int), []),
    )
