from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
import pytest
from coola import objects_are_equal
from jinja2 import Template

from flamme.section import ColumnTemporalContinuousSection
from flamme.section.continuous_temp import (
    create_temporal_figure,
    create_temporal_table,
    create_temporal_table_row,
)


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "col": np.array([1.2, 4.2, np.nan, 2.2]),
            "datetime": pd.to_datetime(["2020-01-03", "2020-02-03", "2020-03-03", "2020-04-03"]),
        }
    )


#####################################################
#     Tests for ColumnTemporalContinuousSection     #
#####################################################


def test_column_temporal_continuous_section_str(dataframe: pd.DataFrame) -> None:
    assert str(
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        )
    ).startswith("ColumnTemporalContinuousSection(")


def test_column_temporal_continuous_section_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.column == "col"


def test_column_temporal_continuous_section_dt_column(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.dt_column == "datetime"


def test_column_temporal_continuous_section_yscale_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ).yscale
        == "auto"
    )


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_column_temporal_continuous_section_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            yscale=yscale,
        ).yscale
        == yscale
    )


def test_column_temporal_continuous_section_period(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert section.period == "M"


def test_column_temporal_continuous_section_figsize_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ).figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_temporal_continuous_section_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnTemporalContinuousSection(
            frame=dataframe, column="col", dt_column="datetime", period="M", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_temporal_continuous_section_get_statistics(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_section_get_statistics_empty_row() -> None:
    section = ColumnTemporalContinuousSection(
        frame=pd.DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_section_get_statistics_empty_column() -> None:
    section = ColumnTemporalContinuousSection(
        frame=pd.DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert objects_are_equal(section.get_statistics(), {})


def test_column_temporal_continuous_section_render_html_body(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_continuous_section_render_html_body_empty_row() -> None:
    section = ColumnTemporalContinuousSection(
        frame=pd.DataFrame({"col": [], "datetime": []}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_continuous_section_render_html_body_empty_column() -> None:
    section = ColumnTemporalContinuousSection(
        frame=pd.DataFrame({}),
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_temporal_continuous_section_render_html_body_args(
    dataframe: pd.DataFrame,
) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_temporal_continuous_section_render_html_toc(dataframe: pd.DataFrame) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_temporal_continuous_section_render_html_toc_args(
    dataframe: pd.DataFrame,
) -> None:
    section = ColumnTemporalContinuousSection(
        frame=dataframe,
        column="col",
        dt_column="datetime",
        period="M",
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


###########################################
#    Tests for create_temporal_figure     #
###########################################


def test_create_temporal_figure(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
        ),
        str,
    )


@pytest.mark.parametrize("yscale", ["linear", "log"])
def test_create_temporal_figure_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            yscale=yscale,
        ),
        str,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_figure_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert isinstance(
        create_temporal_figure(
            frame=dataframe,
            column="col",
            dt_column="datetime",
            period="M",
            figsize=figsize,
        ),
        str,
    )


##########################################
#    Tests for create_temporal_table     #
##########################################


def test_create_temporal_table() -> None:
    assert isinstance(
        create_temporal_table(
            frame=pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    "col2": [1, 2, 3, 4, 5, 6],
                    "datetime": [
                        datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
                        datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                        datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                        datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                        datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
                        datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
                    ],
                },
                schema={
                    "col1": pl.Float64,
                    "col2": pl.Int64,
                    "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                },
            ),
            column="col1",
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


def test_create_temporal_table_empty() -> None:
    assert isinstance(
        create_temporal_table(
            frame=pl.DataFrame(
                {"col1": [], "col2": [], "datetime": []},
                schema={
                    "col1": pl.Float64,
                    "col2": pl.Int64,
                    "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                },
            ),
            column="col1",
            dt_column="datetime",
            period="1mo",
        ),
        str,
    )


##############################################
#    Tests for create_temporal_table_row     #
##############################################


def test_create_temporal_table_row() -> None:
    assert isinstance(
        create_temporal_table_row(
            {
                "step": "2020-01-01",
                "count": 101,
                "nunique": 101,
                "mean": 50.0,
                "std": 29.3,
                "min": 0.0,
                "q01": 1.0,
                "q05": 5.0,
                "q10": 10.0,
                "q25": 25.0,
                "median": 50.0,
                "q75": 75.0,
                "q90": 90.0,
                "q95": 95.0,
                "q99": 99.0,
                "max": 100.0,
            }
        ),
        str,
    )


def test_create_temporal_table_row_none() -> None:
    assert isinstance(
        create_temporal_table_row(
            {
                "step": "2020-01-01",
                "count": 0,
                "nunique": 0,
                "mean": None,
                "std": None,
                "min": None,
                "q01": None,
                "q05": None,
                "q10": None,
                "q25": None,
                "median": None,
                "q75": None,
                "q90": None,
                "q95": None,
                "q99": None,
                "max": None,
            }
        ),
        str,
    )
