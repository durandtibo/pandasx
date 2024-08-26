from __future__ import annotations

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from coola import objects_are_allclose
from jinja2 import Template
from polars.testing import assert_frame_equal

from flamme.section.continuous_drift import (
    ColumnContinuousTemporalDriftSection,
    create_section_template,
    create_temporal_drift_figure,
)
from flamme.utils.data import datetime_range


@pytest.fixture
def dataframe() -> pl.DataFrame:
    rng = np.random.default_rng()
    return pl.DataFrame(
        {
            "col": rng.standard_normal(100),
            "datetime": datetime_range(
                start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                periods=100,
                interval="1d",
                eager=True,
            ),
        },
        schema={"col": pl.Float64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
    )


##########################################################
#     Tests for ColumnContinuousTemporalDriftSection     #
##########################################################


def test_column_continuous_section_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        )
    ).startswith("ColumnContinuousTemporalDriftSection(")


def test_column_continuous_section_frame(dataframe: pl.DataFrame) -> None:
    assert_frame_equal(
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).frame,
        dataframe,
    )


def test_column_continuous_section_column(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).column
        == "col"
    )


def test_column_continuous_section_dt_column(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).dt_column
        == "datetime"
    )


def test_column_continuous_section_period(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).period
        == "1mo"
    )


def test_column_continuous_section_yscale_default(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).yscale
        == "auto"
    )


@pytest.mark.parametrize("yscale", ["log", "linear"])
def test_column_continuous_section_yscale(dataframe: pl.DataFrame, yscale: str) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", yscale=yscale
        ).yscale
        == yscale
    )


def test_column_continuous_section_nbins_default(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).nbins
        is None
    )


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_column_continuous_section_nbins(dataframe: pl.DataFrame, nbins: int) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", nbins=nbins
        ).nbins
        == nbins
    )


def test_column_continuous_section_density_default(dataframe: pl.DataFrame) -> None:
    assert not ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="datetime", period="1mo"
    ).density


@pytest.mark.parametrize("density", [True, False])
def test_column_continuous_section_density(dataframe: pl.DataFrame, density: bool) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", density=density
        ).density
        == density
    )


def test_column_continuous_section_xmin_default(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).xmin
        is None
    )


@pytest.mark.parametrize("xmin", [1.0, "q0.1"])
def test_column_continuous_section_xmin(dataframe: pl.DataFrame, xmin: float | str) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", xmin=xmin
        ).xmin
        == xmin
    )


def test_column_continuous_section_xmax_default(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).xmax
        is None
    )


@pytest.mark.parametrize("xmax", [5.0, "q0.9"])
def test_column_continuous_section_xmax(dataframe: pl.DataFrame, xmax: float | str) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", xmax=xmax
        ).xmax
        == xmax
    )


def test_column_continuous_section_figsize_default(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ).figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_continuous_section_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_continuous_section_get_statistics(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="datetime", period="1mo"
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_get_statistics_empty_row() -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=pl.DataFrame(
            {"col": [], "datetime": []},
            schema={"col": pl.Float64, "datetime": pl.Datetime(time_unit="us", time_zone="UTC")},
        ),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_get_statistics_single_value(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe.with_columns(pl.lit(1).alias("col")),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_get_statistics_only_nans(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe.with_columns(pl.lit(float("nan")).alias("col")),
        column="col",
        dt_column="datetime",
        period="1mo",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_render_html_body(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="datetime", period="1mo"
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_continuous_section_render_html_body_args(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="datetime", period="1mo"
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_continuous_section_render_html_body_empty() -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=pl.DataFrame({}), column="col", dt_column="datetime", period="1mo"
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_continuous_section_render_html_toc(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="datetime", period="1mo"
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_continuous_section_render_html_toc_args(dataframe: pl.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="datetime", period="1mo"
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
#    Tests for create_temporal_drift_figure     #
#################################################


def test_create_temporal_drift_figure(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1mo"
        ),
        plt.Figure,
    )


def test_create_temporal_drift_figure_with_nulls() -> None:
    np.random.default_rng()
    frame = pl.DataFrame(
        {
            "col": [None, *list(range(98)), None],
            "datetime": datetime_range(
                start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                periods=100,
                interval="1d",
                eager=True,
            ),
        },
        schema={
            "col": pl.Float64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )
    assert isinstance(
        create_temporal_drift_figure(frame=frame, column="col", dt_column="datetime", period="1mo"),
        plt.Figure,
    )


def test_create_temporal_drift_figure_only_nulls(dataframe: pl.DataFrame) -> None:
    assert (
        create_temporal_drift_figure(
            frame=dataframe.with_columns(pl.lit(None).alias("col")),
            column="col",
            dt_column="datetime",
            period="1mo",
        )
        is None
    )


def test_create_temporal_drift_figure_only_nans(dataframe: pl.DataFrame) -> None:
    assert (
        create_temporal_drift_figure(
            frame=dataframe.with_columns(pl.lit(float("nan")).alias("col")),
            column="col",
            dt_column="datetime",
            period="1mo",
        )
        is None
    )


def test_create_temporal_drift_figure_empty() -> None:
    assert (
        create_temporal_drift_figure(
            frame=pl.DataFrame(
                {"col": [], "datetime": []},
                schema={
                    "col": pl.Float64,
                    "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                },
            ),
            column="col",
            dt_column="datetime",
            period="1mo",
        )
        is None
    )


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_create_temporal_drift_figure_nbins(dataframe: pl.DataFrame, nbins: int) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", nbins=nbins
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_create_temporal_drift_figure_yscale(dataframe: pl.DataFrame, yscale: str) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", yscale=yscale
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_create_temporal_drift_figure_xmin(
    dataframe: pl.DataFrame, xmin: float | str | None
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", xmin=xmin
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_create_temporal_drift_figure_xmax(
    dataframe: pl.DataFrame, xmax: float | str | None
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", xmax=xmax
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_drift_figure_figsize(
    dataframe: pl.DataFrame, figsize: tuple[float, float]
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1mo", figsize=figsize
        ),
        plt.Figure,
    )


def test_create_temporal_drift_figure_1_group(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="1y"
        ),
        plt.Figure,
    )


def test_create_temporal_drift_figure_2_groups(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="datetime", period="2mo"
        ),
        plt.Figure,
    )


def test_create_temporal_drift_figure_missing_columns() -> None:
    assert (
        create_temporal_drift_figure(
            frame=pl.DataFrame({}), column="col", dt_column="datetime", period="1mo"
        )
        is None
    )
