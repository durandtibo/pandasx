from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from coola import objects_are_allclose
from jinja2 import Template
from pandas._testing import assert_frame_equal

from flamme.section.continuous_drift import (
    ColumnContinuousTemporalDriftSection,
    create_temporal_drift_figure,
)


@pytest.fixture()
def dataframe() -> pd.DataFrame:
    n = 100
    rng = np.random.default_rng()
    return pd.DataFrame(
        {
            "col": rng.standard_normal(n),
            "date": pd.date_range(start="2017-01-01", periods=n, freq="1D"),
        }
    )


##########################################################
#     Tests for ColumnContinuousTemporalDriftSection     #
##########################################################


def test_column_continuous_section_str(dataframe: pd.DataFrame) -> None:
    assert str(
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        )
    ).startswith("ColumnContinuousTemporalDriftSection(")


def test_column_continuous_section_frame(dataframe: pd.DataFrame) -> None:
    assert_frame_equal(
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).frame,
        dataframe,
    )


def test_column_continuous_section_column(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).column
        == "col"
    )


def test_column_continuous_section_dt_column(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).dt_column
        == "date"
    )


def test_column_continuous_section_period(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).period
        == "M"
    )


def test_column_continuous_section_yscale_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).yscale
        == "auto"
    )


@pytest.mark.parametrize("yscale", ["log", "linear"])
def test_column_continuous_section_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M", yscale=yscale
        ).yscale
        == yscale
    )


def test_column_continuous_section_nbins_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).nbins
        is None
    )


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_column_continuous_section_nbins(dataframe: pd.DataFrame, nbins: int) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M", nbins=nbins
        ).nbins
        == nbins
    )


def test_column_continuous_section_density_default(dataframe: pd.DataFrame) -> None:
    assert not ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="date", period="M"
    ).density


@pytest.mark.parametrize("density", [True, False])
def test_column_continuous_section_density(dataframe: pd.DataFrame, density: bool) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M", density=density
        ).density
        == density
    )


def test_column_continuous_section_xmin_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).xmin
        is None
    )


@pytest.mark.parametrize("xmin", [1.0, "q0.1"])
def test_column_continuous_section_xmin(dataframe: pd.DataFrame, xmin: float | str) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M", xmin=xmin
        ).xmin
        == xmin
    )


def test_column_continuous_section_xmax_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).xmax
        is None
    )


@pytest.mark.parametrize("xmax", [5.0, "q0.9"])
def test_column_continuous_section_xmax(dataframe: pd.DataFrame, xmax: float | str) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M", xmax=xmax
        ).xmax
        == xmax
    )


def test_column_continuous_section_figsize_default(dataframe: pd.DataFrame) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M"
        ).figsize
        is None
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_column_continuous_section_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert (
        ColumnContinuousTemporalDriftSection(
            frame=dataframe, column="col", dt_column="date", period="M", figsize=figsize
        ).figsize
        == figsize
    )


def test_column_continuous_section_get_statistics(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="date", period="M"
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_get_statistics_empty_row() -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=pd.DataFrame({"col": [], "date": []}), column="col", dt_column="date", period="M"
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_get_statistics_single_value() -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=pd.DataFrame(
            {
                "col": [1, 1, 1, 1],
                "date": pd.date_range(start="2017-01-01", periods=4, freq="1D"),
            }
        ),
        column="col",
        dt_column="date",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_get_statistics_only_nans() -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=pd.DataFrame(
            {
                "col": [np.nan, np.nan, np.nan, np.nan],
                "date": pd.date_range(start="2017-01-01", periods=4, freq="1D"),
            }
        ),
        column="col",
        dt_column="date",
        period="M",
    )
    assert objects_are_allclose(section.get_statistics(), {})


def test_column_continuous_section_render_html_body(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="date", period="M"
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_continuous_section_render_html_body_args(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="date", period="M"
    )
    assert isinstance(
        Template(section.render_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_column_continuous_section_render_html_body_empty() -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=pd.DataFrame({}), column="col", dt_column="date", period="M"
    )
    assert isinstance(Template(section.render_html_body()).render(), str)


def test_column_continuous_section_render_html_toc(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="date", period="M"
    )
    assert isinstance(Template(section.render_html_toc()).render(), str)


def test_column_continuous_section_render_html_toc_args(dataframe: pd.DataFrame) -> None:
    section = ColumnContinuousTemporalDriftSection(
        frame=dataframe, column="col", dt_column="date", period="M"
    )
    assert isinstance(
        Template(section.render_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#################################################
#    Tests for create_temporal_drift_figure     #
#################################################


def test_create_temporal_drift_figure(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(frame=dataframe, column="col", dt_column="date", period="M"),
        plt.Figure,
    )


def test_create_temporal_drift_figure_empty() -> None:
    assert (
        create_temporal_drift_figure(
            frame=pd.DataFrame({"col": [], "date": []}), column="col", dt_column="date", period="M"
        )
        is None
    )


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_create_temporal_drift_figure_nbins(dataframe: pd.DataFrame, nbins: int) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", nbins=nbins
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_create_temporal_drift_figure_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", yscale=yscale
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_create_temporal_drift_figure_xmin(
    dataframe: pd.DataFrame, xmin: float | str | None
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", xmin=xmin
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_create_temporal_drift_figure_xmax(
    dataframe: pd.DataFrame, xmax: float | str | None
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", xmax=xmax
        ),
        plt.Figure,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_drift_figure_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", figsize=figsize
        ),
        plt.Figure,
    )


def test_create_temporal_drift_figure_1_group(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(frame=dataframe, column="col", dt_column="date", period="Y"),
        plt.Figure,
    )


def test_create_temporal_drift_figure_2_groups(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe.iloc[:50, :], column="col", dt_column="date", period="M"
        ),
        plt.Figure,
    )


def test_create_temporal_drift_figure_missing_columns() -> None:
    assert (
        create_temporal_drift_figure(
            frame=pd.DataFrame({}), column="col", dt_column="date", period="Y"
        )
        is None
    )
