from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flamme.section.continuous_drift import create_temporal_drift_figure


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


#################################################
#    Tests for create_temporal_drift_figure     #
#################################################


def test_create_temporal_drift_figure(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(frame=dataframe, column="col", dt_column="date", period="M"),
        str,
    )


def test_create_temporal_drift_figure_empty() -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=pd.DataFrame({"col": [], "date": []}), column="col", dt_column="date", period="M"
        ),
        str,
    )


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_create_temporal_drift_figure_nbins(dataframe: pd.DataFrame, nbins: int) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", nbins=nbins
        ),
        str,
    )


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_create_temporal_drift_figure_yscale(dataframe: pd.DataFrame, yscale: str) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", yscale=yscale
        ),
        str,
    )


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_create_temporal_drift_figure_xmin(
    dataframe: pd.DataFrame, xmin: float | str | None
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", xmin=xmin
        ),
        str,
    )


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_create_temporal_drift_figure_xmax(
    dataframe: pd.DataFrame, xmax: float | str | None
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", xmax=xmax
        ),
        str,
    )


@pytest.mark.parametrize("figsize", [(7, 3), (1.5, 1.5)])
def test_create_temporal_drift_figure_figsize(
    dataframe: pd.DataFrame, figsize: tuple[float, float]
) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe, column="col", dt_column="date", period="M", figsize=figsize
        ),
        str,
    )


def test_create_temporal_drift_figure_1_group(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(frame=dataframe, column="col", dt_column="date", period="Y"),
        str,
    )


def test_create_temporal_drift_figure_2_groups(dataframe: pd.DataFrame) -> None:
    assert isinstance(
        create_temporal_drift_figure(
            frame=dataframe.iloc[:50, :], column="col", dt_column="date", period="M"
        ),
        str,
    )
