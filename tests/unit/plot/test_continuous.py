from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt

from flamme.plot import (
    boxplot_continuous,
    boxplot_continuous_temporal,
    hist_continuous,
    hist_continuous2,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#######################################
#    Tests for boxplot_continuous     #
#######################################


def test_boxplot_continuous() -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous(ax=ax, array=np.arange(101))


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_boxplot_continuous_xmin(xmin: float | str | None) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous(ax=ax, array=np.arange(101), xmin=xmin)


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_boxplot_continuous_xmax(xmax: float | str | None) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous(ax=ax, array=np.arange(101), xmax=xmax)


def test_boxplot_continuous_xmin_xmax() -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous(ax=ax, array=np.arange(101), xmin="q0.1", xmax="q0.9")


def test_boxplot_continuous_empty() -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous(ax=ax, array=np.array([]))


################################################
#    Tests for boxplot_continuous_temporal     #
################################################


@pytest.fixture()
def data_temp() -> list[np.ndarray]:
    rng = np.random.default_rng()
    return [rng.standard_normal(100) for i in range(10)]


def test_boxplot_continuous_temporal(data_temp: Sequence[np.ndarray]) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous_temporal(ax=ax, data=data_temp, steps=list(range(len(data_temp))))


@pytest.mark.parametrize("ymin", [1.0, "q0.1", None, "q1"])
def test_boxplot_continuous_temporal_ymin(
    data_temp: Sequence[np.ndarray], ymin: float | str | None
) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous_temporal(ax=ax, data=data_temp, steps=list(range(len(data_temp))), ymin=ymin)


@pytest.mark.parametrize("ymax", [100.0, "q0.9", None, "q0"])
def test_boxplot_continuous_temporal_ymax(
    data_temp: Sequence[np.ndarray], ymax: float | str | None
) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous_temporal(ax=ax, data=data_temp, steps=list(range(len(data_temp))), ymax=ymax)


def test_boxplot_continuous_temporal_ymin_ymax(data_temp: Sequence[np.ndarray]) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous_temporal(
        ax=ax, data=data_temp, steps=list(range(len(data_temp))), ymin="q0.1", ymax="q0.9"
    )


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_boxplot_continuous_temporal_yscale(data_temp: Sequence[np.ndarray], yscale: str) -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous_temporal(
        ax=ax, data=data_temp, steps=list(range(len(data_temp))), yscale=yscale
    )


def test_boxplot_continuous_temporal_empty() -> None:
    _fig, ax = plt.subplots()
    boxplot_continuous_temporal(ax=ax, data=[], steps=[])


def test_boxplot_continuous_temporal_incorrect_lengths() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="data and steps have different lengths"):
        boxplot_continuous_temporal(ax=ax, data=[np.ones(5), np.zeros(4)], steps=[1, 2, 3])


####################################
#    Tests for hist_continuous     #
####################################


def test_hist_continuous() -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101))


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_hist_continuous_nbins(nbins: int) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), nbins=nbins)


@pytest.mark.parametrize("density", [True, False])
def test_hist_continuous_density(density: bool) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), density=density)


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_hist_continuous_yscale(yscale: str) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), yscale=yscale)


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_hist_continuous_xmin(xmin: float | str | None) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), xmin=xmin)


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_hist_continuous_xmax(xmax: float | str | None) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), xmax=xmax)


def test_hist_continuous_xmin_xmax() -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), xmin="q0.1", xmax="q0.9")


@pytest.mark.parametrize("cdf", [True, False])
def test_hist_continuous_cdf(cdf: bool) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), cdf=cdf)


@pytest.mark.parametrize("quantile", [True, False])
def test_hist_continuous_quantile(quantile: bool) -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), quantile=quantile)


def test_hist_continuous_empty() -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.array([]))


def test_hist_continuous_nan() -> None:
    _fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.array([1, 2, 3, np.nan, 4, np.nan, np.nan]))


#####################################
#    Tests for hist_continuous2     #
#####################################


def test_hist_continuous2() -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51))


@pytest.mark.parametrize("label1", ["one", "two"])
def test_hist_continuous2_label1(label1: str) -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), label1=label1)


@pytest.mark.parametrize("label2", ["one", "two"])
def test_hist_continuous2_label2(label2: str) -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), label2=label2)


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_hist_continuous2_nbins(nbins: int) -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), nbins=nbins)


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_hist_continuous2_yscale(yscale: str) -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), yscale=yscale)


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_hist_continuous2_xmin(xmin: float | str | None) -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), xmin=xmin)


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_hist_continuous2_xmax(xmax: float | str | None) -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), xmax=xmax)


def test_hist_continuous2_xmin_xmax() -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), xmin="q0.1", xmax="q0.9")


def test_hist_continuous2_empty() -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.array([]), array2=np.array([]))


def test_hist_continuous2_array1_empty() -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.array([]), array2=np.arange(101))


def test_hist_continuous2_array2_empty() -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.array([]))


def test_hist_continuous2_nan() -> None:
    _fig, ax = plt.subplots()
    hist_continuous2(
        ax=ax,
        array1=np.array([1, 2, 3, np.nan, 4, np.nan, np.nan]),
        array2=np.array([5, np.nan, 7, np.nan, np.nan]),
    )
