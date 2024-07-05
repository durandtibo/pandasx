from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot as plt

from flamme.plot import hist_continuous, hist_continuous2

####################################
#    Tests for hist_continuous     #
####################################


def test_hist_continuous() -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101))


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_hist_continuous_nbins(nbins: int) -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), nbins=nbins)


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_hist_continuous_yscale(yscale: str) -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), yscale=yscale)


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_hist_continuous_xmin(xmin: float | str | None) -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), xmin=xmin)


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_hist_continuous_xmax(xmax: float | str | None) -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), xmax=xmax)


def test_hist_continuous_xmin_xmax() -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), xmin="q0.1", xmax="q0.9")


@pytest.mark.parametrize("cdf", [True, False])
def test_hist_continuous_cdf(cdf: bool) -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), cdf=cdf)


@pytest.mark.parametrize("quantile", [True, False])
def test_hist_continuous_quantile(quantile: bool) -> None:
    fig, ax = plt.subplots()
    hist_continuous(ax=ax, array=np.arange(101), quantile=quantile)


#####################################
#    Tests for hist_continuous2     #
#####################################


def test_hist_continuous2() -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51))


@pytest.mark.parametrize("label1", ["one", "two"])
def test_hist_continuous2_label1(label1: str) -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), label1=label1)


@pytest.mark.parametrize("label2", ["one", "two"])
def test_hist_continuous2_label2(label2: str) -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), label2=label2)


@pytest.mark.parametrize("nbins", [1, 2, 4])
def test_hist_continuous2_nbins(nbins: int) -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), nbins=nbins)


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_hist_continuous2_yscale(yscale: str) -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), yscale=yscale)


@pytest.mark.parametrize("xmin", [1.0, "q0.1", None, "q1"])
def test_hist_continuous2_xmin(xmin: float | str | None) -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), xmin=xmin)


@pytest.mark.parametrize("xmax", [100.0, "q0.9", None, "q0"])
def test_hist_continuous2_xmax(xmax: float | str | None) -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), xmax=xmax)


def test_hist_continuous2_xmin_xmax() -> None:
    fig, ax = plt.subplots()
    hist_continuous2(ax=ax, array1=np.arange(101), array2=np.arange(51), xmin="q0.1", xmax="q0.9")
