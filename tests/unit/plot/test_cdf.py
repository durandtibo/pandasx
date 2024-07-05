from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from flamme.plot import plot_cdf

##############################
#     Tests for plot_cdf     #
##############################


def test_plot_cdf() -> None:
    fig, ax = plt.subplots()
    plot_cdf(ax, array=np.arange(101), nbins=10)


def test_plot_cdf_empty() -> None:
    fig, ax = plt.subplots()
    plot_cdf(ax, array=np.array([]), nbins=10)


def test_plot_cdf_nan() -> None:
    fig, ax = plt.subplots()
    plot_cdf(ax, array=np.array([1, 2, np.nan, 3, np.nan]))
