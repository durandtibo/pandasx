from __future__ import annotations

from matplotlib import pyplot as plt

from flamme.plot.utils import axvline_median, axvline_quantile


#####################################
#    Tests for axvline_quantile     #
#####################################


def test_axvline_quantile() -> None:
    _fig, ax = plt.subplots()
    axvline_quantile(ax, quantile=1.0, label="my_label")


###################################
#    Tests for axvline_median     #
###################################


def test_axvline_median() -> None:
    _fig, ax = plt.subplots()
    axvline_median(ax, median=1.0)
