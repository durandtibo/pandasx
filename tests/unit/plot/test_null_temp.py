from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from flamme.plot import plot_null_temporal

########################################
#     Tests for plot_null_temporal     #
########################################


def test_plot_null_temporal() -> None:
    _fig, ax = plt.subplots()
    plot_null_temporal(
        ax, nulls=[1, 2, 3, 4], totals=[10, 12, 14, 16], labels=["jan", "feb", "mar", "apr"]
    )


def test_plot_null_temporal_empty() -> None:
    _fig, ax = plt.subplots()
    plot_null_temporal(ax, nulls=[], totals=[], labels=[])


def test_plot_null_temporal_incorrect_total() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="nulls .* and totals .* have different lengths"):
        plot_null_temporal(
            ax, nulls=[1, 2, 3, 4], totals=[10, 12, 14, 16, 18], labels=["jan", "feb", "mar", "apr"]
        )


def test_plot_null_temporal_incorrect_labels() -> None:
    _fig, ax = plt.subplots()
    with pytest.raises(RuntimeError, match="nulls .* and labels .* have different lengths"):
        plot_null_temporal(
            ax,
            nulls=[1, 2, 3, 4],
            totals=[10, 12, 14, 16],
            labels=["jan", "feb", "mar", "apr", "may"],
        )
