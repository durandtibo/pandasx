from __future__ import annotations

__all__ = ["figure2html"]

from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt

from flamme.utils.figure import figure2html, readable_xticklabels

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@pytest.fixture()
def axes() -> Axes:
    fig, ax = plt.subplots()
    ax.hist(np.arange(10), bins=10)
    return ax


#################################
#     Tests for figure2html     #
#################################


def test_figure2html() -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig), str)


@pytest.mark.parametrize("close_fig", [True, False])
def test_figure2html_close_fig(close_fig: bool) -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig, close_fig=close_fig), str)


@pytest.mark.parametrize("reactive", [True, False])
def test_figure2html_reactive(reactive: bool) -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig, reactive=reactive), str)


##########################################
#     Tests for readable_xticklabels     #
##########################################


def test_readable_xticklabels(axes: Axes) -> None:
    readable_xticklabels(axes)
    assert len(axes.get_xticks()) <= 100


@pytest.mark.parametrize("max_num_xticks", [2, 5, 100])
def test_readable_xticklabels_max_num_xticks(axes: Axes, max_num_xticks: int) -> None:
    readable_xticklabels(axes, max_num_xticks=max_num_xticks)
    assert len(axes.get_xticks()) <= max_num_xticks


@pytest.mark.parametrize("xticklabel_max_len", [2, 5, 100])
def test_readable_xticklabels_xticklabel_max_len(axes: Axes, xticklabel_max_len: int) -> None:
    readable_xticklabels(axes, xticklabel_max_len=xticklabel_max_len)
    assert len(axes.get_xticks()) <= 100


@pytest.mark.parametrize("xticklabel_min", [2, 5, 100])
def test_readable_xticklabels_xticklabel_min(axes: Axes, xticklabel_min: int) -> None:
    readable_xticklabels(axes, xticklabel_min=xticklabel_min)
    assert len(axes.get_xticks()) <= 100
