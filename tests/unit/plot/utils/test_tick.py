from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from matplotlib import pyplot as plt

from flamme.plot.utils import readable_xticklabels

if TYPE_CHECKING:
    from matplotlib.axes import Axes


@pytest.fixture()
def axes() -> Axes:
    _fig, ax = plt.subplots()
    ax.hist(np.arange(10), bins=10)
    return ax


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
