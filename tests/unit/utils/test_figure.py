from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from flamme.utils.figure import figure2html

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


def test_figure2html_none() -> None:
    fig, _ = plt.subplots()
    assert (
        figure2html(fig)
        == "<span>&#9888;</span> No figure is generated because of missing or incorrect data"
    )
