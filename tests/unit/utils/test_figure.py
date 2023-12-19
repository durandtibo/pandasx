from __future__ import annotations

__all__ = ["figure2html"]

from matplotlib import pyplot as plt

from flamme.utils.figure import figure2html

#################################
#     Tests for figure2html     #
#################################


def test_figure2html() -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(fig), str)
