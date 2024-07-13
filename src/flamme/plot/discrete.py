r"""Contain plotting functions to analyze discrete values."""

from __future__ import annotations

__all__ = ["bar_discrete"]

from typing import TYPE_CHECKING

import numpy as np

from flamme.plot.utils import auto_yscale_discrete, readable_xticklabels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


def bar_discrete(
    ax: Axes,
    names: Sequence,
    counts: Sequence[int],
    yscale: str = "auto",
) -> None:
    r"""Plot the histogram of an array containing discrete values.

    Args:
        ax: The axes of the matplotlib figure to update.
        names: The name of the values to plot.
        counts: The number of value occurrences.
        yscale: The y-axis scale. If ``'auto'``, the
            ``'linear'`` or ``'log'/'symlog'`` scale is chosen based
            on the distribution.

    Example usage:

    ```pycon

    >>> from matplotlib import pyplot as plt
    >>> from flamme.plot import bar_discrete
    >>> fig, ax = plt.subplots()
    >>> bar_discrete(ax, names=["a", "b", "c", "d"], counts=[5, 100, 42, 27])

    ```
    """
    n = len(names)
    if n == 0:
        return
    x = np.arange(n)
    ax.bar(x, counts, width=0.9 if n < 50 else 1, color="tab:blue")
    if yscale == "auto":
        yscale = auto_yscale_discrete(min_count=min(counts), max_count=max(counts))
    ax.set_yscale(yscale)
    ax.set_xticks(x, labels=map(str, names))
    readable_xticklabels(ax, max_num_xticks=100)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_xlabel("values")
    ax.set_ylabel("number of occurrences")
