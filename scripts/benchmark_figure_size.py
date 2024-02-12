# noqa: INP001
r"""Contain functions to compare the size of matplotlib and plotly
figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly
import plotly.express as px
from matplotlib import pyplot as plt

from flamme.utils.figure import figure2html
from flamme.utils.io import save_text


def matplotlib_histogram(data: np.ndarray, bins: int = 100) -> str:
    r"""Create a matplotlib histogram figure encoded as a string.

    Args:
        data: Specifies the data used to generate the histogram.
        bins: Specifies the number of bins in the histogram.

    Returns:
        The histogram figure encoded as a string.
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(data, bins=bins, alpha=0.4)
    return figure2html(fig, close_fig=True)


def plotly_histogram(data: np.ndarray, bins: int = 100) -> str:
    r"""Create a plotly histogram figure encoded as a string.

    Args:
        data: Specifies the data used to generate the histogram.
        bins: Specifies the number of bins in the histogram.

    Returns:
        The histogram figure encoded as a string.
    """
    fig = px.histogram(x=data, nbins=bins)
    return plotly.io.to_html(fig, full_html=False)


def main() -> None:
    r"""Define the main function to analyze the size of different
    figures."""
    lines = []
    for size in [1000, 10000, 100000, 1000000]:
        data = np.random.randn(size)
        figm = matplotlib_histogram(data)
        figp = plotly_histogram(data)
        lines.append(f"<h2>size={size:,}</h2>")
        lines.append(figm)
        lines.append(figp)

    path = Path.cwd().joinpath("tmp/figures.html")
    save_text("\n".join(lines), path)


if __name__ == "__main__":
    main()
