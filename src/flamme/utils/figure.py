r"""Contain utility functions to manage matplotlib figures."""

from __future__ import annotations

__all__ = ["figure2html"]

import base64
import io

from matplotlib import pyplot as plt


def figure2html(fig: plt.Figure, reactive: bool = True, close_fig: bool = False) -> str:
    r"""Convert a matplotlib figure to a string that can be used in a
    HTML file.

    Args:
        fig: The figure to convert.
        reactive: If ``True``, the generated is configured to be
            reactive to the screen size.
        close_fig: If ``True``, the figure is closed after it is
            converted to HTML format.

    Returns:
        The converted figure to a string.

    Example usage:

    ```pycon

    >>> from matplotlib import pyplot as plt
    >>> from flamme.utils.figure import figure2html
    >>> fig, ax = plt.subplots()
    >>> string = figure2html(fig)

    ```
    """
    fig.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    data = base64.b64encode(img.getvalue()).decode("utf-8")
    if close_fig:
        plt.close(fig)
    style = 'style="width:100%; height:auto;" ' if reactive else False
    return f'<img {style}src="data:image/png;charset=utf-8;base64, {data}">'
