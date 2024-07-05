r"""Contain scale utility functions."""

from __future__ import annotations

__all__ = ["auto_yscale_continuous"]


import numpy as np

from flamme.utils.array import nonnan


def auto_yscale_continuous(array: np.ndarray, nbins: int | None = None) -> str:
    r"""Find a good scale for y-axis based on the data distribution.

    Args:
        array: The data to use to find the scale.
        nbins: The number of bins in the histogram.

    Returns:
        The scale for y-axis.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from flamme.section.utils import auto_yscale_continuous
    >>> auto_yscale_continuous(np.arange(100))
    linear

    ```
    """
    if nbins is None:
        nbins = 100
    array = nonnan(array)
    counts = np.histogram(array, bins=nbins)[0]
    nonzero_count = [c for c in counts if c > 0]
    if len(nonzero_count) <= 2 or (max(nonzero_count) / max(min(nonzero_count), 1)) < 50:
        return "linear"
    if np.nanmin(array) <= 0.0:
        return "symlog"
    return "log"
