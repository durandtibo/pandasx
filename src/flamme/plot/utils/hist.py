r"""Contain utility functions to generate histograms."""

from __future__ import annotations

__all__ = ["find_nbins"]

import math


def find_nbins(bin_size: float, min: float, max: float) -> int:  # noqa: A002
    r"""Find the number of bins from the bin size and the range of
    values.

    Args:
        bin_size: The target bin size.
        min: The minimum value.
        max: The maximum value.

    Returns:
        The number of bins.

    Raises:
        RuntimeError: if the bin size is invalid.
        RuntimeError: if the max value is invalid.

    Example usage:

    ```pycon

    >>> from flamme.plot.utils import find_nbins
    >>> nbins = find_nbins(bin_size=1, min=0, max=10)
    >>> nbins
    11

    ```
    """
    if bin_size <= 0:
        msg = f"Incorrect bin_size {bin_size}. bin_size must be greater than 0"
        raise RuntimeError(msg)
    if max < min:
        msg = f"Incorrect max {max}. max must be greater or equal to min: {min}"
        raise RuntimeError(msg)
    if min == max:
        return 1
    return math.ceil((max - min + 1) / bin_size)
