from __future__ import annotations

import numpy as np

from flamme.utils.array import nonnan

############################
#     Tests for nonnan     #
############################


def test_nonnan_empty() -> None:
    assert np.array_equal(nonnan(np.asarray([])), np.asarray([]))


def test_nonnan_1d() -> None:
    assert np.array_equal(
        nonnan(np.asarray([1, 2, float("nan"), 5, 6])), np.asarray([1.0, 2.0, 5.0, 6.0])
    )


def test_nonnan_2d() -> None:
    assert np.array_equal(
        nonnan(np.asarray([[1, 2, float("nan")], [4, 5, 6]])), np.asarray([1.0, 2.0, 4.0, 5.0, 6.0])
    )
