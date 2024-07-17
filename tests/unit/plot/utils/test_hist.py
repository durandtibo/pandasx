from __future__ import annotations

import numpy as np
import pytest

from flamme.plot.utils.hist import adjust_nbins, find_nbins

##################################
#     Tests for adjust_nbins     #
##################################


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_adjust_nbins_int(dtype: np.dtype) -> None:
    assert adjust_nbins(nbins=100, array=np.array([1, 4, 5, 6], dtype=dtype)) == 6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjust_nbins_float(dtype: np.dtype) -> None:
    assert adjust_nbins(nbins=100, array=np.array([1, 4, 5, 6], dtype=dtype)) == 100


def test_adjust_nbins_initial() -> None:
    assert adjust_nbins(nbins=100, array=np.array([100, 0])) == 100


def test_adjust_nbins_1() -> None:
    assert adjust_nbins(nbins=100, array=np.array([1, 1])) == 1


################################
#     Tests for find_nbins     #
################################


def test_find_nbins_min_0_max_10() -> None:
    assert find_nbins(bin_size=1, min=0, max=10) == 11


def test_find_nbins_min_0_max_9() -> None:
    assert find_nbins(bin_size=2, min=0, max=9) == 5


def test_find_nbins_min_equals_max() -> None:
    assert find_nbins(bin_size=1, min=2, max=2) == 1


@pytest.mark.parametrize("bin_size", [0, -0.1])
def test_find_nbins_incorrect_bin_size(bin_size: float) -> None:
    with pytest.raises(RuntimeError, match="Incorrect bin_size"):
        find_nbins(bin_size=bin_size, min=0, max=2)


def test_find_nbins_incorrect_max() -> None:
    with pytest.raises(RuntimeError, match="Incorrect max"):
        find_nbins(bin_size=1, min=5, max=2)
