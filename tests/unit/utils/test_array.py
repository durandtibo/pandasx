from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from flamme.utils.array import filter_range, nonnan, rand_replace

##################################
#     Tests for filter_range     #
##################################


@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_filter_range(dtype: np.dtype) -> None:
    assert np.array_equal(
        filter_range(np.arange(10, dtype=dtype), xmin=1, xmax=5),
        np.array([1, 2, 3, 4, 5], dtype=dtype),
    )


def test_filter_range_inf() -> None:
    assert np.array_equal(
        filter_range(np.arange(10), xmin=float("-inf"), xmax=float("inf")),
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )


def test_filter_range_empty() -> None:
    assert np.array_equal(filter_range(np.array([]), xmin=1, xmax=5), np.array([]))


############################
#     Tests for nonnan     #
############################


def test_nonnan_empty() -> None:
    assert np.array_equal(nonnan(np.array([])), np.array([]))


def test_nonnan_1d() -> None:
    assert np.array_equal(
        nonnan(np.array([1, 2, float("nan"), 5, 6])), np.array([1.0, 2.0, 5.0, 6.0])
    )


def test_nonnan_2d() -> None:
    assert np.array_equal(
        nonnan(np.array([[1, 2, float("nan")], [4, 5, 6]])), np.array([1.0, 2.0, 4.0, 5.0, 6.0])
    )


##################################
#     Tests for rand_replace     #
##################################


def test_rand_replace_prob_0() -> None:
    assert objects_are_equal(rand_replace(np.arange(10), value=-1, prob=0.0), np.arange(10))


def test_rand_replace_empty() -> None:
    assert objects_are_equal(rand_replace(np.array([]), value=-1, prob=0.4), np.array([]))


def test_rand_replace_same_seed() -> None:
    assert objects_are_equal(
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(1)),
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(1)),
    )


def test_rand_replace_different_seeds() -> None:
    assert not objects_are_equal(
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(1)),
        rand_replace(np.arange(100), value=-1, prob=0.4, rng=np.random.default_rng(2)),
    )
