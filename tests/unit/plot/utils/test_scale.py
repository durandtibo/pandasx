from __future__ import annotations

import numpy as np
import pytest

from flamme.plot.utils import auto_yscale_continuous, auto_yscale_discrete

############################################
#     Tests for auto_yscale_continuous     #
############################################


@pytest.mark.parametrize("nbins", [1, 5, 10, 100, 1000])
def test_auto_yscale_continuous_nbins(nbins: int) -> None:
    assert auto_yscale_continuous(np.arange(100), nbins=nbins) == "linear"


@pytest.mark.parametrize(
    "array",
    [
        np.ones(100),
        np.arange(100),
        np.eye(10).flatten(),
        np.asarray([*list(range(100)), float("nan")]),
        np.asarray([]),
    ],
)
def test_auto_yscale_continuous_linear(array: np.ndarray) -> None:
    assert auto_yscale_continuous(array, nbins=10) == "linear"


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([1] * 100 + list(range(1, 11))),
        np.asarray([10] * 1000 + list(range(1, 11))),
        np.asarray([1] * 100 + list(range(1, 11)) + [float("nan")]),
    ],
)
def test_auto_yscale_continuous_log(array: np.ndarray) -> None:
    assert auto_yscale_continuous(array, nbins=10) == "log"


@pytest.mark.parametrize(
    "array",
    [
        np.asarray([1] * 100 + [-1, 10, 100]),
        np.asarray([100] * 1000 + [0, 10, 20]),
        np.asarray([100] * 1000 + [-1, 10, 20, float("nan")]),
    ],
)
def test_auto_yscale_continuous_symlog(array: np.ndarray) -> None:
    assert auto_yscale_continuous(array, nbins=10) == "symlog"


##########################################
#     Tests for auto_yscale_discrete     #
##########################################


@pytest.mark.parametrize("max_count", [1, 5, 10])
def test_auto_yscale_discrete_linear(max_count: int) -> None:
    assert auto_yscale_discrete(min_count=1, max_count=max_count) == "linear"


@pytest.mark.parametrize("max_count", [50, 100, 1000])
def test_auto_yscale_discrete_log(max_count: int) -> None:
    assert auto_yscale_discrete(min_count=1, max_count=max_count) == "log"


def test_auto_yscale_discrete_threshold_50() -> None:
    assert auto_yscale_discrete(min_count=1, max_count=49) == "linear"
    assert auto_yscale_discrete(min_count=1, max_count=50) == "log"


def test_auto_yscale_discrete_threshold_100() -> None:
    assert auto_yscale_discrete(min_count=1, max_count=99, threshold=100) == "linear"
    assert auto_yscale_discrete(min_count=1, max_count=100, threshold=100) == "log"


def test_auto_yscale_discrete_min_count_0() -> None:
    assert auto_yscale_discrete(min_count=0, max_count=5) == "linear"
