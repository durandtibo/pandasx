import numpy as np

from flamme.utils.range import find_range

################################
#     Tests for find_range     #
################################


def test_find_range_xmin_none_xmax_none() -> None:
    assert find_range(np.arange(101)) == (None, None)


def test_find_range_xmin_str_xmax_none() -> None:
    assert find_range(np.arange(101), xmin="q0.01") == (1.0, None)


def test_find_range_xmin_none_xmax_str() -> None:
    assert find_range(np.arange(101), xmax="q0.99") == (None, 99.0)


def test_find_range_xmin_str_xmax_str() -> None:
    assert find_range(np.arange(101), xmin="q0.1", xmax="q0.9") == (10.0, 90.0)


def test_find_range_xmin_float_xmax_none() -> None:
    assert find_range(np.arange(101), xmin=5.0) == (5.0, None)


def test_find_range_xmin_none_xmax_float() -> None:
    assert find_range(np.arange(101), xmax=95.0) == (None, 95.0)


def test_find_range_xmin_float_xmax_float() -> None:
    assert find_range(np.arange(101), xmin=0.25, xmax=0.75) == (0.25, 0.75)


def test_find_range_xmin_0_xmax_1() -> None:
    assert find_range(np.arange(101), xmin="q0", xmax="q1") == (0.0, 100.0)
