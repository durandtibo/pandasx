import numpy as np

from flamme.utils.range import find_range

################################
#     Tests for find_range     #
################################


def test_find_range_xmin_none_xmax_none() -> None:
    assert find_range(np.arange(101)) == (0, 100)


def test_find_range_xmin_str_xmax_none() -> None:
    assert find_range(np.arange(101), xmin="q0.01") == (1.0, 100)


def test_find_range_xmin_none_xmax_str() -> None:
    assert find_range(np.arange(101), xmax="q0.99") == (0, 99.0)


def test_find_range_xmin_str_xmax_str() -> None:
    assert find_range(np.arange(101), xmin="q0.1", xmax="q0.9") == (10.0, 90.0)


def test_find_range_xmin_float_xmax_none() -> None:
    assert find_range(np.arange(101), xmin=5.0) == (5.0, 100)


def test_find_range_xmin_none_xmax_float() -> None:
    assert find_range(np.arange(101), xmax=95.0) == (0, 95.0)


def test_find_range_xmin_float_xmax_float() -> None:
    assert find_range(np.arange(101), xmin=0.25, xmax=0.75) == (0.25, 0.75)


def test_find_range_xmin_0_xmax_1() -> None:
    assert find_range(np.arange(101), xmin="q0", xmax="q1") == (0.0, 100.0)


def test_find_range_nan_none() -> None:
    assert find_range(
        np.array(
            [float("nan"), 0, float("nan"), 1, 2, 3, 4, 5, 6, float("nan"), 7, 8, 9, float("nan")]
        )
    ) == (0, 9)


def test_find_range_nan_str() -> None:
    assert find_range(
        np.array(
            [
                float("nan"),
                0,
                float("nan"),
                1,
                2,
                3,
                10,
                4,
                5,
                6,
                float("nan"),
                7,
                8,
                9,
                float("nan"),
            ]
        ),
        xmin="q0.1",
        xmax="q0.9",
    ) == (1, 9)
