from __future__ import annotations

from collections.abc import Iterable

from coola import objects_are_allclose
from pytest import mark, raises

from flamme.utils.mathnan import LowNaN, remove_nan, sortnan

################################
#     Tests for remove_nan     #
################################


@mark.parametrize(
    "data,output",
    [
        ([float("nan"), float("-inf"), -2, 1.2], [float("-inf"), -2, 1.2]),
        ((float("nan"), float("-inf"), -2, 1.2), (float("-inf"), -2, 1.2)),
        (["a", float("nan"), "b", "c", float("nan")], ["a", "b", "c"]),
        ([], [])
    ],
)
def test_remove_nan(data: list, output: list) -> None:
    assert remove_nan(data) == output


#############################
#     Tests for sortnan     #
#############################


def test_sortnan() -> None:
    assert objects_are_allclose(
        sortnan(
            [4, float("nan"), 2, 1.2, 7.9, -2, float("nan"), float("inf"), float("-inf")],
        ),
        [float("nan"), float("nan"), float("-inf"), -2, 1.2, 2, 4, 7.9, float("inf")],
        equal_nan=True,
    )


def test_sortnan_reverse() -> None:
    assert objects_are_allclose(
        sortnan(
            [4, float("nan"), 2, 1.2, 7.9, -2, float("nan"), float("inf"), float("-inf")],
            reverse=True,
        ),
        [float("inf"), 7.9, 4, 2, 1.2, -2, float("-inf"), float("nan"), float("nan")],
        equal_nan=True,
    )


@mark.parametrize(
    "data",
    [
        [-5.86, 4.54, 14.84, 0.37, 1.44, 8.21, -12.16, -9.12, 17.94, 19.91],
        [4, 2, 1.2, 7.9, -2],
        (4, 2, 1.2, 7.9, -2),
        [],
        [True, False, False, True],
        {8, 3, 2, 5, 4, 7, 1, 9},
    ],
)
@mark.parametrize("reverse", (True, False))
def test_sortnan_compatibility(data: Iterable, reverse: bool) -> None:
    assert sortnan(data, reverse=reverse) == sorted(data, reverse=reverse)


############################
#     Tests for LowNaN     #
############################


@mark.parametrize("value", [-5, 0, 2, 4.2, 42, float("-inf"), float("inf"), float("nan")])
def test_lownan_ge(value: float | int) -> None:
    assert not LowNaN() >= value


def test_lownan_ge_incorrect_type() -> None:
    with raises(TypeError, match="'>=' not supported between instances of 'float' and 'str'"):
        LowNaN().__ge__("abc")


@mark.parametrize("value", [-5, 0, 2, 4.2, 42, float("-inf"), float("inf"), float("nan")])
def test_lownan_gt(value: float | int) -> None:
    assert not LowNaN() > value


def test_lownan_gt_incorrect_type() -> None:
    with raises(TypeError, match="'>' not supported between instances of 'float' and 'str'"):
        LowNaN().__gt__("abc")


@mark.parametrize("value", [-5, 0, 2, 4.2, 42, float("-inf"), float("inf"), float("nan")])
def test_lownan_le(value: float | int) -> None:
    assert LowNaN() <= value


def test_lownan_le_incorrect_type() -> None:
    with raises(TypeError, match="'<=' not supported between instances of 'float' and 'str'"):
        LowNaN().__le__("abc")


@mark.parametrize("value", [-5, 0, 2, 4.2, 42, float("-inf"), float("inf"), float("nan")])
def test_lownan_lt(value: float | int) -> None:
    assert LowNaN() < value


def test_lownan_lt_incorrect_type() -> None:
    with raises(TypeError, match="'<' not supported between instances of 'float' and 'str'"):
        LowNaN().__lt__("abc")
