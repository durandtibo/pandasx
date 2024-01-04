from __future__ import annotations

import numpy as np
import pandas as pd
from coola import objects_are_allclose
from pytest import mark

from flamme.section.utils import (
    auto_yscale_continuous,
    compute_statistics,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)

#############################
#     Tests for tags2id     #
#############################


def test_tags2id_empty() -> None:
    assert tags2id([]) == ""


def test_tags2id_1_tag() -> None:
    assert tags2id(["meow"]) == "meow"


def test_tags2id_2_tags() -> None:
    assert tags2id(["super", "meow"]) == "super-meow"


################################
#     Tests for tags2title     #
################################


def test_tags2title_empty() -> None:
    assert tags2title([]) == ""


def test_tags2title_1_tag() -> None:
    assert tags2title(["meow"]) == "meow"


def test_tags2title_2_tags() -> None:
    assert tags2title(["super", "meow"]) == "meow | super"


#################################
#     Tests for valid_h_tag     #
#################################


def test_valid_h_tag_0() -> None:
    assert valid_h_tag(0) == 1


def test_valid_h_tag_1() -> None:
    assert valid_h_tag(1) == 1


def test_valid_h_tag_6() -> None:
    assert valid_h_tag(6) == 6


def test_valid_h_tag_7() -> None:
    assert valid_h_tag(7) == 6


#####################################
#     Tests for render_html_toc     #
#####################################


def test_render_html_toc_no_tags_and_number() -> None:
    assert render_html_toc() == '<li><a href="#"> </a></li>'


def test_render_html_toc_no_tags() -> None:
    assert render_html_toc(number="1.2.") == '<li><a href="#">1.2. </a></li>'


def test_render_html_toc_tags() -> None:
    assert (
        render_html_toc(number="1.2.", tags=("super", "meow"))
        == '<li><a href="#super-meow">1.2. meow</a></li>'
    )


def test_render_html_toc_tags_without_number() -> None:
    assert render_html_toc(tags=("super", "meow")) == '<li><a href="#super-meow"> meow</a></li>'


def test_render_html_toc_max_depth() -> None:
    assert render_html_toc(depth=2, max_depth=2) == ""


############################################
#     Tests for auto_yscale_continuous     #
############################################


@mark.parametrize("nbins", [1, 5, 10, 100, 1000])
def test_auto_yscale_continuous_nbins(nbins: int) -> None:
    assert auto_yscale_continuous(np.arange(100), nbins=nbins) == "linear"


@mark.parametrize(
    "array",
    [
        np.ones(100),
        np.arange(100),
        np.eye(10).flatten(),
        np.asarray(list(range(100)) + [float("nan")]),
        np.asarray([]),
    ],
)
def test_auto_yscale_continuous_linear(array: np.ndarray) -> None:
    assert auto_yscale_continuous(array, nbins=10) == "linear"


@mark.parametrize(
    "array",
    [
        np.asarray([1] * 100 + list(range(1, 11))),
        np.asarray([10] * 1000 + list(range(1, 11))),
        np.asarray([1] * 100 + list(range(1, 11)) + [float("nan")]),
    ],
)
def test_auto_yscale_continuous_log(array: np.ndarray) -> None:
    assert auto_yscale_continuous(array, nbins=10) == "log"


@mark.parametrize(
    "array",
    [
        np.asarray([1] * 100 + [-1, 10, 100]),
        np.asarray([100] * 1000 + [0, 10, 20]),
        np.asarray([100] * 1000 + [-1, 10, 20, float("nan")]),
    ],
)
def test_auto_yscale_continuous_symlog(array: np.ndarray) -> None:
    assert auto_yscale_continuous(array, nbins=10) == "symlog"


########################################
#     Tests for compute_statistics     #
########################################


@mark.parametrize("data", [np.asarray([]), pd.Series([], dtype=object)])
def test_compute_statistics_empty(data: np.ndarray | pd.Series) -> None:
    assert objects_are_allclose(
        compute_statistics(data),
        {
            "count": 0,
            "num_nulls": 0,
            "num_non_nulls": 0,
            "nunique": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": float("nan"),
            "q001": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "q999": float("nan"),
            "max": float("nan"),
        },
        equal_nan=True,
    )


@mark.parametrize(
    "data",
    [
        np.asarray([np.nan] + list(range(101)) + [np.nan]),
        pd.Series([np.nan] + list(range(101)) + [np.nan]),
    ],
)
def test_compute_statistics(data: np.ndarray | pd.Series) -> None:
    assert objects_are_allclose(
        compute_statistics(data),
        {
            "count": 103,
            "num_nulls": 2,
            "num_non_nulls": 101,
            "nunique": 102,
            "mean": 50.0,
            "std": 29.300170647967224,
            "skewness": 0.0,
            "kurtosis": -1.2,
            "min": 0.0,
            "q001": 0.1,
            "q01": 1.0,
            "q05": 5.0,
            "q10": 10.0,
            "q25": 25.0,
            "median": 50.0,
            "q75": 75.0,
            "q90": 90.0,
            "q95": 95.0,
            "q99": 99.0,
            "q999": 99.9,
            "max": 100.0,
        },
        atol=1e-2,
    )


@mark.parametrize(
    "data",
    [
        np.asarray([1, 1, 1, 1, 1]),
        pd.Series([1, 1, 1, 1, 1]),
    ],
)
def test_compute_statistics_single_numeric_value(data: np.ndarray | pd.Series) -> None:
    assert objects_are_allclose(
        compute_statistics(data),
        {
            "count": 5,
            "num_nulls": 0,
            "num_non_nulls": 5,
            "nunique": 1,
            "mean": 1.0,
            "std": 0.0,
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": 1.0,
            "q001": 1.0,
            "q01": 1.0,
            "q05": 1.0,
            "q10": 1.0,
            "q25": 1.0,
            "median": 1.0,
            "q75": 1.0,
            "q90": 1.0,
            "q95": 1.0,
            "q99": 1.0,
            "q999": 1.0,
            "max": 1.0,
        },
        equal_nan=True,
    )


@mark.parametrize(
    "data",
    [
        np.asarray([np.nan, np.nan, np.nan, np.nan]),
        pd.Series([np.nan, np.nan, np.nan, np.nan]),
    ],
)
def test_compute_statistics_only_nans(data: np.ndarray | pd.Series) -> None:
    assert objects_are_allclose(
        compute_statistics(data),
        {
            "count": 4,
            "num_nulls": 4,
            "num_non_nulls": 0,
            "nunique": 1,
            "mean": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
            "min": float("nan"),
            "q001": float("nan"),
            "q01": float("nan"),
            "q05": float("nan"),
            "q10": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q90": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
            "q999": float("nan"),
            "max": float("nan"),
        },
        equal_nan=True,
    )
