import numpy as np
from pytest import mark

from flamme.section.utils import (
    auto_yscale_continuous,
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
