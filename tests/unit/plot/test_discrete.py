from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal
from matplotlib import pyplot as plt

from flamme.plot import bar_discrete
from flamme.plot.discrete import (
    _prepare_counts_bar_discrete_temporal,
    _prepare_steps_bar_discrete_temporal,
    _prepare_values_bar_discrete_temporal,
    bar_discrete_temporal,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


#################################
#    Tests for bar_discrete     #
#################################


def test_bar_discrete() -> None:
    fig, ax = plt.subplots()
    bar_discrete(ax=ax, names=["a", "b", "c", "d"], counts=[5, 100, 42, 27])


@pytest.mark.parametrize("yscale", ["linear", "log", "auto"])
def test_bar_discrete_yscale(yscale: str) -> None:
    fig, ax = plt.subplots()
    bar_discrete(ax=ax, names=["a", "b", "c", "d"], counts=[5, 100, 42, 27], yscale=yscale)


def test_bar_discrete_empty() -> None:
    fig, ax = plt.subplots()
    bar_discrete(ax=ax, names=[], counts=[])


##########################################
#    Tests for bar_discrete_temporal     #
##########################################


def test_bar_discrete_temporal() -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(ax, counts=np.ones((5, 20)))


def test_bar_discrete_temporal_with_values() -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(ax, counts=np.ones((5, 20)), values=list(range(5)))


def test_bar_discrete_temporal_with_steps() -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(ax, counts=np.ones((5, 20)), steps=list(range(20)))


def test_bar_discrete_temporal_with_values_and_steps() -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(ax, counts=np.ones((5, 20)), values=list(range(5)), steps=list(range(20)))


@pytest.mark.parametrize("proportion", [True, False])
def test_bar_discrete_temporal_proportion_ones(proportion: bool) -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(
        ax,
        counts=np.ones((5, 20)),
        values=list(range(5)),
        steps=list(range(20)),
        proportion=proportion,
    )


@pytest.mark.parametrize("proportion", [True, False])
def test_bar_discrete_temporal_proportion_zeros(proportion: bool) -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(
        ax,
        counts=np.zeros((5, 20)),
        values=list(range(5)),
        steps=list(range(20)),
        proportion=proportion,
    )


def test_bar_discrete_temporal_empty() -> None:
    fig, ax = plt.subplots()
    bar_discrete_temporal(ax, counts=np.zeros((0, 0)))


@pytest.mark.parametrize("values", [("A", "B", "C", "D", "E"), ["A", "B", "C", "D", "E"]])
def test_prepare_values_bar_discrete_temporal_values(values: Sequence) -> None:
    assert _prepare_values_bar_discrete_temporal(values=values, num_values=5) == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


def test_prepare_values_bar_discrete_temporal_values_none() -> None:
    assert _prepare_values_bar_discrete_temporal(values=None, num_values=5) == [
        None,
        None,
        None,
        None,
        None,
    ]


def test_prepare_values_bar_discrete_temporal_values_incorrect() -> None:
    with pytest.raises(RuntimeError, match="values length .* do not match with the count matrix"):
        _prepare_values_bar_discrete_temporal(values=[1, 2, 3], num_values=5)


@pytest.mark.parametrize("steps", [("A", "B", "C", "D", "E"), ["A", "B", "C", "D", "E"]])
def test_prepare_steps_bar_discrete_temporal_steps(steps: Sequence) -> None:
    assert _prepare_steps_bar_discrete_temporal(steps=steps, num_steps=5) == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]


def test_prepare_steps_bar_discrete_temporal_steps_none() -> None:
    assert _prepare_steps_bar_discrete_temporal(steps=None, num_steps=5) == [0, 1, 2, 3, 4]


def test_prepare_steps_bar_discrete_temporal_steps_incorrect() -> None:
    with pytest.raises(RuntimeError, match="steps length .* do not match with the count matrix"):
        _prepare_steps_bar_discrete_temporal(steps=[1, 2, 3], num_steps=5)


def test_prepare_counts_bar_discrete_temporal_proportion_false() -> None:
    assert objects_are_equal(
        _prepare_counts_bar_discrete_temporal(
            counts=np.array([[1, 2, 3, 0, 1], [0, 2, 1, 0, 3]]), proportion=False
        ),
        np.array([[1, 2, 3, 0, 1], [0, 2, 1, 0, 3]]),
    )


def test_prepare_counts_bar_discrete_temporal_proportion_true() -> None:
    assert objects_are_equal(
        _prepare_counts_bar_discrete_temporal(
            counts=np.array([[1, 2, 3, 0, 1], [0, 2, 1, 0, 3]]), proportion=True
        ),
        np.array([[1.0, 0.5, 0.75, 0.0, 0.25], [0.0, 0.5, 0.25, 0.0, 0.75]]),
    )
