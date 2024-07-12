from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from flamme.plot import bar_discrete

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
