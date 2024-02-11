__all__ = []

import pytest
from matplotlib import pyplot as plt


@pytest.fixture(autouse=True)
def _close_plt_figure() -> None:
    plt.close()
