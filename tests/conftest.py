__all__ = ["close_plt_figure"]


from matplotlib import pyplot as plt
from pytest import fixture


@fixture(autouse=True)
def close_plt_figure() -> None:
    plt.close()
