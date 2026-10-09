"""Use a noninteractive backend and release figures between tests."""

import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    import matplotlib.pyplot as plt

    yield

    plt.close("all")
