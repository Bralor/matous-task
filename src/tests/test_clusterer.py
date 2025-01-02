from typing import Dict

import numpy
import pytest

from src.clusterer.clusterer import main


def test_dummy():
    assert main() == 1


@pytest.fixture
def testing_sample() -> numpy.ndarray:
    """
    Returns the two-dimensional array as a testing sample.

    :return: 2D array,
    :rtype: numpy.ndarray
    """
    return numpy.array([[1, 2], [2, 3], [1, 1], [8, 8], [9, 9], [10, 10]])


@pytest.fixture
def testing_config() -> Dict[str, int]:
    """
    Returns the object with the boundary conditions.

    :return: The initial setup,
    :rtype: Dict[str, int].
    """
    return {"n_clusters": 3, "n_init": 10, "max_iter": 300, "random_state": 42}
