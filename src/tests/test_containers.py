import numpy
import pytest

from clusterer.containers import Container


@pytest.fixture
def testing_json_container(json_filepath: str = "src/tests/test.json"):
    """
    Yield the testing Container with JSON file input.

    Args:
        json_filepath (str): The path for the config.
    """
    container = Container(
        config={
            "algorithm": "kmeans",
            "input": {"format": "json",
                      "filepath": json_filepath},
            "output": {"format": "json",
                       "filepath": 'test_output.json'},
            "hyperparameters": {"n_clusters": 3,
                                "random_state": 42,
                                "max_iter": 300},
        }
    )
    yield container


@pytest.fixture
def testing_numpy_container(npy_filepath: str = "src/tests/test.npy"):
    """
    Yield the testing Container with NPY file input.

    Args:
        npy_filepath (str): The path for the config.
    """
    container = Container(
        config={
            "algorithm": "kmeans",
            "input": {"format": "npy",
                      "filepath": npy_filepath},
            "output": {"format": "npy",
                       "filepath": 'test_output.npy'},
            "hyperparameters": {"n_clusters": 3,
                                "random_state": 42,
                                "max_iter": 300},
        }
    )
    yield container


def test_factory_initialization_with_json_file(testing_json_container):
    testing_format = testing_json_container.config.input.format()
    testing_input_filepath = testing_json_container.config.input.filepath()
    testing_output_filepath = testing_json_container.config.output.filepath()
    testing_max_iter = testing_json_container.config.hyperparameters.max_iter()

    assert testing_format == "json"
    assert testing_max_iter == 300
    assert testing_output_filepath == 'test_output.json'
    assert testing_input_filepath == "src/tests/test.json"


def test_factory_initialization_with_npy_file(testing_numpy_container):
    testing_format = testing_numpy_container.config.input.format()
    testing_input_filepath = testing_numpy_container.config.input.filepath()
    testing_output_filepath = testing_numpy_container.config.output.filepath()
    testing_max_iter = \
        testing_numpy_container.config.hyperparameters.max_iter()

    assert testing_format == "npy"
    assert testing_max_iter == 300
    assert testing_output_filepath == 'test_output.npy'
    assert testing_input_filepath == "src/tests/test.npy"


def test_provider_reads_numpy_file(testing_numpy_container: Container):
    testing_instance = testing_numpy_container.read_source()
    assert numpy.array_equal(testing_instance, numpy.array([[1, 1], [2, 1]]))


def test_provider_reads_json_file(testing_json_container: Container):
    testing_instance = testing_json_container.read_source()
    assert numpy.array_equal(testing_instance, numpy.array([[1, 2], [3, 4]]))
