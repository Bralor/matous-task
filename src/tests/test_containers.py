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
            "input": {
                "format": "json",
                "filepath": json_filepath,
            }  # TODO: Add hyperparameters
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
            "input": {
                "format": "npy",
                "filepath": npy_filepath,
            }  # TODO: Add hyperparameters
        }
    )
    yield container


def test_factory_initialization_with_json_file(testing_json_container):
    test_filepath = testing_json_container.config.input.filepath()
    test_format = testing_json_container.config.input.format()

    assert test_filepath == "src/tests/test.json"
    assert test_format == "json"


def test_factory_initialization_with_npy_file(testing_numpy_container):
    test_filepath = testing_numpy_container.config.input.filepath()
    test_format = testing_numpy_container.config.input.format()

    assert test_filepath == "src/tests/test.npy"
    assert test_format == "npy"
