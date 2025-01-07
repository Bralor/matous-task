from typing import Dict

import numpy
import pytest

from src.clusterer.clusterer import main, DataLoader


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


@pytest.mark.parametrize(
    "source",
    ["src/tests/foo.txt", "src/tests/test.pdf", "src/tests/sample.xlsx"],
)
def test_file_with_incorrect_suffix_instead_of_json(source: str):
    with pytest.raises(Exception):
        DataLoader.from_json(source)


@pytest.mark.parametrize(
    "source",
    ["src/tests/foo.json", "src/tests/bar.json"],
)
def test_invalid_inputs_from_reading_json(source: str):
    tested_none_type = DataLoader.from_json(source)
    assert tested_none_type is None


def test_general_exception_while_reading_json(mocker):
    mock_file = "src/tests/test.json"

    # Mocking the open function to raise a generic exception
    mocker.patch("builtins.open", side_effect=Exception("Mocked exception"))

    result = DataLoader.from_json(mock_file)
    assert result is None


def test_existing_key_inside_json_file():
    existing_file = DataLoader.from_json("src/tests/test.json")

    existing_content = numpy.array(existing_file.__dict__.get("data"))
    expected_content = numpy.array([[1.1, 2.2], [3.3, 4.4]])

    assert numpy.array_equal(existing_content, expected_content)


@pytest.mark.parametrize(
    "source",
    ["src/tests/foo.txt", "src/tests/test.pdf", "src/tests/sample.xlsx"],
)
def test_file_with_incorrect_suffix_instead_of_npy(source: str):
    with pytest.raises(Exception):
        DataLoader.from_numpy(source)


@pytest.mark.parametrize(
    "source",
    ["src/tests/foo.npy", "src/tests/bar.npy"],
)
def test_invalid_inputs_from_reading_numpy_file(source: str):
    tested_none_type = DataLoader.from_numpy(source)
    assert tested_none_type is None


def test_general_exception_while_reading_npy(mocker):
    mock_file = "src/tests/test.npy"

    # Mocking the open function to raise a generic exception
    mocker.patch("builtins.open", side_effect=Exception("Mocked exception"))

    result = DataLoader.from_numpy(mock_file)
    assert result is None


def test_values_within_reader_object():
    existing_npy_file = DataLoader.from_numpy("src/tests/test.npy")

    existing_content = numpy.array(existing_npy_file.__dict__.get("data"))
    expected_content = numpy.array([[1, 1], [2, 1]])

    assert numpy.array_equal(existing_content, expected_content)
