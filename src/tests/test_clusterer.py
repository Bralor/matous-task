import json
from typing import Dict
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from clusterer.clusterer import DataLoader


@pytest.fixture
def testing_sample() -> npt.NDArray[np.int64]:
    """
    Create a testing sample.

    Returns:
        npt.NDArray: The two-dimensional array as a testing sample.
    """
    return np.array([[1, 2], [2, 3], [1, 1], [8, 8], [9, 9], [10, 10]])


@pytest.fixture
def testing_config() -> Dict[str, int]:
    """
    Create an object with testing hyperparameters.

    Returns:
        Dict[str, int]: The object with the boundary conditions
    """
    return {"n_clusters": 3, "n_init": 10, "max_iter": 300, "random_state": 42}


@pytest.mark.parametrize(
    "source",
    ["src/tests/foo.txt", "src/tests/test.pdf", "src/tests/sample.xlsx"],
)
def test_file_with_incorrect_suffix_instead_of_json(source: str):
    with pytest.raises(ValueError):
        DataLoader.from_json(source)


@pytest.mark.parametrize(
    "source",
    ["src/tests/foo.json", "src/tests/bar.json"],
)
def test_invalid_inputs_from_reading_json(source: str):
    with pytest.raises(FileNotFoundError):
        DataLoader.from_json(source)


def test_json_decode_exception_while_reading_json(mocker):
    mock_file = "src/tests/test.json"

    # Mock the np.load function to raise a generic exception
    with patch("builtins.open", side_effect=json.JSONDecodeError(
            "Mocked decode error", doc="", pos=0)):
        with pytest.raises(IOError,
                           match=f"Invalid JSON format in file: {mock_file}"):
            DataLoader.from_json(mock_file)


def test_existing_key_inside_json_file():
    existing_file = DataLoader.from_json("src/tests/test.json")

    existing_content = np.array(existing_file.data)
    expected_content = np.array([[1, 2], [3, 4]])

    assert np.array_equal(existing_content, expected_content)


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
    with pytest.raises(FileNotFoundError):
        DataLoader.from_numpy(source)


def test_general_exception_while_reading_npy(mocker):
    mock_file = "src/tests/test.npy"

    with patch("numpy.load", side_effect=OSError("Mocked exception")):
        with pytest.raises(OSError,
                           match="Unable to read file"
                                 f" or invalid file format: {mock_file}"):
            DataLoader.from_numpy(mock_file)


def test_values_within_reader_object():
    existing_npy_file = DataLoader.from_numpy("src/tests/test.npy")

    existing_content = np.array(existing_npy_file.data)
    expected_content = np.array([[1, 1], [2, 1]])

    assert np.array_equal(existing_content, expected_content)
