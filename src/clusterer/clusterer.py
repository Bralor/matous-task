import json
import pathlib
from typing import Any
from typing_extensions import Self  # <3.11

import numpy as np
import numpy.typing as npt


def main() -> int:
    """
    Dummy function for the setup.

    Returns:
        int: A dummy value.
    """
    return 1


class DataLoader:
    """
    Returns the content of the data source.
    """

    # PEP-0484 --v
    def __init__(self, data: npt.NDArray[Any], filename: str) -> None:
        self.data: npt.NDArray[Any] = data
        self.filename = filename

    @classmethod
    def from_numpy(cls, source: str, suffix: str = ".npy") -> Self:
        """
        Create a DataLoader instance from a file with .NPY suffix.

        Args:
            source (str): Path to the numpy file to be loaded.
            suffix (str): Extension of the supported file. Defaults to '.npy'.

        Returns:
            DataLoader: An instance of DataLoader with loaded JSON content.

        Raises:
            ValueError: If the file suffix is invalid.
            FileNotFoundError: If the file does not exist.
            OSError: If the file is not readable or is not in a valid format.
        """
        if not pathlib.Path(source).suffix == suffix:
            raise ValueError("Input file does not have"
                             f" the correct suffix: {suffix}")

        try:
            file_content = np.load(source)

        except FileNotFoundError as err:
            raise FileNotFoundError(f"File not found: {source}") from err
        except OSError as err:
            raise OSError("Unable to read file"
                          f" or invalid file format: {source}") from err
        else:
            return cls(data=file_content, filename=source)

    @classmethod
    def from_json(cls, source: str, suffix: str = ".json") -> Self:
        """
        Create a DataLoader instance from a JSON file.

        Args:
            source (str): Path to the JSON file.
            suffix (str): Extension of the supported file. Defaults to '.json'.

        Returns:
            DataLoader: An instance of DataLoader with loaded JSON content.

        Raises:
            ValueError: If the file suffix is invalid.
            FileNotFoundError: If the file does not exist.
            IOError: If there is an issue while reading the file.
        """
        if not pathlib.Path(source).suffix == suffix:
            raise ValueError("Input file does not have"
                             f" the correct suffix: {suffix}")

        try:
            with open(source, "r") as file:
                data = json.load(file)

        except FileNotFoundError as err:
            raise FileNotFoundError(f"File not found: {source}") from err
        except json.JSONDecodeError as err:
            raise IOError(f"Invalid JSON format in file: {source}") from err
        else:
            return cls(data=np.array(data), filename=source)
