import json
import pathlib
from typing import Optional, Any
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

    def __init__(self, data: npt.NDArray[Any], filename: str) -> None:  # PEP-0484
        self.data: npt.NDArray[Any] = data
        self.filename = filename

    @classmethod
    def from_numpy(cls, source: str, suffix: str = ".npy") -> Optional[Self]:
        """
        Create a DataLoader instance from a file with .NPY suffix.

        Args:
            source (str): Path to the numpy file to be loaded.
            suffix (str): Extension of the supported file. Defaults to '.npy'.

        Returns:
            DataLoader: An instance of DataLoader with loaded numpy content.
        """
        if not pathlib.Path(source).suffix == suffix:
            raise Exception("Input file does not have correct suffix.")

        try:
            file_content = np.load(source)

        except FileNotFoundError as err:
            print(repr(err))
            return None
        except Exception as err:
            print(f"Issue while reading .NPY: {source}. {repr(err)}")
            return None
        else:
            return cls(data=file_content, filename=source)

    @classmethod
    def from_json(cls, source: str, suffix: str = ".json") -> Optional[Self]:
        """
        Create a DataLoader instance from a JSON file.

        Args:
            source (str): Path to the JSON file.
            suffix (str): Extension of the supported file. Defaults to '.json'.

        Returns:
            DataLoader: An instance of DataLoader with loaded .json content.
        """
        if not pathlib.Path(source).suffix == suffix:
            raise Exception("Input file does not have correct suffix.")

        try:
            with open(source, "r") as file:
                data = json.load(file)

        except FileNotFoundError as err:
            print(repr(err))
            return None
        except Exception as err:
            print(f"Issue while reading .JSON: {source}. {repr(err)}")
            return None
        else:
            return cls(data=np.array(data), filename=source)
