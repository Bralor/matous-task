import json
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans  # type: ignore
from typing_extensions import Self  # <3.11


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


class ClusterBaseProcessor(ABC):
    """
    Abstract base class for clustering algorithms.

    This class defines a common interface for clustering algorithms, including
    methods:
    - for initialization,
    - fitting data,

    All clustering processors must inherit from this class and implement
    the required methods.
    """

    def __init__(self, **kwargs) -> None:
        self.params = kwargs

    @abstractmethod
    def fit(self, data):
        """
        Fits the clustering algorithm to the given data.

        Args:
            data (array-like): The input data to fit the clustering algorithm.

        Returns:
            self: The fitted clustering processor instance.
        """
        pass


class KMeansProcessor(ClusterBaseProcessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = KMeans(**kwargs)

    def fit(self, data):
        return self.model.fit_predict(data)
