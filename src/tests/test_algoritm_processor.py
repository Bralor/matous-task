from typing import Dict, NamedTuple, Union

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.datasets import make_blobs

from clusterer.clusterer import KMeansProcessor

KwargsDateTypes = Dict[str, Union[int, str]]
ExpectedValuesDateTypes = Dict[str, Union[int, str]]


class SampleData(NamedTuple):
    """Immutable container for sample data."""
    data: npt.NDArray[np.float64]
    labels: npt.NDArray[np.int32]


@pytest.fixture
def sample_data() -> SampleData:
    """Fixture to generate sample data for clustering.

    Returns:
        SampleData: NamedTuple containing the generated data and labels.
    """
    data, labels = make_blobs(n_samples=100, centers=3, random_state=42)
    return SampleData(data=data, labels=labels)


@pytest.mark.parametrize(
    "n_clusters, random_state",
    [
        (3, 42),
        (4, 123),
        (2, 0),
    ],
)
def test_kmeans_processor_initialization(n_clusters: int, random_state: int):
    """Test initialization with default parameters."""
    testing_processor = KMeansProcessor(n_clusters=n_clusters,
                                        random_state=random_state)

    assert testing_processor.model.n_clusters == n_clusters
    assert testing_processor.model.random_state == random_state


@pytest.mark.parametrize(
    "n_clusters",
    [2, 3, 5, 7]
)
def test_kmeans_processor_fit_method(sample_data: SampleData, n_clusters: int):
    """Test fitting the KMeans model with different numbers of clusters."""
    testing_data, _ = sample_data
    testing_processor = KMeansProcessor(n_clusters=n_clusters, random_state=42)
    testing_predictions = testing_processor.fit(testing_data)

    # Predictions
    assert len(testing_predictions) == len(testing_data)
    assert np.issubdtype(testing_predictions.dtype, np.integer)

    # Check number of clusters
    unique_clusters = np.unique(testing_predictions)
    assert len(unique_clusters) == n_clusters


def test_kmeans_fit_method_with_empty_data():
    testing_processor = KMeansProcessor(n_clusters=3, random_state=42)
    testing_empty_data = np.array([]).reshape(0, 2)

    with pytest.raises(ValueError):
        testing_processor.fit(testing_empty_data)


@pytest.mark.parametrize(
    "kwargs, expected_values",
    [
        ({}, {"n_clusters": 8, "init": "k-means++", "max_iter": 300}),
        ({"n_clusters": 4}, {"n_clusters": 4,
                             "init": "k-means++",
                             "max_iter": 300}),
        ({"n_clusters": 3, "init": "random", "max_iter": 500},
         {"n_clusters": 3, "init": "random", "max_iter": 500}),
        ({"random_state": 42},
         {"n_clusters": 8,
          "init": "k-means++",
          "max_iter": 300,
          "random_state": 42}),
    ],
)
def test_initialization_kmeans_with_kwargs(
        kwargs: KwargsDateTypes,
        expected_values: ExpectedValuesDateTypes
):
    """Test initialization with various combinations of kwargs."""
    processor = KMeansProcessor(**kwargs)

    for key, expected_value in expected_values.items():
        assert getattr(processor.model, key) == expected_value


def test_kmeans_fit_result_consistency(sample_data: SampleData):
    """Test consistency of fit method with model labels_."""
    testing_data = sample_data.data
    testing_processor = KMeansProcessor(n_clusters=3, random_state=42)
    testing_predictions = testing_processor.fit(testing_data)

    # Check the labels
    assert np.array_equal(testing_predictions, testing_processor.model.labels_)
