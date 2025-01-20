import pytest

from src.clusterer.clusterer import main


def test_dummy():
    assert main() == 1


def test_reverse_dummy():
    with pytest.raises(AssertionError):
        assert main() == 0
