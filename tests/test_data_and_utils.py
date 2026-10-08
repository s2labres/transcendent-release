import os
import tempfile
import numpy as np
import pytest

from transcend import data, utils


def test_cache_and_load_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "subdir", "test_cache.p")
        dummy_data = {"key": [1, 2, 3], "arr": np.array([0.1, 0.2])}

        data.cache_data(dummy_data, cache_path)
        assert os.path.exists(cache_path)

        loaded_data = data.load_cached_data(cache_path)
        assert loaded_data["key"] == dummy_data["key"]
        assert np.allclose(loaded_data["arr"], dummy_data["arr"])


def test_configure_logger():
    # Should run without error
    utils.configure_logger()
