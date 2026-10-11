"""Stage 5 tests for the guarded depth-model wrapper (torch optional)."""

import pytest

from foodmacros.env_checks import has_package
from foodmacros.weight import depth_model


def test_is_available_matches_torch_presence():
    assert depth_model.is_available() is has_package("torch")


@pytest.mark.skipif(has_package("torch"), reason="torch installed; would attempt a download")
def test_load_raises_clear_error_without_torch():
    with pytest.raises(RuntimeError, match="torch"):
        depth_model.load_depth_model("midas_small")


def test_predict_depth_raises_without_torch():
    if has_package("torch"):
        pytest.skip("torch installed")
    with pytest.raises(RuntimeError, match="torch"):
        depth_model.predict_depth(None, None)
