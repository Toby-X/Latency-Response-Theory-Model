import numpy as np
import pytest

from lart import fit_lart, generate_lart_data


def test_generator_shapes_and_support():
    data = generate_lart_data(n_models=12, n_items=7, seed=4)
    assert data.response.shape == (12, 7)
    assert data.latency.shape == (12, 7)
    assert set(np.unique(data.response)) <= {0, 1}
    assert np.all(data.latency > 0)


def test_fit_rejects_nonpositive_latency():
    response = np.array([[0, 1], [1, 0]])
    latency = np.array([[2.0, 0.0], [1.0, 4.0]])
    with pytest.raises(ValueError, match="strictly positive"):
        fit_lart(response, latency)
