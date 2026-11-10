import numpy as np
import pytest
import torch

from latent_equivariance.learned import (
    EquivariantMLP, NetworkEnclosure, ball_samples, representatives, sine_signs,
)


def test_group_average_intertwines_full_and_latent_actions():
    torch.manual_seed(17)
    model = EquivariantMLP(6, 2, width=5)
    x = torch.randn(13, 6, dtype=torch.float64)
    for a, b in zip(sine_signs(6), sine_signs(2)):
        actual = model(x * torch.tensor(a))
        expected = model(x) * torch.tensor(b)
        torch.testing.assert_close(actual, expected, atol=2e-15, rtol=2e-14)
    with pytest.raises(TypeError):
        model(x.float())


def test_monotone_scalar_network_encloses_analytic_range():
    model = EquivariantMLP(1, 1, width=1)
    with torch.no_grad():
        for layer, weight in zip((model.net[0], model.net[2], model.net[4]), (2., .5, -3.)):
            layer.weight.fill_(weight)
            layer.bias.zero_()
    enclosure = NetworkEnclosure.from_model(model)
    lo = np.array([[-2.], [.2], [-.4]])
    hi = np.array([[2.], [.5], [-.4]])
    a, b = enclosure.enclose(lo, hi)
    exact_min = -3 * np.tanh(.5 * np.tanh(2 * hi))
    exact_max = -3 * np.tanh(.5 * np.tanh(2 * lo))
    np.testing.assert_allclose(a, exact_min, atol=1e-14)
    np.testing.assert_allclose(b, exact_max, atol=1e-14)


def test_network_box_regression_with_signed_weights_and_biases():
    torch.manual_seed(9)
    model = EquivariantMLP(2, 3, width=7)
    enclosure = NetworkEnclosure.from_model(model)
    rng = np.random.default_rng(21)
    lo = rng.uniform(-2, 1, size=(7, 2))
    hi = lo + rng.uniform(.001, .8, size=(7, 2))
    a, b = enclosure.enclose(lo, hi)
    # This is a regression check, not validation of floating-point bounds.
    for _ in range(30):
        x = rng.uniform(lo, hi)
        actual = model(torch.tensor(x)).detach().numpy()
        np.testing.assert_allclose(actual, enclosure(x), atol=2e-15)
        assert np.all(actual >= a - 1e-14)
        assert np.all(actual <= b + 1e-14)
    with pytest.raises(ValueError):
        enclosure.enclose(hi, lo)


def test_orbit_representatives_and_registered_sample_strata():
    x = np.array([[0., -2., -1., .5], [1., 0., -1., 0.]])
    for sign in sine_signs(4):
        np.testing.assert_array_equal(representatives(x * sign), representatives(x))
    samples, strata = ball_samples(100, 32, 1.5, 1000)
    norm = np.linalg.norm(samples, axis=1)
    assert np.all(norm[strata == 0] < 1.5)
    np.testing.assert_allclose(norm[strata == 1], 1.5, atol=1e-15)
    np.testing.assert_array_equal(samples, ball_samples(100, 32, 1.5, 1000)[0])

