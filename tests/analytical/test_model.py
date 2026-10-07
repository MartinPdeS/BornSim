import numpy as np
import pytest
from bornsim import AnalyticalMedium
from bornsim.model import angular_scattering, optical_properties


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_quadratic_contrast_scaling(correlation):
    low = optical_properties(
        medium=AnalyticalMedium(
            index_std=0.01,
            correlation=correlation,
        ),
        wavelength=633e-9,
    )
    high = optical_properties(
        medium=AnalyticalMedium(
            index_std=0.02,
            correlation=correlation,
        ),
        wavelength=633e-9,
    )
    assert high["mu_s"] == pytest.approx(4 * low["mu_s"])
    assert high["g"] == pytest.approx(low["g"])
    assert low["mu_s_prime"] == pytest.approx(low["mu_s"] * (1 - low["g"]))


def test_small_correlation_length_analytic_limit():
    medium = AnalyticalMedium(correlation_length=1e-12)
    k0 = 2 * np.pi / 633e-9
    spectrum = (2 * medium.background_index * medium.index_std) ** 2 * (2 * np.pi) ** 1.5 * medium.correlation_length**3
    expected = k0**4 * spectrum / (6 * np.pi)
    result = optical_properties(
        medium=medium,
        wavelength=633e-9,
    )
    assert result["mu_s"] == pytest.approx(expected, rel=1e-7)
    assert result["g"] == pytest.approx(0, abs=1e-8)


def test_zero_contrast():
    assert optical_properties(
        medium=AnalyticalMedium(index_std=0),
        wavelength=633e-9,
    ) == {"mu_s": 0, "g": None, "mu_s_prime": 0}


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_independent_angular_integration(correlation):
    medium = AnalyticalMedium(
        correlation_length=250e-9,
        correlation=correlation,
    )
    theta = np.linspace(0, np.pi, 100001)
    intensity = angular_scattering(
        medium=medium,
        wavelength=633e-9,
        theta=theta,
    )
    reference = (
        2
        * np.pi
        * np.sum((intensity[1:] * np.sin(theta[1:]) + intensity[:-1] * np.sin(theta[:-1])) * np.diff(theta) / 2)
    )
    result = optical_properties(
        medium=medium,
        wavelength=633e-9,
    )
    assert result["mu_s"] == pytest.approx(reference, rel=1e-7)
    assert 0 < result["g"] < 1


@pytest.mark.parametrize(
    "kwargs",
    [{"background_index": 0}, {"index_std": -1}, {"correlation_length": float("nan")}, {"correlation": "unknown"}],
)
def test_invalid_medium(kwargs):
    with pytest.raises(ValueError):
        AnalyticalMedium(**kwargs)


def test_invalid_wavelength_and_angle():
    with pytest.raises(ValueError):
        angular_scattering(
            medium=AnalyticalMedium(),
            wavelength=0,
            theta=[0],
        )
    with pytest.raises(ValueError):
        angular_scattering(
            medium=AnalyticalMedium(),
            wavelength=633e-9,
            theta=[-1],
        )
