from bornsim.medium.random_medium import GaussianMedium, ExponentialMedium
import numpy as np
import pytest
from .reference import infinite_medium_scattering, infinite_medium_properties
from bornsim.units import ureg


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_quadratic_contrast_scaling(correlation):
    low = infinite_medium_properties(
        medium={
            "gaussian": GaussianMedium,
            "exponential": ExponentialMedium,
        }[correlation](
            refractive_index_std=0.01,
            background_refractive_index=1.33,
            correlation_length=1e-07 * ureg.meter,
        ),
        wavelength=6.33e-07 * ureg.meter,
    )

    high = infinite_medium_properties(
        medium={
            "gaussian": GaussianMedium,
            "exponential": ExponentialMedium,
        }[correlation](
            refractive_index_std=0.02,
            background_refractive_index=1.33,
            correlation_length=1e-07 * ureg.meter,
        ),
        wavelength=6.33e-07 * ureg.meter,
    )

    assert high["mu_s"].to("1 / meter").magnitude == pytest.approx(4 * low["mu_s"].to("1 / meter").magnitude)

    assert high["g"] == pytest.approx(low["g"])

    assert low["mu_s_prime"].to("1 / meter").magnitude == pytest.approx(
        low["mu_s"].to("1 / meter").magnitude * (1 - low["g"])
    )


def test_small_correlation_length_analytic_limit():
    medium = GaussianMedium(
        correlation_length=1e-12 * ureg.meter,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
    )

    k0 = 2 * np.pi / 6.33e-07

    spectrum = (
        (2 * medium.background_refractive_index * medium.refractive_index_std) ** 2
        * (2 * np.pi) ** 1.5
        * medium.correlation_length.to("meter").magnitude ** 3
    )

    expected = k0**4 * spectrum / (6 * np.pi)

    result = infinite_medium_properties(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
    )

    assert result["mu_s"].to("1 / meter").magnitude == pytest.approx(expected, rel=1e-07)

    assert result["g"] == pytest.approx(0, abs=1e-08)


def test_zero_contrast():
    assert infinite_medium_properties(
        medium=GaussianMedium(
            refractive_index_std=0,
            background_refractive_index=1.33,
            correlation_length=1e-07 * ureg.meter,
        ),
        wavelength=6.33e-07 * ureg.meter,
    ) == {"mu_s": 0 * (1 / ureg.meter), "g": None, "mu_s_prime": 0 * (1 / ureg.meter)}


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_independent_angular_integration(correlation):
    medium = {
        "gaussian": GaussianMedium,
        "exponential": ExponentialMedium,
    }[correlation](
        correlation_length=2.5e-07 * ureg.meter,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
    )

    theta = np.linspace(0, np.pi, 100001)

    intensity = (
        infinite_medium_scattering(
            medium=medium,
            wavelength=6.33e-07 * ureg.meter,
            theta=theta * ureg.radian,
        )
        .to("1 / meter / steradian")
        .magnitude
    )

    reference = (
        2
        * np.pi
        * np.sum((intensity[1:] * np.sin(theta[1:]) + intensity[:-1] * np.sin(theta[:-1])) * np.diff(theta) / 2)
    )

    result = infinite_medium_properties(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
    )

    assert result["mu_s"].to("1 / meter").magnitude == pytest.approx(reference, rel=1e-07)

    assert 0 < result["g"] < 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"background_refractive_index": 0},
        {"refractive_index_std": -1},
        {"correlation_length": float("nan") * ureg.meter},
    ],
)
def test_invalid_medium(kwargs):
    with pytest.raises(ValueError):
        GaussianMedium(
            **{
                "background_refractive_index": 1.33,
                "refractive_index_std": 0.01,
                "correlation_length": 1e-07 * ureg.meter,
                **kwargs,
            }
        )


def test_invalid_wavelength_and_angle():
    with pytest.raises(ValueError):
        infinite_medium_scattering(
            medium=GaussianMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=1e-07 * ureg.meter,
            ),
            wavelength=0 * ureg.meter,
            theta=[0] * ureg.radian,
        )

    with pytest.raises(ValueError):
        infinite_medium_scattering(
            medium=GaussianMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=1e-07 * ureg.meter,
            ),
            wavelength=6.33e-07 * ureg.meter,
            theta=[-1] * ureg.radian,
        )
