import numpy as np
import pytest
from bornsim import AnalyticalMedium
from bornsim.model import angular_scattering, optical_properties
from bornsim.units import ureg


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_quadratic_contrast_scaling(correlation):
    low = optical_properties(
        medium=AnalyticalMedium(
            refractive_index_std=0.01,
            correlation=correlation,
            background_refractive_index=1.33,
            correlation_length=100e-9 * ureg.meter,
            smoothness=1.5,
        ),
        wavelength=6.33e-07 * ureg.meter,
    )

    high = optical_properties(
        medium=AnalyticalMedium(
            refractive_index_std=0.02,
            correlation=correlation,
            background_refractive_index=1.33,
            correlation_length=100e-9 * ureg.meter,
            smoothness=1.5,
        ),
        wavelength=6.33e-07 * ureg.meter,
    )

    assert high["mu_s"].to("1 / meter").magnitude == pytest.approx(4 * low["mu_s"].to("1 / meter").magnitude)

    assert high["g"] == pytest.approx(low["g"])

    assert low["mu_s_prime"].to("1 / meter").magnitude == pytest.approx(
        low["mu_s"].to("1 / meter").magnitude * (1 - low["g"])
    )


def test_small_correlation_length_analytic_limit():
    medium = AnalyticalMedium(
        correlation_length=1e-12 * ureg.meter,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation="gaussian",
        smoothness=1.5,
    )

    k0 = 2 * np.pi / 633e-9

    spectrum = (
        (2 * medium.background_refractive_index * medium.refractive_index_std) ** 2
        * (2 * np.pi) ** 1.5
        * medium.correlation_length.to("meter").magnitude ** 3
    )

    expected = k0**4 * spectrum / (6 * np.pi)

    result = optical_properties(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
    )

    assert result["mu_s"].to("1 / meter").magnitude == pytest.approx(expected, rel=1e-7)

    assert result["g"] == pytest.approx(0, abs=1e-8)


def test_zero_contrast():
    assert optical_properties(
        medium=AnalyticalMedium(
            refractive_index_std=0,
            background_refractive_index=1.33,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        ),
        wavelength=6.33e-07 * ureg.meter,
    ) == {"mu_s": 0 * (1 / ureg.meter), "g": None, "mu_s_prime": 0 * (1 / ureg.meter)}


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_independent_angular_integration(correlation):
    medium = AnalyticalMedium(
        correlation_length=2.5e-07 * ureg.meter,
        correlation=correlation,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        smoothness=1.5,
    )

    theta = np.linspace(0, np.pi, 100001)

    intensity = (
        angular_scattering(
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

    result = optical_properties(
        medium=medium,
        wavelength=6.33e-07 * ureg.meter,
    )

    assert result["mu_s"].to("1 / meter").magnitude == pytest.approx(reference, rel=1e-7)

    assert 0 < result["g"] < 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"background_refractive_index": 0},
        {"refractive_index_std": -1},
        {"correlation_length": float("nan") * ureg.meter},
        {"correlation": "unknown"},
    ],
)
def test_invalid_medium(kwargs):
    with pytest.raises(ValueError):
        AnalyticalMedium(
            **{
                "background_refractive_index": 1.33,
                "refractive_index_std": 0.01,
                "correlation_length": 100e-9 * ureg.meter,
                "correlation": "gaussian",
                "smoothness": 1.5,
                **kwargs,
            }
        )


def test_invalid_wavelength_and_angle():
    with pytest.raises(ValueError):
        angular_scattering(
            medium=AnalyticalMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            wavelength=0 * ureg.meter,
            theta=[0] * ureg.radian,
        )

    with pytest.raises(ValueError):
        angular_scattering(
            medium=AnalyticalMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            wavelength=6.33e-07 * ureg.meter,
            theta=[-1] * ureg.radian,
        )
