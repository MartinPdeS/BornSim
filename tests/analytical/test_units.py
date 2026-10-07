import numpy as np
import pytest
from TypedUnit import ureg as typedunit_registry

from bornsim import AnalyticalMedium, Result, Solver, Source
from bornsim.model import angular_scattering, optical_properties
from bornsim.units import Angle, Length, Quantity, ureg


def test_shared_registry_and_scalar_source():
    assert ureg is typedunit_registry
    source = Source(wavelength=633 * ureg.nanometer)
    assert isinstance(source.wavelength, Length)
    assert source.wavelength.to("nanometer").magnitude == pytest.approx(633)
    assert Source(wavelength=633e-9).wavelength == source.wavelength
    with pytest.raises(ValueError, match="scalar"):
        Source(wavelength=[500, 600] * ureg.nanometer)


def test_scaled_units_match_si_analytical_reference():
    medium = AnalyticalMedium(
        background_index=1.33 * ureg.refractive_index_units,
        index_std=1 * ureg.percent,
        correlation_length=0.1 * ureg.micrometer,
    )
    reference = AnalyticalMedium(
        background_index=1.33,
        index_std=0.01,
        correlation_length=100e-9,
    )
    theta = np.array([0, 45, 90, 180]) * ureg.degree
    result = Solver(source=Source(wavelength=0.633 * ureg.micrometer)).solve(
        target=medium,
        angles=theta,
    )
    assert isinstance(result.angles, Angle)
    assert isinstance(result.differential, Quantity)
    np.testing.assert_allclose(result.angles.to("degree").magnitude, theta.magnitude)
    expected = angular_scattering(
        medium=reference,
        wavelength=633e-9,
        theta=np.deg2rad(theta.magnitude),
    )
    np.testing.assert_allclose(result.azimuth_average().differential.to("1 / meter / steradian").magnitude[0], expected)
    np.testing.assert_allclose(
        result.azimuth_average().differential.to("1 / centimeter / steradian").magnitude[0], expected / 100
    )
    coefficients = optical_properties(
        medium=reference,
        wavelength=633e-9,
    )
    np.testing.assert_allclose(result.mu_s.to("1 / centimeter").magnitude, [coefficients["mu_s"] / 100])
    np.testing.assert_allclose(result.mu_s_prime.to("1 / meter").magnitude, [coefficients["mu_s_prime"]])
    np.testing.assert_allclose(result.g.to("dimensionless").magnitude, [coefficients["g"]])
    np.testing.assert_allclose(
        angular_scattering(
            medium=medium,
            wavelength=633 * ureg.nanometer,
            theta=theta,
        ),
        expected,
    )
    assert optical_properties(
        medium=medium,
        wavelength=633 * ureg.nanometer,
    ) == pytest.approx(coefficients)
    np.testing.assert_allclose(result.plot().axes[0].lines[0].get_xdata(), theta.magnitude)
    # Plotting converts quantities even if a user chooses different display units.
    result = Result(
        source=result.source,
        kind=result.kind,
        differential=result.differential.to("1 / centimeter / steradian"),
        angles=result.angles.to("degree"),
        azimuths=result.azimuths,
    )
    np.testing.assert_allclose(result.plot().axes[0].lines[0].get_ydata(), expected)


@pytest.mark.parametrize(
    "operation, name",
    [
        (lambda: Source(wavelength=1 * ureg.second), "wavelength"),
        (lambda: AnalyticalMedium(correlation_length=1 * ureg.second), "correlation_length"),
        (lambda: AnalyticalMedium(background_index=1 * ureg.meter), "background_index"),
        (lambda: AnalyticalMedium(index_std=1 * ureg.meter), "index_std"),
        (
            lambda: Solver(source=Source()).solve(
                target=AnalyticalMedium(),
                angles=[1] * ureg.meter,
            ),
            "angles",
        ),
        (
            lambda: angular_scattering(
                medium=AnalyticalMedium(),
                wavelength=633e-9,
                theta=[1] * ureg.meter,
            ),
            "theta",
        ),
        (
            lambda: optical_properties(
                medium=AnalyticalMedium(),
                wavelength=1 * ureg.second,
            ),
            "wavelength",
        ),
        (
            lambda: Result(
                source=Source(),
                kind="analytical",
                differential=np.ones((1, 2)) * ureg.meter,
            ),
            "differential",
        ),
    ],
)
def test_incompatible_dimensions_are_rejected(operation, name):
    with pytest.raises(ValueError, match=name):
        operation()
