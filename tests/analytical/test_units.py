import numpy as np
import pytest
from TypedUnit import ureg as typedunit_registry

from bornsim import AnalyticalMedium, Result, Solver, Source
from bornsim.model import angular_scattering, optical_properties
from bornsim.units import Angle, Length, Quantity, ureg, validate_units


def test_shared_registry_and_scalar_source():
    assert ureg is typedunit_registry

    source = Source(wavelength=633 * ureg.nanometer)

    assert isinstance(source.wavelength, Length)

    assert source.wavelength.to("nanometer").magnitude == pytest.approx(633)

    np.testing.assert_allclose(
        Source(wavelength=6.33e-07 * ureg.meter).wavelength.to("nanometer").magnitude,
        source.wavelength.magnitude,
    )

    with pytest.raises(ValueError, match="scalar"):
        Source(wavelength=[500, 600] * ureg.nanometer)


def test_scaled_units_match_si_analytical_reference():
    medium = AnalyticalMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=0.1 * ureg.micrometer,
        correlation="gaussian",
        smoothness=1.5,
    )

    reference = AnalyticalMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
        correlation="gaussian",
        smoothness=1.5,
    )

    theta = np.array([0, 45, 90, 180]) * ureg.degree

    result = Solver(source=Source(wavelength=0.633 * ureg.micrometer)).solve(
        target=medium,
        angles=theta * ureg.radian,
    )

    assert isinstance(result.angles, Angle)

    assert isinstance(result.differential, Quantity)

    np.testing.assert_allclose(result.angles.to("degree").magnitude, theta.magnitude)

    expected = angular_scattering(
        medium=reference,
        wavelength=6.33e-07 * ureg.meter,
        theta=np.deg2rad(theta.magnitude) * ureg.radian,
    )

    np.testing.assert_allclose(result.azimuth_average().differential.to("1 / meter / steradian").magnitude[0], expected)

    np.testing.assert_allclose(
        result.azimuth_average().differential.to("1 / centimeter / steradian").magnitude[0], expected / 100
    )

    coefficients = optical_properties(
        medium=reference,
        wavelength=6.33e-07 * ureg.meter,
    )

    np.testing.assert_allclose(
        result.mu_s.to("1 / centimeter").magnitude, [coefficients["mu_s"].to("1 / centimeter").magnitude]
    )

    np.testing.assert_allclose(
        result.mu_s_prime.to("1 / meter").magnitude, [coefficients["mu_s_prime"].to("1 / meter").magnitude]
    )

    np.testing.assert_allclose(result.g.to("dimensionless").magnitude, [coefficients["g"]])

    np.testing.assert_allclose(
        angular_scattering(
            medium=medium,
            wavelength=633 * ureg.nanometer,
            theta=theta * ureg.radian,
        )
        .to(expected.units)
        .magnitude,
        expected.magnitude,
    )

    physical_coefficients = optical_properties(
        medium=medium,
        wavelength=633 * ureg.nanometer,
    )

    for name in ("mu_s", "mu_s_prime"):
        np.testing.assert_allclose(
            physical_coefficients[name].to(coefficients[name].units).magnitude,
            coefficients[name].magnitude,
        )

    assert physical_coefficients["g"] == pytest.approx(coefficients["g"])

    np.testing.assert_allclose(result.plot().axes[0].lines[0].get_xdata(), theta.magnitude)

    # Plotting converts quantities even if a user chooses different display units.
    result = Result(
        source=result.source,
        kind=result.kind,
        differential=result.differential.to("1 / centimeter / steradian").to("1 / meter / steradian"),
        angles=result.angles.to("degree").to("radian"),
        azimuths=result.azimuths.to("radian"),
    )

    np.testing.assert_allclose(result.plot().axes[0].lines[0].get_ydata(), expected)


@pytest.mark.parametrize(
    "operation, name",
    [
        (lambda: Source(wavelength=1 * ureg.second), "wavelength"),
        (
            lambda: AnalyticalMedium(
                correlation_length=1 * ureg.second,
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation="gaussian",
                smoothness=1.5,
            ),
            "correlation_length",
        ),
        (
            lambda: AnalyticalMedium(
                background_refractive_index=1 * ureg.meter,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            "background_refractive_index",
        ),
        (
            lambda: AnalyticalMedium(
                refractive_index_std=1 * ureg.meter,
                background_refractive_index=1.33,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            "refractive_index_std",
        ),
        (
            lambda: Solver(
                source=Source(
                    wavelength=633e-9 * ureg.meter,
                )
            ).solve(
                target=AnalyticalMedium(
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=100e-9 * ureg.meter,
                    correlation="gaussian",
                    smoothness=1.5,
                ),
                angles=[1] * ureg.meter,
            ),
            "angles",
        ),
        (
            lambda: angular_scattering(
                medium=AnalyticalMedium(
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=100e-9 * ureg.meter,
                    correlation="gaussian",
                    smoothness=1.5,
                ),
                wavelength=6.33e-07 * ureg.meter,
                theta=[1] * ureg.meter,
            ),
            "theta",
        ),
        (
            lambda: optical_properties(
                medium=AnalyticalMedium(
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=100e-9 * ureg.meter,
                    correlation="gaussian",
                    smoothness=1.5,
                ),
                wavelength=1 * ureg.second,
            ),
            "wavelength",
        ),
        (
            lambda: Result(
                source=Source(
                    wavelength=633e-9 * ureg.meter,
                ),
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


@pytest.mark.parametrize(
    "value, unit, scalar, message",
    [
        (2, "meter", True, "units"),
        (2 * ureg.second, "meter", True, "compatible"),
        ([1, 2] * ureg.meter, "meter", True, "scalar"),
        (1 * ureg.dimensionless, "radian", True, "angular units"),
    ],
)
def test_validate_units_rejects_invalid_inputs(value, unit, scalar, message):
    with pytest.raises(ValueError, match=message):
        validate_units(
            value,
            unit=unit,
            name="physical_value",
            scalar=scalar,
        )


def test_validation_preserves_supplied_quantities():
    from bornsim import Grid, Layer, RandomMedium, Rotation

    length = 60 * ureg.nanometer

    assert (
        validate_units(
            length,
            unit="meter",
            name="length",
            scalar=True,
        )
        is None
    )

    source = Source(wavelength=length)

    grid = Grid(
        shape=(4, 4, 4),
        spacing=length,
    )

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=length,
        correlation="gaussian",
    )

    layer = Layer(
        lower=0 * ureg.nanometer,
        upper=length,
        refractive_index=1.4,
    )

    angle = 30 * ureg.degree

    rotation = Rotation(
        axis=(0, 0, 1),
        angle=angle,
    )

    assert source.wavelength is length

    assert grid.spacing is length

    assert medium.correlation_length is length

    assert layer.upper is length

    assert rotation.angle is angle

    assert length.units == ureg.nanometer
