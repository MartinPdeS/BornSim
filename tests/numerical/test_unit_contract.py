from bornsim import Grid
import numpy as np
import pytest
from TypedUnit import ureg as typedunit_registry
from bornsim import Result, Source
from bornsim.units import Length, ureg, validate_units


def test_shared_registry_and_scalar_source():
    assert ureg is typedunit_registry

    source = Source(wavelength=633 * ureg.nanometer)

    assert isinstance(source.wavelength, Length)

    assert source.wavelength.to("nanometer").magnitude == pytest.approx(633)

    np.testing.assert_allclose(
        Source(wavelength=6.33e-07 * ureg.meter).wavelength.to("nanometer").magnitude, source.wavelength.magnitude
    )

    with pytest.raises(ValueError, match="scalar"):
        Source(wavelength=[500, 600] * ureg.nanometer)


@pytest.mark.parametrize(
    "operation, name",
    [
        (lambda: Source(wavelength=1 * ureg.second), "wavelength"),
        (
            lambda: Result(
                source=Source(wavelength=6.33e-07 * ureg.meter),
                kind="volume",
                differential=np.ones((1, 2)) * ureg.meter,
                azimuth_averaged=True,
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
    from bornsim import Layer, GaussianMedium, Rotation

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

    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=length,
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
