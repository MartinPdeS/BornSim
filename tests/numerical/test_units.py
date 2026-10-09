import numpy as np
import pytest

from bornsim import (
    AngularData,
    AngularSampling,
    AnalyticalMedium,
    Box,
    Cylinder,
    Grid,
    Layer,
    Rotation,
    Sphere,
    RandomMedium,
    Result,
    Solver,
    Source,
    Volume,
)
from bornsim.series import BornSeries
from bornsim.green import GreenOperator
from bornsim.ensemble import ensemble_scattering
from bornsim.media import random_volume
from bornsim.units import ureg
from bornsim import Directions


def test_unitful_volume_matches_si_and_retains_complex_amplitudes():
    medium = RandomMedium(
        correlation="gaussian",
        correlation_length=100 * ureg.nanometer,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        smoothness=1.5,
    )

    volume = random_volume(
        medium=medium,
        shape=(2, 2, 2),
        spacing=50 * ureg.nanometer,
        seed=42,
    )

    reference = random_volume(
        medium=RandomMedium(
            correlation="gaussian",
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            smoothness=1.5,
        ),
        shape=(2, 2, 2),
        spacing=5e-08 * ureg.meter,
        seed=42,
    )

    np.testing.assert_allclose(volume.delta_refractive_index, reference.delta_refractive_index)

    explicit = Volume(
        delta_refractive_index=volume.delta_refractive_index,
        spacing=0.05 * ureg.micrometer,
        background_refractive_index=1.33,
    )

    np.testing.assert_allclose(explicit.positions.to("meter").magnitude, reference.positions.to("meter").magnitude)

    directions = np.array([[0, 0, 1], [1, 0, 0]]) * ureg.dimensionless

    result = Solver(
        source=Source(wavelength=633 * ureg.nanometer),
        order=2,
    ).solve_cut(
        target=volume,
        directions=Directions(vectors=directions),
    )

    expected = BornSeries(
        shape=reference.delta_refractive_index.shape,
        spacing=reference.spacing.to("meter"),
        background_refractive_index=reference.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions.magnitude),
        order=2,
    ).solve(volume=reference)

    np.testing.assert_allclose(result.amplitudes.to("meter").magnitude, expected.amplitudes)

    assert np.iscomplexobj(result.amplitudes.magnitude)

    assert np.any(result.amplitudes.magnitude.imag != 0)

    np.testing.assert_allclose(result.differential.to("1 / meter / steradian").magnitude, expected.differential)

    np.testing.assert_allclose(result.field_norms.to("dimensionless").magnitude, expected.field_norms)

    np.testing.assert_allclose(
        BornSeries(
            shape=volume.delta_refractive_index.shape,
            spacing=volume.spacing.to("meter"),
            background_refractive_index=volume.background_refractive_index,
            wavelength=633 * ureg.nanometer,
            directions=Directions(vectors=directions),
            order=2,
        )
        .solve(volume=volume)
        .differential,
        expected.differential,
    )


def test_unitful_ensemble_preserves_seed_and_standard_error():
    options = dict(shape=(2, 2, 2), realizations=3, seed=42, azimuth_samples=4, polar_samples=16)

    result = Solver(
        source=Source(wavelength=633 * ureg.nanometer),
        order=2,
    ).ensemble(
        medium=RandomMedium(
            correlation="gaussian",
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            smoothness=1.5,
        ),
        spacing=50 * ureg.nanometer,
        angles=[0, 90, 180] * ureg.degree,
        **{"shape": (12, 12, 12), **options},
    )

    expected = ensemble_scattering(
        medium=RandomMedium(
            correlation="gaussian",
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            smoothness=1.5,
        ),
        wavelength=6.33e-07 * ureg.meter,
        spacing=5e-08 * ureg.meter,
        order=2,
        angles=[0, np.pi / 2, np.pi] * ureg.radian,
        **{"shape": (12, 12, 12), **options},
    )

    np.testing.assert_allclose(result.differential.magnitude, expected["directional_differential"])

    np.testing.assert_allclose(result.stderr.to("1 / meter / steradian").magnitude, expected["directional_stderr"])

    segments = result.plot().axes[0].containers[0].lines[2][0].get_segments()

    errors = [(segment[1, 1] - segment[0, 1]) / 2 for segment in segments]

    np.testing.assert_allclose(errors, expected["directional_stderr"][0, :, 0])

    np.testing.assert_allclose(result.mu_s.to("1 / meter").magnitude, expected["mu_s"])

    numeric = ensemble_scattering(
        medium=RandomMedium(
            correlation="gaussian",
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            smoothness=1.5,
        ),
        wavelength=633 * ureg.nanometer,
        spacing=50 * ureg.nanometer,
        order=2,
        angles=[0, 90, 180] * ureg.degree,
        **{"shape": (12, 12, 12), **options},
    )

    np.testing.assert_allclose(numeric["mean"], expected["mean"])


@pytest.mark.parametrize(
    "operation, name",
    [
        (
            lambda: Volume(
                delta_refractive_index=np.zeros((2, 2, 2)),
                spacing=1 * ureg.second,
                background_refractive_index=1.33,
            ),
            "spacing",
        ),
        (
            lambda: Volume(
                delta_refractive_index=np.ones((2, 2, 2)) * ureg.meter,
                spacing=5e-08 * ureg.meter,
                background_refractive_index=1.33,
            ),
            "delta_refractive_index",
        ),
        (
            lambda: random_volume(
                medium=RandomMedium(
                    correlation="gaussian",
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=100e-9 * ureg.meter,
                    smoothness=1.5,
                ),
                spacing=1 * ureg.second,
                shape=(12, 12, 12),
            ),
            "spacing",
        ),
        (
            lambda: Solver(
                source=Source(
                    wavelength=633e-9 * ureg.meter,
                )
            ).ensemble(
                medium=RandomMedium(
                    correlation="gaussian",
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=100e-9 * ureg.meter,
                    smoothness=1.5,
                ),
                spacing=1 * ureg.second,
                shape=(12, 12, 12),
            ),
            "spacing",
        ),
        (
            lambda: Solver(
                source=Source(
                    wavelength=633e-9 * ureg.meter,
                )
            ).ensemble(
                medium=RandomMedium(
                    correlation="gaussian",
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=100e-9 * ureg.meter,
                    smoothness=1.5,
                ),
                angles=[0] * ureg.meter,
                shape=(12, 12, 12),
                spacing=50e-9 * ureg.meter,
            ),
            "angles",
        ),
        (
            lambda: BornSeries(
                shape=(2, 2, 2),
                spacing=5e-08 * ureg.meter,
                background_refractive_index=1.33,
                wavelength=1 * ureg.second,
                directions=Directions(vectors=[[0, 0, 1]]),
            ).solve(
                volume=Volume(
                    delta_refractive_index=np.zeros((2, 2, 2)),
                    spacing=5e-08 * ureg.meter,
                    background_refractive_index=1.33,
                )
            ),
            "wavelength",
        ),
        (
            lambda: Solver(
                source=Source(
                    wavelength=633e-9 * ureg.meter,
                )
            ).solve_cut(
                target=Volume(
                    delta_refractive_index=np.zeros((2, 2, 2)),
                    spacing=5e-08 * ureg.meter,
                    background_refractive_index=1.33,
                ),
                directions=Directions(vectors=[[0, 0, 1]] * ureg.meter),
            ),
            "directions",
        ),
    ],
)
def test_numerical_dimensions_are_checked(operation, name):
    with pytest.raises(ValueError, match=name):
        operation()


@pytest.mark.parametrize(
    "operation, parameter",
    [
        (lambda: Source(wavelength=633e-9), "wavelength"),
        (
            lambda: Grid(
                spacing=50e-9,
                shape=(12, 12, 12),
            ),
            "spacing",
        ),
        (
            lambda: RandomMedium(
                correlation_length=60e-9,
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation="matern",
                smoothness=1.5,
            ),
            "correlation_length",
        ),
        (
            lambda: AnalyticalMedium(
                correlation_length=60e-9,
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation="gaussian",
                smoothness=1.5,
            ),
            "correlation_length",
        ),
        (
            lambda: Volume(
                delta_refractive_index=np.zeros((2, 2, 2)),
                spacing=50e-9,
                background_refractive_index=1.33,
            ),
            "spacing",
        ),
        (lambda: Sphere(radius=50e-9, refractive_index=1.34), "radius"),
        (lambda: Box(size=(50e-9,) * 3, refractive_index=1.34), "size"),
        (lambda: Layer(lower=0, upper=50 * ureg.nanometer, refractive_index=1.34), "lower"),
        (lambda: Cylinder(radius=50 * ureg.nanometer, height=100e-9, refractive_index=1.34), "height"),
        (lambda: Sphere(radius=50 * ureg.nanometer, centre=(0, 0, 0), refractive_index=1.34), "centre"),
        (
            lambda: Rotation(
                angle=0,
                axis=(0.0, 0.0, 1.0),
            ),
            "angle",
        ),
        (lambda: AngularSampling(angles=[0, np.pi]), "angles"),
        (lambda: AngularSampling(start=0), "start"),
        (lambda: AngularSampling(end=np.pi), "end"),
        (
            lambda: RandomMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="matern",
                smoothness=1.5,
            ).spectral_weight(q_squared=np.array([0, 1e12])),
            "q_squared",
        ),
        (
            lambda: GreenOperator(
                shape=(2, 2, 2),
                spacing=50e-9,
                wavelength=633 * ureg.nanometer,
                background_refractive_index=1.33,
            ),
            "spacing",
        ),
        (
            lambda: BornSeries(
                grid=Grid(
                    shape=(12, 12, 12),
                    spacing=50e-9 * ureg.meter,
                ),
                wavelength=633e-9,
                background_refractive_index=1.33,
                directions=Directions(vectors=[[0, 0, 1]]),
            ),
            "wavelength",
        ),
        (
            lambda: AngularData(
                differential=[[1]],
                angles=[0] * ureg.radian,
            ),
            "differential",
        ),
        (
            lambda: Result(
                source=Source(
                    wavelength=633e-9 * ureg.meter,
                ),
                kind="analytical",
                differential=[[1]],
                angles=[0] * ureg.radian,
            ),
            "differential",
        ),
    ],
)
def test_dimensional_public_inputs_reject_bare_values(operation, parameter):
    with pytest.raises(ValueError, match=parameter + ".*explicit.*units"):
        operation()


def test_public_methods_reject_bare_physical_values():
    sphere = Sphere(
        radius=50 * ureg.nanometer,
        refractive_index=1.34,
    )

    for parameter, operation in (
        ("positions", lambda: sphere.mask(positions=np.zeros((1, 3)))),
        ("offset", lambda: sphere.translated(offset=(0, 0, 0))),
        (
            "about",
            lambda: sphere.rotated(
                rotation=Rotation(
                    axis=(0.0, 0.0, 1.0),
                    angle=0 * ureg.radian,
                ),
                about=(0, 0, 0),
            ),
        ),
    ):
        with pytest.raises(ValueError, match=parameter + ".*explicit.*units"):
            operation()

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    grid = Grid(
        shape=(2, 2, 2),
        spacing=50e-9 * ureg.meter,
    )

    volume = medium.to_volume(grid=grid)

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        )
    )

    for operation in (
        lambda: solver.solve(
            target=AnalyticalMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=100e-9 * ureg.meter,
                correlation="gaussian",
                smoothness=1.5,
            ),
            angles=[0],
        ),
        lambda: solver.solve_cut(target=volume, angles=[0]),
        lambda: solver.ensemble(
            medium=medium,
            spacing=50e-9,
            shape=(12, 12, 12),
        ),
    ):
        with pytest.raises(ValueError, match="requires an explicit quantity"):
            operation()

    result = solver.solve(
        target=AnalyticalMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="gaussian",
            smoothness=1.5,
        )
    )

    with pytest.raises(ValueError, match="azimuth.*explicit.*units"):
        result.meridian(azimuth=0)


def test_angles_require_angular_units_and_dimensionless_statistics_accept_numbers():
    with pytest.raises(ValueError, match="explicit angular units"):
        AngularSampling(start=0 * ureg.dimensionless)

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=60 * ureg.nanometer,
        smoothness=1.5,
        correlation="matern",
    )

    assert medium.correlation_length.to("meter").magnitude == pytest.approx(60e-9)

    np.testing.assert_array_equal(
        medium.spectral_weight(q_squared=np.array([0]) / ureg.meter**2),
        [1],
    )


@pytest.mark.parametrize("unit", ["dimensionless", "refractive_index_units", "percent"])
@pytest.mark.parametrize("parameter", ["background_refractive_index", "refractive_index_std"])
def test_refractive_index_statistics_reject_all_quantities(unit, parameter):
    with pytest.raises(ValueError, match=parameter + ".*without units"):
        RandomMedium(
            **{
                "background_refractive_index": 1.33,
                "refractive_index_std": 0.01,
                "correlation_length": 100e-9 * ureg.meter,
                "correlation": "matern",
                "smoothness": 1.5,
                **{parameter: 1 * ureg.Unit(unit)},
            }
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Sphere(radius=50 * ureg.nanometer, refractive_index=1.34 * ureg.dimensionless),
        lambda: Volume(
            delta_refractive_index=np.zeros((2, 2, 2)) * ureg.dimensionless,
            grid=Grid(
                shape=(2, 2, 2),
                spacing=50e-9 * ureg.meter,
            ),
            background_refractive_index=1.33,
        ),
        lambda: GreenOperator(
            shape=(2, 2, 2),
            spacing=50 * ureg.nanometer,
            wavelength=633 * ureg.nanometer,
            background_refractive_index=1.33 * ureg.dimensionless,
        ),
    ],
)
def test_refractive_index_quantities_are_rejected_at_geometry_field_and_kernel_boundaries(operation):
    with pytest.raises(ValueError, match="refractive_index.*without units"):
        operation()


@pytest.mark.parametrize(
    "operation, parameter",
    [
        (lambda: Source(wavelength=[633] * ureg.nanometer), "wavelength"),
        (
            lambda: Grid(
                spacing=[50] * ureg.nanometer,
                shape=(12, 12, 12),
            ),
            "spacing",
        ),
        (
            lambda: RandomMedium(
                correlation_length=[60] * ureg.nanometer,
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation="matern",
                smoothness=1.5,
            ),
            "correlation_length",
        ),
        (lambda: AngularSampling(start=[0] * ureg.degree), "start"),
        (lambda: Sphere(radius=[50] * ureg.nanometer, refractive_index=1.34), "radius"),
    ],
)
def test_unitful_scalar_parameters_reject_arrays(operation, parameter):
    with pytest.raises(ValueError, match=parameter + ".*scalar"):
        operation()


def test_physical_configuration_preserves_supplied_units_and_quantity_arrays():
    source = Source(wavelength=633 * ureg.nanometer)

    grid = Grid(
        shape=(2, 2, 2),
        spacing=0.05 * ureg.micrometer,
    )

    sampling = AngularSampling(
        start=0 * ureg.degree,
        end=np.pi * ureg.radian,
        n_points=3,
    )

    medium = RandomMedium(
        correlation_length=100 * ureg.nanometer,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation="matern",
        smoothness=1.5,
    )

    sphere = Sphere(
        radius=0.1 * ureg.micrometer,
        refractive_index=1.34,
    )

    assert source.wavelength.units == ureg.nanometer

    assert grid.spacing.units == grid.positions.units == ureg.micrometer

    assert medium.correlation_length.units == ureg.nanometer

    assert sphere.radius.units == ureg.micrometer

    assert sampling.angles.units == ureg.degree

    np.testing.assert_allclose(sampling.angles.magnitude, [0, 90, 180])

    assert not sampling.angles.magnitude.flags.writeable

    positions = np.array([[0, 0, 0], [150, 0, 0]]) * ureg.nanometer

    np.testing.assert_array_equal(sphere.mask(positions=positions), [True, False])

    result = Solver(source=source, sampling=sampling).solve(
        target=AnalyticalMedium(
            correlation_length=100 * ureg.nanometer,
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation="gaussian",
            smoothness=1.5,
        )
    )

    assert result.angles.units == ureg.degree

    assert result.mu_s.is_compatible_with("1 / meter")

    assert result.differential.is_compatible_with("1 / meter / steradian")
