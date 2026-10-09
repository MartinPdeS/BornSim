from bornsim.medium.random_medium import GaussianMedium, WhittleMaternMedium
from bornsim import EnsembleSampling
from bornsim import AngularSampling
import numpy as np
import pytest
from bornsim import AngularData, Box, Cylinder, Grid, Layer, Rotation, Sphere, Result, Solver, Source, Volume
from bornsim.series import BornSeries
from bornsim.green import GreenOperator
from bornsim.ensemble import ensemble_scattering
from bornsim.units import ureg
from bornsim import Directions


def test_unitful_volume_matches_si_and_retains_complex_amplitudes():
    medium = GaussianMedium(
        correlation_length=100 * ureg.nanometer,
        background_refractive_index=1.33,
        refractive_index_std=0.01,
    )

    volume = medium.to_volume(
        seed=42,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=50 * ureg.nanometer,
        ),
    )

    reference = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=1e-07 * ureg.meter,
    ).to_volume(
        seed=42,
        grid=Grid(
            shape=(2, 2, 2),
            spacing=5e-08 * ureg.meter,
        ),
    )

    np.testing.assert_allclose(volume.delta_refractive_index, reference.delta_refractive_index)

    explicit = Volume(
        delta_refractive_index=volume.delta_refractive_index,
        background_refractive_index=1.33,
        grid=Grid(
            shape=np.shape(volume.delta_refractive_index),
            spacing=0.05 * ureg.micrometer,
        ),
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
        background_refractive_index=reference.background_refractive_index,
        wavelength=6.33e-07 * ureg.meter,
        directions=Directions(vectors=directions.magnitude),
        order=2,
        grid=Grid(
            shape=reference.delta_refractive_index.shape,
            spacing=reference.spacing.to("meter"),
        ),
    ).solve(volume=reference)

    np.testing.assert_allclose(result.amplitudes.to("meter").magnitude, expected.amplitudes)

    assert np.iscomplexobj(result.amplitudes.magnitude)

    assert np.any(result.amplitudes.magnitude.imag != 0)

    np.testing.assert_allclose(result.differential.to("1 / meter / steradian").magnitude, expected.differential)

    np.testing.assert_allclose(result.field_norms.to("dimensionless").magnitude, expected.field_norms)

    np.testing.assert_allclose(
        BornSeries(
            background_refractive_index=volume.background_refractive_index,
            wavelength=633 * ureg.nanometer,
            directions=Directions(vectors=directions),
            order=2,
            grid=Grid(
                shape=volume.delta_refractive_index.shape,
                spacing=volume.spacing.to("meter"),
            ),
        )
        .solve(volume=volume)
        .differential,
        expected.differential,
    )


def test_unitful_ensemble_preserves_seed_and_standard_error():
    options = dict(
        grid=Grid(
            shape=(2, 2, 2),
            spacing=50 * ureg.nanometer,
        ),
        ensemble_sampling=EnsembleSampling(
            realizations=3,
            seed=42,
        ),
        sampling=AngularSampling(
            azimuth_samples=4,
            polar_samples=16,
        ),
    )

    result = Solver(
        source=Source(wavelength=633 * ureg.nanometer),
        order=2,
    ).ensemble(
        medium=GaussianMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=1e-07 * ureg.meter,
        ),
        **options,
    )

    expected = ensemble_scattering(
        medium=GaussianMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=1e-07 * ureg.meter,
        ),
        wavelength=6.33e-07 * ureg.meter,
        order=2,
        **options,
    )

    np.testing.assert_allclose(result.differential.magnitude, expected["directional_differential"])

    np.testing.assert_allclose(result.stderr.to("1 / meter / steradian").magnitude, expected["directional_stderr"])

    segments = result.plot().axes[0].containers[0].lines[2][0].get_segments()

    errors = [(segment[1, 1] - segment[0, 1]) / 2 for segment in segments]

    np.testing.assert_allclose(errors, expected["directional_stderr"][0, :, 0])

    np.testing.assert_allclose(result.mu_s.to("1 / meter").magnitude, expected["mu_s"])

    numeric = ensemble_scattering(
        medium=GaussianMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=1e-07 * ureg.meter,
        ),
        wavelength=633 * ureg.nanometer,
        order=2,
        **options,
    )

    np.testing.assert_allclose(numeric["mean"], expected["mean"])


@pytest.mark.parametrize(
    "operation, name",
    [
        (
            lambda: Volume(
                delta_refractive_index=np.zeros((2, 2, 2)),
                background_refractive_index=1.33,
                grid=Grid(
                    shape=np.shape(np.zeros((2, 2, 2))),
                    spacing=1 * ureg.second,
                ),
            ),
            "spacing",
        ),
        (
            lambda: Volume(
                delta_refractive_index=np.ones((2, 2, 2)) * ureg.meter,
                background_refractive_index=1.33,
                grid=Grid(
                    shape=np.shape(np.ones((2, 2, 2)) * ureg.meter),
                    spacing=5e-08 * ureg.meter,
                ),
            ),
            "delta_refractive_index",
        ),
        (
            lambda: GaussianMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=1e-07 * ureg.meter,
            ).to_volume(
                grid=Grid(
                    shape=(12, 12, 12),
                    spacing=1 * ureg.second,
                )
            ),
            "spacing",
        ),
        (
            lambda: Solver(source=Source(wavelength=6.33e-07 * ureg.meter)).ensemble(
                medium=GaussianMedium(
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=1e-07 * ureg.meter,
                ),
                grid=Grid(
                    shape=(12, 12, 12),
                    spacing=1 * ureg.second,
                ),
            ),
            "spacing",
        ),
        (
            lambda: Solver(source=Source(wavelength=6.33e-07 * ureg.meter)).ensemble(
                medium=GaussianMedium(
                    background_refractive_index=1.33,
                    refractive_index_std=0.01,
                    correlation_length=1e-07 * ureg.meter,
                ),
                grid=Grid(
                    shape=(12, 12, 12),
                    spacing=5e-08 * ureg.meter,
                ),
                sampling=AngularSampling(angles=[0] * ureg.meter),
            ),
            "angles",
        ),
        (
            lambda: BornSeries(
                background_refractive_index=1.33,
                wavelength=1 * ureg.second,
                directions=Directions(vectors=[[0, 0, 1]]),
                grid=Grid(
                    shape=(2, 2, 2),
                    spacing=5e-08 * ureg.meter,
                ),
            ).solve(
                volume=Volume(
                    delta_refractive_index=np.zeros((2, 2, 2)),
                    background_refractive_index=1.33,
                    grid=Grid(
                        shape=np.shape(np.zeros((2, 2, 2))),
                        spacing=5e-08 * ureg.meter,
                    ),
                )
            ),
            "wavelength",
        ),
        (
            lambda: Solver(source=Source(wavelength=6.33e-07 * ureg.meter)).solve_cut(
                target=Volume(
                    delta_refractive_index=np.zeros((2, 2, 2)),
                    background_refractive_index=1.33,
                    grid=Grid(
                        shape=np.shape(np.zeros((2, 2, 2))),
                        spacing=5e-08 * ureg.meter,
                    ),
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
        (lambda: Source(wavelength=6.33e-07), "wavelength"),
        (
            lambda: Grid(
                spacing=5e-08,
                shape=(12, 12, 12),
            ),
            "spacing",
        ),
        (
            lambda: WhittleMaternMedium(
                correlation_length=6e-08,
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                smoothness=1.5,
            ),
            "correlation_length",
        ),
        (
            lambda: Volume(
                delta_refractive_index=np.zeros((2, 2, 2)),
                background_refractive_index=1.33,
                grid=Grid(
                    shape=np.shape(np.zeros((2, 2, 2))),
                    spacing=5e-08,
                ),
            ),
            "spacing",
        ),
        (
            lambda: Sphere(
                radius=5e-08,
                refractive_index=1.34,
            ),
            "radius",
        ),
        (
            lambda: Box(
                size=(5e-08,) * 3,
                refractive_index=1.34,
            ),
            "size",
        ),
        (
            lambda: Layer(
                lower=0,
                upper=50 * ureg.nanometer,
                refractive_index=1.34,
            ),
            "lower",
        ),
        (
            lambda: Cylinder(
                radius=50 * ureg.nanometer,
                height=1e-07,
                refractive_index=1.34,
            ),
            "height",
        ),
        (
            lambda: Sphere(
                radius=50 * ureg.nanometer,
                centre=(0, 0, 0),
                refractive_index=1.34,
            ),
            "centre",
        ),
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
            lambda: WhittleMaternMedium(
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                correlation_length=1e-07 * ureg.meter,
                smoothness=1.5,
            ).spectral_weight(q_squared=np.array([0, 1000000000000.0])),
            "q_squared",
        ),
        (
            lambda: GreenOperator(
                shape=(2, 2, 2), spacing=5e-08, wavelength=633 * ureg.nanometer, background_refractive_index=1.33
            ),
            "spacing",
        ),
        (
            lambda: BornSeries(
                grid=Grid(
                    shape=(12, 12, 12),
                    spacing=5e-08 * ureg.meter,
                ),
                wavelength=6.33e-07,
                background_refractive_index=1.33,
                directions=Directions(vectors=[[0, 0, 1]]),
            ),
            "wavelength",
        ),
        (lambda: AngularData(differential=[[1]], angles=[0] * ureg.radian), "differential"),
        (
            lambda: Result(
                source=Source(wavelength=6.33e-07 * ureg.meter),
                kind="volume",
                differential=[[1]],
                angles=[0] * ureg.radian,
                azimuth_averaged=True,
            ),
            "differential",
        ),
    ],
)
def test_dimensional_public_inputs_reject_bare_values(operation, parameter):
    with pytest.raises(ValueError, match=parameter + ".*explicit.*units"):
        operation()


def test_angles_require_angular_units_and_dimensionless_statistics_accept_numbers():
    with pytest.raises(ValueError, match="explicit angular units"):
        AngularSampling(start=0 * ureg.dimensionless)

    medium = WhittleMaternMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=60 * ureg.nanometer,
        smoothness=1.5,
    )

    assert medium.correlation_length.to("meter").magnitude == pytest.approx(6e-08)

    np.testing.assert_array_equal(medium.spectral_weight(q_squared=np.array([0]) / ureg.meter**2), [1])


@pytest.mark.parametrize("unit", ["dimensionless", "refractive_index_units", "percent"])
@pytest.mark.parametrize("parameter", ["background_refractive_index", "refractive_index_std"])
def test_refractive_index_statistics_reject_all_quantities(unit, parameter):
    with pytest.raises(ValueError, match=parameter + ".*without units"):
        GaussianMedium(
            **{
                "background_refractive_index": 1.33,
                "refractive_index_std": 0.01,
                "correlation_length": 1e-07 * ureg.meter,
                **{parameter: 1 * ureg.Unit(unit)},
            }
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: Sphere(
            radius=50 * ureg.nanometer,
            refractive_index=1.34 * ureg.dimensionless,
        ),
        lambda: Volume(
            delta_refractive_index=np.zeros((2, 2, 2)) * ureg.dimensionless,
            grid=Grid(
                shape=(2, 2, 2),
                spacing=5e-08 * ureg.meter,
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
            lambda: WhittleMaternMedium(
                correlation_length=[60] * ureg.nanometer,
                background_refractive_index=1.33,
                refractive_index_std=0.01,
                smoothness=1.5,
            ),
            "correlation_length",
        ),
        (lambda: AngularSampling(start=[0] * ureg.degree), "start"),
        (
            lambda: Sphere(
                radius=[50] * ureg.nanometer,
                refractive_index=1.34,
            ),
            "radius",
        ),
    ],
)
def test_unitful_scalar_parameters_reject_arrays(operation, parameter):
    with pytest.raises(ValueError, match=parameter + ".*scalar"):
        operation()
