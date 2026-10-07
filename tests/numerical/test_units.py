import numpy as np
import pytest

from bornsim import RandomMedium, Solver, Source, Volume
from bornsim.series import BornSeries
from bornsim.ensemble import ensemble_scattering
from bornsim.media import random_volume
from bornsim.units import ureg


def test_unitful_volume_matches_si_and_retains_complex_amplitudes():
    medium = RandomMedium(
        correlation="gaussian",
        correlation_length=100 * ureg.nanometer,
    )
    volume = random_volume(
        medium=medium,
        shape=(2, 2, 2),
        spacing=50 * ureg.nanometer,
        seed=42,
    )
    reference = random_volume(
        medium=RandomMedium(correlation="gaussian"),
        shape=(2, 2, 2),
        spacing=50e-9,
        seed=42,
    )
    np.testing.assert_allclose(volume.delta_index, reference.delta_index)
    explicit = Volume(
        delta_index=volume.delta_index * ureg.dimensionless,
        spacing=0.05 * ureg.micrometer,
    )
    np.testing.assert_allclose(explicit.positions, reference.positions)
    directions = np.array([[0, 0, 1], [1, 0, 0]]) * ureg.dimensionless
    result = Solver(
        source=Source(wavelength=633 * ureg.nanometer),
        order=2,
    ).solve_cut(
        target=volume,
        directions=directions,
    )
    expected = BornSeries(
        shape=reference.delta_index.shape,
        spacing=reference.spacing,
        background_index=reference.background_index,
        wavelength=633e-9,
        directions=directions.magnitude,
        order=2,
    ).solve(volume=reference)
    np.testing.assert_allclose(result.amplitudes.to("meter").magnitude, expected.amplitudes)
    assert np.iscomplexobj(result.amplitudes.magnitude)
    assert np.any(result.amplitudes.magnitude.imag != 0)
    np.testing.assert_allclose(result.differential.to("1 / meter / steradian").magnitude, expected.differential)
    np.testing.assert_allclose(result.field_norms.to("dimensionless").magnitude, expected.field_norms)
    np.testing.assert_allclose(
        BornSeries(
            shape=volume.delta_index.shape,
            spacing=volume.spacing,
            background_index=volume.background_index,
            wavelength=633 * ureg.nanometer,
            directions=directions,
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
        medium=RandomMedium(correlation="gaussian"),
        spacing=50 * ureg.nanometer,
        angles=[0, 90, 180] * ureg.degree,
        **options,
    )
    expected = ensemble_scattering(
        medium=RandomMedium(correlation="gaussian"),
        wavelength=633e-9,
        spacing=50e-9,
        order=2,
        angles=[0, np.pi / 2, np.pi],
        **options,
    )
    np.testing.assert_allclose(result.differential.magnitude, expected["directional_differential"])
    np.testing.assert_allclose(result.stderr.to("1 / meter / steradian").magnitude, expected["directional_stderr"])
    segments = result.plot().axes[0].containers[0].lines[2][0].get_segments()
    errors = [(segment[1, 1] - segment[0, 1]) / 2 for segment in segments]
    np.testing.assert_allclose(errors, expected["directional_stderr"][0, :, 0])
    np.testing.assert_allclose(result.mu_s.to("1 / meter").magnitude, expected["mu_s"])
    numeric = ensemble_scattering(
        medium=RandomMedium(correlation="gaussian"),
        wavelength=633 * ureg.nanometer,
        spacing=50 * ureg.nanometer,
        order=2,
        angles=[0, 90, 180] * ureg.degree,
        **options,
    )
    np.testing.assert_allclose(numeric["mean"], expected["mean"])


@pytest.mark.parametrize(
    "operation, name",
    [
        (
            lambda: Volume(
                delta_index=np.zeros((2, 2, 2)),
                spacing=1 * ureg.second,
            ),
            "spacing",
        ),
        (
            lambda: Volume(
                delta_index=np.ones((2, 2, 2)) * ureg.meter,
                spacing=50e-9,
            ),
            "delta_index",
        ),
        (
            lambda: random_volume(
                medium=RandomMedium(correlation="gaussian"),
                spacing=1 * ureg.second,
            ),
            "spacing",
        ),
        (
            lambda: Solver(source=Source()).ensemble(
                medium=RandomMedium(correlation="gaussian"),
                spacing=1 * ureg.second,
            ),
            "spacing",
        ),
        (
            lambda: Solver(source=Source()).ensemble(
                medium=RandomMedium(correlation="gaussian"),
                angles=[0] * ureg.meter,
            ),
            "angles",
        ),
        (
            lambda: BornSeries(
                shape=(2, 2, 2),
                spacing=50e-9,
                background_index=1.33,
                wavelength=1 * ureg.second,
                directions=[[0, 0, 1]],
            ).solve(
                volume=Volume(
                    delta_index=np.zeros((2, 2, 2)),
                    spacing=50e-9,
                )
            ),
            "wavelength",
        ),
        (
            lambda: Solver(source=Source()).solve_cut(
                target=Volume(
                    delta_index=np.zeros((2, 2, 2)),
                    spacing=50e-9,
                ),
                directions=[[0, 0, 1]] * ureg.meter,
            ),
            "directions",
        ),
    ],
)
def test_numerical_dimensions_are_checked(operation, name):
    with pytest.raises(ValueError, match=name):
        operation()
