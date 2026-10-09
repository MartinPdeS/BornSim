"""Shared configurations and independent fixed-volume scattering semantics."""

from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from bornsim import AngularSampling, Grid, RandomMedium, Result, Solver, Source, StructuredMedium, Volume
from bornsim.series import BornSeries
from bornsim.units import ureg


def test_shared_grid_preserves_coordinates_seeded_field_and_identity():
    grid = Grid(
        shape=(2, 3, 4),
        spacing=25 * ureg.nanometer,
    )

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )

    legacy = medium.to_volume(
        shape=grid.shape,
        spacing=grid.spacing.to("meter"),
        seed=42,
    )

    assert volume.grid is grid

    assert volume.spacing.to("meter").magnitude == pytest.approx(25e-9)

    np.testing.assert_array_equal(volume.delta_refractive_index, legacy.delta_refractive_index)

    np.testing.assert_allclose(grid.positions.to("meter").magnitude[0, 0, 0], [-12.5e-9, -25e-9, -37.5e-9])

    np.testing.assert_array_equal(volume.positions, grid.positions)

    assert volume.volume.to("meter**3").magnitude == pytest.approx(24 * (25e-9) ** 3)

    structured = StructuredMedium(
        background_refractive_index=1.33,
    )

    assert structured.to_volume(grid=grid).grid is grid

    composed = StructuredMedium(
        background_refractive_index=1.33,
    )

    composed.add_background(medium=medium)

    np.testing.assert_array_equal(
        composed.to_volume(grid=grid, seed=42).delta_refractive_index, volume.delta_refractive_index
    )

    with pytest.raises(FrozenInstanceError):
        grid.spacing = 1


def test_full_solve_retains_coherent_directional_amplitudes_normalization_and_archive(tmp_path):
    grid = Grid(
        shape=(2, 3, 2),
        spacing=3e-08 * ureg.meter,
    )

    medium = RandomMedium(
        refractive_index_std=0.001,
        background_refractive_index=1.33,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    volume = medium.to_volume(grid=grid, seed=42)

    sampling = AngularSampling(
        start=180 * ureg.degree,
        end=0 * ureg.degree,
        n_points=3,
        polar_samples=16,
        azimuth_samples=4,
    )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        sampling=sampling,
        order=3,
    )

    result = solver.solve(target=volume)

    assert result.kind == "volume"

    assert result.realizations is None

    assert result.stderr is None

    assert result.amplitudes is not None

    amplitudes = result.directional_amplitudes.magnitude

    assert amplitudes.shape == (3, 3, 4, 2, 3)

    intensity = np.sum(np.abs(np.cumsum(amplitudes, axis=0)) ** 2, axis=(-1, -2)) / (
        2 * volume.volume.to("meter**3").magnitude
    )

    np.testing.assert_allclose(result.directional_differential.magnitude, intensity, rtol=1e-13)

    np.testing.assert_allclose(result.differential.magnitude, intensity, rtol=1e-13)

    np.testing.assert_allclose(
        result.directional_phase_function.magnitude, intensity / result.mu_s.magnitude[:, None, None]
    )

    engine = BornSeries(
        grid=grid,
        background_refractive_index=volume.background_refractive_index,
        wavelength=solver.source.wavelength.to("meter"),
        directions=sampling.directions,
        order=3,
    )

    direct = engine.solve(volume=volume)

    np.testing.assert_array_equal(amplitudes, direct.amplitudes.reshape(3, 19, 4, 2, 3)[:, :3])

    ensemble = solver.ensemble(
        medium=medium,
        grid=grid,
        realizations=1,
        seed=42,
    )

    for name in ("differential", "directional_differential", "mu_s", "g", "mu_s_prime"):
        np.testing.assert_allclose(getattr(result, name).magnitude, getattr(ensemble, name).magnitude, rtol=1e-13)

    restored = Result.load(path=result.save(path=tmp_path / "full.npz"))

    np.testing.assert_array_equal(restored.directional_amplitudes.magnitude, amplitudes)

    np.testing.assert_array_equal(
        restored.directional_phase_function.magnitude, result.directional_phase_function.magnitude
    )

    assert restored.provenance["sampling"] == sampling.metadata

    figure = restored.plot_phase_function(
        view="3d",
        backend="matplotlib",
    )

    figure.canvas.draw()


def test_default_solve_is_full_and_explicit_cuts_remain_unnormalized():
    volume = Volume(
        delta_refractive_index=np.full((2, 2, 2), 0.001),
        spacing=2e-08 * ureg.meter,
        background_refractive_index=1.33,
    )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        order=1,
    )

    full = solver.solve(target=volume)

    assert full.directional_phase_function.shape == (1, 121, 8)

    cut = solver.solve_cut(
        target=volume,
        angles=[0, np.pi] * ureg.radian,
    )

    assert cut.amplitudes.shape == (1, 2, 2, 3)

    assert cut.mu_s is None

    with pytest.raises(ValueError, match="integrated mu_s"):
        _ = cut.phase_function

    zero = Volume(
        delta_refractive_index=np.zeros((2, 2, 2)),
        grid=volume.grid,
        background_refractive_index=1.33,
    )

    empty = solver.solve(target=zero)

    np.testing.assert_array_equal(empty.mu_s.magnitude, [0])

    assert np.isnan(empty.g.magnitude[0])

    with pytest.raises(ValueError, match="positive"):
        _ = empty.directional_phase_function


def test_ensemble_is_for_random_media_and_composed_random_backgrounds():
    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        order=1,
    )

    grid = Grid(
        shape=(2, 2, 2),
        spacing=50e-9 * ureg.meter,
    )

    medium = StructuredMedium(
        background_refractive_index=1.33,
    )

    with pytest.raises(ValueError, match="use Solver.solve"):
        solver.ensemble(
            medium=medium,
            grid=grid,
        )

    medium.add_background(
        medium=RandomMedium(
            background_refractive_index=1.33,
            refractive_index_std=0.01,
            correlation_length=100e-9 * ureg.meter,
            correlation="matern",
            smoothness=1.5,
        )
    )

    result = solver.ensemble(
        medium=medium,
        grid=grid,
        realizations=2,
    )

    assert result.kind == "ensemble"

    assert result.realizations == 2

    assert np.all(np.isfinite(result.stderr.magnitude))


@pytest.mark.parametrize(
    "options",
    [
        {"shape": (1, 2, 2)},
        {"shape": (2, 2)},
        {"shape": (True, 2, 2)},
        {"shape": (33, 2, 2)},
        {"spacing": 0 * ureg.meter},
        {"spacing": np.nan * ureg.meter},
        {"spacing": 1 * ureg.second},
    ],
)
def test_invalid_grid_settings(options):
    with pytest.raises(ValueError):
        Grid(**{"shape": (12, 12, 12), "spacing": 50e-9 * ureg.meter, **options})


@pytest.mark.parametrize(
    "options",
    [
        {"angles": [] * ureg.radian},
        {"angles": [[0]] * ureg.radian},
        {"angles": [-1] * ureg.radian},
        {"angles": [np.nan] * ureg.radian},
        {"angles": [0] * 182 * ureg.radian},
        {"angles": [1] * ureg.meter},
        {"start": -1 * ureg.radian},
        {"end": (np.pi + 0.1) * ureg.radian},
        {"start": np.nan * ureg.radian},
        {"end": np.inf * ureg.radian},
        {"start": [0] * ureg.radian},
        {"end": [np.pi] * ureg.radian},
        {"start": 1 * ureg.meter},
        {"end": 1 * ureg.second},
        {"end": -1 * ureg.radian, "n_points": 1},
        {"n_points": 0},
        {"n_points": 182},
        {"n_points": True},
        {"n_points": 3.0},
        {"angles": [0] * ureg.radian, "start": 0 * ureg.radian},
        {"angles": [0] * ureg.radian, "end": np.pi * ureg.radian},
        {"angles": [0] * ureg.radian, "n_points": 1},
        {"polar_samples": 15},
        {"azimuth_samples": 3},
        {"polar_samples": True},
    ],
)
def test_invalid_angular_sampling(options):
    with pytest.raises(ValueError):
        AngularSampling(**options)


def test_configuration_inputs_are_copied_and_conflicting_settings_are_rejected():
    angles = np.array([0, np.pi / 2, np.pi])

    sampling = AngularSampling(angles=angles * ureg.radian)

    angles[:] = 0

    np.testing.assert_array_equal(sampling.angles.to("radian").magnitude, [0, np.pi / 2, np.pi])

    with pytest.raises(ValueError):
        sampling.angles[0] = 1

    with pytest.raises(FrozenInstanceError):
        sampling.polar_samples = 64

    grid = Grid(
        shape=(2, 2, 2),
        spacing=50e-9 * ureg.meter,
    )

    medium = RandomMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.01,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    with pytest.raises(ValueError, match="not both"):
        medium.to_volume(grid=grid, spacing=1e-09 * ureg.meter)

    with pytest.raises(ValueError, match="match delta_refractive_index"):
        Volume(
            delta_refractive_index=np.zeros((2, 3, 2)),
            grid=grid,
            background_refractive_index=1.33,
        )

    with pytest.raises(ValueError, match="solve_cut"):
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            )
        ).solve(
            target=medium.to_volume(grid=grid),
            sampling=sampling,
            angles=[0] * ureg.radian,
        )

    with pytest.raises(ValueError, match="not both"):
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            )
        ).ensemble(
            medium=medium,
            grid=grid,
            sampling=sampling,
            azimuth_samples=8,
        )

    with pytest.raises(TypeError, match="sampling"):
        Solver(
            source=Source(
                wavelength=633e-9 * ureg.meter,
            ),
            sampling=object(),
        )

    with pytest.raises(TypeError, match="grid"):
        medium.to_volume(grid=object())

    with pytest.raises(ValueError, match="too large"):
        sampling.check_work(
            grid=Grid(
                shape=(32, 32, 32),
                spacing=50e-9 * ureg.meter,
            ),
            order=12,
            realizations=32,
        )


@pytest.mark.parametrize(
    "options, expected_degrees",
    [
        ({"start": 30 * ureg.degree, "end": np.pi / 2 * ureg.radian, "n_points": 3}, [30, 60, 90]),
        ({"start": np.pi * ureg.radian, "end": 0 * ureg.degree, "n_points": 3}, [180, 90, 0]),
        ({"start": 45 * ureg.degree, "n_points": 1}, [45]),
        ({"n_points": 3}, [0, 90, 180]),
        (
            {"start": 0.5 * ureg.radian, "end": 0.5 * ureg.radian, "n_points": 2},
            np.rad2deg([0.5, 0.5]),
        ),
    ],
)
def test_angular_sampling_ranges(options, expected_degrees):
    sampling = AngularSampling(**options)

    np.testing.assert_allclose(sampling.angles.to("degree").magnitude, expected_degrees)

    assert not sampling.angles.magnitude.flags.writeable


def test_angular_sampling_range_matches_explicit_grid_and_defaults():
    sampling = AngularSampling(
        start=0 * ureg.degree,
        end=180 * ureg.degree,
        n_points=181,
        polar_samples=16,
        azimuth_samples=4,
    )

    explicit = AngularSampling(
        angles=np.linspace(0, 180, 181) * ureg.degree,
        polar_samples=16,
        azimuth_samples=4,
    )

    np.testing.assert_allclose(
        sampling.angles.to("radian").magnitude, explicit.angles.to("radian").magnitude, atol=1e-15
    )

    np.testing.assert_allclose(sampling.directions.vectors, explicit.directions.vectors, atol=1e-15)

    assert sampling.polar_samples == 16

    assert sampling.azimuth_samples == 4

    default = AngularSampling()

    np.testing.assert_array_equal(default.angles.to("radian").magnitude, np.linspace(0, np.pi, 121))

    assert default.metadata["angles_rad"] == default.angles.tolist()
