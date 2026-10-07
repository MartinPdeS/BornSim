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
    medium = RandomMedium()
    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )
    legacy = medium.to_volume(
        shape=grid.shape,
        spacing=grid.spacing,
        seed=42,
    )
    assert volume.grid is grid
    assert volume.spacing == pytest.approx(25e-9)
    np.testing.assert_array_equal(volume.delta_index, legacy.delta_index)
    np.testing.assert_allclose(grid.positions[0, 0, 0], [-12.5e-9, -25e-9, -37.5e-9])
    np.testing.assert_array_equal(volume.positions, grid.positions)
    assert volume.volume == pytest.approx(24 * (25e-9) ** 3)
    structured = StructuredMedium()
    assert structured.to_volume(grid=grid).grid is grid
    composed = StructuredMedium()
    composed.add_background(medium=medium)
    np.testing.assert_array_equal(composed.to_volume(grid=grid, seed=42).delta_index, volume.delta_index)
    with pytest.raises(FrozenInstanceError):
        grid.spacing = 1


def test_full_solve_retains_coherent_directional_amplitudes_normalization_and_archive(tmp_path):
    grid = Grid(
        shape=(2, 3, 2),
        spacing=30e-9,
    )
    medium = RandomMedium(index_std=0.001)
    volume = medium.to_volume(grid=grid, seed=42)
    sampling = AngularSampling(
        angles=[180, 90, 0] * ureg.degree,
        polar_samples=16,
        azimuth_samples=4,
    )
    solver = Solver(
        source=Source(),
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
    intensity = np.sum(np.abs(np.cumsum(amplitudes, axis=0)) ** 2, axis=(-1, -2)) / (2 * volume.volume)
    np.testing.assert_allclose(result.directional_differential.magnitude, intensity, rtol=1e-13)
    np.testing.assert_allclose(result.differential.magnitude, intensity, rtol=1e-13)
    np.testing.assert_allclose(
        result.directional_phase_function.magnitude, intensity / result.mu_s.magnitude[:, None, None]
    )
    engine = BornSeries(
        grid=grid,
        background_index=volume.background_index,
        wavelength=solver.source.wavelength,
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
        np.testing.assert_array_equal(getattr(result, name).magnitude, getattr(ensemble, name).magnitude)
    restored = Result.load(path=result.save(path=tmp_path / "full.npz"))
    np.testing.assert_array_equal(restored.directional_amplitudes.magnitude, amplitudes)
    np.testing.assert_array_equal(
        restored.directional_phase_function.magnitude, result.directional_phase_function.magnitude
    )
    assert restored.provenance["sampling"] == sampling.metadata
    figure = restored.plot_phase_function(view="3d")
    figure.canvas.draw()


def test_default_solve_is_full_and_explicit_cuts_remain_unnormalized():
    volume = Volume(
        delta_index=np.full((2, 2, 2), 0.001),
        spacing=20e-9,
    )
    solver = Solver(
        source=Source(),
        order=1,
    )
    full = solver.solve(target=volume)
    assert full.directional_phase_function.shape == (1, 121, 8)
    cut = solver.solve_cut(
        target=volume,
        angles=[0, np.pi],
    )
    assert cut.amplitudes.shape == (1, 2, 2, 3)
    assert cut.mu_s is None
    with pytest.raises(ValueError, match="integrated mu_s"):
        _ = cut.phase_function
    zero = Volume(
        delta_index=np.zeros((2, 2, 2)),
        grid=volume.grid,
    )
    empty = solver.solve(target=zero)
    np.testing.assert_array_equal(empty.mu_s.magnitude, [0])
    assert np.isnan(empty.g.magnitude[0])
    with pytest.raises(ValueError, match="positive"):
        _ = empty.directional_phase_function


def test_ensemble_is_for_random_media_and_composed_random_backgrounds():
    solver = Solver(
        source=Source(),
        order=1,
    )
    grid = Grid(shape=(2, 2, 2))
    medium = StructuredMedium()
    with pytest.raises(ValueError, match="use Solver.solve"):
        solver.ensemble(
            medium=medium,
            grid=grid,
        )
    medium.add_background(medium=RandomMedium())
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
        {"spacing": 0},
        {"spacing": np.nan},
        {"spacing": 1 * ureg.second},
    ],
)
def test_invalid_grid_settings(options):
    with pytest.raises(ValueError):
        Grid(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"angles": []},
        {"angles": [[0]]},
        {"angles": [-1]},
        {"angles": [np.nan]},
        {"angles": [0] * 182},
        {"angles": [1] * ureg.meter},
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
    sampling = AngularSampling(angles=angles)
    angles[:] = 0
    np.testing.assert_array_equal(sampling.angles, [0, np.pi / 2, np.pi])
    with pytest.raises(ValueError):
        sampling.angles[0] = 1
    with pytest.raises(FrozenInstanceError):
        sampling.polar_samples = 64
    grid = Grid(shape=(2, 2, 2))
    medium = RandomMedium()
    with pytest.raises(ValueError, match="not both"):
        medium.to_volume(grid=grid, spacing=1e-9)
    with pytest.raises(ValueError, match="match delta_index"):
        Volume(
            delta_index=np.zeros((2, 3, 2)),
            grid=grid,
        )
    with pytest.raises(ValueError, match="solve_cut"):
        Solver(source=Source()).solve(
            target=medium.to_volume(grid=grid),
            sampling=sampling,
            angles=[0],
        )
    with pytest.raises(ValueError, match="not both"):
        Solver(source=Source()).ensemble(
            medium=medium,
            grid=grid,
            sampling=sampling,
            azimuth_samples=8,
        )
    with pytest.raises(TypeError, match="sampling"):
        Solver(
            source=Source(),
            sampling=object(),
        )
    with pytest.raises(TypeError, match="grid"):
        medium.to_volume(grid=object())
    with pytest.raises(ValueError, match="too large"):
        sampling.check_work(grid=Grid(shape=(32, 32, 32)), order=12, realizations=32)
