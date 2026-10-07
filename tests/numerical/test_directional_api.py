"""Public directional semantics, geometry transforms and archive migration."""

import json
import numpy as np
import pytest
from bornsim import (
    AngularData,
    AngularSampling,
    Box,
    Grid,
    Result,
    Rotation,
    Solver,
    Source,
    Sphere,
    StructuredMedium,
    RandomMedium,
)
from bornsim._archives import _RESULT_UNITS
from bornsim.units import ureg


def setup_problem():
    grid = Grid(
        shape=(3, 3, 3),
        spacing=40e-9,
    )
    medium = StructuredMedium(background_index=1.33)
    sphere = Sphere(
        radius=50e-9,
        index=1.34,
        centre=(20e-9, 0, 0),
    )
    medium.add_structures(sphere)
    sampling = AngularSampling(
        angles=[0, np.pi / 2, np.pi],
        polar_samples=16,
        azimuth_samples=4,
    )
    solver = Solver(
        source=Source(),
        order=2,
        sampling=sampling,
    )
    return grid, medium, solver


def test_directional_data_and_explicit_cut(tmp_path):
    grid, medium, solver = setup_problem()
    volume = medium.to_volume(grid=grid)
    result = solver.solve(target=volume)
    assert result.differential.shape == (2, 3, 4)
    assert result.amplitudes.shape == (2, 3, 4, 2, 3)
    angular = result.angular
    assert angular.directions.shape == (3, 4, 3)
    assert not angular.differential.magnitude.flags.writeable
    np.testing.assert_allclose(
        angular.differential_cross_section.magnitude, result.differential.magnitude * grid.volume
    )
    averaged = result.azimuth_average()
    np.testing.assert_allclose(averaged.differential.magnitude, result.differential.magnitude.mean(axis=-1))
    assert averaged.amplitudes is None
    assert averaged.azimuth_averaged
    np.testing.assert_allclose(averaged.phase_function.magnitude, result.phase_function.magnitude.mean(axis=-1))
    cut = solver.solve_cut(
        target=volume,
        angles=[0, np.pi / 2],
    )
    assert cut.amplitudes.shape == (2, 2, 2, 3)
    assert cut.mu_s is None
    with pytest.raises(ValueError, match="solve_cut"):
        solver.solve(
            target=volume,
            angles=[0],
        )
    with pytest.raises(ValueError, match="normalization"):
        _ = cut.phase_function
    figure = result.plot_cross_section()
    np.testing.assert_allclose(
        figure.axes[0].lines[0].get_ydata(),
        result.differential_cross_section.to("nanometer**2 / steradian").magnitude[0, :, 0],
    )
    for value in (result, averaged, cut):
        path = tmp_path / f"{value.differential.ndim}-{value.azimuth_averaged}.npz"
        value.save(path=path)
        restored = Result.load(path=path)
        assert restored.azimuth_averaged == value.azimuth_averaged
        np.testing.assert_allclose(restored.differential.magnitude, value.differential.magnitude)
        assert restored.sample_volume == value.sample_volume


def test_averaged_ensemble_error_keeps_angular_covariance():
    grid, _, solver = setup_problem()
    medium = RandomMedium(index_std=0.001)
    result = solver.ensemble(
        medium=medium,
        grid=grid,
        realizations=3,
        seed=42,
    )
    curves = [
        solver.solve(target=medium.to_volume(grid=grid, seed=seed)).differential.magnitude.mean(axis=-1)
        for seed in (42, 43, 44)
    ]
    expected = np.std(curves, axis=0, ddof=1) / np.sqrt(3)
    np.testing.assert_allclose(result.azimuth_average().stderr.magnitude, expected)
    np.testing.assert_allclose(result.angular.azimuth_average().stderr.magnitude, expected)


def test_schema_one_promotes_full_arrays(tmp_path):
    grid, medium, solver = setup_problem()
    result = solver.solve(target=medium.to_volume(grid=grid))
    arrays = {
        "differential": result.differential.magnitude.mean(axis=-1),
        "directional_differential": result.differential.magnitude,
        "directional_amplitudes": result.amplitudes.magnitude,
        "angles": result.angles.magnitude,
        "azimuths": result.azimuths.magnitude,
    }
    metadata = {
        "format": "bornsim-result",
        "schema_version": 1,
        "wavelength_m": 633e-9,
        "kind": "volume",
        "realizations": None,
        "warnings": [],
        "provenance": result.provenance,
        "units": {name: _RESULT_UNITS[name] for name in arrays},
    }
    path = tmp_path / "old.npz"
    np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)
    restored = Result.load(path=path)
    np.testing.assert_allclose(restored.amplitudes.magnitude, result.amplitudes.magnitude)
    np.testing.assert_allclose(restored.differential.magnitude, result.differential.magnitude)
    assert restored.sample_volume == result.sample_volume


def test_rotations_translation_and_overlap():
    box = Box(
        size=(4, 2, 2),
        index=1.34,
    )
    rotation = Rotation(
        axis=(0, 0, 1),
        angle=90 * ureg.degree,
    )
    rotated = box.rotated(rotation=rotation)
    positions = np.array([[1.5, 0, 0], [0, 1.5, 0]])
    np.testing.assert_array_equal(box.mask(positions=positions), [True, False])
    np.testing.assert_array_equal(rotated.mask(positions=positions), [False, True])
    moved = rotated.translated(offset=(3, 0, 0))
    np.testing.assert_array_equal(moved.mask(positions=positions + [3, 0, 0]), [False, True])
    orbited = moved.rotated(
        rotation=rotation,
        about=(0, 0, 0),
    )
    np.testing.assert_allclose(orbited.centre, (0, 3, 0), atol=1e-14)
    sphere = Sphere(
        radius=1,
        index=1.35,
    )
    grid = Grid(
        shape=(3, 3, 3),
        spacing=1,
    )
    for policy, expected in (("replace", 1.35), ("preserve", 1.34), ("error", None)):
        medium = StructuredMedium(overlap=policy)
        medium.add_structures(box, sphere)
        if expected is None:
            with pytest.raises(ValueError, match="overlaps"):
                medium.to_volume(grid=grid)
        else:
            volume = medium.to_volume(grid=grid)
            assert volume.delta_index[1, 1, 1] + volume.background_index == expected
    medium = StructuredMedium(warn_on_clipping=True)
    medium.add_structures(box)
    with pytest.warns(UserWarning, match="clipped"):
        medium.to_volume(grid=grid)
    with pytest.raises(ValueError, match="orthonormal"):
        box.rotated(rotation=np.diag([1, 1, -1]))


def test_compact_representations_and_advanced_imports():
    import bornsim

    grid, medium, solver = setup_problem()
    result = solver.solve(target=medium.to_volume(grid=grid))
    for value in (grid, solver.sampling, solver, result, result.angular):
        assert len(repr(value)) < 300
        assert "array(" not in repr(value)
    assert "BornSeries" not in bornsim.__all__
    with pytest.warns(DeprecationWarning, match="bornsim.series"):
        assert bornsim.BornSeries.__name__ == "BornSeries"


@pytest.mark.parametrize(
    "settings",
    [
        {"differential": [[-1]], "angles": [0]},
        {"differential": [[1]], "angles": [0], "mu_s": [1, 2]},
        {"differential": [[1]], "directions": [[0, 0, 2]]},
        {"differential": [[1]], "angles": [4]},
        {"differential": [[1]], "angles": [0], "sample_volume": -1},
    ],
)
def test_angular_data_validates_physical_arrays(settings):
    with pytest.raises(ValueError):
        AngularData(**settings)
