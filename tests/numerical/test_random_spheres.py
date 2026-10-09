"""Independent geometric checks for seeded, finite sphere collections."""

import numpy as np
import pytest
from bornsim import AngularSampling, EnsembleSampling, Grid, Result, Solver, Source
from bornsim.medium.random_spheres import RandomSphereMedium
from bornsim.units import ureg


def test_centers_are_contained_nonoverlapping_and_reproducible():
    grid = Grid(
        shape=(12, 14, 16),
        spacing=25 * ureg.nanometer,
    )

    medium = RandomSphereMedium(
        background_refractive_index=1.33,
        sphere_refractive_index=1.35,
        radius=30 * ureg.nanometer,
        sphere_count=20,
    )

    centers = medium.sphere_centers(
        grid=grid,
        seed=42,
    )

    repeated_centers = medium.sphere_centers(
        grid=grid,
        seed=42,
    )

    other_centers = medium.sphere_centers(
        grid=grid,
        seed=43,
    )

    np.testing.assert_array_equal(centers.magnitude, repeated_centers.magnitude)

    assert not np.array_equal(centers.magnitude, other_centers.magnitude)

    assert centers.shape == (20, 3)

    assert np.all(np.abs(centers) + medium.radius <= np.asarray(grid.shape) * grid.spacing / 2)

    center_distances = np.sqrt(np.sum((centers[:, None] - centers[None, :]) ** 2, axis=-1))

    assert np.all(center_distances[np.triu_indices(20, k=1)] >= 2 * medium.radius)

    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )

    # Independent mask reference in nanometres, using explicit centre distances.
    positions_nm = grid.positions.to("nanometer").magnitude

    expected_mask = np.zeros(grid.shape, dtype=bool)

    for center_nm in centers.to("nanometer").magnitude:
        expected_mask |= np.linalg.norm(positions_nm - center_nm, axis=-1) <= 30

    np.testing.assert_array_equal(volume.delta_refractive_index, expected_mask * (1.35 - 1.33))

    assert volume.medium is medium and volume.seed == 42


def test_impossible_packing_and_exhausted_attempts_raise():
    grid = Grid(
        shape=(2, 2, 2),
        spacing=1 * ureg.micrometer,
    )

    for radius, count, message in ((1.1, 1, "diameter"), (1, 2, "total volume"), (0.9, 2, "Could only place")):
        medium = RandomSphereMedium(
            background_refractive_index=1.33,
            sphere_refractive_index=1.35,
            radius=radius * ureg.micrometer,
            sphere_count=count,
            max_placement_attempts=2,
        )

        with pytest.raises(ValueError, match=message):
            medium.to_volume(
                grid=grid,
                seed=0,
            )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"radius": 30}, "units"),
        ({"radius": 30 * ureg.second}, "compatible"),
        ({"radius": [30] * ureg.nanometer}, "scalar"),
        ({"radius": 0 * ureg.nanometer}, "positive"),
        ({"radius": np.inf * ureg.nanometer}, "finite"),
        ({"sphere_count": 0}, "sphere_count"),
        ({"sphere_count": 1.5}, "sphere_count"),
        ({"sphere_count": True}, "sphere_count"),
        ({"max_placement_attempts": 1}, "max_placement_attempts"),
        ({"sphere_refractive_index": 1.35 * ureg.dimensionless}, "unitless"),
        ({"background_refractive_index": 1.33 * ureg.dimensionless}, "unitless"),
        ({"sphere_refractive_index": 0.1}, "linearized"),
    ],
)
def test_invalid_sphere_configuration(changes, message):
    configuration = {
        "background_refractive_index": 1.33,
        "sphere_refractive_index": 1.35,
        "radius": 30 * ureg.nanometer,
        "sphere_count": 2,
        **changes,
    }

    with pytest.raises(ValueError, match=message):
        RandomSphereMedium(**configuration)


def test_sphere_ensemble_and_archive_preserve_generation_settings(tmp_path):
    grid = Grid(
        shape=(4, 4, 4),
        spacing=40 * ureg.nanometer,
    )

    medium = RandomSphereMedium(
        background_refractive_index=1.33,
        sphere_refractive_index=1.35,
        radius=35 * ureg.nanometer,
        sphere_count=2,
    )

    source = Source(wavelength=633 * ureg.nanometer)

    solver = Solver(
        source=source,
        order=1,
    )

    sampling = AngularSampling(
        start=0 * ureg.degree,
        end=180 * ureg.degree,
        n_points=5,
        polar_samples=16,
        azimuth_samples=4,
    )

    ensemble_sampling = EnsembleSampling(seeds=(42, 43))

    result = solver.ensemble(
        medium=medium,
        grid=grid,
        sampling=sampling,
        ensemble_sampling=ensemble_sampling,
    )

    assert result.provenance["medium"] == medium.metadata

    assert np.all(np.isfinite(result.mu_s))

    assert np.all(np.isfinite(result.stderr))

    result.save(path=tmp_path / "spheres.npz")

    restored = Result.load(path=tmp_path / "spheres.npz")

    assert restored.provenance == result.provenance

    np.testing.assert_array_equal(restored.differential.magnitude, result.differential.magnitude)
