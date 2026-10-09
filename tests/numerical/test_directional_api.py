"""Public directional semantics, geometry transforms and archive migration."""

from bornsim.medium.random_medium import WhittleMaternMedium

from bornsim import EnsembleSampling
from bornsim import AngularSampling
from bornsim import Grid
import numpy as np
import pytest
from bornsim import AngularData, Box, Result, Rotation, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg


def setup_problem():
    grid = Grid(
        shape=(3, 3, 3),
        spacing=4e-08 * ureg.meter,
    )

    medium = StructuredMedium(background_refractive_index=1.33)

    sphere = Sphere(
        radius=5e-08 * ureg.meter,
        refractive_index=1.34,
        centre=(2e-08, 0, 0) * ureg.meter,
    )

    medium.add_structures(sphere)

    sampling = AngularSampling(
        angles=[0, np.pi / 2, np.pi] * ureg.radian,
        polar_samples=16,
        azimuth_samples=4,
    )

    solver = Solver(
        source=Source(wavelength=6.33e-07 * ureg.meter),
        order=2,
        sampling=sampling,
    )

    return (grid, medium, solver)


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
        angles=[0, np.pi / 2] * ureg.radian,
    )

    assert cut.amplitudes.shape == (2, 2, 2, 3)

    assert cut.mu_s is None

    with pytest.raises(TypeError, match="angles"):
        solver.solve(
            target=volume,
            angles=[0] * ureg.radian,
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

    medium = WhittleMaternMedium(
        refractive_index_std=0.001,
        background_refractive_index=1.33,
        correlation_length=1e-07 * ureg.meter,
        smoothness=1.5,
    )

    result = solver.ensemble(
        medium=medium,
        grid=grid,
        ensemble_sampling=EnsembleSampling(
            realizations=3,
            seed=42,
        ),
    )

    curves = [
        solver.solve(
            target=medium.to_volume(
                grid=grid,
                seed=seed,
            )
        ).differential.magnitude.mean(axis=-1)
        for seed in (42, 43, 44)
    ]

    expected = np.std(curves, axis=0, ddof=1) / np.sqrt(3)

    np.testing.assert_allclose(result.azimuth_average().stderr.magnitude, expected)

    np.testing.assert_allclose(result.angular.azimuth_average().stderr.magnitude, expected)


def test_rotations_translation_and_overlap():
    box = Box(
        size=(4, 2, 2) * ureg.meter,
        refractive_index=1.34,
    )

    rotation = Rotation(
        axis=(0, 0, 1),
        angle=90 * ureg.degree,
    )

    rotated = box.rotated(rotation=rotation)

    positions = np.array([[1.5, 0, 0], [0, 1.5, 0]])

    np.testing.assert_array_equal(box.mask(positions=positions * ureg.meter), [True, False])

    np.testing.assert_array_equal(rotated.mask(positions=positions * ureg.meter), [False, True])

    moved = rotated.translated(offset=(3, 0, 0) * ureg.meter)

    np.testing.assert_array_equal(moved.mask(positions=(positions + [3, 0, 0]) * ureg.meter), [False, True])

    orbited = moved.rotated(
        rotation=rotation,
        about=(0, 0, 0) * ureg.meter,
    )

    np.testing.assert_allclose(orbited.centre, (0, 3, 0), atol=1e-14)

    sphere = Sphere(
        radius=1 * ureg.meter,
        refractive_index=1.35,
    )

    grid = Grid(
        shape=(3, 3, 3),
        spacing=1 * ureg.meter,
    )

    for policy, expected in (("replace", 1.35), ("preserve", 1.34), ("error", None)):
        medium = StructuredMedium(
            overlap=policy,
            background_refractive_index=1.33,
        )

        medium.add_structures(box, sphere)

        if expected is None:
            with pytest.raises(ValueError, match="overlaps"):
                medium.to_volume(grid=grid)
        else:
            volume = medium.to_volume(grid=grid)

            assert volume.delta_refractive_index[1, 1, 1] + volume.background_refractive_index == expected

    medium = StructuredMedium(
        warn_on_clipping=True,
        background_refractive_index=1.33,
    )

    medium.add_structures(box)

    with pytest.warns(UserWarning, match="clipped"):
        medium.to_volume(grid=grid)

    with pytest.raises(ValueError, match="orthonormal"):
        box.rotated(rotation=np.diag([1, 1, -1]))


@pytest.mark.parametrize(
    "settings",
    [
        {"differential": [[-1]] * (1 / ureg.meter / ureg.steradian), "angles": [0] * ureg.radian},
        {
            "differential": [[1]] * (1 / ureg.meter / ureg.steradian),
            "angles": [0] * ureg.radian,
            "mu_s": [1, 2] * (1 / ureg.meter),
        },
        {"differential": [[1]] * (1 / ureg.meter / ureg.steradian), "directions": [[0, 0, 2]]},
        {"differential": [[1]] * (1 / ureg.meter / ureg.steradian), "angles": [4] * ureg.radian},
        {
            "differential": [[1]] * (1 / ureg.meter / ureg.steradian),
            "angles": [0] * ureg.radian,
            "sample_volume": -1 * ureg.meter**3,
        },
    ],
)
def test_angular_data_validates_physical_arrays(settings):
    with pytest.raises(ValueError):
        AngularData(**settings)
