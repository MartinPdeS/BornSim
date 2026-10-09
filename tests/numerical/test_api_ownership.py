"""Immutable angular ownership, materials and reproducible direction selection."""

from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from bornsim import (
    AngularSampling,
    EnsembleSampling,
    Grid,
    Material,
    Result,
    Rotation,
    Solver,
    Source,
    Sphere,
    StructuredMedium,
    RandomMedium,
)
from bornsim.units import ureg


def problem():
    grid = Grid(
        shape=(3, 3, 3),
        spacing=3e-08 * ureg.meter,
    )

    medium = RandomMedium(
        refractive_index_std=0.001,
        background_refractive_index=1.33,
        correlation_length=100e-9 * ureg.meter,
        correlation="matern",
        smoothness=1.5,
    )

    sampling = AngularSampling(
        angles=[0, np.pi / 2, np.pi] * ureg.radian,
        polar_samples=16,
        azimuth_samples=4,
    )

    solver = Solver(
        source=Source(
            wavelength=633e-9 * ureg.meter,
        ),
        order=2,
        sampling=sampling,
    )

    volume = medium.to_volume(grid=grid, seed=42)

    return grid, medium, solver, volume


def test_result_owns_one_angular_object_and_is_immutable(tmp_path):
    _, _, solver, volume = problem()

    result = solver.solve(target=volume)

    assert result.angular is result.angular

    assert result.differential is result.angular.differential

    assert result.amplitudes is result.angular.amplitudes

    clone = Result(
        source=result.source,
        kind=result.kind,
        angular=result.angular,
    )

    assert clone.angular is result.angular

    with pytest.raises(FrozenInstanceError):
        result.kind = "analytical"

    with pytest.raises(FrozenInstanceError):
        result.differential = result.differential * 2

    for name in ("differential", "amplitudes", "angles", "azimuths", "mu_s", "g", "mu_s_prime", "field_norms"):
        values = getattr(result, name).magnitude

        assert not values.flags.writeable

        with pytest.raises(ValueError):
            values.flat[0] = 42

    metadata = result.provenance

    metadata["grid"]["shape"][0] = 99

    assert result.provenance["grid"]["shape"][0] == 3

    path = result.save(path=tmp_path / "immutable.npz")

    restored = Result.load(path=path)

    assert restored.differential is restored.angular.differential

    assert not restored.amplitudes.magnitude.flags.writeable


def test_material_shared_across_shapes_and_transforms():
    material = Material(refractive_index=1.34)

    sphere = Sphere(
        radius=3e-08 * ureg.meter,
        material=material,
    )

    second = sphere.translated(offset=(3e-08, 0, 0) * ureg.meter)

    rotated = second.rotated(
        rotation=Rotation(
            angle=np.pi / 2 * ureg.radian,
            axis=(0.0, 0.0, 1.0),
        )
    )

    assert sphere.material is second.material is rotated.material is material

    assert sphere.refractive_index == material.refractive_index

    medium = StructuredMedium(
        background_refractive_index=1.33,
    )

    medium.add_background(material=Material(refractive_index=1.33))

    medium.add_structures(sphere, second)

    volume = medium.to_volume(
        grid=Grid(
            shape=(3, 3, 3),
            spacing=3e-08 * ureg.meter,
        )
    )

    assert volume.delta_refractive_index[1, 1, 1] == pytest.approx(0.01)

    assert medium.metadata["regions"][0]["refractive_index"] == 1.34

    with pytest.raises(ValueError, match="not both"):
        Sphere(
            radius=3e-08 * ureg.meter,
            material=material,
            refractive_index=1.35,
        )

    with pytest.raises(FrozenInstanceError):
        material.refractive_index = 1.35


def test_explicit_ensemble_seeds_match_independent_samples_and_archive(tmp_path):
    grid, medium, solver, _ = problem()

    configuration = EnsembleSampling(seeds=[42, 11, 104])

    result = solver.ensemble(
        medium=medium,
        grid=grid,
        ensemble_sampling=configuration,
    )

    samples = [
        solver.solve(target=medium.to_volume(grid=grid, seed=seed)).differential.magnitude
        for seed in configuration.seeds
    ]

    np.testing.assert_allclose(result.differential.magnitude, np.mean(samples, axis=0))

    np.testing.assert_allclose(result.stderr.magnitude, np.std(samples, axis=0, ddof=1) / np.sqrt(3))

    assert result.provenance["seeds"] == [42, 11, 104]

    restored = Result.load(path=result.save(path=tmp_path / "seeds.npz"))

    assert restored.provenance["ensemble_sampling"] == configuration.metadata

    consecutive = EnsembleSampling(
        realizations=3,
        seed=42,
    )

    assert consecutive.seeds == (42, 43, 44)

    with pytest.raises(ValueError, match="not both"):
        solver.ensemble(
            medium=medium,
            grid=grid,
            ensemble_sampling=configuration,
            seed=0,
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"seeds": [1, 1]},
        {"seeds": []},
        {"seeds": [1], "seed": 0},
        {"seeds": [-1]},
        {"seed": 2**32 - 1, "realizations": 2},
        {"seeds": [True]},
    ],
)
def test_invalid_ensemble_sampling(settings):
    with pytest.raises(ValueError):
        EnsembleSampling(**settings)


def test_meridians_select_angles_without_interpolation_or_averaging():
    _, _, solver, volume = problem()

    result = solver.solve(target=volume)

    meridian = result.meridian(azimuth=90 * ureg.degree)

    np.testing.assert_array_equal(meridian.differential.magnitude, result.differential.magnitude[..., 1])

    np.testing.assert_array_equal(meridian.amplitudes.magnitude, result.amplitudes.magnitude[:, :, 1])

    np.testing.assert_allclose(meridian.phase_function.magnitude, result.phase_function.magnitude[..., 1])

    assert meridian.meridian_azimuth.to("degree").magnitude == pytest.approx(90)

    wrapped = result.meridian(azimuth=-90 * ureg.degree)

    np.testing.assert_array_equal(wrapped.differential.magnitude, result.differential.magnitude[..., 3])

    with pytest.raises(ValueError, match="not sampled"):
        result.meridian(azimuth=87 * ureg.degree)

    nearest = result.meridian(
        azimuth=87 * ureg.degree,
        method="nearest",
    )

    np.testing.assert_array_equal(nearest.differential.magnitude, meridian.differential.magnitude)

    assert "90 degrees" in meridian.plot().axes[0].get_title()

    assert meridian.plot_phase_function().axes[0].get_ylabel()

    with pytest.raises(ValueError, match="full angular"):
        meridian.plot_phase_function(
            view="3d",
        )


def test_legacy_configuration_warns_and_keeps_values():
    with pytest.warns(DeprecationWarning, match="Grid"):
        grid = Grid._resolve(shape=(2, 2, 2), spacing=3e-08 * ureg.meter)

    assert grid.shape == (2, 2, 2)

    with pytest.warns(DeprecationWarning, match="AngularSampling"):
        AngularSampling._resolve(angles=[0] * ureg.radian)

    with pytest.warns(DeprecationWarning, match="EnsembleSampling"):
        configuration = EnsembleSampling._resolve(realizations=2, seed=42)

    assert configuration.seeds == (42, 43)
