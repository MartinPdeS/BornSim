"""Independent illumination, polarization, rotation and normalization checks."""

import json
import numpy as np
import pytest

from bornsim import AngularSampling, Directions, EnsembleSampling, GaussianMedium, Grid, Result, Solver, Source, Volume
from bornsim.series import BornSeries
from bornsim.units import ureg


@pytest.fixture
def volume():
    grid = Grid(
        shape=(3, 3, 3),
        spacing=40 * ureg.nanometer,
    )

    delta_refractive_index = np.zeros(grid.shape)

    delta_refractive_index[0, 1, 2] = 0.02

    delta_refractive_index[2, 0, 1] = -0.01

    return Volume(
        grid=grid,
        background_refractive_index=1.33,
        delta_refractive_index=delta_refractive_index,
        extinction_coefficient=0.0004,
    )


def solve_cut(*, volume, incident_direction=(0, 0, 1), polarization=None, observations=None, order=3):
    source = Source(
        wavelength=633 * ureg.nanometer,
        incident_direction=incident_direction,
        polarization=polarization,
    )

    solver = Solver(
        source=source,
        order=order,
    )

    directions = Directions(vectors=np.eye(3) if observations is None else observations)

    return solver.solve_cut(
        target=volume,
        directions=directions,
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"incident_direction": [0, 0, 0]},
        {"incident_direction": None},
        {"incident_direction": [1, 1, 0]},
        {"incident_direction": [0, 0, np.nan]},
        {"incident_direction": [0, 0, np.inf]},
        {"incident_direction": [0j, 0, 1]},
        {"incident_direction": [[0, 0, 1]]},
        {"polarization": [0, 0, 1]},
        {"polarization": [0, 0, 0]},
        {"polarization": [1, 1, 0]},
        {"polarization": [np.nan, 0, 0]},
        {"polarization": [np.inf, 0, 0]},
        {"polarization": [1, 0]},
        {"polarization": [1, 0, 0] * ureg.dimensionless},
        {"incident_direction": [0, 0, 1] * ureg.dimensionless},
    ],
)
def test_invalid_illumination_is_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        Source(
            wavelength=633 * ureg.nanometer,
            **settings,
        )


def test_source_copies_vectors_and_provides_transverse_unpolarized_modes():
    direction = np.array([0.6, 0, 0.8])

    polarization = np.array([0, 1, 0])

    source = Source(
        wavelength=0.633 * ureg.micrometer,
        incident_direction=direction,
        polarization=polarization,
    )

    direction[:] = 0

    polarization[:] = 0

    np.testing.assert_array_equal(source.incident_direction, [0.6, 0, 0.8])

    np.testing.assert_array_equal(source.polarization, [0, 1, 0])

    for vector in (source.incident_direction, source.polarization):
        with pytest.raises(ValueError):
            vector.setflags(write=True)

    unpolarized = Source(
        wavelength=source.wavelength,
        incident_direction=source.incident_direction,
    )

    modes = unpolarized.polarization_vectors

    np.testing.assert_allclose(modes @ modes.conj().T, np.eye(2), atol=1e-15)

    np.testing.assert_allclose(modes @ source.incident_direction, 0, atol=1e-15)

    assert source.wavelength.units == ureg.micrometer


@pytest.mark.parametrize("polarization", [[1, 0, 0], [1 / np.sqrt(2), 1j / np.sqrt(2), 0]])
def test_first_order_oblique_illumination_matches_direct_voxel_sum(volume, polarization):
    # Cycle axes so the supplied polarization is transverse to an oblique beam.
    direction = np.array([0.6, 0, 0.8])

    basis = np.array([[0.8, 0, -0.6], [0, 1, 0], direction])

    electric_vector = np.asarray(polarization) @ basis

    observations = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])

    result = solve_cut(
        volume=volume,
        incident_direction=direction,
        polarization=electric_vector,
        observations=observations,
        order=1,
    )

    positions = volume.positions.to("meter").magnitude.reshape(-1, 3)

    k0 = 2 * np.pi / 633e-9

    contrast = 2 * 1.33 * (volume.delta_refractive_index + 1j * volume.extinction_coefficient)

    phase = np.exp(1j * k0 * 1.33 * ((direction - observations) @ positions.T))

    integral = phase @ contrast.ravel() * volume.spacing.to("meter").magnitude ** 3

    transverse = electric_vector - observations * (observations @ electric_vector)[:, None]

    expected = k0**2 / (4 * np.pi) * integral[:, None] * transverse

    np.testing.assert_allclose(result.amplitudes.to("meter").magnitude[0, :, 0], expected, rtol=1e-12, atol=1e-23)

    expected_intensity = np.sum(abs(expected) ** 2, axis=-1) / volume.volume.to("meter**3").magnitude

    np.testing.assert_allclose(result.differential.magnitude[0], expected_intensity, rtol=1e-12)


@pytest.mark.parametrize("second_weight", [1, 1j, 0.4 + 0.3j])
def test_complex_polarization_superposition_at_every_order(volume, second_weight):
    x_result = solve_cut(volume=volume, polarization=[1, 0, 0])

    y_result = solve_cut(volume=volume, polarization=[0, 1, 0])

    normalization = np.sqrt(1 + abs(second_weight) ** 2)

    polarized = solve_cut(
        volume=volume,
        polarization=np.array([1, second_weight, 0]) / normalization,
    )

    expected = (x_result.amplitudes + second_weight * y_result.amplitudes) / normalization

    np.testing.assert_allclose(polarized.amplitudes.magnitude, expected.magnitude, rtol=1e-11, atol=1e-23)


def test_unpolarized_intensity_and_absorption_are_average_of_pure_modes(volume):
    unpolarized = solve_cut(volume=volume)

    x_result = solve_cut(volume=volume, polarization=[1, 0, 0])

    y_result = solve_cut(volume=volume, polarization=[0, 1, 0])

    for name in ("differential", "term_differential", "mu_a"):
        expected = (getattr(x_result, name) + getattr(y_result, name)) / 2

        np.testing.assert_allclose(getattr(unpolarized, name).magnitude, expected.magnitude, rtol=1e-12)

    # In first order a uniform kappa absorbs 2*k0*kappa, independent of polarization.
    expected_absorption = 2 * (2 * np.pi / 633e-9) * 0.0004

    np.testing.assert_allclose(x_result.mu_a.magnitude[0], expected_absorption, rtol=1e-14)


@pytest.mark.parametrize("polarized", [False, True])
def test_axis_rotation_covariance_including_anisotropy_and_absorption(volume, polarized):
    # Cyclic permutation is an exact grid rotation, with no interpolation error.
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])

    rotated_volume = Volume(
        grid=volume.grid,
        background_refractive_index=volume.background_refractive_index,
        delta_refractive_index=volume.delta_refractive_index.transpose(2, 0, 1),
        extinction_coefficient=volume.extinction_coefficient.transpose(2, 0, 1),
    )

    polarization = np.array([1, 0.5j, 0]) / np.sqrt(1.25) if polarized else None

    original = solve_cut(
        volume=volume,
        polarization=polarization,
    )

    rotated = solve_cut(
        volume=rotated_volume,
        incident_direction=rotation @ np.array([0, 0, 1]),
        polarization=None if polarization is None else rotation @ polarization,
        observations=np.eye(3) @ rotation.T,
    )

    np.testing.assert_allclose(rotated.differential.magnitude, original.differential.magnitude, rtol=1e-11)

    np.testing.assert_allclose(rotated.mu_a.magnitude, original.mu_a.magnitude, rtol=1e-11)

    if polarized:
        np.testing.assert_allclose(
            rotated.amplitudes.magnitude, original.amplitudes.magnitude @ rotation.T, rtol=1e-11, atol=1e-23
        )

    sampling = AngularSampling(
        n_points=9,
        polar_samples=32,
        azimuth_samples=32,
    )

    full_results = []

    for target, source in ((volume, original.source), (rotated_volume, rotated.source)):
        solver = Solver(
            source=source,
            sampling=sampling,
            order=3,
        )

        full_results.append(solver.solve(target=target))

    for name in ("mu_s", "g", "mu_s_prime", "mu_a"):
        np.testing.assert_allclose(
            getattr(full_results[0], name).magnitude, getattr(full_results[1], name).magnitude, rtol=1e-10, atol=1e-12
        )

    assert abs(full_results[0].g.magnitude[-1]) > 1e-3


@pytest.mark.parametrize("polarized", [False, True])
def test_oblique_ensemble_matches_individual_seeded_solves(volume, polarized):
    medium = GaussianMedium(
        background_refractive_index=1.33,
        correlation_length=60 * ureg.nanometer,
        refractive_index_std=0.01,
        extinction_coefficient=0.0004,
    )

    source = Source(
        wavelength=633 * ureg.nanometer,
        incident_direction=[0.6, 0, 0.8],
        polarization=[0.8 / np.sqrt(2), 1j / np.sqrt(2), -0.6 / np.sqrt(2)] if polarized else None,
    )

    sampling = AngularSampling(
        n_points=5,
        polar_samples=16,
        azimuth_samples=8,
    )

    solver = Solver(
        source=source,
        sampling=sampling,
        order=3,
    )

    seeds = EnsembleSampling(
        realizations=3,
        seed=17,
    )

    ensemble = solver.ensemble(
        medium=medium,
        grid=volume.grid,
        ensemble_sampling=seeds,
    )

    individuals = []

    for seed in seeds.seeds:
        sample = medium.to_volume(
            grid=volume.grid,
            seed=seed,
        )

        individuals.append(solver.solve(target=sample))

    for name in ("differential", "mu_s", "mu_s_prime", "mu_a"):
        expected = np.mean([getattr(result, name).magnitude for result in individuals], axis=0)

        np.testing.assert_allclose(getattr(ensemble, name).magnitude, expected, rtol=1e-12)

    mean_amplitude = np.mean([np.cumsum(result.amplitudes.magnitude, axis=0) for result in individuals], axis=0)

    expected_coherent = (
        np.mean(np.sum(abs(mean_amplitude) ** 2, axis=-1), axis=-1) / volume.volume.to("meter**3").magnitude
    )

    np.testing.assert_allclose(ensemble.coherent_differential.magnitude, expected_coherent, rtol=1e-12)

    expected_stderr = np.std([result.differential.magnitude for result in individuals], axis=0, ddof=1) / np.sqrt(3)

    np.testing.assert_allclose(ensemble.stderr.magnitude, expected_stderr, rtol=1e-12)

    np.testing.assert_allclose(
        (ensemble.coherent_differential + ensemble.diffuse_differential).magnitude,
        ensemble.differential.magnitude,
    )


@pytest.mark.parametrize("polarized", [False, True])
def test_oblique_source_archive_round_trip(volume, tmp_path, polarized):
    result = solve_cut(
        volume=volume,
        incident_direction=[0, 1, 0],
        polarization=[1 / np.sqrt(2), 0, 1j / np.sqrt(2)] if polarized else None,
    )

    path = result.save(path=tmp_path / "polarized.npz")

    restored = Result.load(path=path)

    assert restored.source.metadata == result.source.metadata

    assert restored.provenance["source"] == result.source.metadata

    for name in ("amplitudes", "differential", "mu_a", "residual_norms"):
        np.testing.assert_array_equal(getattr(restored, name).magnitude, getattr(result, name).magnitude)


def test_legacy_schema_two_defaults_to_unpolarized_z(volume, tmp_path):
    result = solve_cut(volume=volume)

    path = result.save(path=tmp_path / "legacy.npz")

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    metadata = json.loads(arrays["metadata"].item())

    metadata["schema_version"] = 2

    metadata.pop("source")

    arrays["metadata"] = np.array(json.dumps(metadata))

    np.savez_compressed(path, **arrays)

    restored = Result.load(path=path)

    assert restored.source.polarization is None

    np.testing.assert_array_equal(restored.source.incident_direction, [0, 0, 1])

    np.testing.assert_array_equal(restored.amplitudes.magnitude, result.amplitudes.magnitude)


def test_engine_illumination_configuration_is_immutable(volume):
    direction = np.array([0, 1, 0])

    polarization = np.array([1, 0, 0])

    directions = Directions(vectors=np.eye(3))

    engine = BornSeries(
        grid=volume.grid,
        background_refractive_index=1.33,
        wavelength=633 * ureg.nanometer,
        directions=directions,
        incident_direction=direction,
        polarization=polarization,
        order=3,
    )

    expected = engine.solve(volume=volume)

    direction[:] = 0

    polarization[:] = 0

    actual = engine.solve(volume=volume)

    np.testing.assert_array_equal(actual.amplitudes, expected.amplitudes)
