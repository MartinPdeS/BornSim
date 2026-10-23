"""Ensemble statistics, residuals, persistence and refinement reports."""

import numpy as np
import pytest

from bornsim import AngularSampling, ConvergenceReport, EnsembleSampling, GaussianMedium, Grid, Result, Solver, Source
from bornsim.units import ureg
from tests.numerical.test_series import independent_matrix
from bornsim import Box, Material, StructuredMedium


def test_ensemble_components_and_absorption_archive(tmp_path):
    grid = Grid(shape=(3, 3, 3), spacing=20 * ureg.nanometer)
    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.002,
        correlation_length=40 * ureg.nanometer,
        extinction_coefficient=0.0003,
    )
    sampling = AngularSampling(n_points=9, polar_samples=16, azimuth_samples=16)
    source = Source(wavelength=633 * ureg.nanometer)
    solver = Solver(source=source, sampling=sampling, order=3)
    seeds = EnsembleSampling(realizations=3, seed=42)
    ensemble = solver.ensemble(medium=medium, grid=grid, ensemble_sampling=seeds)
    samples = []
    for seed in seeds.seeds:
        volume = medium.to_volume(grid=grid, seed=seed)
        samples.append(solver.solve(target=volume))
    amplitudes = np.array([np.cumsum(sample.amplitudes.magnitude, axis=0) for sample in samples])
    mean = amplitudes.mean(axis=0)
    expected_coherent = np.sum(abs(mean) ** 2, axis=(-1, -2)) / (2 * grid.volume.to("meter**3").magnitude)
    expected_diffuse = np.mean(np.sum(abs(amplitudes - mean) ** 2, axis=(-1, -2)), axis=0) / (
        2 * grid.volume.to("meter**3").magnitude
    )
    np.testing.assert_allclose(ensemble.coherent_differential.magnitude, expected_coherent, rtol=1e-12)
    np.testing.assert_allclose(ensemble.diffuse_differential.magnitude, expected_diffuse, rtol=1e-10)
    absorption = np.array([sample.mu_a.magnitude for sample in samples])
    np.testing.assert_allclose(ensemble.mu_a.magnitude, absorption.mean(axis=0))
    np.testing.assert_allclose(ensemble.mu_a_stderr.magnitude, absorption.std(axis=0, ddof=1) / np.sqrt(3))
    averaged = ensemble.azimuth_average()
    restored = Result.load(path=averaged.save(path=tmp_path / "statistics.npz"))
    np.testing.assert_array_equal(restored.coherent_differential.magnitude, averaged.coherent_differential.magnitude)
    np.testing.assert_array_equal(restored.residual_norms.magnitude, ensemble.residual_norms.magnitude)
    report = ConvergenceReport(results=(ensemble, restored), labels=("full", "averaged"))
    assert report.rows[-1]["stable"]


def test_residual_against_dense_equation():
    grid = Grid(shape=(2, 2, 2), spacing=20 * ureg.nanometer)
    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.02,
        correlation_length=40 * ureg.nanometer,
        extinction_coefficient=0.001,
    )
    volume = medium.to_volume(grid=grid, seed=42)
    source = Source(wavelength=633 * ureg.nanometer)
    solver = Solver(source=source, order=3)
    result = solver.solve(target=volume)
    green = independent_matrix(volume, 633e-9)
    contrast = 2 * 1.33 * (volume.delta_refractive_index + 1j * volume.extinction_coefficient)
    operator = green * np.repeat(contrast.ravel(), 3)[None, :]
    phase = np.exp(2j * np.pi * 1.33 / 633e-9 * grid.positions.to("meter").magnitude[..., 2])
    initial = np.stack([(phase[..., None] * np.array(p)).ravel() for p in ([1, 0, 0], [0, 1, 0])], axis=1)
    total, term = initial.copy(), initial.copy()
    for order in range(3):
        term = operator @ term
        total += term
        residual = np.linalg.norm(initial + operator @ total - total) / np.linalg.norm(initial)
        assert result.residual_norms.magnitude[order] == pytest.approx(residual, rel=1e-5, abs=1e-14)


def test_absorbing_material_replacement_and_metadata():
    grid = Grid(shape=(4, 4, 4), spacing=20 * ureg.nanometer)
    background = Material(refractive_index=1.33, extinction_coefficient=0.001)
    absorbing = Material(refractive_index=1.335, extinction_coefficient=0.002)
    clear = Material(refractive_index=1.34)
    first = Box(size=[40, 80, 80] * ureg.nanometer, material=absorbing)
    second = Box(size=[20, 80, 80] * ureg.nanometer, centre=[10, 0, 0] * ureg.nanometer, material=clear)
    medium = StructuredMedium()
    medium.add_background(material=background)
    medium.add_structures(first, second)
    volume = medium.to_volume(grid=grid)
    expected = np.full(grid.shape, 0.001)
    expected[first.mask(positions=grid.positions)] = 0.002
    expected[second.mask(positions=grid.positions)] = 0
    np.testing.assert_array_equal(volume.extinction_coefficient, expected)
    assert medium.metadata["regions"][0]["extinction_coefficient"] == 0.002
    assert medium.metadata["extinction_coefficient"] == 0.001


def test_single_realization_has_unknown_absorption_uncertainty():
    grid = Grid(shape=(2, 2, 2), spacing=20 * ureg.nanometer)
    medium = GaussianMedium(
        background_refractive_index=1.33,
        refractive_index_std=0.002,
        correlation_length=40 * ureg.nanometer,
        extinction_coefficient=0.0003,
    )
    source = Source(wavelength=633 * ureg.nanometer)
    solver = Solver(source=source, order=2)
    seeds = EnsembleSampling(realizations=1, seed=42)
    result = solver.ensemble(medium=medium, grid=grid, ensemble_sampling=seeds)
    assert np.all(np.isnan(result.mu_a_stderr.magnitude))
    np.testing.assert_allclose(result.diffuse_differential.magnitude, 0, atol=1e-15)


@pytest.mark.parametrize("tolerance", [0, -1, np.nan])
def test_invalid_convergence_tolerance(tolerance):
    with pytest.raises(ValueError):
        ConvergenceReport(results=(None, None), labels=("a", "b"), tolerance=tolerance)


def test_report_detects_phase_change_with_unchanged_integrals():
    source = Source(wavelength=633 * ureg.nanometer)
    angles = np.linspace(0, np.pi, 101) * ureg.radian
    cosine = np.cos(angles.magnitude)
    legendre_four = (35 * cosine**4 - 30 * cosine**2 + 3) / 8
    results = []
    for amplitude in (0.0, 0.5):
        results.append(
            Result(
                source=source,
                kind="volume",
                angles=angles,
                differential=(1 + amplitude * legendre_four)[None, :] * (1 / ureg.meter / ureg.steradian),
                mu_s=np.array([4 * np.pi]) * (1 / ureg.meter),
                azimuth_averaged=True,
            )
        )
    report = ConvergenceReport(results=tuple(results), labels=("isotropic", "fourth moment"))
    assert report.rows[-1]["changes"]["mu_s"] == 0
    assert report.rows[-1]["changes"]["phase_l1"] > 0.01
    assert not report.rows[-1]["stable"]
