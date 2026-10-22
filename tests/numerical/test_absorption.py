"""Passive sample absorption with a lossless propagation background."""

import numpy as np
import pytest

from bornsim import Directions, Grid, Result, Solver, Source, Volume
from bornsim.units import ureg
from bornsim.series import BornSeries
from tests.numerical.test_series import independent_matrix


def test_absorption_limit_and_archive(tmp_path):
    grid = Grid(shape=(2, 2, 2), spacing=10 * ureg.nanometer)
    volume = Volume(
        grid=grid,
        background_refractive_index=1.33,
        delta_refractive_index=np.zeros(grid.shape),
    )
    absorbing = volume.with_absorption(extinction_coefficient=0.001)
    source = Source(wavelength=633 * ureg.nanometer)
    directions = Directions(vectors=[[0, 0, 1], [1, 0, 0]])
    solver = Solver(source=source, order=2)
    result = solver.solve(target=absorbing, sampling=directions)
    # Incident-field dissipation gives Beer-Lambert intensity attenuation.
    expected = 4 * np.pi * 0.001 / (633e-9)
    assert result.mu_a.to("1/meter").magnitude[0] == pytest.approx(expected)
    assert np.all(result.mu_a.magnitude > 0)
    assert np.all(result.differential.magnitude >= 0)
    path = result.save(path=tmp_path / "absorbing.npz")
    restored = Result.load(path=path)
    np.testing.assert_array_equal(restored.mu_a.magnitude, result.mu_a.magnitude)
    assert not absorbing.extinction_coefficient.flags.writeable
    assert np.all(volume.extinction_coefficient == 0)


@pytest.mark.parametrize("value", [-0.01, np.nan, np.ones((3, 3, 3))])
def test_absorption_validation(value):
    grid = Grid(shape=(2, 2, 2), spacing=10 * ureg.nanometer)
    with pytest.raises(ValueError, match="extinction_coefficient"):
        Volume(
            grid=grid,
            background_refractive_index=1.33,
            delta_refractive_index=np.zeros(grid.shape),
            extinction_coefficient=value,
        )


def test_absorbed_power_against_dense_internal_field():
    grid = Grid(shape=(2, 2, 2), spacing=15 * ureg.nanometer)
    kappa = np.arange(8).reshape(grid.shape) * 0.0002
    volume = Volume(
        grid=grid,
        background_refractive_index=1.33,
        delta_refractive_index=np.full(grid.shape, 0.002),
        extinction_coefficient=kappa,
    )
    directions = Directions(vectors=[[0, 0, 1]])
    engine = BornSeries(
        grid=grid,
        background_refractive_index=1.33,
        wavelength=633 * ureg.nanometer,
        directions=directions,
        order=3,
    )
    result = engine.solve(volume=volume)
    matrix = independent_matrix(volume, 633e-9)
    phase = np.exp(2j * np.pi * 1.33 / 633e-9 * grid.positions.to("meter").magnitude[..., 2])
    contrast = 2 * 1.33 * (volume.delta_refractive_index + 1j * kappa)
    expected = np.zeros(3)
    for polarization in ([1, 0, 0], [0, 1, 0]):
        term = phase[..., None] * np.array(polarization)
        total = term.copy()
        for order in range(3):
            expected[order] += (2 * np.pi / 633e-9) * np.mean(kappa * np.sum(abs(total) ** 2, axis=-1))
            term = (matrix @ (contrast[..., None] * term).ravel()).reshape(*grid.shape, 3)
            total += term
    np.testing.assert_allclose(result.mu_a, expected, rtol=1e-9)
