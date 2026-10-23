"""Weak-contrast spheres: Mie scattering, absorption and optical theorem."""

import numpy as np
import pytest

from bornsim import AngularSampling, Directions, Grid, Material, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg
from tests.analytical.mie_reference import sphere_reference


def test_mie_reference_normalization_and_rayleigh_limit():
    cosine, weights = np.polynomial.legendre.leggauss(64)
    radius, wavelength, host = 1e-9, 633e-9, 1.33
    refractive_index = 1.5
    scattering, absorption, angular = sphere_reference(
        radius_m=radius,
        wavelength_m=wavelength,
        host=host,
        refractive_index=refractive_index,
        cosine=cosine,
    )
    rayleigh = (
        8
        * np.pi
        / 3
        * (2 * np.pi * host / wavelength) ** 4
        * radius**6
        * abs(((refractive_index / host) ** 2 - 1) / ((refractive_index / host) ** 2 + 2)) ** 2
    )
    assert scattering == pytest.approx(rayleigh, rel=0.001)
    assert abs(absorption) < 1e-10 * scattering
    assert 2 * np.pi * np.sum(angular * weights) == pytest.approx(scattering, rel=1e-10)


@pytest.mark.parametrize("kappa", [0.0, 0.0002])
@pytest.mark.parametrize("radius_nm", [24, 36])
@pytest.mark.parametrize("refractive_index", [1.331, 1.34])
def test_sphere_against_exact_mie(kappa, radius_nm, refractive_index):
    grid = Grid(shape=(16, 16, 16), spacing=6 * ureg.nanometer)
    material = Material(refractive_index=refractive_index, extinction_coefficient=kappa)
    sphere = Sphere(radius=radius_nm * ureg.nanometer, material=material)
    medium = StructuredMedium()
    medium.add_background(refractive_index=1.33)
    medium.add_structures(sphere)
    volume = medium.to_volume(grid=grid)
    source = Source(wavelength=633 * ureg.nanometer)
    sampling = AngularSampling(n_points=19, polar_samples=16, azimuth_samples=16)
    solver = Solver(source=source, sampling=sampling, order=4)
    result = solver.solve(target=volume)
    # Match discretized occupied volume to isolate wave-operator accuracy from
    # staircase geometry errors; this is not a nominal-radius convergence claim.
    occupied = np.count_nonzero(volume.delta_refractive_index)
    radius = (3 * occupied * (6e-9) ** 3 / (4 * np.pi)) ** (1 / 3)
    sigma, absorption, angular = sphere_reference(
        radius_m=radius,
        wavelength_m=633e-9,
        host=1.33,
        refractive_index=refractive_index + 1j * kappa,
        cosine=np.cos(result.angles.to("radian").magnitude),
    )
    numeric_sigma = (result.mu_s * volume.volume).to("meter**2").magnitude[-1]
    assert numeric_sigma == pytest.approx(sigma, rel=0.02)
    np.testing.assert_allclose(result.azimuth_average().differential_cross_section.magnitude[-1], angular, rtol=0.03)
    if kappa:
        assert result.absorption_cross_section.to("meter**2").magnitude[-1] == pytest.approx(absorption, rel=0.01)
    else:
        assert np.all(result.mu_a.magnitude == 0)
    directions = Directions(vectors=[[0, 0, 1]])
    forward = solver.solve(target=volume, sampling=directions)
    amplitude = np.cumsum(forward.amplitudes.to("meter").magnitude, axis=0)[-1, 0]
    optical_extinction = 4 * np.pi / (2 * np.pi * 1.33 / 633e-9) * np.mean([amplitude[0, 0].imag, amplitude[1, 1].imag])
    total = numeric_sigma + result.absorption_cross_section.to("meter**2").magnitude[-1]
    assert optical_extinction == pytest.approx(total, rel=0.02)
