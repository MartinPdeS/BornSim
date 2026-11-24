"""Nominal sphere geometry refinement and larger weak-contrast optical sizes."""

import numpy as np
import pytest
from bornsim import AngularSampling, Grid, Material, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg
from tests.reference.mie_reference import sphere_reference


@pytest.mark.parametrize("extinction_coefficient", [0.0, 0.0002])
def test_larger_sphere_reference_truncation_and_normalization(*, extinction_coefficient):
    """Check Mie truncation and angular normalization at the largest tested size."""

    cosine, weights = np.polynomial.legendre.leggauss(64)

    settings = {
        "radius_m": 140e-9,
        "wavelength_m": 633e-9,
        "host": 1.33,
        "refractive_index": 1.34 + 1j * extinction_coefficient,
        "cosine": cosine,
    }

    reference = sphere_reference(**settings)

    refined = sphere_reference(
        **settings,
        multipole_terms=14,
    )

    for actual, expected in zip(reference, refined):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-28)

    scattering, absorption, angular = reference

    assert 2 * np.pi * np.sum(angular * weights) == pytest.approx(scattering, rel=1e-12)

    if extinction_coefficient:
        assert absorption > 0
    else:
        assert abs(absorption) < 1e-10 * scattering


@pytest.mark.parametrize("radius_nm", [40, 100, 140])
@pytest.mark.parametrize("extinction_coefficient", [0.0, 0.0002])
def test_nominal_sphere_mesh_refinement(*, radius_nm, extinction_coefficient):
    """Refine fixed geometry without fitting the reference to occupied voxels.

    The size parameter spans 0.53 to 1.85 at weak contrast. These cases test
    both staircase geometry and propagation; mesh refinement cannot eliminate
    the solver's linearized dielectric-contrast error.
    """

    material = Material(
        refractive_index=1.34,
        extinction_coefficient=extinction_coefficient,
    )

    sphere = Sphere(
        radius=radius_nm * ureg.nanometer,
        material=material,
    )

    medium = StructuredMedium()

    medium.add_background(refractive_index=1.33)

    medium.add_structures(sphere)

    source = Source(wavelength=633 * ureg.nanometer)

    sampling = AngularSampling(
        n_points=19,
        polar_samples=16,
        azimuth_samples=8,
    )

    solver = Solver(
        source=source,
        sampling=sampling,
        order=4,
    )

    expected_scattering, expected_absorption, expected_angular = sphere_reference(
        radius_m=float(sphere.radius.to("meter").magnitude),
        wavelength_m=float(source.wavelength.to("meter").magnitude),
        host=1.33,
        refractive_index=1.34 + 1j * extinction_coefficient,
        cosine=np.cos(sampling.angles.to("radian").magnitude),
    )

    scattering_errors, absorption_errors, angular_errors = [], [], []

    # Keep the nominal radius, box extent, illumination and material fixed.
    for cells_per_axis in (8, 16, 32):
        grid = Grid(
            shape=(cells_per_axis,) * 3,
            spacing=3 * sphere.radius / cells_per_axis,
        )

        volume = medium.to_volume(grid=grid)

        result = solver.solve(target=volume)

        scattering = (result.mu_s * grid.volume).to("meter**2").magnitude[-1]

        averaged = result.azimuth_average()

        angular = averaged.differential_cross_section.to("meter**2 / steradian").magnitude[-1]

        scattering_errors.append(abs(scattering / expected_scattering - 1))

        angular_errors.append(float(np.max(np.abs(angular / expected_angular - 1))))

        if extinction_coefficient:
            absorption = result.absorption_cross_section.to("meter**2").magnitude[-1]

            absorption_errors.append(abs(absorption / expected_absorption - 1))
        else:
            np.testing.assert_array_equal(result.mu_a.magnitude, 0)

    assert scattering_errors[2] < scattering_errors[1] < scattering_errors[0]

    assert scattering_errors[-1] < 0.03

    assert angular_errors[-1] < angular_errors[0]

    assert angular_errors[-1] < 0.05

    if extinction_coefficient:
        assert absorption_errors[2] < absorption_errors[1] < absorption_errors[0]

        assert absorption_errors[-1] < 0.02

    # Small truncation error here does not certify mesh or constitutive accuracy.
    np.testing.assert_allclose(result.mu_s.magnitude[-2], result.mu_s.magnitude[-1], rtol=1e-5)

    assert result.residual_norms.magnitude[-1] < 1e-8

    check_angular_refinement = radius_nm == 140 and extinction_coefficient > 0

    if check_angular_refinement:
        refined_sampling = AngularSampling(
            n_points=19,
            polar_samples=24,
            azimuth_samples=16,
        )

        refined = solver.solve(
            target=volume,
            sampling=refined_sampling,
        )

        np.testing.assert_allclose(refined.mu_s.magnitude, result.mu_s.magnitude, rtol=1e-4)

        np.testing.assert_allclose(refined.g.magnitude, result.g.magnitude, atol=1e-5)
