"""Use infinite-medium formulas to validate a numerical limiting case."""

from bornsim.medium.random_medium import GaussianMedium, ExponentialMedium

import numpy as np
import pytest

from bornsim import AngularSampling, Grid, Solver, Source
from bornsim.units import ureg
from .reference import infinite_medium_scattering, infinite_medium_properties


@pytest.mark.parametrize("correlation", ["gaussian", "exponential"])
def test_first_order_numerical_phase_matches_rayleigh_reference(correlation):
    grid = Grid(
        shape=(3, 3, 3),
        spacing=25 * ureg.nanometer,
    )

    medium = {
        "gaussian": GaussianMedium,
        "exponential": ExponentialMedium,
    }[correlation](
        background_refractive_index=1.33,
        refractive_index_std=0.001,
        correlation_length=75 * ureg.nanometer,
    )

    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )

    source = Source(wavelength=1 * ureg.meter)

    sampling = AngularSampling(
        start=0 * ureg.degree,
        end=180 * ureg.degree,
        n_points=19,
        polar_samples=16,
        azimuth_samples=4,
    )

    solver = Solver(
        source=source,
        sampling=sampling,
        order=1,
    )

    result = solver.solve(target=volume)

    reference_curve = infinite_medium_scattering(
        medium=medium,
        wavelength=source.wavelength,
        theta=sampling.angles,
    )

    reference_properties = infinite_medium_properties(
        medium=medium,
        wavelength=source.wavelength,
    )

    reference_phase = (reference_curve / reference_properties["mu_s"]).to("1 / steradian").magnitude

    np.testing.assert_allclose(
        result.phase_function.magnitude[0],
        np.broadcast_to(reference_phase[:, None], (19, 4)),
        rtol=1e-8,
    )

    assert abs(result.g.magnitude[0]) < 1e-8
