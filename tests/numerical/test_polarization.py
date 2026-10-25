"""PyMieSim-style Jones inputs retain BornSim's incident-intensity convention."""

import numpy as np
import pytest

from bornsim import Directions, Grid, LeftCircular, PolarizationState, RightCircular, Solver, Source, Volume
from bornsim.units import ureg


@pytest.mark.parametrize("angle", [-90, 0, 30, 45, 90, 180, 360])
def test_linear_state_uses_explicit_angles(angle):
    state = PolarizationState(angle=angle * ureg.degree)

    expected = [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]

    np.testing.assert_allclose(state.jones_vector, expected, atol=1e-15)

    assert state.angle.units == ureg.degree


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"angle": 0 * ureg.degree, "jones_vector": [1, 0]},
        {"angle": 45},
        {"angle": 45 * ureg.dimensionless},
        {"angle": 1 * ureg.meter},
        {"angle": np.nan * ureg.degree},
        {"angle": np.inf * ureg.degree},
        {"angle": 1j * ureg.degree},
        {"angle": [0, 45] * ureg.degree},
        {"jones_vector": [0, 0]},
        {"jones_vector": [1, 0, 0]},
        {"jones_vector": [[1, 0]]},
        {"jones_vector": [1, np.inf]},
        {"jones_vector": [np.nan, 0]},
        {"jones_vector": [1, 0] * ureg.dimensionless},
    ],
)
def test_invalid_polarization_state_is_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        PolarizationState(**settings)


@pytest.mark.parametrize("scale", [1e-300, 1, 1e300])
def test_jones_vector_is_normalized_without_losing_relative_phase(scale):
    original = scale * np.array([1, 0.5j])

    state = PolarizationState(jones_vector=original)

    np.testing.assert_allclose(state.jones_vector, np.array([1, 0.5j]) / np.sqrt(1.25), atol=1e-15)

    original[:] = 0

    assert np.linalg.norm(state.jones_vector) == pytest.approx(1)

    with pytest.raises(ValueError):
        state.jones_vector.setflags(write=True)


def test_circular_states_match_pymiesim_signs():
    left = LeftCircular()

    right = RightCircular()

    np.testing.assert_allclose(left.jones_vector, np.array([1, 1j]) / np.sqrt(2))

    np.testing.assert_allclose(right.jones_vector, np.array([1, -1j]) / np.sqrt(2))

    assert isinstance(left, PolarizationState)

    assert isinstance(right, PolarizationState)


@pytest.mark.parametrize("direction", [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0.6, 0, 0.8]])
def test_jones_basis_is_transverse_and_right_handed(direction):
    horizontal_state = PolarizationState(angle=0 * ureg.degree)

    vertical_state = PolarizationState(angle=90 * ureg.degree)

    horizontal = Source(
        wavelength=633 * ureg.nanometer,
        incident_direction=direction,
        polarization=horizontal_state,
    )

    vertical = Source(
        wavelength=633 * ureg.nanometer,
        incident_direction=direction,
        polarization=vertical_state,
    )

    np.testing.assert_allclose(np.dot(horizontal.polarization, direction), 0, atol=1e-15)

    np.testing.assert_allclose(np.dot(vertical.polarization, direction), 0, atol=1e-15)

    np.testing.assert_allclose(np.cross(horizontal.polarization, vertical.polarization), direction, atol=1e-15)


@pytest.mark.parametrize(
    "direction,expected_first",
    [([0, 0, 1], [1, 0, 0]), ([0.6, 0, 0.8], [0.8, 0, -0.6]), ([1, 0, 0], [0, 1, 0])],
)
def test_zero_angle_basis_uses_projected_x_with_y_fallback(direction, expected_first):
    state = PolarizationState(angle=0 * ureg.degree)

    source = Source(
        wavelength=633 * ureg.nanometer,
        incident_direction=direction,
        polarization=state,
    )

    np.testing.assert_allclose(source.polarization, expected_first, atol=1e-15)


@pytest.mark.parametrize("circular_factory,sign", [(LeftCircular, 1), (RightCircular, -1)])
def test_jones_state_and_cartesian_source_give_identical_third_order_scattering(circular_factory, sign):
    grid = Grid(
        shape=(3, 3, 3),
        spacing=40 * ureg.nanometer,
    )

    volume = Volume(
        grid=grid,
        background_refractive_index=1.33,
        delta_refractive_index=np.random.default_rng(42).normal(0, 0.01, grid.shape),
        extinction_coefficient=0.0005,
    )

    directions = Directions(vectors=np.eye(3))

    state = circular_factory()

    cartesian = np.array([0.8, sign * 1j, -0.6]) / np.sqrt(2)

    results = []

    for polarization in (state, cartesian):
        source = Source(
            wavelength=633 * ureg.nanometer,
            incident_direction=[0.6, 0, 0.8],
            polarization=polarization,
        )

        solver = Solver(
            source=source,
            order=3,
        )

        result = solver.solve(
            target=volume,
            sampling=directions,
        )

        results.append(result)

    for name in ("amplitudes", "differential", "mu_a", "residual_norms"):
        np.testing.assert_allclose(
            getattr(results[0], name).magnitude, getattr(results[1], name).magnitude, rtol=1e-12, atol=1e-23
        )

    np.testing.assert_allclose(results[0].source.polarization, results[1].source.polarization, atol=1e-15)
