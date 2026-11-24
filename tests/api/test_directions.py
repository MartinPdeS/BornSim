"""Observation directions are immutable configurations with explicit coordinates."""

from dataclasses import FrozenInstanceError
import numpy as np
import pytest

from bornsim import Directions, Grid, Source, Solver, Volume, AngularSampling
from bornsim.api import Directions as ApiDirections
from bornsim.series import BornSeries
from bornsim.units import ureg


def test_directions_own_immutable_vectors_and_preserve_order():
    input_vectors = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    directions = Directions(vectors=input_vectors)

    input_vectors[:] = 0

    np.testing.assert_array_equal(directions.vectors, [[0, 0, 1], [1, 0, 0]])

    assert len(directions) == 2

    assert ApiDirections is Directions

    with pytest.raises(ValueError):
        directions.vectors[0, 0] = 1

    with pytest.raises(ValueError):
        directions.vectors.setflags(write=True)

    with pytest.raises(FrozenInstanceError):
        directions.vectors = np.zeros((2, 3))


@pytest.mark.parametrize(
    "vectors",
    [[], [0, 0, 1], [[0, 0]], [[0, 0, 0]], [[0, 0, 2]], [[0, 0, np.nan]], [[0, 0, np.inf]], [[0, 0, 1j]]],
)
def test_invalid_vectors_are_rejected_without_normalizing(vectors):
    with pytest.raises(ValueError, match="directions.vectors"):
        Directions(vectors=vectors)


def test_paired_angles_and_broadcast_grids_preserve_spherical_coordinates():
    paired = Directions.from_angles(
        polar_angles=[0, 90, 90] * ureg.degree,
        azimuth_angles=[0, 0, np.pi / 2] * ureg.radian,
    )

    np.testing.assert_allclose(paired.vectors, [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-15)

    grid = Directions.from_angles(
        polar_angles=np.array([0, 90])[:, None] * ureg.degree,
        azimuth_angles=np.array([0, 90])[None, :] * ureg.degree,
    )

    np.testing.assert_allclose(grid.vectors, [[0, 0, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-15)


@pytest.mark.parametrize(
    "polar_angles, azimuth_angles",
    [
        ([0], [0] * ureg.radian),
        ([0] * ureg.radian, [0]),
        ([0] * ureg.meter, [0] * ureg.radian),
        ([-1] * ureg.degree, [0] * ureg.degree),
        ([181] * ureg.degree, [0] * ureg.degree),
        ([0] * ureg.degree, [np.nan] * ureg.degree),
    ],
)
def test_angle_inputs_require_explicit_valid_angular_units(polar_angles, azimuth_angles):
    with pytest.raises(ValueError):
        Directions.from_angles(
            polar_angles=polar_angles,
            azimuth_angles=azimuth_angles,
        )


def test_sampling_solver_and_engine_share_direction_configuration():
    grid = Grid(
        shape=(2, 2, 2),
        spacing=50 * ureg.nanometer,
    )

    volume = Volume(
        delta_refractive_index=np.full(grid.shape, 0.001),
        grid=grid,
        background_refractive_index=1.33,
    )

    source = Source(wavelength=633 * ureg.nanometer)

    directions = Directions(vectors=[[0, 0, 1], [1, 0, 0]])

    solver = Solver(
        source=source,
        order=1,
    )

    result = solver.solve_cut(
        target=volume,
        directions=directions,
    )

    np.testing.assert_array_equal(result.directions.magnitude, directions.vectors)

    engine = BornSeries(
        grid=grid,
        wavelength=source.wavelength,
        background_refractive_index=volume.background_refractive_index,
        directions=directions,
        order=1,
    )

    assert engine.directions is directions

    born_result = engine.solve(volume=volume)

    np.testing.assert_array_equal(result.amplitudes.magnitude, born_result.amplitudes)

    sampling = AngularSampling(angles=[0, 90, 180] * ureg.degree)

    assert isinstance(sampling.directions, Directions)

    with pytest.raises(TypeError, match="Directions"):
        solver.solve_cut(
            target=volume,
            directions=directions.vectors,
        )
