"""Independent voxel checks for finite particle and lattice media."""

import numpy as np
import pytest
from bornsim import CoreShellMedium, CylinderMedium, EllipsoidMedium, PeriodicMedium, Grid, Sphere, StructuredMedium
from bornsim.units import ureg


def test_compatibility_imports():
    from bornsim.geometry import StructuredMedium as historical
    from bornsim.medium.structured import StructuredMedium as canonical
    from bornsim.medium.random_medium import GaussianMedium as old_random
    from bornsim.medium.random import GaussianMedium
    from bornsim.medium.random_spheres import RandomSphereMedium as old_particles
    from bornsim.medium.particles import RandomSphereMedium

    assert historical is canonical is StructuredMedium
    assert old_random is GaussianMedium
    assert old_particles is RandomSphereMedium


def test_core_shell_regions_and_provenance():
    grid = Grid(shape=(9, 9, 9), spacing=1 * ureg.nanometer)

    medium = CoreShellMedium(
        background_refractive_index=1.3,
        core_refractive_index=1.5,
        shell_refractive_index=1.4,
        core_radius=1.1 * ureg.nanometer,
        outer_radius=3.1 * ureg.nanometer,
        centre=np.zeros(3) * ureg.nanometer,
    )

    volume = medium.to_volume(grid=grid)

    assert volume.refractive_index[4, 4, 4] == pytest.approx(1.5)
    assert volume.refractive_index[6, 4, 4] == pytest.approx(1.4)
    assert volume.refractive_index[8, 4, 4] == pytest.approx(1.3)
    assert volume.medium_metadata["model"] == "core_shell"


def test_ellipsoid_axes():
    grid = Grid(shape=(9, 9, 9), spacing=1 * ureg.nanometer)

    medium = EllipsoidMedium(
        background_refractive_index=1.3,
        particle_refractive_index=1.4,
        radii=np.array([3.1, 1.1, 1.1]) * ureg.nanometer,
        centre=np.zeros(3) * ureg.nanometer,
    )

    volume = medium.to_volume(grid=grid)

    assert volume.refractive_index[7, 4, 4] == pytest.approx(1.4)
    assert volume.refractive_index[4, 7, 4] == pytest.approx(1.3)


def test_cylinder_rod():
    grid = Grid(shape=(9, 9, 9), spacing=1 * ureg.nanometer)

    medium = CylinderMedium(
        background_refractive_index=1.3,
        particle_refractive_index=1.4,
        radius=1.1 * ureg.nanometer,
        height=5.1 * ureg.nanometer,
        centre=np.zeros(3) * ureg.nanometer,
        axis=2,
    )

    volume = medium.to_volume(grid=grid)

    assert volume.refractive_index[4, 4, 6] == pytest.approx(1.4)
    assert volume.refractive_index[4, 5, 4] == pytest.approx(1.4)
    assert volume.refractive_index[4, 6, 4] == pytest.approx(1.3)
    assert volume.medium_metadata["model"] == "cylinder"


def test_periodic_array_and_overlap():
    grid = Grid(shape=(9, 9, 9), spacing=1 * ureg.nanometer)

    sphere = Sphere(radius=0.6 * ureg.nanometer, refractive_index=1.4)

    medium = PeriodicMedium(
        background_refractive_index=1.3,
        motif=(sphere,),
        lattice_vectors=np.eye(3) * 4 * ureg.nanometer,
        repetitions=(2, 1, 1),
    )

    volume = medium.to_volume(grid=grid)

    expected = np.zeros(grid.shape)

    expected[2, 4, 4] = 0.1

    expected[6, 4, 4] = 0.1

    np.testing.assert_allclose(volume.delta_refractive_index, expected, atol=1e-15)

    assert volume.medium_metadata["repetitions"] == [2, 1, 1]


@pytest.mark.parametrize("radius", [0, 3.1, 4])
def test_invalid_core_radius(radius):
    with pytest.raises(ValueError):
        CoreShellMedium(
            background_refractive_index=1.3,
            core_refractive_index=1.5,
            shell_refractive_index=1.4,
            core_radius=radius * ureg.nanometer,
            outer_radius=3.1 * ureg.nanometer,
            centre=np.zeros(3) * ureg.nanometer,
        )
