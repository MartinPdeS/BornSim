"""
Random nonoverlapping spheres
=============================

Generate a reproducible collection of identical spheres in a uniform background.
Sphere centres are proposed uniformly within the part of the box that keeps each
sphere fully inside. Overlapping proposals are rejected; the generator raises an
error if it cannot place every requested sphere within its attempt limit.
This is sequential placement, rather than an equilibrium hard-sphere model.

The plotted medium samples material at voxel centres. Choose a spacing fine enough
to resolve each sphere, and use small refractive index contrast for Born scattering.
"""

import matplotlib.pyplot as plt
from bornsim import Grid
from bornsim.medium.random_spheres import RandomSphereMedium
from bornsim.units import ureg


grid = Grid(
    shape=(16, 16, 16),
    spacing=25 * ureg.nanometer,
)

medium = RandomSphereMedium(
    background_refractive_index=1.33,
    sphere_refractive_index=1.35,
    radius=40 * ureg.nanometer,
    sphere_count=12,
)

volume = medium.to_volume(
    grid=grid,
    seed=42,
)

figure = volume.plot_3d(
    backend="matplotlib",
    mode="voxels",
    field="refractive_index",
)

print(f"Generated {medium.sphere_count} nonoverlapping spheres with radius {medium.radius:~P}.")

plt.show()
