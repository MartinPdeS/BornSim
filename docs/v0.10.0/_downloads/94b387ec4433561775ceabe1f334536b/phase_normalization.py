"""
Phase density and polar-angle probability
=========================================

A phase function is a density per steradian. After explicit azimuth averaging,
2*pi*p(theta)*sin(theta) is the polar-angle probability density per radian.
Full solid-angle sampling supplies normalization separately at each Born order.
An angular cut alone cannot provide this normalization.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg

grid = Grid(
    shape=(12, 12, 12),
    spacing=20 * ureg.nanometer,
)

sphere = Sphere(
    radius=70 * ureg.nanometer,
    refractive_index=1.35,
)

medium = StructuredMedium()

medium.add_background(refractive_index=1.33)

medium.add_structures(sphere)

volume = medium.to_volume(grid=grid)

volume.plot_slice(title="Input sphere: central refractive index slice")

plt.show()

# %%
source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(
    source=source,
    order=3,
)

result = solver.solve(target=volume)

averaged = result.azimuth_average()

theta = averaged.angles.to("radian").magnitude

phase_density = averaged.phase_function.to("1 / steradian").magnitude

figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

for order in (1, 2, 3):
    polar_density = 2 * np.pi * phase_density[order - 1] * np.sin(theta)

    axes[0].plot(theta, phase_density[order - 1], label=f"Order {order}")

    axes[1].plot(theta, polar_density, label=f"Order {order}")

axes[0].set(ylabel="Phase density (sr⁻¹)")

axes[1].set(ylabel="Polar-angle density (rad⁻¹)")

for axis in axes:
    axis.set(xlabel="Polar angle (radians)")

    axis.legend()

plt.show()
