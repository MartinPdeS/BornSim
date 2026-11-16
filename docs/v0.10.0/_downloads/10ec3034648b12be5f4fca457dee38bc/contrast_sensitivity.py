"""
Contrast sensitivity of successive Born corrections
===================================================

Compare positive and negative refractive index offsets at fixed geometry.
The first intensity is even in contrast; higher orders can break that symmetry.
All orders use linearized dielectric contrast and the spherical Green self cell.
These small examples illustrate corrections, not a universal convergence bound.
"""

import matplotlib.pyplot as plt

from bornsim import Grid, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg

grid = Grid(
    shape=(12, 12, 12),
    spacing=20 * ureg.nanometer,
)

# %%
source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(
    source=source,
    order=4,
)

scattering_areas = []

for delta_refractive_index in (-0.04, -0.02, 0.02, 0.04):
    sphere = Sphere(
        radius=70 * ureg.nanometer,
        refractive_index=1.33 + delta_refractive_index,
    )

    medium = StructuredMedium()

    medium.add_background(refractive_index=1.33)

    medium.add_structures(sphere)

    volume = medium.to_volume(grid=grid)

    volume.plot_slice(title=f"Refractive index offset {delta_refractive_index:+.2f}")

    plt.show()

    result = solver.solve(target=volume)

    scattering_area = (result.mu_s * result.sample_volume).to("nanometer**2")

    scattering_areas.append(scattering_area)

# %%
# Compare cumulative cross sections on the same scale.
figure, axis = plt.subplots(layout="constrained")

for delta_refractive_index, scattering_area in zip((-0.04, -0.02, 0.02, 0.04), scattering_areas, strict=True):
    axis.plot(range(1, 5), scattering_area.magnitude, "o-", label=f"Offset {delta_refractive_index:+.2f}")

axis.set(xlabel="Cumulative Born order", ylabel="Scattering cross section (nm²)", xticks=range(1, 5))

axis.legend()

plt.show()
