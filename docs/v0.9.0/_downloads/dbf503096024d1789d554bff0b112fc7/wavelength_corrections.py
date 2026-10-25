"""
Wavelength dependence of higher-order corrections
=================================================

Keep one voxelized sphere fixed while changing vacuum wavelength.
Compare cumulative orders through fifth order using the same angular cut.
The material is nondispersive here: its refractive index is held fixed.
A small correction does not assess grid accuracy or the contrast approximation.
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
angles = np.linspace(0, 180, 91) * ureg.degree

figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")

for wavelength, axis in zip((450, 633, 850), axes, strict=True):
    source = Source(wavelength=wavelength * ureg.nanometer)

    solver = Solver(
        source=source,
        order=5,
    )

    result = solver.solve_cut(
        target=volume,
        angles=angles,
    )

    differential = result.differential_cross_section.to("nanometer**2 / steradian")

    for order in (1, 2, 3, 5):
        axis.plot(angles.magnitude, differential.magnitude[order - 1], label=f"Order {order}")

    axis.set(
        title=f"Vacuum wavelength {wavelength} nm",
        xlabel="Scattering angle (degrees)",
        ylabel="Cross section (nm² sr⁻¹)",
    )

    axis.legend()

plt.show()
