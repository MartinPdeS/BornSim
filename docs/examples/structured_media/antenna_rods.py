"""
Finite dielectric rods for antenna-like scatterers
==================================================

Use ``CylinderMedium`` to voxelize a finite dielectric rod. This is useful for
antenna-like geometry studies when a dielectric refractive-index model is
appropriate. Conducting currents, complex permittivity, and metal boundary
conditions require a different electromagnetic solver.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/antenna_rods.png"

import sys
import matplotlib.pyplot as plt
import numpy as np
from bornsim import CylinderMedium, Grid, Rotation, Solver, Source
from bornsim.units import ureg

grid = Grid(
    shape=(16, 16, 24),
    spacing=25 * ureg.nanometer,
)

tilt = Rotation(
    axis=(0, 1, 0),
    angle=18 * ureg.degree,
)

rod = CylinderMedium(
    background_refractive_index=1.33,
    particle_refractive_index=1.50,
    radius=55 * ureg.nanometer,
    height=450 * ureg.nanometer,
    centre=np.zeros(3) * ureg.nanometer,
    axis=2,
    rotation=tilt,
)

volume = rod.to_volume(grid=grid)

# %%
# Inspect the finite rod
# ----------------------
# The cylinder has a circular cross-section and a finite end face. Rotation
# changes the geometry coordinates before voxel-centre sampling.
figure = volume.plot_3d(
    backend="plotly",
    mode="volume",
    field="refractive_index",
)

figure.update_layout(title="Tilted finite dielectric rod")

if "--no-browser" not in sys.argv:
    figure.show(renderer="browser")

# %%
# %%
# Directional scattering
# ----------------------
# This finite rod is solved with the first Born order. The result describes
# dielectric scattering from the voxelized rod, not a conducting antenna.
source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(
    source=source,
    order=1,
)

angles = np.linspace(0, 180, 121) * ureg.degree

result = solver.solve_cut(
    target=volume,
    angles=angles,
)

cross_section = result.differential_cross_section.to("nanometer**2 / steradian")

scattering_figure, scattering_axis = plt.subplots(figsize=(7, 4), layout="constrained")

scattering_axis.plot(angles.to("degree").magnitude, cross_section.magnitude[-1])

scattering_axis.set(
    xlabel="Scattering angle (degrees)",
    ylabel="dσ/dΩ (nm² sr⁻¹)",
    title="Finite dielectric rod · first Born order",
)

plt.show()
