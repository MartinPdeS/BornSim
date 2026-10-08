"""
Layers and scatterers with numerical random fluctuations
========================================================

Compose finite slabs and shaped inclusions using absolute refractive indices.
Later regions replace earlier ones where they overlap. Add a seeded
Whittle–Matérn fluctuation field on the same grid, then solve numerically.
The background outside the voxel box stays homogeneous. This is a finite
sample with edges, not an infinite planar stack calculation.

All Born orders retain the linearized contrast ``2*n0*delta_index``. The
smoothness convention is ``x=sqrt(2*nu)*r/ell``; probability statistics remain
Gaussian. A composed manual volume does not retain inferred generation
metadata: keep the ingredients and seed for reproducibility.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid, Layer, RandomMedium, Solver, Source, Sphere, StructuredMedium, Volume
from bornsim.media import random_volume
from bornsim.units import ureg

grid = Grid(
    shape=(16, 16, 16),
    spacing=25 * ureg.nanometer,
)
structure = StructuredMedium()
structure.add_background(index=1.33)
layer = Layer(
    lower=-200e-9,
    upper=0,
    index=1.335,
)

layer_2 = Layer(
    lower=0,
    upper=200e-9,
    index=1.34,
)

sphere = Sphere(
    radius=70e-9,
    index=1.345,
    centre=(0, 0, 50e-9),
)

structure.add_structures(
    layer,
    layer_2,
    sphere,
)
structured = structure.to_volume(
    grid=grid,
)
statistics = RandomMedium(
    index_std=0.002,
    correlation_length=50e-9,
    smoothness=1.5,
)
fluctuations = random_volume(
    medium=statistics,
    grid=grid,
    seed=42,
)
combined = Volume(
    delta_index=structured.delta_index + fluctuations.delta_index,
    grid=grid,
    background_index=structure.background_index,
)

# %%
# Inspect the structured medium in 3D
# -----------------------------------
# Inspect the deterministic layers and sphere before adding fluctuations.
# Drag to rotate and scroll to zoom in the embedded browser volume.
medium_figure = structured.plot_3d(
    mode="volume",
    field="index",
    opacity_scale="increasing",
)
if "--no-browser" not in sys.argv:
    medium_figure.show(renderer="browser")

# %%
# Inspect the combined medium in 3D
# ---------------------------------
# This interactive volume shows the sampled index used by the solver,
# including the same seeded Matérn fluctuations as the scattering curves.
# Higher-index regions are more opaque; opacity is a display setting.
medium_figure = combined.plot_3d(
    mode="volume",
    field="index",
    opacity_scale="increasing",
)
if "--no-browser" not in sys.argv:
    medium_figure.show(renderer="browser")

# %%
# Scattering curves and a diagnostic cross-section
# --------------------------------------------------
# The slice checks exact voxel values after exploring the full volumes above.
figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
image = axes[0].imshow(
    (combined.background_index + combined.delta_index[:, grid.shape[1] // 2, :]).T,
    origin="lower",
    extent=(-200, 200, -200, 200),
    cmap="viridis",
)
axes[0].set(xlabel="x (nm)", ylabel="z (nm)", title="Layers, sphere and Matérn fluctuations")
figure.colorbar(image, ax=axes[0], label="Refractive index")
solver = Solver(
    source=Source(wavelength=633 * ureg.nanometer),
    order=3,
)

result = solver.solve_cut(
    target=combined,
    angles=np.linspace(0, 180, 91) * ureg.degree,
)
cross_section = result.differential_cross_section.to("nanometer**2 / steradian")
for order, curve in enumerate(cross_section.magnitude, start=1):
    axes[1].plot(
        result.angles.to("degree").magnitude,
        curve,
        label=f"Through order {order}",
    )
axes[1].set(xlabel="Scattering angle (degrees)", ylabel="dσ/dΩ (nm² sr⁻¹)")
axes[1].legend(frameon=False)
plt.show()
