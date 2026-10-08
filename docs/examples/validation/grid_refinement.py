"""
Grid refinement against an exact cube integral
==============================================

Refine a fixed physical cube rather than drawing a new random field on each
grid. Its first-order amplitude has an independent closed-form reference,
so this example isolates voxel midpoint-quadrature error. It does not test
higher-order Green-tensor discretization or random-field synthesis.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid, Solver, Source, Volume
from bornsim.units import ureg

source = Source(wavelength=633 * ureg.nanometer)
solver = Solver(
    source=source,
    order=1,
)
side = 180 * ureg.nanometer
background = 1.33
delta_index = 0.01
direction = np.array([[1.0, 0.0, 0.0]])
k0 = 2 * np.pi / source.wavelength.to("meter").magnitude
length = side.to("meter").magnitude
q = background * k0 * (np.array([0.0, 0.0, 1.0]) - direction[0])
integral = length**3 * np.prod(np.sinc(q * length / (2 * np.pi)))
# Linearized dielectric contrast; y polarization is transverse to +x.
reference = k0**2 * (2 * background * delta_index) * integral / (4 * np.pi)

cells = np.array([2, 4, 8, 16])
relative_errors = []
for count in cells:
    grid = Grid(
        shape=(count,) * 3,
        spacing=side / count,
    )
    volume = Volume(
        delta_index=np.full((count,) * 3, delta_index),
        grid=grid,
        background_index=background,
    )
    result = solver.solve_cut(
        target=volume,
        directions=direction,
    )
    amplitude = result.amplitudes.to("meter").magnitude[0, 0, 1, 1]
    relative_errors.append(abs(amplitude - reference) / abs(reference))

figure, axis = plt.subplots(layout="constrained")
axis.loglog(cells, relative_errors, "o-", label="First-order amplitude error")
axis.loglog(cells, relative_errors[0] * (cells[0] / cells) ** 2, "--", label="Second-order reference slope")
axis.set(xlabel="Cells per axis (fixed 180 nm cube)", ylabel="Relative amplitude error", title="Voxel refinement")
axis.grid(alpha=0.25, which="both")
axis.legend()
plt.show()

# %%
# Inspect the medium in 3D
# ------------------------
# Inspect the actual finite input sample with physical spatial axes.
# Matplotlib permits rotation with an interactive backend.
# Call plt.show() to display the figure; the gallery captures a static image.
# This is the finest grid. The index is constant throughout the sample;
# orthogonal slices show the uniform cube and its physical extent.
medium_figure = volume.plot_3d(
    mode="slices",
    field="index",
)
plt.show()
