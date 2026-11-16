"""
Core–shell and ellipsoidal particles
====================================

Compare finite particle geometries on the same voxel grid. Refractive-index
interfaces use voxel-centre sampling; refine the grid for quantitative studies.
The solver retains its linearized dielectric contrast for both particle types.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from bornsim import CoreShellMedium, EllipsoidMedium, Grid, Solver, Source, Rotation
from bornsim.units import ureg

grid = Grid(
    shape=(16, 16, 16),
    spacing=25 * ureg.nanometer,
)

core_shell = CoreShellMedium(
    background_refractive_index=1.33,
    core_refractive_index=1.35,
    shell_refractive_index=1.34,
    core_radius=60 * ureg.nanometer,
    outer_radius=120 * ureg.nanometer,
    centre=np.zeros(3) * ureg.nanometer,
)

rotation = Rotation(
    axis=(0, 1, 0),
    angle=35 * ureg.degree,
)

ellipsoid = EllipsoidMedium(
    background_refractive_index=1.33,
    particle_refractive_index=1.34,
    radii=np.array([120, 80, 60]) * ureg.nanometer,
    centre=np.zeros(3) * ureg.nanometer,
    rotation=rotation,
)

# %%
# Inspect the core and shell
# --------------------------
# The core replaces the shell inside core_radius; both refractive indices
# are absolute values relative to the same uniform exterior. The shell is
# sampled at voxel centres, so its thickness must be resolved by the grid.
core_shell_volume = core_shell.to_volume(grid=grid)

core_shell_figure = core_shell_volume.plot_3d(
    backend="plotly",
    mode="volume",
    field="refractive_index",
)

core_shell_figure.update_layout(title="Core–shell sphere: 60 nm core, 120 nm outer radius")

if "--no-browser" not in sys.argv:
    core_shell_figure.show(renderer="browser")

# %%
# Inspect an oriented ellipsoid
# -----------------------------
# Rotation maps the local semi-axes to world coordinates. Here the long axis
# is tilted about y; no voxel-array rotation or interpolation is performed.
ellipsoid_volume = ellipsoid.to_volume(grid=grid)

ellipsoid_figure = ellipsoid_volume.plot_3d(
    backend="plotly",
    mode="volume",
    field="refractive_index",
)

ellipsoid_figure.update_layout(title="Ellipsoid: 120 × 80 × 60 nm semi-axes, tilted 35°")

if "--no-browser" not in sys.argv:
    ellipsoid_figure.show(renderer="browser")

# %%
# Compare scattering
# ------------------
# The curves use the x-z observation meridian and cumulative first Born order.
# Differences reflect both geometry and material contrast; this
# comparison does not isolate shape alone or validate against an exact solver.
source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(
    source=source,
    order=1,
)

angles = np.linspace(0, 180, 121) * ureg.degree

figure, axis = plt.subplots(layout="constrained")

for label, volume in (("Core–shell", core_shell_volume), ("Ellipsoid", ellipsoid_volume)):
    result = solver.solve_cut(
        target=volume,
        angles=angles,
    )

    cross_section = result.differential_cross_section.to("nanometer**2 / steradian")

    axis.plot(angles.to("degree").magnitude, cross_section.magnitude[-1], label=label)

axis.set(xlabel="Scattering angle (degrees)", ylabel="dσ/dΩ (nm² sr⁻¹)", title="First Born order")

axis.legend(frameon=False)

plt.show()
