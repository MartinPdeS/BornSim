"""
Core–shell and ellipsoidal particles
====================================

Compare finite particle geometries on the same voxel grid. Refractive-index
interfaces use voxel-centre sampling; refine the grid for quantitative studies.
The solver retains its linearized dielectric contrast for both particle types.
"""

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
    backend="matplotlib",
    mode="voxels",
    field="refractive_index",
)

core_shell_figure.suptitle("Core–shell sphere: 60 nm core, 120 nm outer radius")

plt.show()

# %%
# Inspect an oriented ellipsoid
# -----------------------------
# Rotation maps the local semi-axes to world coordinates. Here the long axis
# is tilted about y; no voxel-array rotation or interpolation is performed.
ellipsoid_volume = ellipsoid.to_volume(grid=grid)

ellipsoid_figure = ellipsoid_volume.plot_3d(
    backend="matplotlib",
    mode="voxels",
    field="refractive_index",
)

ellipsoid_figure.suptitle("Ellipsoid: 120 × 80 × 60 nm semi-axes, tilted 35°")

plt.show()

# %%
# Compare a central slice and scattering
# --------------------------------------
# The cross-sections show the absolute refractive index on the sampled grid.
# The scattering curves use the x-z observation meridian and cumulative first
# Born order. Differences reflect both geometry and material contrast; this
# comparison does not isolate shape alone or validate against an exact solver.
source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(
    source=source,
    order=1,
)

angles = np.linspace(0, 180, 121) * ureg.degree

figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")

half_extent = (np.asarray(grid.shape) * grid.spacing / 2).to("nanometer").magnitude

for panel, (label, volume) in enumerate((("Core–shell", core_shell_volume), ("Ellipsoid", ellipsoid_volume))):
    image = axes[panel].imshow(
        volume.refractive_index[:, grid.shape[1] // 2, :].T,
        origin="lower",
        extent=(-half_extent[0], half_extent[0], -half_extent[2], half_extent[2]),
        vmin=1.33,
        vmax=1.35,
    )

    axes[panel].set(xlabel="x (nm)", ylabel="z (nm)", title=label)

    figure.colorbar(image, ax=axes[panel], label="Refractive index")

    result = solver.solve_cut(
        target=volume,
        angles=angles,
    )

    cross_section = result.differential_cross_section.to("nanometer**2 / steradian")

    axes[2].plot(angles.to("degree").magnitude, cross_section.magnitude[-1], label=label)

axes[2].set(xlabel="Scattering angle (degrees)", ylabel="dσ/dΩ (nm² sr⁻¹)", title="First Born order")

axes[2].legend(frameon=False)

plt.show()
