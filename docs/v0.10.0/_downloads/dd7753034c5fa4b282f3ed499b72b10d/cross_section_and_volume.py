"""
Cross sections and box-normalized coefficients
==============================================

Pad one fixed physical sphere with background voxels at fixed spacing.
The box-normalized scattering coefficient changes with sample volume, while
its product with that volume is the scattering cross section. Padding tests
representation and normalization, not an infinite-medium transport limit.
"""

import matplotlib.pyplot as plt

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
    order=4,
)

figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

for count in (10, 12, 16):
    grid = Grid(
        shape=(count, count, count),
        spacing=20 * ureg.nanometer,
    )

    volume = medium.to_volume(grid=grid)

    result = solver.solve(target=volume)

    scattering_area = (result.mu_s * result.sample_volume).to("nanometer**2")

    axes[0].plot(range(1, 5), result.mu_s.to("1 / meter").magnitude, "o-", label=f"{count}³ box")

    axes[1].plot(range(1, 5), scattering_area.magnitude, "o-", label=f"{count}³ box")

    print(f"Box {count}³, fourth-order scattering cross section: {scattering_area[-1]}")

axes[0].set(ylabel="Box-normalized coefficient (m⁻¹)")

axes[1].set(ylabel="Scattering cross section (nm²)")

for axis in axes:
    axis.set(xlabel="Cumulative Born order", xticks=range(1, 5))

    axis.legend()

plt.show()
