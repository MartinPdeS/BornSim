"""
Finite samples and analytical first-order scattering
====================================================

Compare first-order random-volume ensembles with the infinite-medium
analytical model at the same wavelength and covariance. Increase sample size
at a fixed voxel spacing. These numerical coefficients are cross sections
divided by sample volume; they are not automatically intrinsic transport
coefficients. A finite window and finite synthesis box change the spectrum,
and voxel resolution remains a separate source of error.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import EnsembleSampling, AngularSampling, Grid, AnalyticalMedium, Solver, Source
from bornsim.units import ureg

medium = AnalyticalMedium(
    index_std=0.001,
    correlation_length=60 * ureg.nanometer,
)
solver = Solver(
    source=Source(wavelength=633 * ureg.nanometer),
    order=1,
)
angles = np.linspace(0, 180, 19) * ureg.degree
sampling = AngularSampling(
    angles=angles,
    polar_samples=32,
    azimuth_samples=8,
)
analytical = solver.solve(
    target=medium,
    sampling=sampling,
)
figure, axis = plt.subplots(layout="constrained")
axis.plot(
    angles.magnitude,
    analytical.azimuth_average().differential.to("1 / meter / steradian").magnitude[0],
    "k--",
    linewidth=2,
    label="Analytical infinite medium, first order",
)
for cells in (4, 8, 12):
    grid = Grid(
        shape=(cells,) * 3,
        spacing=30 * ureg.nanometer,
    )
    result = solver.ensemble(
        medium=medium,
        grid=grid,
        ensemble_sampling=EnsembleSampling(
            realizations=8,
            seed=42,
        ),
        sampling=sampling,
    )
    averaged = result.azimuth_average()
    axis.errorbar(
        angles.magnitude,
        averaged.differential.to("1 / meter / steradian").magnitude[0],
        yerr=averaged.stderr.to("1 / meter / steradian").magnitude[0],
        label=f"Finite cube: {cells * 30} nm side, 8 realizations",
    )
    print(f"{cells * 30} nm cube: μs = {result.mu_s[0]}, g = {result.g[0]}")
axis.set(
    xlabel="Scattering angle (degrees)",
    ylabel="Differential scattering (m⁻¹ sr⁻¹)",
    title="First order: finite samples versus infinite medium",
)
axis.legend()
axis.grid(alpha=0.25)
plt.show()

# %%
# Inspect one ensemble realization in 3D
# --------------------------------------
# This is the first realization (seed 42) on the calculation grid,
# not an ensemble average or an infinite-medium material boundary.
# Matplotlib permits rotation with an interactive backend.
# Call plt.show() to display the figure; the gallery captures a static image.
# Use the largest of the three finite cubes compared above.
preview_volume = medium.to_volume(
    grid=grid,
    seed=42,
)
medium_figure = preview_volume.plot_3d(
    mode="slices",
    field="delta_index",
)
plt.show()
