"""
Realization count and sampling uncertainty
==========================================

Increase the number of independent volumes at a fixed grid, covariance, and
Born order. Curves show the estimated mean with one-standard-error bars.
One realization has unknown error, represented by NaN. Standard errors
measure sampling uncertainty, not voxel, quadrature, or finite-size error.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import EnsembleSampling, Grid, AngularSampling, RandomMedium, Solver, Source
from bornsim.units import ureg

grid = Grid(
    shape=(6, 6, 6),
    spacing=40 * ureg.nanometer,
)
medium = RandomMedium(
    correlation="gaussian",
    correlation_length=80 * ureg.nanometer,
)
solver = Solver(
    source=Source(wavelength=633 * ureg.nanometer),
    order=1,
)
angles = np.linspace(0, 180, 19) * ureg.degree
sampling = AngularSampling(
    angles=angles,
    polar_samples=16,
    azimuth_samples=4,
)
counts = [1, 2, 4, 8, 16, 32]
means = []
errors = []
figure, axis = plt.subplots(layout="constrained")
for count in counts:
    result = solver.ensemble(
        medium=medium,
        grid=grid,
        ensemble_sampling=EnsembleSampling(
            realizations=count,
            seed=42,
        ),
        sampling=sampling,
    )
    averaged = result.azimuth_average()
    values = averaged.differential.to("1 / meter / steradian").magnitude[0]
    stderr = averaged.stderr.to("1 / meter / steradian").magnitude[0]
    axis.errorbar(angles.magnitude, values, yerr=None if count == 1 else stderr, label=f"N = {count}")
    means.append(values[0])
    errors.append(stderr[0])
axis.set(
    xlabel="Scattering angle (degrees)",
    ylabel="Differential scattering (m⁻¹ sr⁻¹)",
    title="Finite-sample ensemble means",
)
axis.legend()
axis.grid(alpha=0.25)

# %%
# Sampling error at a fixed direction
# -----------------------------------
# These are nested ensembles, using the first N consecutive seeds. Their
# estimates are correlated across N. A realized error need not decrease
# monotonically; the N**(-1/2) guide describes independent-sample scaling.
figure, axis = plt.subplots(layout="constrained")
axis.loglog(counts[1:], errors[1:], "o-", label="Forward-scattering standard error")
axis.loglog(counts[1:], errors[-1] * np.sqrt(counts[-1] / np.array(counts[1:])), "--", label="N⁻¹ᐟ² guide")
axis.set(
    xlabel="Independent realizations N", ylabel="Standard error (m⁻¹ sr⁻¹)", title="Sampling uncertainty at θ = 0°"
)
axis.legend()
axis.grid(alpha=0.25, which="both")
plt.show()

# %%
# Inspect one ensemble realization in 3D
# --------------------------------------
# This is the first realization (seed 42) on the calculation grid,
# not an ensemble average or an infinite-medium material boundary.
# Matplotlib permits rotation with an interactive backend.
# Call plt.show() to display the figure; the gallery captures a static image.
preview_volume = medium.to_volume(
    grid=grid,
    seed=42,
)
medium_figure = preview_volume.plot_3d(
    backend="matplotlib",
    mode="slices",
    field="delta_index",
)
plt.show()
