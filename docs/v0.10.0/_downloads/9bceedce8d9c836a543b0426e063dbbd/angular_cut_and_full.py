"""
Angular cuts and full directional results
=========================================

Compare a meridian cut with the matching meridian of a full solve through
second order. The cut keeps complex amplitudes but has no integrated scattering
coefficient. Full quadrature additionally supplies finite-sample coefficients.
Changing observation sampling does not change the internal Born field.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import AngularSampling, Grid, Solver, Source, Sphere, StructuredMedium
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
    order=2,
)

sampling = AngularSampling(
    n_points=91,
    polar_samples=24,
    azimuth_samples=16,
)

full_result = solver.solve(
    target=volume,
    sampling=sampling,
)

cut_result = solver.solve_cut(
    target=volume,
    angles=full_result.angles,
)

meridian = full_result.meridian(azimuth=0 * ureg.degree)

relative_difference = np.max(abs(cut_result.differential.magnitude - meridian.differential.magnitude)) / np.max(
    abs(cut_result.differential.magnitude)
)

print(f"Maximum relative meridian difference: {relative_difference:.3e}")

print(f"Full-solve finite-sample scattering coefficient: {full_result.mu_s}")

cut_result.plot_cross_section()

plt.show()
