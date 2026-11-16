"""
Explicit refinement studies and equation residuals
==================================================

Change one numerical control at a time. Grid studies below retain the same
physical sphere and box; extent studies retain voxel spacing and sphere size.
Statistical sample-size studies require independently assessing representative
volume effects and must not be confused with padding a deterministic sphere.
"""

import matplotlib.pyplot as plt

from bornsim import AngularSampling, ConvergenceReport, Grid, Material, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg

material = Material(refractive_index=1.335, extinction_coefficient=0.0005)

sphere = Sphere(radius=40 * ureg.nanometer, material=material)

medium = StructuredMedium()

medium.add_background(refractive_index=1.33)

medium.add_structures(sphere)

source = Source(wavelength=633 * ureg.nanometer)

base_grid = Grid(shape=(12, 12, 12), spacing=10 * ureg.nanometer)

base_volume = medium.to_volume(grid=base_grid)

# %%
# Born order: residual of the cumulative internal field
# ------------------------------------------------------
# residual_norms[M-1] is ||E_inc + K chi E_[M] - E_[M]|| / ||E_inc||.
# This checks the discretized equation, not the omitted quadratic contrast,
# staircase geometry or missing physical effects.
order_results = []

for order in (1, 2, 3, 4):
    solver = Solver(source=source, order=order)

    order_results.append(solver.solve(target=base_volume))

report = ConvergenceReport(results=tuple(order_results), labels=("Order 1", "Order 2", "Order 3", "Order 4"))

print(report.summary())

figure, axis = plt.subplots(layout="constrained")

axis.semilogy([1, 2, 3, 4], order_results[-1].residual_norms.magnitude, "o-")

axis.set(xlabel="Internal-field Born order", ylabel="Relative equation residual", xticks=[1, 2, 3, 4])

plt.show()

# %%
# Grid refinement at fixed box size
# -----------------------------------
solver = Solver(source=source, order=4)

grid_results = []

for count in (8, 12, 16):
    grid = Grid(shape=(count, count, count), spacing=120 / count * ureg.nanometer)

    volume = medium.to_volume(grid=grid)

    grid_results.append(solver.solve(target=volume))

grid_report = ConvergenceReport(results=tuple(grid_results), labels=("8 cubed", "12 cubed", "16 cubed"))

print(grid_report.summary())

# %%
# Angular quadrature refinement on a fixed grid
# -----------------------------------------------
angular_results = []

for samples in (16, 24, 32):
    sampling = AngularSampling(polar_samples=samples, azimuth_samples=samples)

    angular_results.append(solver.solve(target=base_volume, sampling=sampling))

angular_report = ConvergenceReport(results=tuple(angular_results), labels=("16 x 16", "24 x 24", "32 x 32"))

print(angular_report.summary())

# %%
# Finite box extent: compare cross sections, not box-normalized coefficients
# --------------------------------------------------------------------------
# For a fixed sphere, padding adds empty host voxels. mu_s and mu_a change
# because they are divided by box volume. Physical cross sections should be
# stable instead; this illustrates why a generic stability flag needs context.
for count in (12, 16):
    padded_grid = Grid(shape=(count, count, count), spacing=10 * ureg.nanometer)

    padded_volume = medium.to_volume(grid=padded_grid)

    result = solver.solve(target=padded_volume)

    scattering_area = (result.mu_s * result.sample_volume).to("nanometer**2")

    absorption_area = result.absorption_cross_section.to("nanometer**2")

    print(f"Box {count} cubed: scattering area = {scattering_area[-1]}, absorption area = {absorption_area[-1]}")
