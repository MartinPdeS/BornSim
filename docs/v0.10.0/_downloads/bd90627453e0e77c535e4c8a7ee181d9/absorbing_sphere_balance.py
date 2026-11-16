"""
Absorbing material and an optical-theorem check
===============================================

Assign absorption to a reusable Material before voxelization. Compare forward
extinction from the optical theorem against scattering plus dissipation.
Finite-order Born approximations need not satisfy this balance exactly.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Directions, Grid, Material, Solver, Source, Sphere, StructuredMedium
from bornsim.units import ureg

material = Material(
    refractive_index=1.335,
    extinction_coefficient=0.0002,
)

sphere = Sphere(radius=36 * ureg.nanometer, material=material)

medium = StructuredMedium()

medium.add_background(refractive_index=1.33)

medium.add_structures(sphere)

grid = Grid(shape=(16, 16, 16), spacing=6 * ureg.nanometer)

volume = medium.to_volume(grid=grid)

source = Source(wavelength=633 * ureg.nanometer)

solver = Solver(source=source, order=4)

result = solver.solve(target=volume)

forward = Directions(vectors=[[0, 0, 1]])

cut = solver.solve(target=volume, sampling=forward)

amplitudes = np.cumsum(cut.amplitudes.to("meter").magnitude, axis=0)

forward_component = (amplitudes[:, 0, 0, 0] + amplitudes[:, 0, 1, 1]) / 2

host_wavenumber = 2 * np.pi * 1.33 / source.wavelength.to("meter").magnitude

extinction = 4 * np.pi / host_wavenumber * forward_component.imag

scattering = (result.mu_s * result.sample_volume).to("meter**2").magnitude

absorption = result.absorption_cross_section.to("meter**2").magnitude

orders = np.arange(1, 5)

figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

axes[0].plot(orders, extinction * 1e18, "o-", label="Forward optical theorem")

axes[0].plot(orders, (scattering + absorption) * 1e18, "s--", label="Scattering + absorption")

axes[0].set(xlabel="Highest Born order", ylabel="Cross section (nm²)", xticks=orders)

axes[0].legend()

relative_defect = abs(extinction - scattering - absorption) / (scattering + absorption)

axes[1].semilogy(orders, relative_defect, "o-")

axes[1].set(xlabel="Highest Born order", ylabel="Relative energy-balance defect", xticks=orders)

plt.show()

# %%
# Validation scope
# -----------------
# The automated Mie-reference tests additionally check a weak-contrast lossless
# sphere and an absorbing sphere against independently evaluated Mie-series
# cross sections and angular scattering. Those tests match occupied voxel
# volume to a sphere to reduce staircase-volume bias. They are not a universal
# accuracy bound. This example checks internal energy consistency; mesh and
# constitutive errors can remain even if the balance defect is small.
