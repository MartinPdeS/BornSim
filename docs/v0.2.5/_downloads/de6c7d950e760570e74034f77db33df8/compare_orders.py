"""
Compare numerical Born orders
=============================

Compare finite-sample effective scattering coefficients through order three.
The numerical ensemble retains amplitude interference and sampling errors.
Integrated coefficients describe finite samples rather than an infinite
medium. Error bars describe realization sampling, not discretization error.
Field-term diagnostics alone do not certify convergence.
"""

import sys

import matplotlib.pyplot as plt

from bornsim import EnsembleSampling, Grid, RandomMedium, Solver, Source
from bornsim.units import ureg

grid = Grid(
    shape=(8, 8, 8),
    spacing=50 * ureg.nanometer,
)
medium = RandomMedium(
    correlation="gaussian",
    background_index=1.33,
    index_std=0.01,
    correlation_length=100 * ureg.nanometer,
)
ensemble_solver = Solver(
    source=Source(wavelength=633 * ureg.nanometer),
    order=3,
)

result = ensemble_solver.ensemble(
    medium=medium,
    grid=grid,
    ensemble_sampling=EnsembleSampling(
        realizations=4,
        seed=42,
    ),
)
for index, coefficient in enumerate(result.mu_s.to("1 / meter").magnitude, start=1):
    print(f"Through order {index}: effective μs = {coefficient:.5g} m⁻¹")
result.plot()
plt.show()

# %%
# Field-term diagnostics
# ----------------------
# Compare field terms separately from the scattering curves. Decreasing
# terms alone do not certify convergence of the Born series.
result.plot_field_norms()
plt.show()

# %%
# Inspect one ensemble realization in 3D
# --------------------------------------
# This is the first realization (seed 42) on the calculation grid,
# not an ensemble average or an infinite-medium material boundary.
# Drag to rotate and scroll to zoom in the embedded browser view.
# Higher-index regions are more opaque; opacity is not absorption.
preview_volume = medium.to_volume(
    grid=grid,
    seed=42,
)
medium_figure = preview_volume.plot_3d(
    mode="volume",
    field="index",
    opacity_scale="increasing",
)
if "--no-browser" not in sys.argv:
    medium_figure.show(renderer="browser")
