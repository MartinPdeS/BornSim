"""
Save results and inspect calculation provenance
===============================================

Save a numerical result and restore it with its physical units, complex
amplitudes, diagnostics, and original calculation settings. The versioned
NumPy archive uses JSON metadata and loads without pickle.

This example uses a temporary directory so running the gallery leaves no
data files behind. For a persistent archive, pass your desired filename to
``result.save(path="scattering.npz")``. Archives contain results rather than the
original input voxel field. Retain manually supplied fields separately;
generated-volume provenance records the medium and seed, with a field hash
to identify the exact input. Exact regeneration depends on software versions.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid, RandomMedium, Result, Solver, Source
from bornsim.media import random_volume
from bornsim.units import ureg

grid = Grid(
    shape=(4, 4, 4),
    spacing=40 * ureg.nanometer,
)
medium = RandomMedium(
    correlation="gaussian",
    correlation_length=80 * ureg.nanometer,
)
volume = random_volume(
    medium=medium,
    grid=grid,
    seed=42,
)
solver = Solver(
    source=Source(wavelength=633 * ureg.nanometer),
    order=3,
)

result = solver.solve(target=volume)

with TemporaryDirectory() as directory:
    path = result.save(path=Path(directory) / "scattering.npz")
    restored = Result.load(path=path)

# %%
# Inspect settings and retained amplitudes
# ----------------------------------------
# The arrays have been copied from the archive; they remain usable after the
# temporary directory is removed. Loading preserves the recorded versions.
print("BornSim version:", restored.provenance["bornsim_version"])
print("NumPy version:", restored.provenance["numpy_version"])
print("Seed:", restored.provenance["seed"])
print("Grid:", restored.provenance["grid"])
print("Medium:", restored.provenance["medium"])
print("Coefficient scope:", restored.provenance["coefficient_scope"])
print("Wavelength:", restored.source.wavelength.to("nanometer"))
print("Complex amplitudes retained:", np.iscomplexobj(restored.amplitudes.magnitude))

# %%
# Plot restored scattering and field diagnostics
# ----------------------------------------------
# Full angular sampling supplies normalized phase functions and finite-sample
# coefficients. They are not infinite-medium material transport coefficients.
restored.plot()
restored.plot_field_norms()
plt.show()

# %%
# Inspect the medium in 3D
# ------------------------
# Inspect the actual finite input sample with physical spatial axes.
# Matplotlib permits rotation with an interactive backend.
# Call plt.show() to display the figure; the gallery captures a static image.
medium_figure = volume.plot_3d(
    mode="slices",
    field="delta_index",
)
plt.show()
