"""
Visualize seeded three-dimensional random media
===============================================

Explore the full three-dimensional sample for each spatial covariance model.
Drag to rotate, scroll to zoom, and hover to inspect the refractive index.
Low-index regions are transparent and stronger-index regions are more opaque.
Opacity is a display setting, not absorption. Histograms and central slices
follow as secondary checks of the same seeded fields.

Both models have Gaussian one-point probability distributions. Correlated
voxels in one finite sample are not independent observations, so a voxel
histogram need not match the ensemble Gaussian density closely. BornSim
does not recenter or rescale each realization to impose its mean or variance.
Finite voxel resolution and the cropped synthesis box also affect statistics.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid, RandomMedium
from bornsim.units import ureg

shape = (16, 16, 16)
spacing = 25 * ureg.nanometer
grid = Grid(
    shape=shape,
    spacing=spacing,
)
sigma = 0.01
half_side_nm = shape[0] * spacing.to("nanometer").magnitude / 2
extent = (-half_side_nm, half_side_nm, -half_side_nm, half_side_nm)
values = np.linspace(-4 * sigma, 4 * sigma, 301)
gaussian_density = np.exp(-(values**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
volumes = {}
for correlation in ("gaussian", "exponential"):
    medium = RandomMedium(
        index_std=sigma,
        correlation_length=75 * ureg.nanometer,
        correlation=correlation,
    )
    volume = medium.to_volume(
        grid=grid,
        seed=42,
    )
    volumes[correlation] = volume
    print(f"{correlation}: sample mean = {volume.delta_index.mean():.4g}, sample std = {volume.delta_index.std():.4g}")

# %%
# Gaussian volume with increasing opacity
# ---------------------------------------
# View the full Gaussian sample first. The opacity ramp uses this sample's
# index range; equal opacity across different samples need not mean equal index.
# Pass ``--no-browser`` to run without opening tabs, as the docs builder does.
volume = volumes["gaussian"]
gaussian_figure = volume.plot_3d(
    field="index",
    surface_count=16,
    opacity=0.2,
    opacity_scale="increasing",
)
if "--no-browser" not in sys.argv:
    gaussian_figure.show(renderer="browser")

# %%
# Exponential volume with increasing opacity
# ------------------------------------------
# Compare the exponential sample in its own interactive view with the same
# rendering settings. Correlation length conventions differ between models.
volume = volumes["exponential"]
exponential_figure = volume.plot_3d(
    field="index",
    surface_count=16,
    opacity=0.2,
    opacity_scale="increasing",
)
if "--no-browser" not in sys.argv:
    exponential_figure.show(renderer="browser")

# %%
# Secondary checks: central slices and one-point distributions
# ------------------------------------------------------------
# Slices show one sampled plane, while histograms describe the finite field's
# one-point distribution. Neither replaces the complete spatial volume above.
figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
for row, correlation in enumerate(("gaussian", "exponential")):
    volume = volumes[correlation]
    image_axis, histogram_axis = axes[row]
    image = image_axis.imshow(
        volume.delta_index[:, :, shape[2] // 2].T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-3 * sigma,
        vmax=3 * sigma,
    )
    z_nm = volume.positions[0, 0, shape[2] // 2, 2] * 1e9
    image_axis.set(xlabel="x (nm)", ylabel="y (nm)", title=f"{correlation.capitalize()} covariance; z = {z_nm:g} nm")
    figure.colorbar(image, ax=image_axis, label="Index fluctuation δn", extend="both")
    histogram_axis.hist(volume.delta_index.ravel(), bins=30, density=True, alpha=0.6, label="One finite volume")
    histogram_axis.plot(
        values,
        gaussian_density,
        "k--",
        label="Ensemble Gaussian density",
    )
    histogram_axis.set(xlabel="Index fluctuation δn", ylabel="Probability density", title="One-point distribution")
    histogram_axis.legend(frameon=False, loc="upper right", fontsize=9)
figure.suptitle("Seed 42; shared color scale; ensemble σn = 0.01")
plt.show()
