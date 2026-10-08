"""
Visualize seeded three-dimensional random media
===============================================

Inspect a central slice and a voxel histogram for each spatial covariance
model. A slice shows one plane of a three-dimensional volume. Its color gives
the dimensionless index fluctuation relative to the background index.

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
from bornsim.media import random_volume
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
figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")

for row, correlation in enumerate(("gaussian", "exponential")):
    medium = RandomMedium(
        index_std=sigma,
        correlation_length=75 * ureg.nanometer,
        correlation=correlation,
    )
    volume = random_volume(
        medium=medium,
        grid=grid,
        seed=42,
    )
    volumes[correlation] = volume
    image_axis, histogram_axis = axes[row]
    # Array axes are x, y, z. Transpose so image columns follow x.
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
    print(f"{correlation}: sample mean = {volume.delta_index.mean():.4g}, sample std = {volume.delta_index.std():.4g}")

figure.suptitle("Seed 42; shared color scale; ensemble σn = 0.01")
plt.show()

# %%
# Interactive volumes with index-dependent opacity
# ------------------------------------------------
# The browser views show the same samples as the slices above. Low-index
# regions are transparent and higher-index regions are more opaque. Opacity
# does not represent absorption. Drag to rotate and scroll to zoom.
# Plotly is installed with BornSim, and these browser views are included in
# the documentation. Close the first Matplotlib window to advance to these
# cells when running the script.
# Pass ``--no-browser`` to construct the figures without opening browser tabs;
# the documentation builder uses this option.
show_browser = "--no-browser" not in sys.argv

# %%
# Gaussian volume with increasing opacity
# ---------------------------------------
# The same Gaussian sample is displayed throughout its three-dimensional box.
#
# .. raw:: html
#
#    <iframe src="../../_static/random-medium-gaussian.html" title="Gaussian random medium with increasing index opacity" width="100%" height="560" loading="lazy" style="border:0;"></iframe>
volume = volumes["gaussian"]
gaussian_figure = volume.plot_3d(
    mode="volume",
    field="index",
    surface_count=16,
    opacity=0.2,
    opacity_scale="increasing",
)
if show_browser:
    gaussian_figure.show(renderer="browser")

# %%
# Exponential volume with increasing opacity
# ------------------------------------------
# Compare the exponential sample in its own cell with the same display settings.
#
# .. raw:: html
#
#    <iframe src="../../_static/random-medium-exponential.html" title="Exponential random medium with increasing index opacity" width="100%" height="560" loading="lazy" style="border:0;"></iframe>
volume = volumes["exponential"]
exponential_figure = volume.plot_3d(
    mode="volume",
    field="index",
    surface_count=16,
    opacity=0.2,
    opacity_scale="increasing",
)
if show_browser:
    exponential_figure.show(renderer="browser")

# %%
# Inspect the gaussian sample in 3D
# ---------------------------------
# Inspect the actual finite input sample with physical spatial axes.
# Matplotlib permits rotation with an interactive backend.
# Call plt.show() to display the figure; the gallery captures a static image.
medium_figure = volumes["gaussian"].plot_3d(
    backend="matplotlib",
    mode="slices",
    field="delta_index",
)
plt.show()
# %%
# Inspect the exponential sample in 3D
# ------------------------------------
# Inspect the actual finite input sample with physical spatial axes.
# Matplotlib permits rotation with an interactive backend.
# Call plt.show() to display the figure; the gallery captures a static image.
medium_figure = volumes["exponential"].plot_3d(
    backend="matplotlib",
    mode="slices",
    field="delta_index",
)
plt.show()
