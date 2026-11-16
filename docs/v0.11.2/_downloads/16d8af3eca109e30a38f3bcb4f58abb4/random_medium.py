"""
Visualize seeded three-dimensional random media
===============================================

Explore the full three-dimensional sample for each spatial covariance model.
Drag to rotate, scroll to zoom, and hover to inspect the refractive index.
Regions with lower refractive index are transparent; regions with higher
refractive index are more opaque.
Opacity is a display setting, not absorption. Histograms follow as statistical
checks of the same seeded fields; spatial structure is shown only in 3D.

Both models have Gaussian one-point probability distributions. Correlated
voxels in one finite sample are not independent observations, so a voxel
histogram need not match the ensemble Gaussian density closely. BornSim
does not recenter or rescale each realization to impose its mean or variance.
Finite voxel resolution and the cropped synthesis box also affect statistics.
"""

# sphinx_gallery_thumbnail_path = "_static/gallery_thumbnails/random_medium.png"

import sys
from bornsim.medium.random_medium import GaussianMedium, ExponentialMedium


import matplotlib.pyplot as plt
import numpy as np

from bornsim import Grid
from bornsim.units import ureg

grid_shape = (16, 16, 16)

voxel_spacing = 25 * ureg.nanometer

grid = Grid(
    shape=grid_shape,
    spacing=voxel_spacing,
)

refractive_index_std = 0.01

# Evaluate the ensemble probability density at these refractive-index fluctuations.
refractive_index_fluctuations = np.linspace(-4 * refractive_index_std, 4 * refractive_index_std, 301)

gaussian_probability_density = np.exp(-(refractive_index_fluctuations**2) / (2 * refractive_index_std**2)) / (
    np.sqrt(2 * np.pi) * refractive_index_std
)

volumes_by_covariance = {}

for covariance_model in ("gaussian", "exponential"):
    medium_type = {
        "gaussian": GaussianMedium,
        "exponential": ExponentialMedium,
    }[covariance_model]

    random_medium = medium_type(
        refractive_index_std=refractive_index_std,
        correlation_length=75 * ureg.nanometer,
        background_refractive_index=1.33,
    )

    sampled_volume = random_medium.to_volume(
        grid=grid,
        seed=42,
    )

    volumes_by_covariance[covariance_model] = sampled_volume

    print(
        f"{covariance_model}: sample mean = {sampled_volume.delta_refractive_index.mean():.4g}, sample std = {sampled_volume.delta_refractive_index.std():.4g}"
    )

# %%
# Gaussian voxel volume
# ----------------------
# View the sampled Gaussian field with Plotly. The contour colors represent
# absolute refractive index; opacity does not represent material absorption.
gaussian_volume = volumes_by_covariance["gaussian"]

gaussian_figure = gaussian_volume.plot_3d(
    field="refractive_index",
)

if "--no-browser" not in sys.argv:
    gaussian_figure.show(renderer="browser")

# %%
# Exponential voxel volume
# -------------------------
# Compare the exponential sample in its own Plotly view with the same
# rendering settings. Correlation length conventions differ between models.
exponential_volume = volumes_by_covariance["exponential"]

exponential_figure = exponential_volume.plot_3d(
    field="refractive_index",
)

if "--no-browser" not in sys.argv:
    exponential_figure.show(renderer="browser")

# %%
# Check one-point distributions
# -----------------------------
histogram_figure, histogram_axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

for covariance_column, covariance_model in enumerate(("gaussian", "exponential")):
    sampled_volume = volumes_by_covariance[covariance_model]

    histogram_axis = histogram_axes[covariance_column]

    histogram_axis.hist(
        sampled_volume.delta_refractive_index.ravel(), bins=30, density=True, alpha=0.6, label="One finite volume"
    )

    histogram_axis.plot(
        refractive_index_fluctuations,
        gaussian_probability_density,
        "k--",
        label="Ensemble Gaussian density",
    )

    histogram_axis.set(
        xlabel="Refractive-index fluctuation δn", ylabel="Probability density", title="One-point distribution"
    )

    histogram_axis.legend(frameon=False, loc="upper right", fontsize=9)

plt.show()
