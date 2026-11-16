"""
Oblique illumination and linear, circular, and elliptical polarization
======================================================================

Illuminate the same anisotropic dielectric particle with different plane waves.
Propagation and electric-field vectors are specified in laboratory coordinates.
A PolarizationState supplies a linear angle or complex Jones vector; omitting
it selects unpolarized illumination. LeftCircular uses (1, +i)/sqrt(2),
matching PyMieSim, with time convention exp(-i omega t).

The observation grid stays fixed in the laboratory frame. The plotted angle
is measured from laboratory +z, whereas anisotropy g uses the angle relative
to each incident beam. This third-order example retains the linearized
contrast and spherical Green self cell; it is not a general strong-scattering
solution. Refine geometry, quadrature, and Born order for quantitative work.
"""

import matplotlib.pyplot as plt

from bornsim import AngularSampling, Ellipsoid, Grid, LeftCircular, PolarizationState, Solver, Source, StructuredMedium
from bornsim.units import ureg

particle = Ellipsoid(
    radii=[80, 40, 60] * ureg.nanometer,
    refractive_index=1.35,
)

medium = StructuredMedium()

medium.add_background(refractive_index=1.33)

medium.add_structures(particle)

grid = Grid(
    shape=(12, 12, 12),
    spacing=20 * ureg.nanometer,
)

volume = medium.to_volume(grid=grid)

volume.plot_slice(title="Anisotropic particle: laboratory x-y slice")

plt.show()

# %%
# Define Jones states for an oblique incident beam
# -------------------------------------------------
# For direction (0.6, 0, 0.8), the real transverse basis is
# (0.8, 0, -0.6) and (0, 1, 0). Circular polarization combines them with
# equal magnitudes and a pi/2 relative phase; elliptical uses unequal magnitudes.
# Jones vectors are normalized by PolarizationState; no manual square roots
# or Cartesian projections are needed.
linear_state = PolarizationState(angle=90 * ureg.degree)

circular_state = LeftCircular()

elliptical_state = PolarizationState(jones_vector=[1, 0.5j])

unpolarized_source = Source(wavelength=633 * ureg.nanometer)

linear_source = Source(
    wavelength=633 * ureg.nanometer,
    incident_direction=[0.6, 0, 0.8],
    polarization=linear_state,
)

circular_source = Source(
    wavelength=633 * ureg.nanometer,
    incident_direction=[0.6, 0, 0.8],
    polarization=circular_state,
)

elliptical_source = Source(
    wavelength=633 * ureg.nanometer,
    incident_direction=[0.6, 0, 0.8],
    polarization=elliptical_state,
)

sampling = AngularSampling(
    n_points=91,
    polar_samples=24,
    azimuth_samples=16,
)

results = []

for label, source in (
    ("Unpolarized, +z", unpolarized_source),
    ("Linear, oblique", linear_source),
    ("Circular, oblique", circular_source),
    ("Elliptical, oblique", elliptical_source),
):
    solver = Solver(
        source=source,
        sampling=sampling,
        order=3,
    )

    result = solver.solve(target=volume)

    results.append((label, result))

    print(f"{label}: third-order cross section = {(result.mu_s[-1] * result.sample_volume).to('nanometer**2')}")

    print(f"{label}: third-order anisotropy relative to its beam = {result.g[-1]}")

# %%
# Compare one physical laboratory meridian
# -----------------------------------------
# This meridian is phi=0 for every source. It is not an azimuth average.
# For the oblique sources, the forward direction is theta=36.87 degrees.
figure, axis = plt.subplots(layout="constrained")

for label, result in results:
    cross_section = result.differential_cross_section.to("nanometer**2 / steradian")

    axis.plot(result.angles.to("degree").magnitude, cross_section.magnitude[-1, :, 0], label=label)

axis.set(
    xlabel="Laboratory polar angle (degrees), azimuth 0°",
    ylabel="Differential cross section (nm² sr⁻¹)",
    title="Same particle, third cumulative Born order",
)

axis.legend()

plt.show()

# %%
# Outgoing amplitudes for a pure polarization state
# ---------------------------------------------------
# Pure illumination retains one incident mode, rather than two unpolarized
# modes. The final axis holds laboratory x, y, z field components in metres.
# These complex amplitudes can be projected onto an outgoing transverse
# analyzer before squaring to study polarization-resolved detection.
print(f"Circular-source amplitude array shape: {results[2][1].amplitudes.shape}")
