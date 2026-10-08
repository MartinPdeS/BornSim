"""
Gaussian and exponential spatial covariance
===========================================

Compare two covariance models with the same index standard deviation and
the same length parameter. Their definitions differ: Gaussian covariance is
``sigma_n**2 * exp(-r**2 / (2 * ell**2))``, while exponential covariance is
``sigma_n**2 * exp(-r / ell)``. Equal parameters therefore do not give identical
real-space profiles or scattering spectra.

Both random-volume generators use Gaussian probability distributions; the
covariance choice describes spatial correlations. The phase functions below
come from the analytical infinite-medium first-order model and are normalized
per steradian, not per degree.
"""

import matplotlib.pyplot as plt
import numpy as np

from bornsim import AngularSampling, AnalyticalMedium, Solver, Source
from bornsim.units import ureg

length = 120 * ureg.nanometer
separation = np.linspace(0, 500, 301) * ureg.nanometer
angles = np.linspace(0, 180, 181) * ureg.degree
solver = Solver(source=Source(wavelength=633 * ureg.nanometer))
sampling = AngularSampling(angles=angles)
figure, (covariance_axis, phase_axis) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")

for correlation, color in (("gaussian", "#0072B2"), ("exponential", "#D55E00")):
    medium = AnalyticalMedium(
        index_std=0.01,
        correlation_length=length,
        correlation=correlation,
    )
    ratio = (separation / length).to("dimensionless").magnitude
    normalized_covariance = np.exp(-(ratio**2) / 2) if correlation == "gaussian" else np.exp(-ratio)
    result = solver.solve(
        target=medium,
        sampling=sampling,
    )
    averaged = result.azimuth_average()
    covariance_axis.plot(
        separation.magnitude,
        normalized_covariance,
        color=color,
        label=correlation.capitalize(),
    )
    phase_axis.semilogy(
        angles.magnitude,
        averaged.phase_function.to("1 / steradian").magnitude[0],
        color=color,
        label=f"{correlation.capitalize()}, g = {result.g.magnitude[0]:.3f}",
    )
    print(f"{correlation}: μs = {result.mu_s[0].to('1 / millimeter')}")

covariance_axis.set(xlabel="Separation r (nm)", ylabel="Cn(r) / σn²", title="Spatial covariance")
phase_axis.set(xlabel="Scattering angle (degrees)", ylabel="Phase function p (sr⁻¹)", title="Directional scattering")
for axis in (covariance_axis, phase_axis):
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
figure.suptitle("Same ℓ = 120 nm and σn = 0.01; different covariance models")
plt.show()
